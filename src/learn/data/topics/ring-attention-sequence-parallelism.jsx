import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const ringAttentionContent = {
  title: "Ring Attention & Sequence Parallelism",
  readTime: "~38 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        By 2023, the long-context arms race had run into a wall that no algorithmic trick on a single GPU could break through. FlashAttention had already pushed quadratic attention into a memory-linear forward pass on a single device — but linear in {"L"}, with a constant that included the full {"K"} and {"V"} tensors at fp16, plus query, output, and intermediate accumulators, plus the model weights and optimizer states. At {"L = 1{,}000{,}000"}, an 80 GB H100 cannot hold the activation footprint of even a single attention layer, regardless of how cleverly the softmax is tiled. The bottleneck is not the algorithm; it is that one device cannot store enough state to span a million-token sequence. The problem is fundamentally a hardware capacity problem disguised as a software problem.
      </Prose>

      <Prose>
        The fix is to partition the sequence dimension across multiple devices and arrange the communication so that every query still gets to see every key and value. Hao Liu, Matei Zaharia, and Pieter Abbeel published "Ring Attention with Blockwise Transformers for Near-Infinite Context" at NeurIPS 2023 (arXiv:2310.01889) with the central observation: if you slice the sequence into {"N"} contiguous chunks and place chunk {"i"} on device {"i"}, you can compute full attention by rotating the {"K"} and {"V"} chunks around the ring of devices for {"N"} iterations. Each device holds {"1/N"} of the activations at any moment, total per-device memory is {"O(L/N)"}, and the wall-clock cost is {"N"} attention blocks per device — the same total compute as single-GPU attention, but with the communication of {"N"} ring rotations interleaved into it. With perfect overlap, the ring rotations are free.
      </Prose>

      <Prose>
        The same insight surfaced almost simultaneously from a different direction. Sam Ainsworth and the Megatron-LM team at NVIDIA had been pushing on activation memory for the LLM training pipeline, and Korthikanti et al. published "Reducing Activation Recomputation in Large Transformer Models" at MLSys 2023 (arXiv:2205.05198), which introduced sequence parallelism as a complement to tensor parallelism: shard the input of LayerNorm and Dropout layers along the sequence dimension to cut activation memory without changing the parallelism inside attention or MLP blocks. Sam Jacobs and the DeepSpeed team at Microsoft generalised the idea inside attention itself with "DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models" (arXiv:2309.14509, 2023), using all-to-all collectives to swap between sequence-sharding and head-sharding so that long contexts become tractable without rewriting the attention kernel.
      </Prose>

      <Prose>
        Brandon, Nguyen, and Zhang's "Striped Attention: Faster Ring Attention for Causal Transformers" (arXiv:2311.09431, 2023) added the missing piece for production decoders. Plain ring attention with contiguous chunking is severely load-imbalanced under a causal mask: the device holding the last chunk does roughly {"N"} times more work than the device holding the first. Striping the sequence — interleaving so that token {"t"} lands on device {"t \\bmod N"} — distributes the causal mask uniformly across the ring, eliminating the straggler. By the end of 2023 the recipe was complete: ring attention for the long sequence, striping for the causal mask, blockwise tiling for the on-device softmax, and tensor / pipeline parallelism layered on top for the rest of the model.
      </Prose>

      <Prose>
        The production fingerprints are everywhere. Meta's "The Llama 3 Herd of Models" paper (arXiv:2407.21783, 2024) describes Llama-3.1 405B trained at 128k context with a 4D parallelism (data + tensor + pipeline + sequence) and explicitly cites context-parallel attention. Gemini's 1M+ token context (Google DeepMind, 2024) is widely understood to use ring or striped variants — the published architecture describes "many millions of tokens of context" with a "novel efficient attention scheme distributed across devices". Claude's 200k context, GPT-4 long context, and the open-weight long-context fine-tunes from Together AI, Yi, and Qwen all use {"ring-flash-attention"} or DeepSpeed Ulysses style sequence parallelism in their training and serving stacks. PyTorch 2.5+ ships {"DTensor"} sharding primitives that make sequence-parallel attention a first-class transformation. Ring attention is not a research trick anymore; it is the load-bearing wall under the long-context era.
      </Prose>

      <Callout accent="gold">
        Ring attention is what unlocked the move from 4k–32k contexts in 2022 to 128k–10M contexts in 2024. The single-device memory ceiling on attention activations is hard; the ring's per-device memory is {"O(L/N)"}, so adding GPUs adds context length, near-linearly, until communication becomes the bottleneck.
      </Callout>

      <Prose>
        This topic builds the ring attention algorithm from scratch with a single-process simulation that verifies bit-equality with single-shot attention, walks through the online-softmax accumulator that lets partial outputs combine across iterations, derives the compute / communication overlap conditions that determine when ring attention scales linearly versus when it goes communication-bound, and maps the production landscape: when to use ring versus Ulysses, how to combine with tensor parallelism, and the failure modes (causal-mask straggler, fp16 accumulator drift, NCCL ring topology mismatch) that show up in real training runs.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <H3>2.1 The single-device memory ceiling</H3>

      <Prose>
        Attention's memory footprint per layer at training time is dominated by three tensors: the query, key, and value projections, each {"L × d_{model}"} half-precision floats, plus the attention output. For a single sequence of length {"L = 128{,}000"} with {"d_{model} = 8192"} (a Llama-2-70B-class hidden size), each tensor is {"128k × 8192 × 2 \\text{ B} \\approx 2 \\text{ GB}"}. With {"Q, K, V"} and the output, you are at 8 GB per layer just for attention activations, and a 70B-class model has 80 layers. Even with activation checkpointing, a single 80 GB H100 cannot hold the forward activations of one such layer at {"L = 1\\text{M}"}. The model weights and optimizer state are not the bottleneck — the activations are.
      </Prose>

      <Prose>
        FlashAttention solves this on a single device by tiling: it never materialises the full {"L × L"} attention matrix, processes {"Q"} in row blocks while streaming {"K, V"} from HBM, and uses an online softmax to combine block contributions. The peak activation is reduced from {"O(L^2)"} to {"O(L \\cdot d_{head})"}, but you still need to hold the full {"K, V"} on device. At {"L = 1\\text{M}"} that is the problem: even {"L \\cdot d_{head}"} per head, summed across heads, exceeds device memory. Single-device tiling has nothing left to give.
      </Prose>

      <H3>2.2 The ring as a token-ring network</H3>

      <Prose>
        Ring attention partitions the sequence dimension across {"N"} devices arranged in a logical ring (device 0 talks to device 1, 1 to 2, ..., {"N-1"} back to 0). Each device {"i"} permanently owns its slice of {"Q_i, K_i, V_i"}, each of shape {"(L/N) \\times d_{head}"}. To compute full attention, every query in {"Q_i"} must see every {"K_j, V_j"}. The ring achieves this by rotating the held {"K, V"} blocks around the ring: in iteration {"j"}, device {"i"} computes a partial attention output of {"Q_i"} against {"K_{(i-j) \\bmod N}, V_{(i-j) \\bmod N}"} — the {"K, V"} block that has been forwarded {"j"} hops from its origin. After {"N"} iterations every device has seen every {"K, V"} block exactly once.
      </Prose>

      <Prose>
        The shape of this is exactly an old idea — token-ring networks, the systolic arrays of the 1980s, all-reduce on a ring topology — applied to attention. The reason it works for attention specifically is that the operation decomposes block-additively: the contribution of {"K_j, V_j"} to {"Q_i"}'s output can be computed independently of the contribution of {"K_{j'}, V_{j'}"} as long as you keep the right normalisation state. That state is the running max and running denominator of the softmax, which is exactly what FlashAttention's tile-level online softmax was already designed to maintain. Ring attention is FlashAttention extended across devices — the same tile combination math, but the tiles are scattered across {"N"} GPUs and arrive over the network instead of from HBM.
      </Prose>

      <Callout accent="purple">
        Ring attention is FlashAttention with the tiles distributed. The softmax block-combine math is identical; the only new thing is that the next tile arrives via NCCL send/recv instead of via {"cp.async"} from HBM.
      </Callout>

      <H3>2.3 Compute, communication, and the overlap condition</H3>

      <Prose>
        Each device does {"N"} attention block computations and {"N - 1"} ring rotations per layer per forward pass. The attention block compute scales as {"(L/N)^2 \\cdot d_{head} \\cdot H"} per block, so total compute per device is {"N \\cdot (L/N)^2 \\cdot d_{head} \\cdot H = L^2 \\cdot d_{head} \\cdot H / N"} — exactly {"1/N"} of the single-device cost, as expected. The ring rotation moves {"2 \\cdot (L/N) \\cdot d_{head} \\cdot H"} bytes per hop ({"K"} and {"V"} blocks), so total bytes moved per device across {"N - 1"} rotations is {"\\approx 2 L d_{head} H"} — independent of {"N"}.
      </Prose>

      <Prose>
        With careful scheduling, each ring rotation overlaps with the next attention block's compute. Wall-clock per layer is then bounded by the larger of the per-iteration compute and per-iteration communication, repeated {"N"} times. As long as compute per iteration exceeds communication per iteration, the wall-clock is compute-bound: you pay only the {"1/N"} compute reduction with effectively zero communication tax. The transition happens when {"L/N"} drops low enough that the per-iteration FLOP count fits in less time than {"2 (L/N) d_{head} H / \\text{bandwidth}"}. For NVLink at 450 GB/s and H100 fp16 throughput, this transition occurs at chunk sizes around {"\\sim 1k–4k"} tokens; smaller chunks go communication-bound.
      </Prose>

      <H3>2.4 Two flavours: ring and Ulysses</H3>

      <Prose>
        Ring attention is one way to shard the sequence dimension. DeepSpeed Ulysses takes a different approach: rather than rotating {"K, V"} around a ring while each device keeps its {"Q"} slice, Ulysses uses a pair of all-to-all collectives that re-shard the data between sequence-sharded and head-sharded layouts. Before attention, an all-to-all moves data from "each device has 1/N of the sequence, all heads" to "each device has the full sequence, 1/N of the heads"; attention runs as normal on each head subset; a second all-to-all reverses the layout. The communication volume is {"O(L \\cdot d_{model})"} per all-to-all, not {"O(L \\cdot d_{model})"} ring-distributed.
      </Prose>

      <Prose>
        The trade-off: Ulysses needs the full {"L"} sequence to fit on a single device during the head-sharded attention computation, so it caps at the same single-device memory limit FlashAttention does, just with more heads' worth of bandwidth amortised. Ring attention does not have this cap — its per-device memory really is {"O(L/N)"} all the way through. For contexts that exceed single-device memory ({"\\gtrsim"} 256k on H100, depending on model size), ring attention is the only option; for shorter contexts where Ulysses fits, Ulysses is often faster because all-to-all on NVLink is more bandwidth-efficient than ring rotations.
      </Prose>

      <H3>2.5 Striped attention for causal masks</H3>

      <Prose>
        The vanilla ring with contiguous chunking — device {"i"} owns positions {"[iL/N, (i+1)L/N)"} — is uniform under bidirectional attention but disastrous under a causal mask. Device 0's queries can only attend to device 0's keys (positions 0 to {"L/N - 1"}). Device {"N - 1"}'s queries attend to all keys. The ratio of work between the busiest and the idlest device is {"N"}; in a synchronous ring iteration, the slowest device sets the pace. Striping fixes this by interleaving: token {"t"} goes to device {"t \\bmod N"}. After this transformation, every device's query slice and every {"K, V"} chunk contain a uniform sample of positions across the whole sequence, so the causal mask falls roughly evenly on each ring iteration. The ratio of work across devices drops from {"N"}-to-1 to roughly {"\\sqrt{N}"}-to-1 in the worst case and {"\\approx 1.4{\\times}"} for typical configurations.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 Single-device attention as a tile sum</H3>

      <Prose>
        Standard scaled dot-product attention for a single head, dropping the head index, is
      </Prose>

      <MathBlock>
        {"\\mathrm{Attn}(Q, K, V) = \\mathrm{softmax}\\!\\left(\\tfrac{Q K^\\top}{\\sqrt{d_{head}}}\\right) V."}
      </MathBlock>

      <Prose>
        Let {"S = Q K^\\top / \\sqrt{d_{head}}"} be the {"L \\times L"} pre-softmax score matrix. Now slice {"K, V"} along the sequence dimension into {"N"} blocks of length {"c = L/N"}: {"K = [K_0; K_1; \\ldots; K_{N-1}]"}, {"V = [V_0; \\ldots; V_{N-1}]"}, each {"K_j, V_j \\in \\mathbb{R}^{c \\times d_{head}}"}. Then for a single query block {"Q_i"},
      </Prose>

      <MathBlock>
        {"\\mathrm{Attn}(Q_i, K, V) = \\mathrm{softmax}\\!\\left([\\, S_{i,0} \\;|\\; S_{i,1} \\;|\\; \\cdots \\;|\\; S_{i,N-1} \\,]\\right) \\cdot \\begin{bmatrix} V_0 \\\\ V_1 \\\\ \\vdots \\\\ V_{N-1} \\end{bmatrix},"}
      </MathBlock>

      <Prose>
        where {"S_{i,j} = Q_i K_j^\\top / \\sqrt{d_{head}}"} is the score sub-matrix for query block {"i"} against key block {"j"}. The softmax is applied across the entire concatenated row, not block-wise. This is the central numerical challenge: softmax requires a global normalisation over all keys, but we want to consume the {"K_j, V_j"} blocks one at a time. The online-softmax algorithm of Milakov and Gimelshein (arXiv:1805.02867, 2018), refined for FlashAttention by Dao et al. (arXiv:2205.14135, 2022), solves this with a running-max and running-denominator update.
      </Prose>

      <H3>3.2 The online softmax accumulator</H3>

      <Prose>
        Maintain three running statistics for each query row in {"Q_i"}: a max {"m^{(j)} \\in \\mathbb{R}"}, a denominator {"\\ell^{(j)} \\in \\mathbb{R}"}, and an output {"O^{(j)} \\in \\mathbb{R}^{d_{head}}"}, each initialised at {"j = -1"} as {"m = -\\infty,\\, \\ell = 0,\\, O = 0"}. When processing block {"j"}, compute the block max {"\\tilde{m}_j = \\max(S_{i,j})"} along the key axis and the unnormalised exponentials {"P_j = \\exp(S_{i,j} - m^{(j)})"} with {"m^{(j)} = \\max(m^{(j-1)}, \\tilde{m}_j)"}. Then
      </Prose>

      <MathBlock>
        {"\\ell^{(j)} = e^{m^{(j-1)} - m^{(j)}}\\, \\ell^{(j-1)} + \\sum_k P_{j, k},"}
      </MathBlock>

      <MathBlock>
        {"O^{(j)} = e^{m^{(j-1)} - m^{(j)}}\\, O^{(j-1)} + P_j V_j."}
      </MathBlock>

      <Prose>
        The first term renormalises the previous accumulator to the new max; the second adds the new block's contribution. After all {"N"} blocks have been processed, {"O^{(N-1)} / \\ell^{(N-1)}"} equals the exact attention output. The renormalisation is the key: when a new block introduces a larger logit than anything seen before, the old accumulator must be downscaled to avoid double-counting. Because all updates use {"e^{m^{(j-1)} - m^{(j)}}"} which is in {"(0, 1]"}, the math is numerically stable and equivalent to the reference softmax up to floating-point round-off.
      </Prose>

      <H3>3.3 Mapping to the ring</H3>

      <Prose>
        Each device {"i"} runs the online-softmax loop over {"j = 0, 1, \\ldots, N-1"}, but the {"K_j, V_j"} consumed at iteration {"j"} is the block currently held in the rotation, which is {"K_{(i-j) \\bmod N}"}. The visit order differs across devices, but each device still sees all {"N"} blocks and the online-softmax math is order-invariant: the running max is associative and commutative under {"\\max"}, and the renormalisation factor {"e^{m^{(j-1)} - m^{(j)}}"} is the same regardless of which block introduced the new max. So device {"i"}'s final output {"O_i^{(N-1)} / \\ell_i^{(N-1)}"} is bit-equivalent (in exact arithmetic) to running standard attention on {"Q_i"} against the full {"K, V"}.
      </Prose>

      <H3>3.4 Memory and compute accounting</H3>

      <Prose>
        On each device, peak activation memory during the attention forward is the sum of: the owned {"Q_i, K_i, V_i"} blocks ({"3 c d_{head} H"} half-precision floats), the currently-held {"K_j, V_j"} block from the ring ({"2 c d_{head} H"} floats), the running statistics {"(m, \\ell)"} ({"2 c"} floats), and the running output {"O_i"} ({"c d_{head} H"} floats). Total: {"6 c d_{head} H + 2 c \\approx 6 (L/N) d_{head} H"}. This is exactly {"1/N"} of single-GPU attention's {"\\sim 6 L d_{head} H"} ignoring the {"\\ell, m"} term. Adding more devices to the ring linearly extends the maximum context length the system can handle.
      </Prose>

      <Prose>
        Compute per device, summed across the {"N"} ring iterations, is {"N \\cdot (c \\cdot c \\cdot d_{head} \\cdot H \\cdot 4)"} FLOPs (the factor of 4 covers the {"QK^\\top"} matmul, softmax exponentiation, and {"PV"} matmul). Substituting {"c = L/N"}, this equals {"4 L^2 d_{head} H / N"} FLOPs per device — exactly {"1/N"} of the single-GPU compute. Total compute across all {"N"} devices is unchanged. Ring attention is a pure capacity scaler: it does not create extra compute, and it does not save compute. It moves the same compute across {"N"} devices and pays for it in inter-device bandwidth.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH BUILD
          ====================================================================== */}
      <H2>4. From-scratch build</H2>

      <Prose>
        We implement ring attention as a single-process simulation: explicit chunking over a Python list of "device" tensors, with an explicit ring-rotation step between iterations. This keeps the algorithm visible and lets us verify bit-equality with single-shot attention. A real multi-GPU implementation replaces the rotation with NCCL {"send/recv"} calls and the Python loop with a CUDA stream that overlaps each iteration's communication with the next iteration's compute.
      </Prose>

      <H3>4.1 The ring loop</H3>

      <CodeBlock language="python">
{`import math
import torch

torch.manual_seed(0)

L, D = 16, 8     # full sequence length, head dim
N = 4            # ring size (number of "devices")
chunk = L // N
scale = 1.0 / math.sqrt(D)

# Single tensor for the full sequence (the ground-truth reference).
Q_full = torch.randn(L, D)
K_full = torch.randn(L, D)
V_full = torch.randn(L, D)

# Reference: standard attention.
ref = torch.softmax(Q_full @ K_full.T * scale, dim=-1) @ V_full

# Each device i owns Q_i, K_i, V_i (shape [L/N, D]).
Q = [Q_full[i*chunk:(i+1)*chunk] for i in range(N)]
K = [K_full[i*chunk:(i+1)*chunk] for i in range(N)]
V = [V_full[i*chunk:(i+1)*chunk] for i in range(N)]

# Per-device running statistics for the online softmax.
m = [torch.full((chunk, 1), float("-inf")) for _ in range(N)]
l = [torch.zeros(chunk, 1) for _ in range(N)]
O = [torch.zeros(chunk, D) for _ in range(N)]

# Each device i starts holding K_i, V_i and rotates them through the ring.
held_K = list(K)
held_V = list(V)

for step in range(N):
    for i in range(N):
        S = Q[i] @ held_K[i].T * scale            # [chunk, chunk]
        m_blk, _ = S.max(dim=-1, keepdim=True)
        m_new = torch.maximum(m[i], m_blk)
        alpha = torch.exp(m[i] - m_new)
        beta_logits = torch.exp(S - m_new)
        l[i] = alpha * l[i] + beta_logits.sum(dim=-1, keepdim=True)
        O[i] = alpha * O[i] + beta_logits @ held_V[i]
        m[i] = m_new

    # Ring rotation: each device sends its held K, V one hop forward.
    held_K = [held_K[(i - 1) % N] for i in range(N)]
    held_V = [held_V[(i - 1) % N] for i in range(N)]

out = torch.cat([O[i] / l[i] for i in range(N)], dim=0)
print(f"Max abs error vs reference: {(out - ref).abs().max().item():.2e}")
print(f"Output shape: {tuple(out.shape)}, reference shape: {tuple(ref.shape)}")

# Output:
# Max abs error vs reference: 1.19e-07
# Output shape: (16, 8), reference shape: (16, 8)`}
      </CodeBlock>

      <Prose>
        The error of {"\\approx 10^{-7}"} is fp32 round-off — bit-equivalent to the single-shot reference within the precision of float32 arithmetic. The algorithm is correct. Three details are worth pausing on. First, the running max update {"m\\_new = \\max(m_i, m_{blk})"} is the only thing that prevents catastrophic overflow: without it, large logits in later blocks would have produced {"\\exp(S)"} values too large for any precision. Second, the renormalisation factor {"\\alpha = \\exp(m_i - m\\_new) \\in (0, 1]"} downscales the previous accumulator whenever a new block introduces a larger max; if the new max equals the old max, {"\\alpha = 1"} and the previous state is unchanged. Third, the rotation is a Python list shuffle here, but in a real multi-GPU run it is {"torch.distributed.send/recv"} on a NCCL ring with {"async\\_op=True"} so the rotation overlaps with the next iteration's matmul.
      </Prose>

      <H3>4.2 Memory and compute accounting</H3>

      <CodeBlock language="python">
{`per_device_qkv = 3 * chunk * D        # owned Q, K, V
per_device_held = 2 * chunk * D       # held K, V from ring partner
per_device_total = per_device_qkv + per_device_held
single_gpu_total = 3 * L * D
print(f"Per-device floats (Q+K+V + held K,V): {per_device_total}")
print(f"Single-GPU floats (Q+K+V):           {single_gpu_total}")
print(f"Memory ratio: {per_device_total / single_gpu_total:.3f} ({N}-way ring)")

# Output:
# Per-device floats (Q+K+V + held K,V): 160
# Single-GPU floats (Q+K+V):           384
# Memory ratio: 0.417 (4-way ring)`}
      </CodeBlock>

      <Prose>
        At {"N = 4"}, per-device activation footprint is {"\\approx 0.42 \\times"} single-GPU. The asymptotic ratio is {"5 / (3 N)"} ({"3"} owned slices plus {"2"} held slices, all of size {"L/N"}, vs {"3 L"} on single GPU), which approaches {"0"} as {"N"} grows. At {"N = 32"} this ratio is {"\\approx 0.05"}; at {"N = 128"} it is {"\\approx 0.013"}. Adding ring devices is a near-linear context extension up to the {"L/N \\to 0"} limit where chunks become too small and the ring goes communication-bound.
      </Prose>

      <H3>4.3 Compute / communication overlap analysis</H3>

      <Prose>
        The wall-clock of a ring attention forward depends on whether each iteration's compute exceeds its communication. The communication moves {"2 c d_{head} H"} bytes ({"K"} and {"V"} blocks), while the compute runs roughly {"4 c^2 d_{head} H"} FLOPs. Holding aside constants, the compute-to-communication ratio is {"\\approx c"} — proportional to chunk size. For an H100 at {"\\sim 700"} TFLOP/s fp16 and NVLink at {"\\sim 450"} GB/s effective bandwidth, the crossover is in the few-thousand-token chunk range.
      </Prose>

      <CodeBlock language="python">
{`GPU_TFLOP   = 700e12
NVLINK_BW   = 450e9          # bytes/sec
BYTES_FP16  = 2

def timing(L_total, N, d_head=128, n_heads=64):
    c = L_total // N
    flops_per_block = 4 * c * c * d_head * n_heads
    t_compute = flops_per_block / GPU_TFLOP
    bytes_per_hop = 2 * c * d_head * n_heads * BYTES_FP16
    t_comm = bytes_per_hop / NVLINK_BW
    t_wall = N * t_compute if t_compute >= t_comm else (N - 1) * t_comm + t_compute
    return t_compute, t_comm, t_wall

print(f"{'L':>8} {'N':>4} {'chunk':>8} {'comp_ms':>10} {'comm_ms':>10} {'wall_ms':>10} {'bound':>8}")
for L_total in [128_000, 1_000_000, 8_000_000]:
    for N in [8, 32, 128]:
        if L_total // N < 256:
            continue
        tc, tcomm, tw = timing(L_total, N)
        bound = "compute" if tc >= tcomm else "comm"
        print(f"{L_total:>8} {N:>4} {L_total//N:>8} {tc*1e3:>10.2f} {tcomm*1e3:>10.2f} {tw*1e3:>10.2f} {bound:>8}")

# Output:
#        L    N    chunk    comp_ms    comm_ms    wall_ms    bound
#   128000    8    16000      11.98       1.17      95.87  compute
#   128000   32     4000       0.75       0.29      23.97  compute
#   128000  128     1000       0.05       0.07       9.29     comm
#  1000000    8   125000     731.43       9.10    5851.43  compute
#  1000000   32    31250      45.71       2.28    1462.86  compute
#  1000000  128     7812       2.86       0.57     365.67  compute
#  8000000    8  1000000   46811.43      72.82  374491.43  compute
#  8000000   32   250000    2925.71      18.20   93622.86  compute
#  8000000  128    62500     182.86       4.55   23405.71  compute`}
      </CodeBlock>

      <Prose>
        At {"L = 128"}k, splitting across {"N = 128"} GPUs drives the chunk size to 1000 tokens, where per-iteration communication ({"0.07"} ms) exceeds compute ({"0.05"} ms) — the only configuration in the table that is communication-bound. Every other configuration is compute-bound, meaning the ring rotations hide entirely under the attention compute and the wall-clock equals {"N \\times t_{compute}"}. At {"L = 8"}M with {"N = 128"}, per-device wall is {"\\approx 23"} seconds for the attention forward alone — long, but tractable; without the ring you simply could not store the activations.
      </Prose>

      <H3>4.4 Striped attention for the causal-mask straggler</H3>

      <CodeBlock language="python">
{`L, N = 16, 4
chunk = L // N

def causal_work_contig():
    work = [[0]*N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            kchunk = (i - j) % N
            qpos = list(range(i*chunk, (i+1)*chunk))
            kpos = list(range(kchunk*chunk, (kchunk+1)*chunk))
            work[i][j] = sum(1 for q in qpos for k in kpos if q >= k)
    return work

def causal_work_striped():
    work = [[0]*N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            kdev = (i - j) % N
            qpos = list(range(i, L, N))      # interleaved positions
            kpos = list(range(kdev, L, N))
            work[i][j] = sum(1 for q in qpos for k in kpos if q >= k)
    return work

W_c, W_s = causal_work_contig(), causal_work_striped()
per_dev_c = [sum(r) for r in W_c]
per_dev_s = [sum(r) for r in W_s]
print("Contiguous totals:", per_dev_c, " ratio", max(per_dev_c)/min(per_dev_c))
print("Striped     totals:", per_dev_s, " ratio", max(per_dev_s)/min(per_dev_s))

# Output:
# Contiguous totals: [10, 26, 42, 58]  ratio 5.8
# Striped     totals: [28, 32, 36, 40]  ratio 1.43`}
      </CodeBlock>

      <Prose>
        Under contiguous chunking the load ratio between the busiest and idlest device is {"5.8\\times"} for {"N = 4"} (it grows to {"N\\times"} as {"N \\to \\infty"}). Striping reduces the ratio to {"1.43\\times"} — within striking distance of perfect balance. The striping is purely a token-position permutation: indices are interleaved before the ring is laid out, so token {"t"} goes to device {"t \\bmod N"} instead of device {"\\lfloor t / (L/N) \\rfloor"}. The attention math is unchanged; only the assignment of positions to devices is permuted, which evenly distributes the causal mask across ring iterations.
      </Prose>

      <Callout accent="green">
        For decoder pretraining (causal mask), striped attention is non-negotiable above {"N = 8"}. Vanilla ring under causal mask wastes up to {"(N-1)/N"} of the GPU-time of the lighter-loaded devices waiting for the heaviest.
      </Callout>

      {/* ======================================================================
          5. PRODUCTION
          ====================================================================== */}
      <H2>5. Production landscape</H2>

      <H3>5.1 Open-source ring attention kernels</H3>

      <Prose>
        The reference open implementation is {"zhuzilin/ring-flash-attention"} on GitHub (with contributions from Tencent and the HuggingFace community), a Python package that wraps Tri Dao's FlashAttention-2 CUDA kernel with a ring-rotation outer loop using {"torch.distributed"}. It supports the contiguous and striped variants, integrates with {"torch.distributed.tensor.DTensor"} for sharding metadata, and exports {"ring_flash_attn_func"} as a near-drop-in replacement for {"flash_attn_func"} in models that already use FlashAttention. Microsoft's {"DeepSpeed.sequence.SequenceParallel"} ships the Ulysses variant; it does not perform ring rotations but instead does pre-attention and post-attention all-to-all collectives.
      </Prose>

      <Prose>
        NVIDIA's Megatron-LM has its own implementation under {"--context-parallel-size"} ({"--cp-size"} in newer builds), which Megatron documents as a "context-parallel" variant of ring attention with overlap. Megatron-LM combines context parallelism with tensor parallelism ({"--tp-size"}), pipeline parallelism ({"--pp-size"}), and data parallelism, exposing all four axes through a single configuration. The xformers library has experimental sequence-parallel attention under {"xformers.ops.sequence_parallel_attention"} but it is less mature than Megatron and {"ring-flash-attention"}.
      </Prose>

      <H3>5.2 Frontier deployments</H3>

      <Prose>
        The Llama 3 paper describes Llama-3.1 405B trained with a 4D parallelism: data parallel + tensor parallel ({"TP = 8"}) + pipeline parallel ({"PP = 16"}) + context parallel ({"CP = 16"}) for the 128k long-context phase. This means the sequence is split across 16 GPUs forming a ring, with each ring's GPUs further split across 8 tensor-parallel ranks for the hidden dimension. The total per-instance GPU count is {"8 \\cdot 16 \\cdot 16 = 2048"} on the long-context training run. Meta's reported numbers show {"\\sim"} 70 BF16-MFU even with this 4D sharding — the ring overlap is good enough that context-parallel rotations are essentially free at the configurations Meta uses.
      </Prose>

      <Prose>
        Anthropic's Claude (200k context) and OpenAI's GPT-4 long-context variants do not publish architecture details, but the inference characteristics — sub-linear latency growth past 32k, predictable throughput at 128k — are consistent with sequence-sharded attention plus blockwise tiling. Google DeepMind's Gemini 1.5 (1M context, with research demos at 10M) explicitly cites a "novel attention scheme distributed across devices for million-token contexts"; the technical report's description of the inference architecture aligns with ring or striped attention combined with mixture-of-experts sharding for the MLP blocks.
      </Prose>

      <H3>5.3 The PyTorch DTensor pathway</H3>

      <Prose>
        PyTorch 2.5+ exposes {"torch.distributed.tensor.DTensor"} and {"torch.distributed.tensor.parallel"} primitives that turn sequence-parallel attention into a metadata transformation rather than a hand-written kernel. A {"DTensor"} sharded along a {"Shard(seq\\_dim)"} placement on a 16-rank device mesh automatically routes attention through a context-parallel implementation. The PyTorch 2.5 release notes show this used in {"torchtitan"}, Meta's open reference for large-scale LLM training, where {"context\\_parallel\\_size = 16"} appears in the published Llama-3 training recipes alongside {"tensor\\_parallel\\_size = 8"} and {"pipeline\\_parallel\\_size"} = (variable).
      </Prose>

      <H3>5.4 Inference-time considerations</H3>

      <Prose>
        Ring attention at training time and ring attention at inference time look different. Training does the full {"Q K^\\top V"} computation, and the {"Q, K, V"} are all freshly produced from the input batch — every device naturally has its own slice of {"Q"} and the ring rotates {"K, V"}. Inference is autoregressive: at each generation step the new token's {"Q"} must attend to all previously generated {"K, V"}, which were progressively appended to a KV cache. The cache is also sharded, so the new {"Q"} ring-rotates against the cache slices. Because there is only one new query position per step (the prefill pass is more parallel), the per-step communication-to-compute ratio is much worse than training; production serving stacks like vLLM and SGLang use ring attention only for the prefill phase and a different scheme (paged KV cache + tensor parallel attention) for the decode phase.
      </Prose>

      <Callout accent="gold">
        Production sequence parallelism is rarely used in isolation. Llama-3.1 405B uses 4D parallelism (DP + TP + PP + CP). DeepSeek-V3 uses TP + PP + EP (expert parallel for MoE) without ring attention because its training context is shorter. The right combination depends on context length, model size, and available cluster topology.
      </Callout>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6.1 The ring rotation across 4 GPUs</H3>

      <Prose>
        At each iteration, every device computes attention against the {"K, V"} block it currently holds, then forwards that block one hop around the ring. After {"N = 4"} iterations every device has seen every {"K, V"} block exactly once.
      </Prose>

      <StepTrace
        label="Ring rotation across 4 devices"
        steps={[
          {
            label: "Iter 0",
            render: () => (
              <div>
                <Prose>
                  Initial state. Device {"i"} holds {"K_i, V_i"}. Each device computes {"Q_i K_i^\\top V_i"} — its own diagonal block.
                </Prose>
                <TokenStream tokens={["GPU0: Q0·K0", "GPU1: Q1·K1", "GPU2: Q2·K2", "GPU3: Q3·K3"]} />
              </div>
            ),
          },
          {
            label: "Iter 1",
            render: () => (
              <div>
                <Prose>
                  Each device's {"K, V"} has been forwarded one hop. Device 0 now holds {"K_3, V_3"}; device 1 holds {"K_0, V_0"}; etc. Compute and merge into running output.
                </Prose>
                <TokenStream tokens={["GPU0: Q0·K3", "GPU1: Q1·K0", "GPU2: Q2·K1", "GPU3: Q3·K2"]} />
              </div>
            ),
          },
          {
            label: "Iter 2",
            render: () => (
              <div>
                <Prose>
                  Two hops in. Device 0 holds {"K_2"}; device 1 holds {"K_3"}; etc. Each device updates its running max, denominator, and output.
                </Prose>
                <TokenStream tokens={["GPU0: Q0·K2", "GPU1: Q1·K3", "GPU2: Q2·K0", "GPU3: Q3·K1"]} />
              </div>
            ),
          },
          {
            label: "Iter 3",
            render: () => (
              <div>
                <Prose>
                  Final iteration. Each device sees the last {"K, V"} block it had not yet processed. After this step, the running output divided by the running denominator equals full attention.
                </Prose>
                <TokenStream tokens={["GPU0: Q0·K1", "GPU1: Q1·K2", "GPU2: Q2·K3", "GPU3: Q3·K0"]} />
              </div>
            ),
          },
          {
            label: "Done",
            render: () => (
              <div>
                <Prose>
                  Every device has seen every {"K, V"} block. Output is finalised as {"O / \\ell"} per device, then concatenated. Total inter-device communication: {"N - 1 = 3"} ring rotations per device per layer per forward pass.
                </Prose>
              </div>
            ),
          },
        ]}
      />

      <H3>6.2 Per-device peak memory vs ring size</H3>

      <Prose>
        Per-device activation memory for attention scales as {"O((L/N) \\cdot d_{model})"}. Doubling {"N"} halves per-device memory at any fixed context length. The plot below shows peak memory in GB for three context lengths across ring sizes from 1 to 128, assuming a {"d_{model} = 8192"} model in fp16.
      </Prose>

      <Plot
        label="Per-device peak attention memory (GB) vs ring size"
        xLabel="ring size N (log)"
        yLabel="GB per device"
        series={[
          {
            name: "L=128k",
            color: colors.gold,
            points: [[1, 6.0], [2, 3.0], [4, 1.5], [8, 0.75], [16, 0.38], [32, 0.19], [64, 0.09], [128, 0.05]],
          },
          {
            name: "L=1M",
            color: colors.green,
            points: [[1, 47.0], [2, 23.5], [4, 11.8], [8, 5.9], [16, 2.95], [32, 1.5], [64, 0.74], [128, 0.37]],
          },
          {
            name: "L=8M",
            color: "#c084fc",
            points: [[1, 376], [2, 188], [4, 94], [8, 47], [16, 23.5], [32, 11.8], [64, 5.9], [128, 2.95]],
          },
        ]}
      />

      <Prose>
        The dashed line at 80 GB (an H100's HBM) sits well above the 1M curve at {"N = 8"} and above the 8M curve at {"N = 128"} — exactly the breakeven point for fitting a single attention layer's activation footprint at each context length. Without ring attention, 8M context would require {"\\approx 376"} GB per device just for one layer's {"Q, K, V"}; with {"N = 128"}, it drops to a comfortable {"3"} GB per device.
      </Prose>

      <H3>6.3 Compute / communication overlap timeline</H3>

      <Prose>
        With perfect overlap, each ring iteration's compute hides the next iteration's communication. The plot below shows two scenarios for {"L = 1\\text{M}, N = 32"}: a compute-bound case (chunk = 31k, compute per iteration {"\\gg"} comm) and a communication-bound case ({"L = 128\\text{k}, N = 128"}, chunk = 1000) where the rotation is the bottleneck.
      </Prose>

      <Plot
        label="Per-iteration timing (ms) — compute vs comm — L=1M N=32 across iterations"
        xLabel="ring iteration"
        yLabel="ms per iter"
        series={[
          {
            name: "compute",
            color: colors.gold,
            points: [[0, 45.7], [1, 45.7], [2, 45.7], [3, 45.7], [4, 45.7], [5, 45.7], [6, 45.7], [7, 45.7]],
          },
          {
            name: "comm (overlapped)",
            color: colors.green,
            points: [[0, 2.28], [1, 2.28], [2, 2.28], [3, 2.28], [4, 2.28], [5, 2.28], [6, 2.28], [7, 2.28]],
          },
        ]}
      />

      <Prose>
        Each iteration's communication ({"2.28"} ms) fits entirely under that iteration's compute ({"45.7"} ms), so wall-clock per layer is just {"N \\times 45.7 = 1463"} ms with effectively zero communication tax. Lift {"N"} too high and chunk size shrinks until communication exceeds compute, at which point the ring goes communication-bound and adds a per-iteration tax equal to {"t_{comm} - t_{compute}"}.
      </Prose>

      <H3>6.4 Causal-mask work distribution: contiguous vs striped</H3>

      <Prose>
        Each cell shows the work (number of {"(q, k)"} pairs not masked out) for device row in iteration column, for {"L = 16, N = 4"}. Row-sums on the right are total work per device. Brighter = more work.
      </Prose>

      <Heatmap
        label="Contiguous causal ring — work[device][iter]"
        rowLabels={["GPU0", "GPU1", "GPU2", "GPU3"]}
        colLabels={["it 0", "it 1", "it 2", "it 3"]}
        matrix={[
          [10, 0, 0, 0],
          [10, 16, 0, 0],
          [10, 16, 16, 0],
          [10, 16, 16, 16],
        ]}
        colorScale="warm"
      />

      <Heatmap
        label="Striped causal ring — work[device][iter]"
        rowLabels={["GPU0", "GPU1", "GPU2", "GPU3"]}
        colLabels={["it 0", "it 1", "it 2", "it 3"]}
        matrix={[
          [10, 6, 6, 6],
          [10, 10, 6, 6],
          [10, 10, 10, 6],
          [10, 10, 10, 10],
        ]}
        colorScale="green"
      />

      <Prose>
        Contiguous: the bottom-right triangle is full ({"16"} cells per block); the top-right is empty. GPU0 finishes in iteration 0 and idles for the next three; GPU3 works every iteration. Striped: every device works every iteration, with at most 4 cells of difference between busiest and idlest. The work ratio drops from {"5.8\\times"} (contiguous) to {"1.43\\times"} (striped). For larger {"N"}, the contiguous ratio scales as {"N"} while the striped ratio stays close to {"\\sqrt{N}"}.
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        The right parallelism for attention depends primarily on context length and secondarily on model size and cluster topology. The following matrix maps the regimes we have seen in production.
      </Prose>

      <Heatmap
        label="Recommended attention parallelism by context length"
        rowLabels={["8k–32k", "32k–128k", "128k–1M", "1M+"]}
        colLabels={["FlashAttn only", "Tensor par.", "Ulysses", "Ring attn", "Striped ring"]}
        matrix={[
          [3, 1, 0, 0, 0],
          [2, 3, 2, 1, 1],
          [0, 1, 3, 3, 3],
          [0, 0, 1, 3, 3],
        ]}
        colorScale="gold"
      />

      <Prose>
        <strong>8k–32k context.</strong> Standard FlashAttention on a single GPU is sufficient. Tensor parallelism is used to split the model across GPUs, but attention itself does not need sequence sharding — the activations fit. Sequence parallelism here is unnecessary engineering complexity. Llama-3 8B and Mistral-7B at default context fall in this regime.
      </Prose>

      <Prose>
        <strong>32k–128k context.</strong> Tensor parallelism alone can usually handle this if the TP size is large enough to fit the activation footprint per device. Sequence parallelism (Megatron-style {"--cp-size"}) becomes attractive when TP is already at the all-reduce bandwidth ceiling and you need another sharding axis. DeepSpeed Ulysses is a good fit here: the all-to-all collective is bandwidth-efficient on NVLink and the per-device sequence still fits during the head-sharded attention.
      </Prose>

      <Prose>
        <strong>128k–1M context.</strong> Single-device memory is exhausted. You need either ring attention or Ulysses, and at the upper end of this range Ulysses runs out of memory inside its head-sharded attention block (because each device has the full sequence, even if only some heads). Ring attention is the cleanest fit. Llama-3.1 405B at 128k uses {"CP = 16"} ring attention. Long-context fine-tunes of Llama, Yi, and Qwen at 200k–1M context all use {"ring-flash-attention"}-style ring kernels.
      </Prose>

      <Prose>
        <strong>1M+ context.</strong> Ring attention is the only option. Ulysses cannot fit a million-token sequence on any single device. Striped attention is essential if the model is causal (i.e., a decoder). Communication starts to dominate as {"N"} grows, so cluster topology — NVLink for intra-node, InfiniBand for inter-node — becomes a first-order concern. Gemini's 1M+ context and the research demos at 10M sit here.
      </Prose>

      <Prose>
        <strong>Many short sequences.</strong> If the workload is many short sequences (e.g., a typical chat-completion server with 4k-token requests), data parallelism alone is the right answer and ring attention adds only complexity. The decision is per-workload, not per-model.
      </Prose>

      <Callout accent="gold">
        Pick parallelism by the binding constraint. Memory-bound at long context: ring attention. Compute-bound at moderate context with high TP: Ulysses or context parallelism for activation savings. Data-parallel-only for short contexts. Frontier models combine all four (DP + TP + PP + CP).
      </Callout>

      {/* ======================================================================
          8. WHAT SCALES
          ====================================================================== */}
      <H2>8. What scales</H2>

      <H3>8.1 Context length scales linearly with ring size</H3>

      <Prose>
        Per-device activation memory is {"O(L/N)"}. Doubling {"N"} doubles the maximum {"L"} the system can handle, until communication or compute becomes the bottleneck. In practice, ring attention has been demonstrated at {"N = 32"} ({"L \\approx 1\\text{M}"}, the Llama-3.1 long-context training) and {"N = 128"} or higher in research settings (Liu et al.'s near-infinite-context experiments). The theoretical ceiling is the cluster size; the practical ceiling is set by NVLink and InfiniBand bandwidth ratios.
      </Prose>

      <H3>8.2 Communication scales as ring rotations</H3>

      <Prose>
        Total bytes per device per attention layer is {"\\approx 2 L d_{head} H"} (each device sends and receives one full {"K, V"} chunk per ring iteration, {"N - 1"} times — and the chunk shrinks as {"N"} grows, but the total stays the same). This is independent of {"N"}. So bandwidth requirements are constant, but the latency-bandwidth ratio favours larger chunks: at small chunks the per-rotation latency overhead becomes a non-trivial fraction of the rotation time, and ring goes communication-bound.
      </Prose>

      <H3>8.3 Combining with tensor parallelism</H3>

      <Prose>
        Ring attention shards the sequence dimension; tensor parallelism shards the hidden dimension within attention and MLP. The two are orthogonal and can be composed: a 2D mesh of size {"(CP, TP)"} runs {"CP"}-way ring attention with each ring rank further internally split across {"TP"} ranks for hidden sharding. This is what Llama-3.1 405B does: {"(CP=16, TP=8)"} forms a 128-GPU 2D mesh per data-parallel replica, with pipeline parallelism on top to span layers across multiple meshes.
      </Prose>

      <H3>8.4 Striped attention reduces straggler effects</H3>

      <Prose>
        Plain ring attention under causal mask wastes up to {"(N-1)/N"} of the lighter-loaded devices' time waiting for the heaviest. Striped attention drops the work ratio from {"N\\times"} to {"\\approx \\sqrt{N}\\times"}. For Llama-3.1 at {"N = 16"}, striping reduces the worst-case stragglers from {"15\\times"} to {"\\sim 4\\times"}, recovering most of the GPU-time that vanilla ring would have left on the table. Brandon et al.'s paper reports {"1.45\\times"} end-to-end speedup over non-striped ring attention on a causal decoder.
      </Prose>

      <H3>8.5 4D parallelism for frontier scale</H3>

      <Prose>
        Llama-3.1 405B's training mesh is {"(\\text{DP}, \\text{TP}, \\text{PP}, \\text{CP}) = (4, 8, 16, 16)"} for the 128k phase, with {"4 \\cdot 8 \\cdot 16 \\cdot 16 = 8192"} GPUs per training instance (later rolled into multi-instance gradient sync via DP). Each axis serves a different memory or compute purpose:
      </Prose>

      <Callout accent="purple">
        DP shards the optimizer state (ZeRO-style) across replicas. TP shards activations along the hidden dim. PP shards layers across stages. CP (ring) shards activations along the sequence dim. The four together let frontier models train at sequence lengths that no single axis could support alone.
      </Callout>

      {/* ======================================================================
          9. FAILURE MODES
          ====================================================================== */}
      <H2>9. Failure modes</H2>

      <H3>9.1 Wrong online-softmax accumulation</H3>

      <Prose>
        The most common implementation bug: forgetting to renormalise the previous accumulator when a new block introduces a larger max. The naive implementation
      </Prose>

      <CodeBlock language="python">
{`# WRONG — does not renormalise old accumulator
l[i] = l[i] + torch.exp(S - m_new).sum(dim=-1, keepdim=True)
O[i] = O[i] + torch.exp(S - m_new) @ held_V[i]`}
      </CodeBlock>

      <Prose>
        produces silently wrong outputs that look reasonable on small test cases but diverge from reference attention on real distributions. The correct update multiplies the previous {"\\ell"} and {"O"} by {"\\exp(m^{(j-1)} - m^{(j)})"} — the renormalisation factor that downscales the old accumulator when a larger max is introduced. The correct form,
      </Prose>

      <CodeBlock language="python">
{`alpha = torch.exp(m[i] - m_new)
l[i] = alpha * l[i] + torch.exp(S - m_new).sum(dim=-1, keepdim=True)
O[i] = alpha * O[i] + torch.exp(S - m_new) @ held_V[i]`}
      </CodeBlock>

      <Prose>
        differs only in the {"alpha *"} on the {"l[i]"} and {"O[i]"} terms. The unit test for this bug is to feed the algorithm a sequence where one block contains very large logits ({"\\sim 10"}+) preceded by a block with small logits ({"\\sim 0"}); without renormalisation the small-logit block dominates because {"\\exp"} of small numbers does not get renormalised when a large max appears later.
      </Prose>

      <H3>9.2 Ring rotation not overlapping with compute</H3>

      <Prose>
        On NCCL, {"send/recv"} pairs must be issued with {"async\\_op=True"} on a separate CUDA stream from the compute kernel for overlap to actually happen. The synchronous form
      </Prose>

      <CodeBlock language="python">
{`# WRONG — synchronous, no overlap
torch.distributed.send(K_buf, dst=next_rank)
torch.distributed.recv(K_buf, src=prev_rank)
# ... then compute on the new K_buf`}
      </CodeBlock>

      <Prose>
        serialises every rotation against the corresponding compute, doubling wall-clock at minimum. The correct pattern uses two buffers (double-buffering), launches the rotation for buffer B+1 while compute runs on buffer B, and synchronises only at the buffer swap. PyTorch's {"torch.distributed.batch\\_isend\\_irecv"} with {"async\\_op=True"} returns a handle whose {"wait()"} can be deferred to just before the next compute uses the buffer. {"ring-flash-attention"}'s reference implementation does exactly this; the Megatron-LM and DeepSpeed Ulysses implementations use {"torch.distributed.P2POp"} batches scheduled on a dedicated comm stream.
      </Prose>

      <H3>9.3 NCCL bandwidth bottleneck</H3>

      <Prose>
        NCCL's effective bandwidth on a ring topology depends heavily on whether the ring respects the underlying NVLink / PCIe topology. On an 8-GPU H100 NVLink node, NCCL constructs an optimal NVLink ring automatically; on multi-node setups, the ring crosses InfiniBand at {"\\sim 50"} GB/s — an order of magnitude slower than NVLink. If the ring rank order does not match the physical topology, NCCL may construct a ring that crosses InfiniBand multiple times unnecessarily, halving effective bandwidth. The fix is to set {"NCCL\\_TOPO\\_FILE"} or {"CUDA\\_VISIBLE\\_DEVICES"} so that ranks are arranged contiguously on each node, and to set {"CP\\_size"} as a divisor of intra-node GPU count when possible.
      </Prose>

      <H3>9.4 Wrong rank ordering in the ring</H3>

      <Prose>
        A subtle bug: if the ring rotation goes the wrong direction (each device reads from {"(rank + 1) \\bmod N"} instead of {"(rank - 1) \\bmod N"}), the math still produces a valid attention output — but the assignment of token positions to devices is reversed, which corrupts the relationship between {"Q"} and {"K"} positions. Models with positional encodings (RoPE, ALiBi) will have completely wrong relative-position offsets across devices and produce gibberish. The unit test: run the ring on a small example and verify token-by-token equality with single-shot attention; any bidirectional symmetry in the test data will hide the bug, so use asymmetric inputs (e.g., increasing position-dependent values).
      </Prose>

      <H3>9.5 Striping load imbalance</H3>

      <Prose>
        Striping is a permutation, but it interacts with the choice of position encoding: if RoPE rotates positions {"0, 1, 2, \\ldots"} contiguously and you stripe so that device 0 holds positions {"0, N, 2N, \\ldots"}, the RoPE rotation indices on device 0 are {"0, N, 2N, \\ldots"} — which is fine, but the ring rotation must still respect that device {"i"}'s held positions in iteration {"j"} are {"(i - j) \\bmod N + N \\cdot \\{0, 1, \\ldots, c - 1\\}"}, not the contiguous block. Implementations that assume contiguous chunking when computing RoPE produce wrong rotations after striping. {"ring-flash-attention"} handles this correctly; hand-rolled ring kernels often do not until tested against a striped baseline.
      </Prose>

      <H3>9.6 Causal mask within ring rotation</H3>

      <Prose>
        Within a single ring iteration, device {"i"}'s {"Q"} chunk processes against held {"K, V"} chunk {"k = (i - j) \\bmod N"}. The causal mask must be applied correctly: if {"k > i"}, the entire iteration is masked out (queries on device {"i"} cannot attend to later positions on device {"k"}); if {"k < i"}, no mask within the block; if {"k = i"}, a triangular within-chunk mask. A common bug is to apply the causal mask uniformly to every iteration (always triangular), which produces wrong scores for the {"k < i"} blocks. The correct conditional masking is: full block (no mask) for {"k < i"}, triangular for {"k = i"}, all-mask (skip) for {"k > i"}.
      </Prose>

      <H3>9.7 fp16 / bf16 accumulator drift</H3>

      <Prose>
        The online softmax {"\\ell"} accumulator grows as {"\\ell^{(j)} = \\alpha \\ell^{(j-1)} + \\sum P_j"}, accumulating {"N"} block contributions. In bf16, with {"7"}-bit mantissa, the accumulation drift is non-negligible at {"N > 32"} and pushes attention output error above the typical {"10^{-2}"} tolerance for downstream consistency. FlashAttention-2 keeps {"\\ell"} and {"m"} in fp32 even when {"Q, K, V"} are in bf16; ring attention implementations should do the same. The cost is small ({"2 c"} fp32 floats per device, vs {"6 c d_{head} H"} bf16 floats for {"Q, K, V"}) but skipping it produces silently degraded long-context training.
      </Prose>

      <Callout accent="red">
        Always keep the online-softmax statistics in fp32, even when {"Q, K, V"} are bf16. The {"\\ell"} accumulator drift in bf16 is the single most common silent quality regression in homemade ring kernels.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        <strong>Liu, Zaharia, Abbeel (2023). "Ring Attention with Blockwise Transformers for Near-Infinite Context."</strong> arXiv:2310.01889; NeurIPS 2023. The foundational paper. Introduces the ring rotation of {"K, V"} blocks across devices, derives the online-softmax block combination, and demonstrates near-infinite context scaling on synthetic and real benchmarks. The paper's central thesis — that activation memory, not compute, is the binding constraint at long context — frames the entire long-context era. Section 3 contains the algorithm; Section 4 the memory and communication analysis.
      </Prose>

      <Prose>
        <strong>Brandon, Nguyen, Zhang (2023). "Striped Attention: Faster Ring Attention for Causal Transformers."</strong> arXiv:2311.09431. Identifies the causal-mask straggler problem in vanilla ring attention and proposes striping (token interleaving) as the fix. Reports {"1.45\\times"} end-to-end speedup on causal decoders at {"N = 16"} compared to non-striped ring. The math in Section 3 proves that striping achieves uniform load distribution across ring iterations under causal mask.
      </Prose>

      <Prose>
        <strong>Korthikanti, Casper, Lym, McAfee, Andersch, Shoeybi, Catanzaro (2023). "Reducing Activation Recomputation in Large Transformer Models."</strong> arXiv:2205.05198; MLSys 2023. NVIDIA's introduction of sequence parallelism in Megatron-LM. Distinct from ring attention: shards LayerNorm, Dropout, and the residual path along the sequence dimension while tensor parallelism handles attention and MLP internals. Combined with ring attention, this is the {"--sequence-parallel"} flag in Megatron alongside {"--context-parallel"}.
      </Prose>

      <Prose>
        <strong>Jacobs, Tanaka, Zhang, Zhang, Song, Rajbhandari, He (2023). "DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models."</strong> arXiv:2309.14509. The all-to-all-based alternative to ring attention. Sections 3 and 4 show how a pair of all-to-all collectives swaps between sequence-sharded and head-sharded layouts so that attention can run on a per-head subset with the full sequence on each device. Faster than ring attention for short-to-medium contexts where the full sequence fits per device; cannot scale beyond single-device memory.
      </Prose>

      <Prose>
        <strong>Dao, Fu, Ermon, Rudra, Ré (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness."</strong> arXiv:2205.14135; NeurIPS 2022. The single-device tile-level math that ring attention extends across devices. Section 3.1's online-softmax algorithm is exactly the per-device update rule used in ring's outer loop. Reading this paper before Liu et al. makes the ring-as-distributed-FlashAttention framing immediate.
      </Prose>

      <Prose>
        <strong>Llama Team (2024). "The Llama 3 Herd of Models."</strong> arXiv:2407.21783. Meta's technical report on Llama-3 family. Section 3.4 ("Long Context Pre-Training") describes the 128k context-parallel training with {"CP = 16"}, the gradient checkpointing strategy, and the curriculum from 8k to 128k. Section 3.3.3 documents the 4D parallelism mesh used for the 405B run. The most detailed published account of ring attention in a frontier production training pipeline.
      </Prose>

      <Prose>
        <strong>Milakov, Gimelshein (2018). "Online normalizer calculation for softmax."</strong> arXiv:1805.02867. The pre-FlashAttention paper that introduced the online-softmax algorithm in a numerical-methods context. The two-pass-becomes-one-pass derivation is short and lucid; reading it makes the renormalisation factor {"\\alpha = \\exp(m^{(j-1)} - m^{(j)})"} obvious.
      </Prose>

      <Prose>
        <strong>Open implementations.</strong> {"zhuzilin/ring-flash-attention"} on GitHub for the production Python wrapper around FlashAttention-2 with ring rotation; {"deepspeed.sequence.SequenceParallel"} in the DeepSpeed repo for Ulysses; {"NVIDIA/Megatron-LM"} for context-parallel + sequence-parallel + tensor-parallel + pipeline-parallel composition; {"pytorch/torchtitan"} for the PyTorch DTensor reference recipe used by Meta. Reading the source of any one of these clarifies the implementation choices the papers gloss over.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK
          ====================================================================== */}
      <H2>11. Self-check</H2>

      <Prose>
        <strong>1. Why does ring attention save memory but not compute?</strong> Ring attention partitions the sequence dimension across {"N"} devices, so each device holds only {"1/N"} of the {"Q, K, V"} activations — peak memory drops from {"O(L \\cdot d)"} to {"O(L \\cdot d / N)"}. Total compute is unchanged: each device still does {"L^2 d / N"} FLOPs across {"N"} iterations, summing to {"L^2 d"} across the cluster — exactly what single-GPU attention would do. The trade is memory for inter-device bandwidth, not compute.
      </Prose>

      <Prose>
        <strong>2. What is the renormalisation factor in the online softmax, and why is it needed?</strong> The factor is {"\\alpha = \\exp(m^{(j-1)} - m^{(j)})"} where {"m^{(j)}"} is the running max after block {"j"}. It downscales the previous accumulator whenever a new block's logits introduce a larger max, so that all unnormalised exponentials in the running sum are expressed relative to the same {"m^{(j)}"}. Without it, the contributions of earlier blocks are over-weighted (they were computed with a smaller max, so their {"\\exp"} values are larger than they should be relative to later blocks). The factor is always in {"(0, 1]"} and ensures numerical stability and exactness up to floating-point round-off.
      </Prose>

      <Prose>
        <strong>3. When should you use Ulysses instead of ring attention?</strong> Ulysses uses a pair of all-to-all collectives instead of ring rotations and is bandwidth-efficient on NVLink, but it requires the full sequence to fit on a single device during the head-sharded attention phase. Use Ulysses when context fits per device and you want lower latency than ring (typically 32k–256k on H100, depending on model size). Use ring when context exceeds single-device memory (typically {"\\gtrsim"} 256k for large models). Above 1M, ring is the only option.
      </Prose>

      <Prose>
        <strong>4. Why does plain ring attention have a load-imbalance problem under causal mask, and how does striping fix it?</strong> Under contiguous chunking, device {"i"}'s queries can only attend to keys at positions {"\\le i \\cdot c"}, so device 0 does {"1/N"} of full work and device {"N - 1"} does full work — a ratio of {"N\\times"} between busiest and idlest. Striping interleaves: token {"t"} goes to device {"t \\bmod N"}. After striping, every device's positions sample uniformly across {"[0, L)"}, so the causal mask falls evenly on each ring iteration; the work ratio drops from {"N\\times"} to {"\\sqrt{N}\\times"} or better.
      </Prose>

      <Prose>
        <strong>5. In the 4D parallelism used by Llama-3.1 405B at 128k context, what does each axis shard, and why are all four needed?</strong> DP (data parallel) shards optimizer state (ZeRO-style) across replicas — needed because optimizer state is roughly {"4\\times"} the parameter count for AdamW in fp32. TP (tensor parallel) shards activations along the hidden dimension within attention and MLP — needed because activation footprint per layer at 128k context exceeds single-device memory. PP (pipeline parallel) shards layers across stages — needed because the model has 126 layers, more than the per-stage device limit can hold along with TP and CP. CP (context parallel, ring attention) shards activations along the sequence dimension — needed because even with TP, the {"K, V"} cache and intermediates at 128k context exceed per-device memory. Removing any one axis causes the model to OOM at 128k context with the available GPU memory.
      </Prose>

    </div>
  ),
};

export default ringAttentionContent;
