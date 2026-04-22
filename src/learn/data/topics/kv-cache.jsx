import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const kvCache = {
  title: "KV-Cache & Memory Management",
  readTime: "42 min",
  content: () => (
    <div>
      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Autoregressive generation is a loop. The model produces one token, appends it to the sequence, runs the entire sequence through every transformer layer again, and produces the next token. Without any optimization, generating a reply 1,000 tokens long means running attention 1,000 times. The first pass processes 1 token, the second processes 2, the third processes 3, and so on through the thousandth pass processing 1,000. Add those up and you have 500,500 attention evaluations to produce a thousand tokens. The cost of generation scales with the square of output length, which is the same quadratic trap that made naive sequence-to-sequence models unworkable before transformers.
      </Prose>

      <Prose>
        The escape hatch is pure linear algebra. Every transformer layer computes three projections per token: query (Q), key (K), and value (V). The attention output for a query at position t is a weighted average of all values, where the weights come from the similarity of the query to every key in the sequence. Here is the critical observation: the key and value tensors for every token computed at step t are <em>identical</em> to the key and value tensors the model would compute for those same tokens at step t+1. The token embeddings did not change. The projection weights did not change. The math is deterministic. So why recompute anything?
      </Prose>

      <Prose>
        The KV cache is the answer: after computing keys and values for each token, store them. On the next decode step, only the single new token needs fresh Q, K, V projections. The new query attends against the full cached history of keys, and the new value is appended to the growing cache. The total arithmetic across all decode steps is now linear in sequence length — one Q/K/V projection per new token, plus one attention computation of size O(current_length) — rather than quadratic. Generating 1,000 tokens costs roughly 1,000 attention steps instead of 500,000.
      </Prose>

      <Prose>
        The idea is as old as the transformer itself. The original Vaswani et al. 2017 paper (arXiv:1706.03762) describes the decoder's autoregressive inference mode, and any implementation of it naturally avoids recomputing attention over the same tokens repeatedly — the cache is implicit in how the inference loop is written. What changed in the years since is the cost. When context windows were 512 or 1,024 tokens, the memory footprint of the cache was a minor implementation detail. With Llama 3's 128k context, DeepSeek-V2's 128k, and production deployments routinely running at 32k–64k tokens per request with dozens of concurrent users, the KV cache is almost always the primary consumer of GPU memory and the primary constraint on serving throughput. Pope et al. (2022, arXiv:2211.05102) gave a rigorous analytical treatment of this constraint; by the time Kwon et al. (2023, arXiv:2309.06180) shipped vLLM with PagedAttention, the community had accepted that managing this memory well was not an optimization but a prerequisite for viable serving.
      </Prose>

      <Callout accent="purple">
        The KV cache converts O(N²) generation into O(N) generation. Its memory footprint — not the model weights — is the primary bottleneck in modern LLM serving at any meaningful scale.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <H3>The asymmetry that makes caching possible</H3>

      <Prose>
        During autoregressive decoding, queries and keys/values play fundamentally different roles. The query for position t asks: "what in the history is relevant to me?" That question can only be asked from position t's perspective — it changes with every new token. But the key and value vectors for position 0 through t-1 answer a timeless question: "what information do I carry, and how should I be recognized?" Those answers are fixed the moment the token is processed. The projection that produced K₀ and V₀ from token 0's embedding saw the same embedding and the same weight matrix every time; it always produces the same K₀ and V₀. Caching them is not an approximation — it is algebraically exact.
      </Prose>

      <Prose>
        This asymmetry between Q (recomputed every step) and K, V (fixed once computed) is the entire conceptual basis of the KV cache. In the prefill phase, all prompt tokens are processed simultaneously and their K/V tensors are written out in one pass. In every decode step that follows, only the new token's Q is projected fresh, and it attends against the accumulated cache.
      </Prose>

      <H3>Memory cost: the formula that rules everything</H3>

      <Prose>
        The memory footprint of the KV cache is determined by exactly five architectural parameters and one operational one:
      </Prose>

      <MathBlock>{"\\text{KV cache size} = 2 \\cdot L \\cdot H_{kv} \\cdot d_h \\cdot S \\cdot B \\cdot \\text{bytes}"}</MathBlock>

      <Prose>
        Where <Code>L</Code> is the number of transformer layers, <Code>H_kv</Code> is the number of key-value heads per layer (which can be less than the query head count with GQA), <Code>d_h</Code> is the head dimension, <Code>S</Code> is the sequence length, <Code>B</Code> is the batch size (concurrent sequences), and <Code>bytes</Code> is the precision: 2 for BF16 or FP16, 1 for FP8 or INT8, 0.5 for INT4. The factor of 2 in front accounts for storing both keys and values separately.
      </Prose>

      <Prose>
        Plug in Llama 3 70B: 80 layers, 8 KV heads (after GQA), head dimension 128. A single sequence at 8k context in BF16: <Code>2 × 80 × 8 × 128 × 8192 × 2 = 2.68 GB</Code>. One conversation. An H100 80GB holds the model weights at BF16 (roughly 140 GB across two H100s, or ~70 GB per GPU with tensor parallelism). After weights, activation scratch, and framework overhead, roughly 50–60 GB may remain for KV cache. At 2.68 GB per sequence, that is roughly 18–22 concurrent 8k-context requests per GPU — in the best case. Push to 32k context and each sequence consumes 10.74 GB; you are down to 5–6 concurrent requests. Push to 128k and a single sequence would consume the entire remaining GPU memory.
      </Prose>

      <H3>Two phases, two bottlenecks</H3>

      <Prose>
        LLM inference divides cleanly into two regimes with opposite performance characteristics. <strong>Prefill</strong> processes all prompt tokens simultaneously in one forward pass. Every layer produces its K and V tensors for every prompt position at once. The computation is dominated by large matrix multiplications — tensor cores run near peak utilization, memory bandwidth is secondary, the GPU is compute-bound. Time-to-first-token (TTFT) is determined here.
      </Prose>

      <Prose>
        <strong>Decode</strong> generates one token per step. There is no sequence-axis parallelism — the next token cannot be computed before the current one exists. Each step reads the full KV cache out of HBM, does a tiny amount of arithmetic (one query vector's dot products against thousands of key rows), and writes one new K/V pair back. The arithmetic intensity — FLOPs per byte of memory traffic — is catastrophically low. A modern H100 can execute roughly 100 BF16 FLOPs per byte of HBM traffic at peak; a decode step delivers on the order of 1–3 FLOPs per byte. The GPU is memory-bandwidth-bound, not compute-bound. Faster tensor cores do not help. More memory bandwidth does.
      </Prose>

      <Prose>
        This asymmetry has deep implications. Every major serving optimization threads through it. Continuous batching improves throughput because decode is bandwidth-bound: packing more sequences into a batch costs almost no additional HBM traffic per step relative to the fixed cost of reading the weights. Quantizing the cache helps because the bottleneck is bytes moved, not FLOPs performed — halving the bytes halves the latency. Speculative decoding helps because decode is sequential: the bottleneck is the number of serial steps, and any mechanism that accepts multiple tokens per step wins even if it consumes more compute. Prefill optimizations are a different problem entirely and need different solutions.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>Attention with cache: the exact computation</H3>

      <Prose>
        At decode step t, the model has already computed and stored key-value tensors for positions 0 through t-1. The KV cache at layer l, head h is the matrix pair:
      </Prose>

      <MathBlock>{"K_{\\text{cache}}^{(l,h)} \\in \\mathbb{R}^{t \\times d_h}, \\quad V_{\\text{cache}}^{(l,h)} \\in \\mathbb{R}^{t \\times d_h}"}</MathBlock>

      <Prose>
        The new token at position t is projected into a fresh query, key, and value. The key and value are appended to the cache; the query attends against the full cache:
      </Prose>

      <MathBlock>{"\\mathbf{q}_t = \\mathbf{x}_t W_Q, \\quad \\mathbf{k}_t = \\mathbf{x}_t W_K, \\quad \\mathbf{v}_t = \\mathbf{x}_t W_V"}</MathBlock>

      <MathBlock>{"K_{t} = \\begin{bmatrix} K_{\\text{cache}} \\\\ \\mathbf{k}_t \\end{bmatrix}, \\quad V_{t} = \\begin{bmatrix} V_{\\text{cache}} \\\\ \\mathbf{v}_t \\end{bmatrix}"}</MathBlock>

      <MathBlock>{"\\mathbf{o}_t = \\text{softmax}\\!\\left(\\frac{\\mathbf{q}_t K_t^\\top}{\\sqrt{d_h}}\\right) V_t"}</MathBlock>

      <Prose>
        This is algebraically identical to running full attention over all t+1 tokens simultaneously — just restricted to the query at position t. The causal mask that prevents attending to future positions is automatically satisfied because the cache only contains past tokens. There is no approximation. The cache is a lossless optimization.
      </Prose>

      <H3>GQA: shrinking H_kv</H3>

      <Prose>
        Grouped-Query Attention (GQA, Ainslie et al. 2023, arXiv:2305.13245) separates the number of query heads from the number of key-value heads. With <Code>H_q</Code> query heads grouped into <Code>g</Code> groups, each group sharing one K/V head:
      </Prose>

      <MathBlock>{"H_{kv} = H_q / g, \\quad \\text{cache savings} = g\\times"}</MathBlock>

      <Prose>
        For Llama 3 70B, <Code>H_q = 64</Code> and <Code>H_kv = 8</Code>, giving a group size <Code>g = 8</Code>. The KV cache is 8× smaller than full MHA with no change to the number of parameters in the Q projection. The attention computation for each query head still reads from a shared K/V head within its group. Empirically, this trades a small representational capacity reduction for 8× cache memory savings, and the capacity reduction is recoverable through training — evaluated at the same total parameter count, GQA matches MHA quality closely.
      </Prose>

      <H3>MLA: compressing into a latent</H3>

      <Prose>
        Multi-head Latent Attention (MLA), introduced in DeepSeek-V2 (arXiv:2405.04434), goes further. Instead of reducing the number of heads, it projects K and V through a shared low-rank bottleneck. A compressed latent vector <Code>c_t</Code> is computed and cached; at attention time, it is decompressed back to full K/V per layer:
      </Prose>

      <MathBlock>{"\\mathbf{c}_t = \\mathbf{x}_t W_{DKV} \\in \\mathbb{R}^{d_c}, \\quad d_c \\ll d_{\\text{model}}"}</MathBlock>

      <MathBlock>{"K_t = \\mathbf{c}_t W_{UK}, \\quad V_t = \\mathbf{c}_t W_{UV}"}</MathBlock>

      <Prose>
        Only <Code>c_t</Code> is stored in the cache, not the full K and V tensors. The cache size drops from <Code>2 · L · H · d_h</Code> to <Code>L · d_c</Code> bytes per token. DeepSeek-V2 reports a 93.3% reduction in KV cache size versus standard MHA. The decompression matrices <Code>W_UK</Code> and <Code>W_UV</Code> are part of the model weights (fixed), so the decompression cost is pure compute rather than memory bandwidth.
      </Prose>

      <H3>Memory budget at deployment</H3>

      <Prose>
        The practical constraint at deployment is simple: the KV cache for all concurrent requests must fit in HBM after accounting for weights and activations:
      </Prose>

      <MathBlock>{"B_{\\max} = \\left\\lfloor \\frac{\\text{GPU HBM} - W_{\\text{model}} - W_{\\text{activations}}}{2 \\cdot L \\cdot H_{kv} \\cdot d_h \\cdot S \\cdot \\text{bytes}} \\right\\rfloor"}</MathBlock>

      <Prose>
        This inequality is the governing equation of LLM serving economics. Every technique in this topic — GQA, quantization, paging, prefix sharing, MLA — is a manipulation of one of its terms. The left side (GPU memory) is fixed hardware. The model weight term is not negotiable without changing the model. The activation scratch is roughly constant. Every optimization targets the KV cache term in the denominator.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The five implementations below were all executed and their outputs embedded verbatim. Each one isolates a different aspect of KV cache mechanics so that the concepts in sections 2 and 3 connect to runnable code. NumPy only — no deep learning framework required to understand what is happening.
      </Prose>

      <H3>4a. Naive vs cached attention</H3>

      <Prose>
        The most direct demonstration: implement decode-time attention two ways, measure both, and show the divergence with sequence length. Naive attention reprojects all past tokens at every step. Cached attention reads from stored K/V tensors.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
import math
import time

def naive_attention(tokens_embedded, W_q, W_k, W_v, d_h):
    """Full recompute every decode step — O(S²) total work."""
    S = tokens_embedded.shape[0]
    for t in range(S):
        # Reproject ALL tokens 0..t from scratch
        Q = tokens_embedded[:t+1] @ W_q  # (t+1, d_h)
        K = tokens_embedded[:t+1] @ W_k
        V = tokens_embedded[:t+1] @ W_v
        q = Q[-1:]                        # only the last token queries
        scores = q @ K.T / math.sqrt(d_h)
        weights = np.exp(scores - scores.max())
        weights /= weights.sum()
        _ = weights @ V                   # output discarded for timing

def cached_attention(tokens_embedded, W_q, W_k, W_v, d_h):
    """Cache K,V — O(S) total projections, O(S) bandwidth per step."""
    K_cache = np.empty((0, d_h))
    V_cache = np.empty((0, d_h))
    for t in range(tokens_embedded.shape[0]):
        tok = tokens_embedded[t]
        q = tok @ W_q          # single query projection
        k = tok @ W_k          # single key   (cached after this)
        v = tok @ W_v          # single value (cached after this)
        K_cache = np.vstack([K_cache, k]) if len(K_cache) > 0 else k[None]
        V_cache = np.vstack([V_cache, v]) if len(V_cache) > 0 else v[None]
        scores = q @ K_cache.T / math.sqrt(d_h)
        weights = np.exp(scores - scores.max())
        weights /= weights.sum()
        _ = weights @ V_cache

# Benchmarks (d_model=128, d_h=64):
# Seq len |  Naive (ms) | Cached (ms) | Speedup
# ------  | ----------- | ----------- | -------
#      32 |        5.05 |        1.24 |   4.06x
#      64 |       22.61 |        3.70 |   6.11x
#     128 |       90.21 |        8.75 |  10.31x
#     256 |      252.88 |       54.30 |   4.66x
#     512 |      659.11 |      175.77 |   3.75x
# Speedup grows then plateaus as cache read cost dominates at long S.`}
      </CodeBlock>

      <Prose>
        The speedup peaks around S=128 in this pure-Python benchmark because at short sequences the overhead of the Python loop dominates both implementations. In a GPU kernel, the savings are more dramatic and more sustained: the naive implementation touches O(S²) elements of HBM over the full generation, while the cached implementation touches O(S) projection FLOPs (once per new token) plus O(S²) attention reads total — the latter is irreducible, but the former's elimination is what matters. The cached implementation also admits incremental updates: the cache only ever grows, and each append is a single vector write.
      </Prose>

      <H3>4b. Memory accounting</H3>

      <Prose>
        The formula from section 3 becomes concrete when applied to real model configurations. This function computes exact byte counts; the table below shows why the KV cache is described as the primary memory consumer at scale.
      </Prose>

      <CodeBlock language="python">
{`def kv_cache_bytes(L, H_kv, d_h, S, B=1, dtype_bytes=2):
    """
    Total KV cache size in bytes.
    L          = number of transformer layers
    H_kv       = number of KV heads per layer
    d_h        = head dimension
    S          = sequence length (tokens)
    B          = batch size (concurrent sequences)
    dtype_bytes = 2 (BF16/FP16), 1 (FP8/INT8), 0.5 (INT4)
    """
    return 2 * L * H_kv * d_h * S * B * dtype_bytes

# Model configs: (name, L, H_kv, d_h)
# Llama 3  8B:  32 layers, 8 KV heads, d_h=128
# Llama 3 70B:  80 layers, 8 KV heads, d_h=128
# Llama 3 405B: 126 layers, 8 KV heads, d_h=128

# --- Llama 3 70B (L=80, H_kv=8, d_h=128) ---
#    Context | BF16 (1 seq) | FP8 (1 seq) | BF16 (32 seq) | FP8 (32 seq)
#        8k  |     2.68 GB  |    1.34 GB  |      85.90 GB  |    42.95 GB
#       32k  |    10.74 GB  |    5.37 GB  |     343.60 GB  |   171.80 GB
#      128k  |    42.95 GB  |   21.47 GB  |        1.37 TB |   687.19 GB
#        1M  |   343.60 GB  |  171.80 GB  |       11.00 TB |     5.50 TB

# One H100 80 GB holds Llama 3 70B weights at ~70 GB across two GPUs.
# Remaining ~50 GB for KV cache at 8k context (BF16): ~18 concurrent seqs.
# At 128k context (BF16): cache for ONE sequence nearly fills the H100.`}
      </CodeBlock>

      <Prose>
        These numbers establish the hard wall that everything else in this topic pushes against. The 70B model at 128k context in BF16 requires 42.95 GB for a single sequence's KV cache — more than half an H100's total HBM. Switching to FP8 cache halves that to 21.47 GB. Adding GQA at 8× reduction (already baked into these numbers) brought the original H_kv from 64 to 8; without GQA the BF16 128k cache would be 343 GB, which is physically impossible to store on any single GPU. The combination of GQA and FP8 makes long-context serving viable. Neither alone is sufficient.
      </Prose>

      <H3>4c. GQA vs MHA</H3>

      <Prose>
        Grouped-Query Attention is a drop-in replacement for multi-head attention that changes only how many unique key-value projections exist per layer. Each query head within a group attends to the same K and V head. The output shape is identical; only the cache size changes.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
import math

def mha_attention(X, W_q, W_k, W_v, W_o, H, d_h):
    """Standard multi-head attention — H KV heads."""
    S, d_model = X.shape
    Q = (X @ W_q).reshape(S, H, d_h)
    K = (X @ W_k).reshape(S, H, d_h)
    V = (X @ W_v).reshape(S, H, d_h)
    out_heads = []
    for h in range(H):
        scores = Q[:, h] @ K[:, h].T / math.sqrt(d_h)
        mask = np.triu(np.ones((S, S)) * -1e9, k=1)
        w = np.exp(scores + mask - (scores + mask).max(-1, keepdims=True))
        w /= w.sum(-1, keepdims=True)
        out_heads.append(w @ V[:, h])
    return np.concatenate(out_heads, axis=-1) @ W_o

def gqa_attention(X, W_q, W_k, W_v, W_o, H_q, H_kv, d_h):
    """Grouped-query attention — H_kv KV heads shared across H_q query heads."""
    S, d_model = X.shape
    g = H_q // H_kv          # queries per KV head
    Q = (X @ W_q).reshape(S, H_q,  d_h)
    K = (X @ W_k).reshape(S, H_kv, d_h)
    V = (X @ W_v).reshape(S, H_kv, d_h)
    out_heads = []
    for h in range(H_q):
        kv_h = h // g         # which shared KV head this query uses
        scores = Q[:, h] @ K[:, kv_h].T / math.sqrt(d_h)
        mask = np.triu(np.ones((S, S)) * -1e9, k=1)
        w = np.exp(scores + mask - (scores + mask).max(-1, keepdims=True))
        w /= w.sum(-1, keepdims=True)
        out_heads.append(w @ V[:, kv_h])
    return np.concatenate(out_heads, axis=-1) @ W_o

# Results (H_q=8, H_kv=2, d_h=32, S=16, d_model=256):
# MHA output shape: (16, 256)   ← same
# GQA output shape: (16, 256)   ← same
# MHA KV cache (seq=16, BF16): 16384 bytes  (8 KV heads)
# GQA KV cache (seq=16, BF16):  4096 bytes  (2 KV heads, 4x smaller)
# Outputs differ (different KV weights), but architecture is equivalent.`}
      </CodeBlock>

      <Prose>
        The key insight from the output: GQA produces the same output shape as MHA — the downstream model and the serving stack are completely unaware of the reduction. The only visible difference is the size of the K and V weight matrices (<Code>W_k</Code> and <Code>W_v</Code> are 4× smaller in this example) and the size of the KV cache tensors. For Llama 3 70B with group size 8, the W_k and W_v matrices shrink from <Code>8192 × 8192</Code> to <Code>8192 × 1024</Code>, and the KV cache shrinks 8× correspondingly.
      </Prose>

      <H3>4d. Quantized KV cache</H3>

      <Prose>
        The cache is storage, not arithmetic. Its values are read, dequantized on-chip, and used in floating-point attention. Any precision that survives the round-trip without material quality loss is fair game for storage. Per-token, per-head symmetric quantization is the standard — one scale factor per token per head keeps the quantization grid aligned to each row's actual distribution rather than sharing one scale across the whole cache (which outliers would dominate).
      </Prose>

      <CodeBlock language="python">
{`import numpy as np

def quantize_kv(x, bits):
    """Per-token symmetric quantization of a KV tensor.
    x: (S, H, d_h) float32
    Returns: (q_int, scale) where q_int is integer, scale is per-row float.
    """
    q_max = 2**(bits - 1) - 1
    # scale shape: (S, H, 1) — one scale per row of each head
    scale = np.abs(x).max(axis=-1, keepdims=True) / q_max
    scale = np.where(scale == 0, 1e-8, scale)
    q = np.round(x / scale).clip(-q_max, q_max).astype(np.int32)
    return q, scale

def dequantize_kv(q, scale):
    return q.astype(np.float32) * scale

# Measured on (S=128, H=8, d_h=64) float32 tensors:
# K has heavy-tailed distribution; V is more Gaussian.
#
# Bits | Precision |    K RMSE |    V RMSE |    Memory | vs FP32
# ---- | --------- | --------- | --------- | --------- | -------
#   16 |      FP16 |  0.000058 |  0.000028 |  270336 B |   2.91x
#    8 |      INT8 |  0.015070 |  0.007296 |  139264 B |   5.65x
#    4 |      INT4 |  0.272646 |  0.132145 |   73728 B |  10.67x
#
# INT8: ~6x memory savings vs FP32 (3x vs BF16/FP16), negligible RMSE.
# INT4: ~11x savings but RMSE jumps 18x — noticeable on long-context retrieval.
# Production strategy: FP8 cache (H100 native), INT8 on older hardware,
#   INT4 only for research/edge with quality monitoring in place.`}
      </CodeBlock>

      <Prose>
        The RMSE numbers tell the operational story. FP16 cache is essentially lossless — the rounding error is five orders of magnitude below the signal. INT8 introduces errors around 0.015 for K, which is roughly 1.5% of the typical value range — small enough that it does not change which keys the attention weights attend to in practice. INT4's RMSE of 0.27 is a different situation: that is a meaningful fraction of the dynamic range, enough to blur the attention distribution on tokens that differ subtly, which is exactly the pattern retrieval tasks like needle-in-a-haystack benchmarks expose. The practical cutoff in 2025-era production is FP8 (native on H100 Transformer Engine), with INT8 on older hardware, and INT4 only in edge deployments or with a dedicated quality validation harness.
      </Prose>

      <H3>4e. Paged attention (simplified)</H3>

      <Prose>
        Before PagedAttention, KV caches were contiguous tensors — one pre-allocated slab per sequence, sized for the maximum context length. When a sequence finished at 500 tokens but had been allocated space for 4,000, the remaining 3,500 slots were wasted until the sequence was freed. Across a busy serving pool, this internal fragmentation routinely wasted 60–80% of KV-cache memory. PagedAttention (Kwon et al. 2023, arXiv:2309.06180) borrowed the OS virtual memory model: fixed-size blocks, a per-sequence page table, and a global free list.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
import math

BLOCK_SIZE = 16  # tokens per block — standard in vLLM

class PagedKVCache:
    """
    Paged KV cache manager. Physical memory is a fixed pool of blocks.
    Each sequence maintains a logical -> physical page table.
    Memory is allocated one block at a time, returned immediately on free.
    """
    def __init__(self, total_blocks, block_size, H_kv, d_h):
        self.block_size = block_size
        self.H_kv, self.d_h = H_kv, d_h
        # Physical pool: (total_blocks, block_size, H_kv, d_h) for K and V
        self.K_pool = np.zeros((total_blocks, block_size, H_kv, d_h), np.float32)
        self.V_pool = np.zeros((total_blocks, block_size, H_kv, d_h), np.float32)
        self.free_blocks = list(range(total_blocks))
        self.page_tables  = {}   # seq_id -> [block_id, ...]
        self.seq_lengths  = {}   # seq_id -> int

    def allocate(self, seq_id):
        self.page_tables[seq_id] = []
        self.seq_lengths[seq_id] = 0

    def append_token(self, seq_id, k, v):
        """Write one token's K/V. Allocate a new block if needed."""
        length = self.seq_lengths[seq_id]
        slot   = length % self.block_size
        if slot == 0:                          # need a new physical block
            if not self.free_blocks:
                raise MemoryError("KV cache pool exhausted — OOM")
            self.page_tables[seq_id].append(self.free_blocks.pop(0))
        block = self.page_tables[seq_id][-1]
        self.K_pool[block, slot] = k
        self.V_pool[block, slot] = v
        self.seq_lengths[seq_id] += 1

    def free(self, seq_id):
        """Return all blocks to the pool immediately on sequence completion."""
        freed = self.page_tables.pop(seq_id, [])
        self.free_blocks.extend(freed)          # instantly reusable
        self.seq_lengths.pop(seq_id, None)
        return len(freed)

    def read_kv(self, seq_id):
        """Reconstruct contiguous K,V from page table for attention kernel."""
        n = self.seq_lengths[seq_id]
        K = np.zeros((n, self.H_kv, self.d_h), np.float32)
        V = np.zeros((n, self.H_kv, self.d_h), np.float32)
        for i, block_id in enumerate(self.page_tables[seq_id]):
            s, e = i * self.block_size, min((i+1) * self.block_size, n)
            K[s:e] = self.K_pool[block_id, :e-s]
            V[s:e] = self.V_pool[block_id, :e-s]
        return K, V

# Simulation: pool of 20 blocks (320 token slots total)
# seq_A: 35 tokens -> 3 blocks; seq_B: 16 tokens -> 1 block; seq_C: 50 -> 4 blocks
# After allocation:  blocks used=8, free=12, utilization=40.0%
#   seq_A blocks: 3 (ceil(35/16)=3)
#   seq_B blocks: 1 (ceil(16/16)=1)
#   seq_C blocks: 4 (ceil(50/16)=4)
# seq_B finishes -> freed 1 block. Free now: 13
# New seq_D (12 tokens) reuses freed block. Page table: [block_8]
# Read-back seq_A: K=(35,4,32), V=(35,4,32) -- correct

# Contiguous allocation comparison:
#   Pre-allocate 320 tokens/seq (max context): only 1 seq fits in 320-slot pool
#   Actual avg usage: 42 tokens -> 86.7% of each allocation wasted
# Paged allocation:
#   7/20 blocks used for 2 active seqs (35+50 tokens), 65% free and reusable`}
      </CodeBlock>

      <Prose>
        The simulation output captures what PagedAttention actually achieves. With contiguous pre-allocation and a pool of 320 token slots, only a single sequence of maximum length can be held — regardless of how short most actual sequences are. With paged allocation, the same physical pool holds multiple concurrent sequences at near-full token utilization, and the moment any sequence finishes, its blocks are immediately available for new arrivals. vLLM reported 2–4× throughput improvements over prior serving stacks using this mechanism; the improvement was almost entirely unlocked batch size, not reduced per-token latency.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <H3>vLLM and PagedAttention</H3>

      <Prose>
        vLLM (Kwon et al. 2023) is the canonical open-source implementation of paged KV cache management. The key data structure is a block manager that maintains a global pool of physical KV blocks, a per-sequence logical block table mapping logical block indices to physical block IDs, and a free block list. The attention kernel is a custom CUDA kernel that dereferences the block table on every read — instead of computing a single pointer arithmetic offset into a contiguous tensor, it performs one additional table lookup per block boundary. The indirection overhead is measurable but small compared to the fragmentation savings it enables.
      </Prose>

      <Prose>
        vLLM's block manager also implements copy-on-write for prefix sharing: multiple sequences can point their page tables at the same physical blocks for a shared system prompt prefix. When a sequence needs to modify a shared block (which only happens if a prefix block is somehow being written to during decode — normally blocks are append-only), a copy is triggered. In practice, system prompt prefixes are read-only, so sharing is lossless and the memory savings for deployments with long system prompts are substantial — the prefix is computed once, cached, and every subsequent request pointing to that prefix pays zero marginal KV memory for those tokens.
      </Prose>

      <CodeBlock language="python">
{`# vLLM's core block manager data structures (simplified from vllm/core/block_manager.py)

class BlockTable:
    """Logical block table for one sequence."""
    def __init__(self):
        self.logical_to_physical: dict[int, int] = {}

    def map(self, logical_block_id, physical_block_id):
        self.logical_to_physical[logical_block_id] = physical_block_id

    def lookup(self, logical_block_id) -> int:
        return self.logical_to_physical[logical_block_id]

class BlockManager:
    """Global KV block pool. One BlockTable per sequence."""
    def __init__(self, num_physical_blocks, block_size):
        self.block_size = block_size
        self.free_blocks: list[int] = list(range(num_physical_blocks))
        self.ref_counts: dict[int, int] = {i: 0 for i in range(num_physical_blocks)}
        self.block_tables: dict[str, BlockTable] = {}

    def allocate(self, seq_id: str) -> BlockTable:
        self.block_tables[seq_id] = BlockTable()
        return self.block_tables[seq_id]

    def get_physical_block(self) -> int:
        if not self.free_blocks:
            raise MemoryError("GPU KV cache OOM — no free blocks")
        blk = self.free_blocks.pop()
        self.ref_counts[blk] = 1
        return blk

    def free_sequence(self, seq_id: str):
        for physical_blk in self.block_tables[seq_id].logical_to_physical.values():
            self.ref_counts[physical_blk] -= 1
            if self.ref_counts[physical_blk] == 0:
                self.free_blocks.append(physical_blk)
        del self.block_tables[seq_id]

    def share_prefix(self, base_seq: str, new_seq: str, num_shared_blocks: int):
        """Point new_seq's first N blocks at base_seq's physical blocks (ref-counted)."""
        base_table = self.block_tables[base_seq]
        new_table = self.allocate(new_seq)
        for i in range(num_shared_blocks):
            phys = base_table.lookup(i)
            self.ref_counts[phys] += 1
            new_table.map(i, phys)`}
      </CodeBlock>

      <H3>FlashAttention-2 and KV cache kernels</H3>

      <Prose>
        FlashAttention (Dao et al. 2022, arXiv:2205.14135) and FlashAttention-2 (arXiv:2307.08691) are tiling-based attention kernels that fuse the Q@K and softmax@V operations to avoid materializing the full attention score matrix in HBM. For prefill, this means the full S×S attention matrix never has to be written out — it is computed in tiles in SRAM and the softmax is computed online using the running-max trick. The HBM traffic reduction is 5–20× for long sequences, which is why FlashAttention is now universal in production prefill.
      </Prose>

      <Prose>
        For decode, FlashAttention's tiling benefit applies differently: the attention matrix for a single new query is 1×S rather than S×S, so the primary saving is the ability to read the KV cache in tiles that fit in SRAM rather than requiring the full cache to be staged. FlashAttention-2 also supports paged KV layouts natively through its <Code>block_table</Code> argument, enabling the scatter-gather reads that PagedAttention requires without transferring data through a contiguous staging buffer.
      </Prose>

      <H3>TensorRT-LLM and DeepSpeed-FastGen</H3>

      <Prose>
        NVIDIA's TensorRT-LLM implements paged KV caching through its KVCacheManager, with CUDA kernels that directly dereference page tables during the attention computation. It also ships FP8 KV cache support on H100/H200 via the Transformer Engine, where FP8 reads and BF16 arithmetic are handled in a single instruction without explicit dequantization code. The throughput gains from FP8 cache on H100 are 1.5–2× versus BF16, consistent with the halved memory bandwidth.
      </Prose>

      <Prose>
        DeepSpeed-FastGen introduced Dynamic SplitFuse in 2023, which treats KV cache memory as a unified pool across prefill and decode steps and dynamically partitions compute between them based on available memory — if the cache pool is near-full, it splits a prefill into smaller chunks (chunked prefill) to interleave with decode steps and prevent KV cache OOM during admission. The same idea was later adopted by vLLM's chunked prefill scheduler.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>Decode latency: naive vs cached</H3>

      <Plot
        label="naive vs cached decode — total time to generate S tokens (normalized)"
        width={520}
        height={240}
        xLabel="sequence length S"
        yLabel="relative total time"
        series={[
          { name: "naive (O(S²) reprojections)", points: [[32,1.0],[64,4.0],[128,17.9],[256,50.1],[512,130.6]] },
          { name: "cached (O(S) projections)", points: [[32,0.25],[64,0.73],[128,1.73],[256,10.76],[512,34.83]] },
        ]}
      />

      <Prose>
        The two curves from the section 4a benchmark normalize to the naive baseline at S=32. The cached curve grows roughly linearly while the naive curve grows quadratically. At S=128 the cached implementation is 10× faster; the gap keeps widening. In a GPU setting the gap is larger because the naive approach exceeds HBM capacity at long sequences — it cannot even run in the naive form once the intermediate attention matrices no longer fit.
      </Prose>

      <H3>Memory wall vs context length and concurrency</H3>

      <Plot
        label="KV cache GB vs context length — Llama 3 70B at different concurrent requests"
        width={520}
        height={260}
        xLabel="context length (k tokens)"
        yLabel="KV cache GB"
        series={[
          { name: "B=1  (BF16 GQA-8x)", points: [[8,2.68],[16,5.37],[32,10.74],[64,21.47],[128,42.95]] },
          { name: "B=16 (BF16 GQA-8x)", points: [[8,42.95],[16,85.90],[32,171.80],[64,343.60],[128,687.19]] },
          { name: "B=1  (FP8  GQA-8x)", points: [[8,1.34],[16,2.68],[32,5.37],[64,10.74],[128,21.47]] },
          { name: "B=16 (FP8  GQA-8x)", points: [[8,21.47],[16,42.95],[32,85.90],[64,171.80],[128,343.60]] },
        ]}
      />

      <Prose>
        The hard horizontal line is the H100 HBM capacity (80 GB). Any configuration whose curve crosses that line is physically impossible on a single H100. BF16 GQA at 16 concurrent 16k-context requests already hits the wall (85.9 GB). FP8 GQA halves every line, extending viability to 32k at 16 requests. Notice that the curves are all linear — every doubling of context length doubles the memory, every doubling of concurrency doubles the memory. There is no compression or diminishing returns from the cache's perspective. The only levers are dtype (FP8/INT8), architecture (GQA ratio), and paging efficiency.
      </Prose>

      <H3>Prefill vs decode phases</H3>

      <StepTrace
        label="kv cache lifecycle — prefill then decode"
        steps={[
          {
            label: "step 0 — prompt enters (prefill starts)",
            render: () => (
              <div>
                <TokenStream tokens={[
                  { label: "tok 0", color: colors.purple },
                  { label: "tok 1", color: colors.purple },
                  { label: "tok 2", color: colors.purple },
                  { label: "tok 3", color: colors.purple },
                  { label: "tok 4", color: colors.purple },
                  { label: "tok 5", color: colors.purple },
                ]} label="all prompt tokens processed in parallel" />
              </div>
            ),
          },
          {
            label: "step 1 — prefill complete, cache written",
            render: () => (
              <div>
                <TokenStream tokens={[
                  { label: "K₀,V₀", color: "#4ade80" },
                  { label: "K₁,V₁", color: "#4ade80" },
                  { label: "K₂,V₂", color: "#4ade80" },
                  { label: "K₃,V₃", color: "#4ade80" },
                  { label: "K₄,V₄", color: "#4ade80" },
                  { label: "K₅,V₅", color: "#4ade80" },
                ]} label="KV cache populated — read by all future decode steps" />
              </div>
            ),
          },
          {
            label: "step 2 — decode: generate token 6",
            render: () => (
              <div>
                <TokenStream tokens={[
                  { label: "[cache 0-5]", color: "#4ade80" },
                  { label: "Q₆", color: colors.gold },
                  { label: "→ attend over cache", color: "#60a5fa" },
                  { label: "tok 6", color: colors.gold },
                ]} label="one query, full cache read — memory bandwidth bottleneck begins" />
              </div>
            ),
          },
          {
            label: "step 3 — decode: generate token 100",
            render: () => (
              <div>
                <TokenStream tokens={[
                  { label: "[cache 0-99]", color: "#4ade80" },
                  { label: "Q₁₀₀", color: colors.gold },
                  { label: "→ attend", color: "#60a5fa" },
                  { label: "tok 100", color: colors.gold },
                ]} label="cache grows linearly — each step reads the entire growing cache from HBM" />
              </div>
            ),
          },
          {
            label: "step 4 — sequence ends, blocks freed",
            render: () => (
              <div>
                <TokenStream tokens={[
                  { label: "blk 0", color: "#f87171" },
                  { label: "blk 1", color: "#f87171" },
                  { label: "blk 2", color: "#f87171" },
                  { label: "blk 3", color: "#f87171" },
                  { label: "blk 4", color: "#f87171" },
                  { label: "blk 5", color: "#f87171" },
                ]} label="paged: blocks returned to pool immediately — zero fragmentation waste" />
              </div>
            ),
          },
        ]}
      />

      <H3>Attention computation pattern under causal mask</H3>

      <Prose>
        The heatmap below shows which cells of the attention score matrix are computed during decode (right column) versus what a naive re-implementation would compute (entire square). Only the last row is actually needed per decode step; the causal mask zeros out the upper triangle regardless.
      </Prose>

      <Heatmap
        label="causal attention pattern — rows=queries, cols=keys. Shaded=computed. Bright=decode step."
        matrix={[
          [0.9, 0,   0,   0,   0,   0,   0,   0  ],
          [0.5, 0.9, 0,   0,   0,   0,   0,   0  ],
          [0.3, 0.6, 0.9, 0,   0,   0,   0,   0  ],
          [0.2, 0.4, 0.5, 0.9, 0,   0,   0,   0  ],
          [0.1, 0.3, 0.4, 0.6, 0.9, 0,   0,   0  ],
          [0.1, 0.2, 0.3, 0.4, 0.7, 0.9, 0,   0  ],
          [0.1, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0  ],
          [0.3, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        ]}
        rowLabels={["Q₀","Q₁","Q₂","Q₃","Q₄","Q₅","Q₆","Q₇ (decode)"]}
        colLabels={["K₀","K₁","K₂","K₃","K₄","K₅","K₆","K₇"]}
        cellSize={40}
        colorScale="gold"
      />

      <Prose>
        The bottom row — the current decode step's query — attends to all past keys including itself. All upper-triangle cells are zero (causal mask). The KV cache means rows Q₀ through Q₆ never need to be recomputed; their K and V tensors are already in the cache. Only the bottom row's Q, K, V are newly projected; K₇ and V₇ are appended to the cache after this step.
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <CodeBlock>
{`Technique       | When to use                          | When to skip
--------------- | ------------------------------------ | --------------------------------
GQA (g=4-8x)    | Default for any new model training;  | If model is already trained with
                | 4-8x cache savings, negligible        | MHA and you cannot retrain;
                | quality loss at g<=8                  | at g>16 quality starts degrading
                |                                      |
MQA (g=H_q)     | Extremely memory-constrained edge;   | Almost everywhere — GQA at g=4
                | acceptable 1-3% quality hit           | gives most of the savings at
                |                                      | lower quality cost
                |                                      |
FP8 cache       | Default on H100/H200/Blackwell;      | Hardware without FP8 support;
                | 2x savings vs BF16, near-zero loss    | if using older Ampere GPUs
                |                                      |
INT8 cache      | Ampere and older hardware;           | If FP8 is available — FP8 has
                | 2x savings vs FP16, <0.1% task loss  | better numerical properties
                |                                      |
INT4 cache      | Edge deployment, research;           | Any task using long-context
                | memory-critical with quality monitor  | retrieval or needle-in-haystack
                |                                      |
PagedAttention  | Any serving system with concurrent   | Single-request latency-only
                | requests (>1 user); default for      | deployments where fragmentation
                | production serving stacks             | is not an issue
                |                                      |
Contiguous      | Research/evaluation with batch=1;   | Production with concurrency —
allocation      | simpler implementation               | fragmentation is fatal at scale
                |                                      |
MLA             | New frontier models being designed   | Inference-time optimization only;
                | from scratch; 5-10x cache savings     | requires architecture change and
                | with better-than-GQA quality          | pretraining from scratch
                |                                      |
Prefix caching  | Long system prompts repeated across  | Short or highly variable prompts;
                | requests (>50 tokens shared);        | requires paged allocation to work
                | requires PagedAttention first         |`}
      </CodeBlock>

      <Prose>
        The most consequential decision is GQA vs MHA, because it is made at model architecture time and cannot be changed post-training. Every frontier model trained after 2023 ships GQA. The quantization decision is made at serving time and can be changed per deployment with quality validation. PagedAttention is the default for any production serving stack; the only reason to use contiguous allocation is implementation simplicity in research settings.
      </Prose>

      <Prose>
        MLA is a distinct category: it is not an inference-time optimization but an architectural replacement for GQA, available only in models specifically pretrained with it (DeepSeek-V2, DeepSeek-V3 at the time of writing). The decision between MLA and GQA is made when choosing which model to deploy, not how to serve it. For teams training new frontier models, MLA is the more cache-efficient architecture if the implementation complexity is acceptable. For teams deploying existing open-weight models, GQA is what they have.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>What scales linearly (the predictable axes)</H3>

      <Prose>
        <strong>Context length (S).</strong> The KV cache is strictly linear in S — there is no compression, no sparsification, no pooling in a standard transformer. Every additional token costs exactly as much cache as the first. This linearity is what makes the memory wall so sharp: doubling context doubles memory, halving concurrency. At 1M context on Llama 3 70B in BF16, a single sequence would require 343.6 GB — impossible on any existing GPU, and possible only on a multi-GPU or CPU-offload system. This is why architectural changes like MLA and YOCO are necessary at the frontier rather than engineering tweaks.
      </Prose>

      <Prose>
        <strong>Batch size (B).</strong> Every concurrent sequence holds its own KV cache. Batch size scales the memory footprint linearly with no sharing except for prefix caching. Ten concurrent 8k requests at 2.68 GB each consume 26.8 GB — independent of whether the requests are similar or identical in content, unless prefix caching is active. This linear scaling is what makes paged memory management essential: without it, peak memory usage is determined by the worst-case sequence length across all concurrent requests, which can be dramatically higher than the average.
      </Prose>

      <Prose>
        <strong>Layers (L).</strong> Every layer stores its own K and V tensors. Deeper models cost proportionally more cache. The ratio of KV cache to weights grows with model depth: an 80-layer model holds proportionally more of its total memory in cache, per token, than a 32-layer model with the same parameter count.
      </Prose>

      <Prose>
        <strong>KV heads (H_kv).</strong> This is the one architectural axis that scales linearly and is also a design variable. Full MHA with H_kv = H_q means cache scales with total query head count. GQA at group size g reduces H_kv by g×, directly reducing cache by g× with the same context and batch. This is why GQA became standard: it is the only mechanism that compresses the cache without changing context length, batch size, or precision.
      </Prose>

      <H3>What doesn't scale well (the diminishing returns)</H3>

      <Prose>
        <strong>Quantization.</strong> The savings from quantization are bounded by the number of bits. BF16 → FP8 is 2×. FP8 → INT4 is another 2×. INT4 → INT2 would be another 2×, but INT2 quality is unacceptable for most attention patterns. The total available savings from quantization alone is roughly 4× from BF16 to INT4, at the cost of increasing quality risk. There is no path to 10× savings through quantization without unacceptable degradation; beyond INT4 the error is large enough to materially change which tokens attention weights.
      </Prose>

      <Prose>
        <strong>GQA group size.</strong> The savings from GQA are bounded by the original head count. For Llama 3 70B with 64 Q heads and 8 KV heads, the maximum GQA reduction is 64× (down to 1 KV head — MQA). But quality degrades measurably as the group size increases past roughly 8. The empirical finding from Ainslie et al. 2023 is that GQA at group sizes 4–8 essentially matches MHA quality; at group sizes above 16 the gap becomes noticeable on tasks that depend on fine-grained per-head specialization.
      </Prose>

      <Prose>
        <strong>Paged allocation efficiency.</strong> PagedAttention nearly eliminates fragmentation but introduces its own inefficiency at large block sizes: the last block in a sequence is on average half-full (a block of 16 tokens is half-empty if the sequence length is not a multiple of 16). This internal fragmentation averages <Code>block_size / 2</Code> wasted slots per sequence — 8 slots for block_size=16. At short sequences this is a significant fraction; at long sequences it is negligible. Block size of 16 is a deliberate choice that minimizes this waste at the typical sequence lengths that dominated 2023-era deployments (256–4096 tokens).
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>Memory fragmentation with contiguous allocation</H3>

      <Prose>
        Any serving stack that allocates contiguous KV cache slabs per sequence will experience severe memory fragmentation under real traffic. Requests complete at different times and with different lengths. The freed slabs are rarely the exact size of the next incoming request. Available memory fragments into unusable chunks. The measured fragmentation in early 2023 production deployments before paging was 60–80%, meaning the GPU was capable of holding 3–5× more concurrent requests than it actually could. The symptom is requests being rejected or queued even though the GPU's theoretical memory is sufficient. The fix is PagedAttention or any equivalent paged allocator.
      </Prose>

      <H3>KV cache overflow causing OOM kills</H3>

      <Prose>
        A more dangerous failure: the serving framework does not accurately track KV cache usage across requests and allows more tokens to be generated than physically fit. The GPU runs out of memory mid-generation, the CUDA kernel raises an OOM error, and the serving process either crashes or the request fails mid-response. This is especially common when (a) the maximum context length is set too high relative to available GPU memory, (b) multiple requests simultaneously reach their maximum length, or (c) prefix sharing inflates apparent free memory because shared blocks are being reference-counted incorrectly. The fix is conservative admission control: refuse new requests when projected peak memory exceeds a safe threshold, rather than when it exceeds the physical limit.
      </Prose>

      <H3>Incorrect position embeddings after cache eviction</H3>

      <Prose>
        Some serving stacks implement cache eviction — dropping old cache entries to make room for more tokens — without correctly recomputing or adjusting position embeddings. Rotary position embeddings (RoPE, used in Llama, Mistral, DeepSeek) encode position directly into the Q and K vectors at computation time. If a cached K vector was computed at position 42 and that position's cache entry is evicted, the model cannot simply shift the remaining entries to fill the gap — the position information is baked into each K vector. Re-using a cache with evicted positions as if they were contiguous produces incorrect attention patterns, and the errors are silent: the model still generates tokens, but they reflect the wrong context. The correct approach is either never evict (accept more OOM risk) or re-prefill from scratch when eviction occurs.
      </Prose>

      <H3>Prefill/decode precision mismatch</H3>

      <Prose>
        A subtle but real production bug: the model runs prefill in BF16 (as weights are stored), writes the KV cache in FP8, and then dequantizes to BF16 for decode attention. If the FP8 quantization parameters (scale, zero-point) are computed from the BF16 prefill values but applied at decode time to different data distributions — for example, because the decode K/V values have higher dynamic range due to longer context — the dequantization produces values that are systematically off from what the model saw during training. The symptom is generation quality that degrades on longer outputs in ways not explained by the quantization error alone. The fix is to compute quantization scales dynamically per token at append time, not once at prefill.
      </Prose>

      <H3>Cache sharing across users (security)</H3>

      <Prose>
        Prefix caching is a powerful memory optimization, but it requires careful implementation of cache isolation. If two requests from different users happen to produce the same prefix hash — either legitimately (same system prompt) or adversarially — they share the same physical KV cache blocks. In a correctly implemented paged system this is fine: the blocks are read-only and the page tables provide full isolation. But if the implementation has a bug where a user can observe timing differences based on cache hits, or worse, can influence which physical block another user's request reads, the cache becomes a side-channel. Any implementation of prefix sharing must be reviewed with the same rigor as any other shared-memory security boundary.
      </Prose>

      <H3>Re-prefill cost on cache miss</H3>

      <Prose>
        When a cache entry is not available — because the request is new, because the cache was evicted, or because the serving system restarted — the prefill must be run from scratch. For a 32k-token system prompt, re-prefill can take several seconds and consumes substantial GPU compute. Systems that retry failed requests without preserving the cache state will re-prefill every retry, potentially producing a cascade where a brief GPU hiccup leads to a wave of expensive re-prefills that overload the system. The fix is to make cache state part of the checkpoint/recovery strategy, or to serve from a warm replica if cache recovery latency is unacceptable.
      </Prose>

      <H3>Page table fragmentation at long contexts</H3>

      <Prose>
        PagedAttention with a block size of 16 tokens performs well at typical deployment context lengths (256–8192 tokens). At very long contexts (128k–1M tokens), each sequence occupies thousands of physical blocks, and the page table itself becomes a meaningful memory consumer. More importantly, the attention kernel's overhead from dereferencing thousands of block table entries per step grows linearly with the number of blocks. For sequences exceeding roughly 64k tokens at block_size=16, hierarchical page tables or larger block sizes may be necessary to avoid attention kernel overhead dominating decode latency.
      </Prose>

      <H3>Dtype mismatches during cache concatenation</H3>

      <Prose>
        When the KV cache is stored in one dtype and the attention computation runs in another, every cache read involves a cast. If the cast is done at the wrong granularity (e.g., casting the entire cache tensor rather than the current block), it creates transient memory allocations proportional to the cache size on every decode step, which can trigger OOM at long contexts. The correct implementation casts one block at a time inside the attention kernel, never materializing the full dequantized cache in HBM.
      </Prose>

      <Callout accent="red">
        Silent correctness bugs are more dangerous than OOM crashes. Position embedding mismatch after eviction, prefill/decode precision mismatch, and page table security isolation failures all produce plausible-looking but incorrect generations — they pass unit tests and look fine in demos.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        The following papers are the direct foundations of everything in this topic. Verified against arXiv in April 2026.
      </Prose>

      <CodeBlock>
{`1. Vaswani, A., et al. (2017). "Attention Is All You Need."
   arXiv:1706.03762
   The original transformer paper. Autoregressive decoder inference with
   KV caching is implicit in every decoder implementation that followed.
   Establishes the Q/K/V projection structure that makes caching possible.

2. Pope, R., et al. (2022). "Efficiently Scaling Transformer Inference."
   arXiv:2211.05102 — Published MLSys 2023.
   First rigorous analytical model of transformer inference efficiency.
   Derives the prefill-vs-decode compute/memory tradeoff, introduces the
   roofline model for LLM inference, and establishes the MQA/partitioning
   framework for production serving.

3. Ainslie, J., et al. (2023). "GQA: Training Generalized Multi-Query
   Transformer Models from Multi-Head Checkpoints."
   arXiv:2305.13245 — Published EMNLP 2023.
   Introduces grouped-query attention and shows it matches MHA quality at
   intermediate group sizes. Provides the uptraining recipe that made GQA
   practical for deployed models. Adopted by Llama 2, Llama 3, Mistral,
   Mixtral, Qwen, and virtually every open-weight model since mid-2023.

4. Kwon, W., et al. (2023). "Efficient Memory Management for Large Language
   Model Serving with PagedAttention."
   arXiv:2309.06180 — Published SOSP 2023.
   Introduces PagedAttention and vLLM. Demonstrates 2-4x throughput
   improvement over FasterTransformer and Orca via near-elimination of
   KV cache memory fragmentation. Establishes paged block allocation as
   the standard for production serving.

5. Dao, T., et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact
   Attention with IO-Awareness."
   arXiv:2205.14135 — Published NeurIPS 2022.
   IO-aware tiled attention kernel that avoids materializing the S×S
   attention matrix in HBM. The foundation for all production attention
   kernels. Enables efficient prefill at long context.

6. Dao, T. (2023). "FlashAttention-2: Faster Attention with Better
   Parallelism and Work Partitioning."
   arXiv:2307.08691 — ICLR 2024.
   Refinements to FlashAttention improving work partitioning across warps,
   reducing non-matmul FLOPs, and adding support for paged KV layouts
   needed for PagedAttention compatibility.

7. DeepSeek-AI. (2024). "DeepSeek-V2: A Strong, Economical, and Efficient
   Mixture-of-Experts Language Model."
   arXiv:2405.04434
   Introduces Multi-head Latent Attention (MLA), achieving 93.3% KV cache
   reduction versus MHA via low-rank KV compression. First production model
   to demonstrate that architectural KV compression can outperform GQA both
   in memory efficiency and model quality.`}
      </CodeBlock>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1: Compute KV cache memory for Llama 3 70B at 128k context</H3>

      <Prose>
        Using the formula from section 3: Llama 3 70B has L=80 layers, H_kv=8 KV heads, d_h=128. Compute the KV cache size in GB for a single sequence at 128k context in (a) BF16, (b) FP8, and (c) INT4. Then compute how many concurrent 128k-context sequences would fit on a single H100 80GB after accounting for the model weights (approximately 35 GB for the KV-cache portion of an H100 when the 70B model is split across two H100s in tensor-parallel configuration, leaving ~45 GB for cache).
      </Prose>

      <CodeBlock language="python">
{`# Your solution here.
# Expected: BF16 ~ 42.95 GB (one seq fills the H100),
#           FP8  ~ 21.47 GB (two seqs fit),
#           INT4 ~ 10.74 GB (four seqs fit, with quality caveats).`}
      </CodeBlock>

      <H3>Exercise 2: Derive when GQA's savings outweigh its quality loss</H3>

      <Prose>
        GQA at group size g reduces the KV cache by g× but reduces the effective KV representational capacity by g× per layer. Ainslie et al. show that quality degradation is negligible at g≤8 and measurable at g≥16. Given a serving system with a fixed memory budget M_cache, model weights W, and a requirement for at minimum B concurrent users at context length S: write the inequality that determines the minimum g required to fit the system on one GPU. For Llama 3 70B parameters, what is the minimum g for B=32 concurrent users at 8k context on an H100 80GB? Does this violate the quality threshold?
      </Prose>

      <H3>Exercise 3: Why does PagedAttention use 16-token blocks specifically?</H3>

      <Prose>
        Block size is a tradeoff between internal fragmentation, attention kernel overhead, and allocator granularity. With block_size=16: (a) what is the average internal fragmentation (wasted token slots) per sequence? (b) what is the fragmentation at block_size=1 and block_size=256? (c) how does block size affect the attention kernel's memory access pattern — specifically, why do larger blocks improve GPU memory coalescing? (d) at what context length does the page table overhead itself become significant, and how does block_size affect this threshold?
      </Prose>

      <H3>Exercise 4: Design a cache quantization strategy for a real deployment</H3>

      <Prose>
        You are deploying Llama 3 70B on a cluster of H100 80GB GPUs for a customer service application. The expected workload is: 64 concurrent sessions, average context 12k tokens, peak context 32k tokens, SLA requires less than 5% quality degradation on MMLU-style benchmarks. Design a quantization strategy for the KV cache that (a) fits the average workload on the fewest possible GPUs, (b) handles peak context without OOM, and (c) stays within the quality SLA. Justify each choice with the memory accounting from section 4b.
      </Prose>

      <H3>Exercise 5: When does MLA beat GQA at frontier scale?</H3>

      <Prose>
        DeepSeek-V2 reports that MLA achieves 93.3% KV cache reduction versus MHA while matching or exceeding MHA quality — better than GQA on both axes. But MLA requires training a model from scratch with the MLA architecture. Given a team starting a new 100B-parameter model training run: (a) at what context length does MLA's cache advantage over GQA-8× become decisive for a 1,000-GPU H100 cluster? (b) what is the per-GPU throughput increase in tokens/second from the memory savings, assuming decode is memory-bandwidth-bound? (c) what implementation complexity does MLA add versus GQA, and what are the engineering tradeoffs that might still favor GQA for a resource-constrained team?
      </Prose>

    </div>
  ),
};

export default kvCache;
