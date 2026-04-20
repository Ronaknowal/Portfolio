import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { StepTrace, TokenStream, Plot } from "../../components/viz";

const kvCache = {
  title: "KV-Cache & Memory Management",
  readTime: "16 min",
  content: () => (
    <div>
      <Prose>
        Every optimization in LLM serving begins with understanding the KV cache. Without it, generating a thousand-token response would mean recomputing attention over every previous token at every new step — one, then two, then three, all the way up to a thousand, summing to roughly half a million attention operations for a reply that should have cost a thousand. The decoder would spend almost all of its life redoing work it had already done, and the cost of a long response would grow with the square of its length for no good reason. The KV cache is the observation that none of that recomputation is necessary, and the engineering that flows from accepting it.
      </Prose>

      <Prose>
        With the cache in place, each new token costs roughly the same amount of compute as the last — attention becomes linear in sequence length for generation, not quadratic. But the cache has its own cost, and the cost is severe: it dominates GPU memory during long-context inference, it scales with every concurrent user, and the way you manage it is the single biggest determinant of whether your serving stack is fast, cheap, and available, or slow, expensive, and falling over under load. Everything else in this section is, in one form or another, a technique for extracting more throughput out of the memory this cache is already consuming.
      </Prose>

      <H2>What the cache actually stores</H2>

      <Prose>
        Every transformer layer projects each input token into three vectors — query, key, and value — used by self-attention. During autoregressive generation, a useful asymmetry appears: the keys and values computed for every past token never change. Their projection depended only on the token embedding at that position and the weights of the layer, both of which are fixed. Only the current token's query gets computed fresh, and that query attends against the full history of keys and values. Caching the K and V tensors of every past token means the model never has to reproject them; only the one new position needs projection work per step. Every layer, every attention head, every token is a row of keys and a row of values sitting in GPU memory waiting to be read again.
      </Prose>

      <Prose>
        The arithmetic of how large that memory gets is deceptively simple and deceptively brutal. For a model with L layers, H KV heads per layer, head dimension d_h, sequence length S, batch size B, and a precision in bytes:
      </Prose>

      <MathBlock>{"\\text{KV cache size} = 2 \\cdot L \\cdot H \\cdot d_h \\cdot S \\cdot B \\cdot \\text{bytes}"}</MathBlock>

      <Prose>
        The factor of two is for keys and values held separately. Bytes is the precision: 2 for FP16 or BF16, 1 for FP8, half a byte for INT4. Plug numbers in and the scale stops being abstract. For Llama 3 70B — 80 layers, 8 KV heads after Grouped-Query Attention, head dimension 128 — a single 8k-context sequence in BF16 comes out to roughly 2.6 GB. One request. Thirty-two concurrent 8k sessions is 84 GB, already past the 80 GB of a single H100. Push context to 32k and batching evaporates. The model weights themselves — 140 GB in BF16 — are a one-time fixed cost paid across all users of that GPU. The KV cache is a per-user cost paid every time someone opens a chat window, and it is almost always the thing that runs out first.
      </Prose>

      <CodeBlock language="python">
{`def attention_with_kv_cache(q_new, k_new, v_new, k_cache, v_cache):
    """Attention step for one new token, given cached keys/values of previous tokens."""
    # Append the new token's K and V to the cache
    k_full = torch.cat([k_cache, k_new], dim=-2)  # (batch, heads, seq+1, d_h)
    v_full = torch.cat([v_cache, v_new], dim=-2)

    # Query attends to ALL past keys, including the new one
    scores = q_new @ k_full.transpose(-2, -1) / math.sqrt(k_full.size(-1))
    weights = F.softmax(scores, dim=-1)
    out = weights @ v_full

    return out, k_full, v_full  # return new cache for next step`}
      </CodeBlock>

      <Prose>
        The kernel in that snippet is doing the minimum possible work for one step of generation. One query vector against a key matrix that may be thousands of rows tall, one softmax, one weighted sum over the value matrix. But the keys and values for every previous token have to be streamed out of high-bandwidth memory to participate in that one computation, used once, and dropped until the next step. That read pattern is the quiet villain of the whole performance story.
      </Prose>

      <H2>Two phases, two bottlenecks</H2>

      <Prose>
        LLM inference has two sharply different regimes, and the optimizations that help one often do nothing for the other. The first is <em>prefill</em>: the whole prompt goes through the model in parallel. All prompt tokens are processed in a single forward pass, every layer producing its keys and values for every prompt position at once. The GPU is compute-bound — matrix multiplications at full utilization, tensor cores spinning, memory bandwidth secondary because each byte read from HBM contributes to many arithmetic operations. The KV cache for the entire prompt is written out in one big burst at the end of the pass.
      </Prose>

      <Prose>
        The second is <em>decode</em>: one token at a time, each attending to the growing cache. There is no parallelism along the sequence axis — the next token cannot be computed until the current one has been. Every step reads the full KV cache out of HBM, does a tiny amount of arithmetic against it (one query vector's worth per head), and writes a single new row back. The arithmetic intensity is catastrophic. A modern GPU can do roughly a hundred BF16 FLOPs per byte of HBM traffic at peak; decode steps deliver on the order of a single-digit number of FLOPs per byte. The GPU spends most of its wall-clock time waiting for memory, not computing. Decode is memory-bandwidth-bound, not compute-bound, and throwing faster tensor cores at it does almost nothing.
      </Prose>

      <Prose>
        This asymmetry shapes every subsequent optimization in inference. Continuous batching works because decode is bandwidth-bound: stacking more users into one batch costs almost no extra memory traffic per step. Speculative decoding works because decode is sequential: the bottleneck is steps, not FLOPs, and trading extra compute for fewer serial steps is a good bargain. Quantizing the cache helps because the bottleneck is bytes moved, and halving the bytes doubles effective bandwidth. Prefill optimizations — chunked prefill, prefix sharing — are a different problem with a different answer. Any serving stack that treats the two phases with a single strategy leaves huge throughput on the table.
      </Prose>

      <StepTrace
        label="prefill vs decode — the two inference phases"
        steps={[
          { label: "1. prefill — prompt tokens in parallel", render: () => (
            <TokenStream tokens={[
              { label: "tok 0", color: "#c084fc" },
              { label: "tok 1", color: "#c084fc" },
              { label: "tok 2", color: "#c084fc" },
              { label: "tok 3", color: "#c084fc" },
              { label: "tok 4", color: "#c084fc" },
              { label: "tok 5", color: "#c084fc" },
              { label: "→ full KV cache", color: "#4ade80" },
            ]} />
          ) },
          { label: "2. decode — one new token at a time", render: () => (
            <TokenStream tokens={[
              { label: "[cache 0-5]", color: "#4ade80" },
              { label: "tok 6", color: "#e2b55a" },
              { label: "→ append", color: "#60a5fa" },
            ]} />
          ) },
          { label: "3. decode step N — cache keeps growing", render: () => (
            <TokenStream tokens={[
              { label: "[cache 0-99]", color: "#4ade80" },
              { label: "tok 100", color: "#e2b55a" },
              { label: "→ append", color: "#60a5fa" },
            ]} />
          ) },
        ]}
      />

      <Prose>
        A related consequence: prefill latency and decode latency are effectively independent. Time-to-first-token is set by how fast the GPU can crunch a parallel matmul; inter-token latency is set by how fast it can stream the cache out of HBM. A model can have fast TTFT and slow tokens-per-second, or the reverse, and the tuning for each lives in different parts of the stack.
      </Prose>

      <H2>GQA and MQA — shrinking the cache</H2>

      <Prose>
        The memory formula above has one term that architecture can attack directly: H, the number of attention heads for keys and values. Query heads are load-bearing — they are how the model asks questions of the past — but the number of key and value heads does not have to match. If you could reduce the K and V head count aggressively while keeping Q heads high, the cache shrinks proportionally, and so does the HBM traffic during decode. This is the observation that Multi-Query Attention and Grouped-Query Attention are built on.
      </Prose>

      <Prose>
        Multi-Query Attention (MQA, Shazeer 2019) is the extreme: one K head and one V head, shared across every Q head in the layer. The cache shrinks by a factor equal to the original number of heads — 32× or 64× on a typical large model. Quality takes a small but measurable hit, because collapsing all attention heads onto one K/V projection removes some of the representational richness that multi-head attention was designed to provide. Grouped-Query Attention (GQA, Ainslie et al. 2023) is the middle ground that most of the field settled on: groups of Q heads share a single K/V head, with the group size tunable. Llama 2 introduced GQA at the frontier; Llama 3 70B uses 64 Q heads but only 8 KV heads, for an 8× reduction. Mistral, Mixtral, Qwen, DeepSeek, and almost every modern open-weight model ship some variant of GQA.
      </Prose>

      <Prose>
        The practical effect is that, for a given memory budget, you can either serve roughly 8× more concurrent requests, support an 8× longer context per request, or split the savings across both. The architectural cost is small: careful evaluations show GQA at a sensible group size matches full multi-head attention at the same parameter count, because the lost K/V capacity is recoverable by the attention pattern itself — the model learns to route information through shared heads in a way that does not meaningfully reduce what it can represent. MQA's collapse to a single head is more visibly damaging, which is why GQA rather than MQA is what shipped. Every byte of cache you can avoid storing is a byte you do not have to read back during decode, and the bandwidth savings compound with the memory savings.
      </Prose>

      <H2>Quantization of the cache</H2>

      <Prose>
        The rest of the model may be in BF16 or FP16, but the KV cache does not have to match. The cache is storage, not computation — its values get read, dequantized on the fly, and used in BF16 arithmetic. You can store it in any precision that survives the round trip without damaging quality. FP8 KV cache is now standard on H100-class hardware, where the Transformer Engine supports it natively and the memory savings are exactly 2× versus BF16. INT8 is common on hardware without FP8 support; INT4 shows up in research and on memory-starved edge deployments. Below INT4, quality degradation becomes visible on long-context retrieval and needle-in-a-haystack evals — the cache is not just being stored, it is being retrieved by attention, and low-precision retrieval eventually blurs distinctions the model was relying on.
      </Prose>

      <CodeBlock language="python">
{`def quantize_kv(k, v, bits=8):
    """Per-token, per-head, symmetric quantization of the KV cache."""
    # Compute per-token max for each head
    k_scale = k.abs().amax(dim=-1, keepdim=True) / (2**(bits-1) - 1)
    v_scale = v.abs().amax(dim=-1, keepdim=True) / (2**(bits-1) - 1)
    k_quant = torch.round(k / (k_scale + 1e-8)).clamp(-127, 127).to(torch.int8)
    v_quant = torch.round(v / (v_scale + 1e-8)).clamp(-127, 127).to(torch.int8)
    return k_quant, v_quant, k_scale, v_scale

# 50% memory reduction vs BF16, with near-zero measured quality loss on most tasks.
# INT4 cuts another 50% but shows degradation on long-context retrieval.`}
      </CodeBlock>

      <Prose>
        The granularity of the scale factor matters. Per-tensor quantization — one scale for the whole cache — is cheap but lossy, because outliers in a few heads dominate the scale and crush the precision available for everything else. Per-token, per-head quantization, as in the snippet above, keeps a small scale factor per row and costs a few extra bytes per token to preserve precision where it matters. Production implementations go further: mixed-precision caches that keep the most recent tokens in BF16 and older ones in FP8, or separate quantization schemes for K and V because their distributions differ in practice — K tends to have heavier tails.
      </Prose>

      <H2>Fragmentation — the problem PagedAttention solved</H2>

      <Prose>
        A subtle but operationally crucial issue, and one that dominated production serving quality in 2023 before the fix landed. Before vLLM, every serving stack stored KV caches as contiguous tensors per sequence. When a new request arrived, you allocated a slab of GPU memory large enough to hold its maximum possible cache — the context window's worth — and that slab stayed pinned to that sequence until it finished. If the user only generated 500 tokens before the conversation ended but you had allocated space for 4,000, the remaining 3,500 slots were wasted. Worse, if a new 2,000-token prefill arrived and no single slab of 2,000 tokens was free, the request had to wait or be evicted, even though the <em>aggregate</em> free memory across the fragmented pool was more than sufficient. Internal fragmentation routinely wasted sixty to eighty percent of KV-cache memory. Effective throughput was a fraction of what the hardware was physically capable of.
      </Prose>

      <Prose>
        PagedAttention, introduced by Kwon et al. in 2023 with the vLLM project, borrowed the mental model directly from operating-system virtual memory. Break the KV cache into fixed-size blocks — sixteen tokens is the typical block size — and manage allocation through a per-sequence page table that maps logical token positions to physical blocks scattered anywhere in a global pool. A 500-token sequence uses thirty-two blocks, wherever they happen to land; when the sequence finishes, those blocks return to the pool and any incoming request can grab them. The attention kernel is modified to dereference the page table on every read, so the fragmented storage is invisible to the model. No internal fragmentation. Near-full memory utilization in practice. vLLM reported throughput improvements of 2–4× over the best prior serving stacks, and the improvement was almost entirely unlocked batch size — the same GPU, the same model, the same requests, just able to hold more of them concurrently because memory was not being wasted.
      </Prose>

      <Prose>
        This is the single largest efficiency win in modern LLM serving. Every major inference engine — vLLM, TensorRT-LLM, SGLang, TGI, LMDeploy — now implements some form of paged KV cache. The PagedAttention topic later in this section goes into the mechanism in depth. The key observation for now is simpler: the cache looks like a single contiguous tensor from the model's perspective and is actually a scatter-gather structure from the allocator's perspective, and that one indirection closes most of the gap between theoretical and achievable throughput.
      </Prose>

      <H3>Prefix sharing across requests</H3>

      <Prose>
        A second memory win flows directly from paged allocation. If multiple concurrent requests share a system prompt, a tool-use header, a few-shot example block, or any other identical leading sequence of tokens, their KV caches for that shared prefix are <em>bit-for-bit identical</em>. The keys and values are a deterministic function of the token ids and the model weights; two requests that see the same tokens through the same weights produce exactly the same K and V. With contiguous per-sequence caches this observation was academic, because you had no way to share a slab across sequences. With a paged cache, you can hash each block of prompt tokens as it comes in, and if a block is already in the cache — from a previous request or a concurrent one — just point the new request's page table at the existing physical block instead of recomputing or reallocating. This is prefix caching, and it is massive for production deployments with long system prompts repeated across every single call. The topic later in this section covers it in detail; the mechanism depends on paging being in place first.
      </Prose>

      <H2>Memory as a throughput lever</H2>

      <Prose>
        The practical calculus of serving economics reduces to this: every KV byte saved is either another concurrent request you can serve, or more context length per existing request. The memory budget is, directly and almost exclusively, the serving budget. An H100 with 80 GB holds roughly 60 GB of KV cache after the 70B model weights and activation scratch are accounted for. If a single request consumes 2.6 GB of cache at 8k context, the GPU can hold about 23 concurrent requests. Halve the cache with GQA and it holds 46. Halve it again with FP8 and it holds 92. Every doubling of effective cache density is a doubling of throughput for the same hardware, and for long enough deployments it is a doubling of revenue per GPU.
      </Prose>

      <Plot
        label="KV cache memory vs. concurrent requests (illustrative, 70B model at 8k context)"
        width={520}
        height={240}
        xLabel="KV cache GB per request"
        yLabel="concurrent requests on H100-80GB"
        series={[
          { name: "BF16 MHA (baseline)", points: [[20, 4], [16, 5], [12, 6], [8, 10], [4, 20]] },
          { name: "BF16 GQA-8x", points: [[2.5, 32], [2, 40], [1.5, 53], [1, 80]] },
          { name: "FP8 GQA-8x + paged", points: [[1.25, 64], [1, 80], [0.75, 106], [0.5, 160]] },
        ]}
      />

      <Prose>
        The three curves tell the story of the last three years of inference progress. The baseline is full multi-head attention with BF16 cache and contiguous allocation — the 2022 configuration. A single H100 could barely reach double digits on concurrent requests at realistic context lengths. Adding GQA with an 8× reduction bumps the same GPU into the tens. Adding FP8 cache and paged allocation pushes into the hundreds. None of these changes touched the model's benchmark quality in any meaningful way; every one is a pure engineering win, extracting more throughput from memory that was always there. Frontier stacks in 2025 combine all of the above with continuous batching, speculative decoding, and prefix sharing, and the resulting throughput on a single GPU is roughly an order of magnitude better than what the same hardware delivered two years earlier running the same model.
      </Prose>

      <Callout accent="gold">
        Every KV byte saved is a byte of throughput. GQA, FP8, paging, and prefix sharing are not alternatives to one another — they stack, and most modern serving stacks apply all four.
      </Callout>

      <H3>Cross-layer KV sharing — the research frontier</H3>

      <Prose>
        A recent line of architectural work pushes on a question the engineering tricks above never asked: do we really need a separate KV cache per layer? Every transformer layer stores its own keys and values, and the cache size scales linearly with L. Multi-head Latent Attention (MLA), introduced by DeepSeek in V2 and refined in V3, projects K and V into a shared low-rank latent that is cached <em>once</em> and decompressed per layer at read time. The storage reduction is five to tenfold depending on the rank, and the paper reports negligible quality loss — the latent is expressive enough to reconstruct what each layer needs. YOCO (You Only Cache Once, Microsoft 2024) goes further, caching K/V for only one layer and having all other layers attend against that single shared cache. Both are architectural changes rather than pure inference tricks — they require the model to be designed and pretrained for the scheme — but they are reshaping what is possible at million-token contexts, where even GQA plus FP8 starts to run out of memory at realistic batch sizes. The KV cache is no longer treated as a fixed cost of the architecture; it is a design variable being actively optimized.
      </Prose>

      <H2>What this section covers next</H2>

      <Prose>
        The KV cache is the shared substrate for every remaining topic in this section. Decoding strategies decide what token to emit given the cached context. Speculative decoding uses a small draft model to propose tokens and the large target model to verify them in parallel, both backed by their own caches. Continuous batching and PagedAttention manage cache memory across concurrent requests. Prefix caching turns duplicate system prompts into a free speedup. Long-context serving confronts the memory wall when contexts stretch past a million tokens, at which point the tricks in this topic stop being enough and architectural redesigns like MLA become necessary.
      </Prose>

      <Prose>
        Each of those is a variation on a single theme: the model itself is expensive to change, but the memory around it is ours to shape. The weights are a fixed capital expense; the cache is a variable operational one, and the marginal improvements in how it is stored, compressed, shared, and streamed are where the throughput gains of the next serving-stack generation will continue to come from.
      </Prose>
    </div>
  ),
};

export default kvCache;
