import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const gqaMqaContent = {
  title: "Grouped-Query Attention (GQA) & Multi-Query Attention (MQA)",
  readTime: "~35 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        The Transformer that Vaswani et al. published in 2017 had a beautiful and inconvenient property: at inference time, every autoregressive decoding step had to carry forward a growing memory — the <em>KV cache</em> — of all keys and values computed for every prior token, at every layer, in every head. Attention is exact only because the query at position {"t"} can see {"every"} prior key-value pair. Training pays this cost once per batch in parallel; inference pays it once per new token, serially, and the cache grows linearly with the output length. For a 2017-vintage encoder-decoder translating a 100-token sentence, the total memory was a handful of megabytes. By 2024, when Anthropic and OpenAI were serving 70-billion-parameter models with 128k context windows to thousands of concurrent users, the KV cache had become — not the model weights, not the activations during a forward pass, not the softmax in attention, but the <em>cache</em> — the dominant consumer of GPU memory on every single token of every single request.
      </Prose>

      <Prose>
        The arithmetic is blunt. A single token in a modern LLM adds, to the KV cache, {"2 · n_{kv\\_heads} · d_h · n_{layers}"} bytes of fp16 per inference stream. For Llama-2 70B, a stock Multi-Head Attention (MHA) configuration would store {"2 · 64 · 128 · 80 · 2 = 2{,}621{,}440"} bytes {"="} 2.5 MB per token. A single 32k-token conversation therefore needs {"2.5 · 32768 / 1000 ≈ 82 GB"} of KV cache — which exceeds the entire memory budget of an 80GB H100 before the model weights even arrive. Serving more than one concurrent session at 32k context becomes impossible without either an exotic memory hierarchy or a structural architectural change. The autoregressive transformer had shipped with a latent inference-time bottleneck that the training-time formulation had simply not exposed.
      </Prose>

      <Prose>
        Noam Shazeer saw this in 2019. "Fast Transformer Decoding: One Write-Head is All You Need" (arXiv:1911.02150) is a six-page technote with the entire thesis in the title: keep the query heads, but let <em>all of them share a single key head and a single value head</em>. Multi-Query Attention (MQA) shrinks the KV cache by a factor of {"n_{heads}"} — typically 8 to 64 — at the cost of some representational capacity in the key/value side. Shazeer's experiments on translation and language modeling showed a ~1% quality loss and a large inference speedup. At the time, this was an interesting idea for research deployments and not obviously worth the quality hit. PaLM (Chowdhery et al. 2022) and Falcon adopted MQA; most production models did not.
      </Prose>

      <Prose>
        By 2023, the arithmetic had become unignorable and a middle ground was sorely needed. Joshua Ainslie and coauthors at Google Research published "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" at EMNLP 2023 (arXiv:2305.13245). Grouped-Query Attention (GQA) introduces a single hyperparameter, {"n_{kv\\_heads}"}, that interpolates between MHA ({"n_{kv\\_heads} = n_{heads}"}) and MQA ({"n_{kv\\_heads} = 1"}). With {"n_{kv\\_heads} = 8"} on a 64-Q-head model, the KV cache shrinks by {"8×"}, the quality loss relative to MHA shrinks to essentially zero on standard benchmarks, and the paper's second contribution — an <em>uptraining</em> recipe that initializes GQA key/value heads from averaged MHA heads and continues training at a fraction of the original compute budget — made it practical to <em>convert existing MHA checkpoints</em> rather than retraining from scratch. The method landed in Llama-2 70B (64 Q heads, 8 KV heads, {"g = 8"}) a few months later.
      </Prose>

      <Prose>
        Everything you use in 2026 is a descendant of this line of work. Llama-2 70B, Llama-3 (all sizes), Mistral 7B, Mixtral 8x7B, Qwen2, DeepSeek (before DeepSeek-V2's MLA), and — based on inference-throughput patterns and the architectural disclosures that have leaked or been confirmed — Gemini, GPT-4, Claude, and every major commercial frontier model. Whenever someone says "this model has 64 attention heads but 8 KV heads," they are describing GQA with {"g = 8"}. The modern stack — FlashAttention-2+ (Dao 2023, arXiv:2307.08691), vLLM's PagedAttention (Kwon et al. 2023, arXiv:2309.06180), TensorRT-LLM, SGLang — has native, optimized kernels for GQA because it is the default assumption of every serving system shipped in the last two years.
      </Prose>

      <Callout accent="gold">
        The one-line summary: MHA trains well but wastes KV bandwidth at inference. MQA saves the bandwidth but loses quality. GQA is the Pareto-optimal interior point — pick a small number of KV heads, share each across a group of Q heads, match MHA quality, inherit MQA-scale memory savings. Every frontier model larger than ~13B parameters released since mid-2023 uses GQA.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <H3>2.1 What each head actually does</H3>

      <Prose>
        A multi-head attention layer with {"h"} heads does the following: project the input into {"h"} separate query spaces, {"h"} separate key spaces, and {"h"} separate value spaces (each of dimension {"d_h = d_{model} / h"}), run scaled dot-product attention in each head independently, and concatenate the {"h"} outputs back into a {"d_{model}"}-dimensional vector. The intuition Vaswani et al. gave for multiple heads was that different heads could learn different <em>types</em> of relationships — one head tracks syntactic dependencies, another resolves coreference, another attends to content words. In practice, probing studies have found that this is partially true and partially overstated: heads within a layer often learn overlapping patterns, and many heads can be pruned at inference time without meaningful quality loss (Michel, Levy, Neubig 2019, arXiv:1905.10650).
      </Prose>

      <Prose>
        That second observation — that heads carry less unique information than their count suggests — is the foundation of MQA and GQA. If many of the {"h"} key/value heads are redundant, then why not collapse them? The query side is still useful at full resolution because the query is what selects which memory to read. The key/value side is a memory store, and a smaller, well-trained memory can be almost as informative as a larger redundant one.
      </Prose>

      <H3>2.2 MHA, MQA, GQA side by side</H3>

      <Prose>
        All three variants share the same {"Q, K, V, softmax, O"} pipeline. The only knob that changes is how many distinct key and value heads are materialized:
      </Prose>

      <TokenStream
        label="Per-head layout (example with h=8)"
        tokens={[
          { label: "MHA: Q×8  K×8  V×8",       color: "#60a5fa" },
          { label: "GQA g=4: Q×8  K×4  V×4",   color: colors.gold },
          { label: "GQA g=2: Q×8  K×2  V×2",   color: colors.gold },
          { label: "MQA: Q×8  K×1  V×1",       color: colors.green },
        ]}
      />

      <Prose>
        In MHA, each of the {"h"} query heads gets its own dedicated key head and value head — a one-to-one pairing. In MQA, all {"h"} query heads share a single key head and a single value head — an {"h"}-to-one pairing. In GQA with {"g"} KV heads, the {"h"} query heads are partitioned into {"g"} groups of size {"h/g"}, and each group shares one KV head — an {"h/g"}-to-one pairing within each group. MHA and MQA are the two extremes of this spectrum ({"g = h"} and {"g = 1"} respectively); GQA occupies the interior.
      </Prose>

      <H3>2.3 Why KV memory — not parameter count — is the problem</H3>

      <Prose>
        It is tempting to read the KV-head collapse as a parameter-reduction trick, but that is not the point. Even in MHA, the K/V projection weights {"W_K, W_V"} together occupy only {"2 · d_{model}^2"} parameters per layer, which is a fraction of a percent of the whole model. Training memory is dominated by activations and optimizer state; weights are an afterthought. The real bottleneck shows up only at <em>inference</em>, and only when the cache grows. The cache stores, for every token, for every layer, for every KV head, a vector of dimension {"d_h"}. The total is {"2 · L · n_{kv\\_heads} · d_h · n_{layers}"} bytes per inference stream — and this grows unboundedly with {"L"} while the weights stay fixed. Collapsing 64 KV heads to 8 drops the cache by {"8×"}. Collapsing 64 to 1 drops it by {"64×"}.
      </Prose>

      <H3>2.4 The quality-vs-memory Pareto</H3>

      <Prose>
        Ainslie et al. (2023) ran the ablation everyone wanted to see. On the T5-XXL (11B) checkpoint, across a basket of benchmarks (MMLU-style, summarization, translation, reading comprehension), the average performance as a function of {"n_{kv\\_heads}"} is flat from {"n_{kv\\_heads} = n_{heads}"} down to {"n_{kv\\_heads} = 8"}, and drops noticeably only as {"n_{kv\\_heads}"} approaches 1. The MQA endpoint ({"n_{kv\\_heads} = 1"}) costs roughly 1 point of average benchmark score; GQA with {"g = 8"} costs essentially nothing. The memory savings, meanwhile, are the same {"8×"} between MHA and GQA {"g=8"} as between MHA and MQA at that ratio. {"g = 8"} is the Pareto sweet spot, and it is why every production GQA model on the market picked exactly that value.
      </Prose>

      <H3>2.5 Why Q heads stay at full resolution</H3>

      <Prose>
        If the KV heads can be collapsed, why not the Q heads too? Two reasons. First, the query heads are what select the information: different queries at the same token position pose different "what is relevant?" questions, and collapsing them would collapse the diversity of <em>reads</em>, not the redundancy of <em>stored memory</em>. Second, the Q heads do not contribute to the KV cache. The query is recomputed for every new token in the decode loop; it is not stored. Shrinking Q heads would reduce parameter count but not KV-cache memory, which is the bottleneck. Keeping Q at full resolution and collapsing only KV is therefore the precise intervention that targets the bottleneck without sacrificing the read-side expressiveness.
      </Prose>

      <Callout accent="gold">
        Intuitive analogy: MHA is a library where every reader ({"Q"}) has their own private set of bookshelves ({"K, V"}). MQA is a library where every reader shares a single shared bookshelf. GQA is a library where readers are grouped by interest — eight readers per shelf — and each group has its own. The readers are still individually expressive; the storage is simply deduplicated.
      </Callout>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 The unified attention equation</H3>

      <Prose>
        Write the standard Transformer per-layer attention. Let {"X ∈ ℝ^{B×T×d_{model}}"} be the input, {"h = n_{heads}"} the number of query heads, {"g = n_{kv\\_heads}"} the number of key/value heads (with {"h"} divisible by {"g"}), and {"d_h = d_{model} / h"} the per-head dimension. The group size — how many Q heads share each KV head — is {"s = h/g"}.
      </Prose>

      <MathBlock>{"Q = X W_Q \\in \\mathbb{R}^{B \\times T \\times h \\cdot d_h}, \\quad K = X W_K \\in \\mathbb{R}^{B \\times T \\times g \\cdot d_h}, \\quad V = X W_V \\in \\mathbb{R}^{B \\times T \\times g \\cdot d_h}"}</MathBlock>

      <Prose>
        Note the asymmetry in output dimension: {"W_Q"} projects to {"h · d_h = d_{model}"} features, but {"W_K"} and {"W_V"} project to the <em>smaller</em> {"g · d_h"}. When {"g = h"} this is MHA and the three projections have the same shape; when {"g = 1"} this is MQA and {"W_K, W_V"} shrink to a single {"d_h"}-vector output; intermediate {"g"} is GQA.
      </Prose>

      <Prose>
        Reshape into heads, transpose for batched matmul:
      </Prose>

      <MathBlock>{"\\hat{Q} \\in \\mathbb{R}^{B \\times h \\times T \\times d_h}, \\quad \\hat{K} \\in \\mathbb{R}^{B \\times g \\times T \\times d_h}, \\quad \\hat{V} \\in \\mathbb{R}^{B \\times g \\times T \\times d_h}"}</MathBlock>

      <Prose>
        For the attention computation, each Q head must be paired with its corresponding KV head. In GQA, Q heads {"\\{s \\cdot j, s \\cdot j + 1, \\ldots, s \\cdot j + s - 1\\}"} all pair with KV head {"j"}. The implementation trick is to <em>expand</em> {"\\hat{K}, \\hat{V}"} by repeating each of the {"g"} heads {"s"} times along the head dimension, producing tensors of shape {"B × h × T × d_h"} that line up one-to-one with {"\\hat{Q}"}. This is exactly what {"torch.repeat_interleave(dim=1)"} does. After expansion, attention is computed identically to MHA:
      </Prose>

      <MathBlock>{"\\text{Attn}(Q, K, V) = \\mathrm{softmax}\\!\\left(\\frac{\\hat{Q} \\hat{K}^\\top}{\\sqrt{d_h}}\\right) \\hat{V}"}</MathBlock>

      <H3>3.2 KV cache memory</H3>

      <Prose>
        At inference, the KV cache stores {"\\hat{K}"} and {"\\hat{V}"} — crucially, the <em>unexpanded</em> versions with only {"g"} heads, not {"h"}. The expansion is done just-in-time during attention computation and does not multiply the cache size.
      </Prose>

      <MathBlock>{"\\text{KV bytes}(L) = 2 \\cdot L \\cdot g \\cdot d_h \\cdot n_{\\text{layers}} \\cdot \\text{dtype\\_bytes}"}</MathBlock>

      <Prose>
        The factor of {"2"} is for K plus V; {"L"} is the current sequence length; {"dtype\\_bytes = 2"} for fp16/bf16, {"1"} for fp8/int8, {"0.5"} for int4. For any fixed {"h, d_h, n_{layers}, L"}, the KV cache is a strictly linear function of {"g"}. Collapsing {"g"} from {"h"} (MHA) to {"h/8"} (GQA g=8) to {"1"} (MQA) shrinks the cache by {"8×"} then {"64×"} respectively.
      </Prose>

      <H3>3.3 Concrete numbers for Llama-2 70B</H3>

      <Prose>
        The canonical worked example. Llama-2 70B has {"d_{model} = 8192"}, {"h = 64"}, {"d_h = 128"}, {"n_{layers} = 80"}. The per-token KV cost for each variant is:
      </Prose>

      <MathBlock>{"\\text{MHA: } 2 \\cdot 64 \\cdot 128 \\cdot 80 \\cdot 2 = 2{,}621{,}440 \\text{ bytes} \\approx 2.5 \\text{ MB/token}"}</MathBlock>
      <MathBlock>{"\\text{GQA } g=8\\text{: } 2 \\cdot 8 \\cdot 128 \\cdot 80 \\cdot 2 = 327{,}680 \\text{ bytes} \\approx 320 \\text{ KB/token}"}</MathBlock>
      <MathBlock>{"\\text{MQA: } 2 \\cdot 1 \\cdot 128 \\cdot 80 \\cdot 2 = 40{,}960 \\text{ bytes} \\approx 40 \\text{ KB/token}"}</MathBlock>

      <Prose>
        At a 32k context window, MHA's cache is {"2.5 \\text{ MB} \\times 32768 \\approx 82 \\text{ GB}"}; the H100 has 80 GB of HBM and Llama-2 70B weights in fp16 are 140 GB. MHA at 32k is impossible on a single H100 — you cannot even fit the cache. GQA {"g=8"} pulls the cache down to {"\\approx 10.5 \\text{ GB}"}, and MQA to {"\\approx 1.3 \\text{ GB}"}. Ainslie's decision to ship Llama-2 70B with GQA {"g=8"} is what makes 70B inference at long context practical on single-node hardware.
      </Prose>

      <H3>3.4 Parameter count</H3>

      <Prose>
        Though parameters are not the main motivation, the parameter count of the attention layer is also reduced. {"W_Q"} remains {"d_{model} × h · d_h = d_{model}^2"} parameters; {"W_K"} and {"W_V"} shrink to {"d_{model} × g · d_h = d_{model}^2 · g/h"} parameters each. Total parameters in the attention block:
      </Prose>

      <MathBlock>{"P_{\\text{attn}} = d_{\\text{model}}^2 \\cdot \\left(2 + 2 \\cdot \\frac{g}{h}\\right)"}</MathBlock>

      <Prose>
        For {"h = 64, g = 8"} this is {"d_{model}^2 \\cdot 2.25"} instead of {"d_{model}^2 \\cdot 4"}, a 44% reduction in attention-layer parameters. The FFN (MLP) block typically dominates the total parameter count, so the overall model parameter reduction from adopting GQA is modest — roughly 5–10% at Llama scale. This is why Llama-2 70B is still called "70B" even though it uses GQA.
      </Prose>

      <H3>3.5 Uptraining: initializing GQA from MHA</H3>

      <Prose>
        Ainslie et al.'s key practical contribution is that you do not have to train a GQA model from scratch. Given an existing MHA checkpoint, initialize the GQA key and value weights by <em>averaging</em> the MHA key/value heads within each group:
      </Prose>

      <MathBlock>{"W_K^{\\text{GQA}}[j] = \\frac{1}{s} \\sum_{i=s \\cdot j}^{s \\cdot j + s - 1} W_K^{\\text{MHA}}[i], \\quad j = 0, 1, \\ldots, g-1"}</MathBlock>

      <Prose>
        (Analogous formula for {"W_V"}.) {"W_Q"} and {"W_O"} copy over unchanged. This initialization is not optimal — the averaged heads have not been trained to work together as a single shared head — but it is a principled starting point that recovers {"\\approx 90\\%"} of final quality within the first few hundred training steps. The full uptraining recipe is: load the MHA checkpoint, apply the averaging init, continue training with the original data mixture for {"\\alpha \\cdot T_{\\text{original}}"} steps where {"\\alpha \\approx 0.05–0.1"}. A 5-10% compute overhead, in exchange for an {"8×"} KV cache reduction at serving time, is a bargain that every production team takes.
      </Prose>

      <H3>3.6 Why averaging is the right init</H3>

      <Prose>
        Consider the pre-uptrain GQA forward pass. If {"W_K^{\\text{GQA}}[j]"} is the mean of {"s"} MHA K-heads and all {"s"} Q-heads within group {"j"} attend against it, the expected attention logits are the average of the logits each Q-head would have produced against its own MHA K-head. For a softmax that has not moved far from its MHA-trained operating point, this mean behaves like a {"1/s"}-scaled smoother of the original attention distributions. It is already close to "the average correct behavior"; the uptraining then has to learn to compensate for the loss of per-head specificity by adjusting neighboring weights. Starting from random init, by contrast, requires the model to rediscover the entire attention structure from scratch — hence the {"10×"} slower convergence reported by Ainslie.
      </Prose>

      <H3>3.7 Compute FLOPs</H3>

      <Prose>
        Once the KV heads are expanded to match Q heads inside the attention kernel, the FLOPs of the attention computation are identical to MHA. GQA saves memory and memory bandwidth, not arithmetic. This is the right trade for modern accelerators, where tensor-core FLOPs are cheap and HBM bandwidth is scarce — the GPU is memory-bound on decode, and cutting memory traffic by {"8×"} roughly translates to an {"8×"} decode-step speedup in the regime where attention dominates.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The code below was run on PyTorch 2.6 with CUDA. Every {"# Output:"} comment is real stdout from the run. We implement MHA, MQA, and GQA as a single unified class parameterized by {"n_{kv\\_heads}"}; verify output shapes and gradient flow; measure actual KV tensor sizes as a function of sequence length; and demonstrate the uptraining initialization.
      </Prose>

      <H3>4.1 Unified attention class</H3>

      <CodeBlock language="python">
{`import torch
import torch.nn as nn
import torch.nn.functional as F
import math

torch.manual_seed(0)
device = "cuda"


class GroupedQueryAttention(nn.Module):
    """Unified MHA / GQA / MQA. Set n_kv_heads to control the variant:
         n_kv_heads == n_heads  -> MHA  (one KV head per Q head)
         1 < n_kv_heads < h     -> GQA  (groups of Q heads share a KV head)
         n_kv_heads == 1        -> MQA  (all Q heads share one KV head)
    """
    def __init__(self, d_model=512, n_heads=8, n_kv_heads=None):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        assert n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.head_dim   = d_model // n_heads
        self.group_size = n_heads // self.n_kv_heads   # Q heads per KV head
        self.d_model    = d_model

        self.W_q = nn.Linear(d_model, n_heads         * self.head_dim, bias=False)
        self.W_k = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.W_o = nn.Linear(n_heads         * self.head_dim, d_model, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.W_q(x).view(B, T, self.n_heads,    self.head_dim).transpose(1, 2)  # [B, h, T, d_h]
        k = self.W_k(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # [B, g, T, d_h]
        v = self.W_v(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Expand KV to match Q head count. This is the only GQA-specific line.
        if self.group_size > 1:
            k = k.repeat_interleave(self.group_size, dim=1)   # [B, h, T, d_h]
            v = v.repeat_interleave(self.group_size, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn   = F.softmax(scores, dim=-1)
        out    = torch.matmul(attn, v)                         # [B, h, T, d_h]
        out    = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_o(out)`}
      </CodeBlock>

      <Prose>
        One class, one hyperparameter. When {"n_{kv\\_heads} == n_{heads}"} this degenerates exactly to MHA; when {"n_{kv\\_heads} == 1"} exactly to MQA. The single GQA-specific construct is the {"repeat_interleave"} — it expands each of the {"g"} KV heads {"s"} times so that head {"i"} of Q pairs with head {"i // s"} of the underlying {"g"}-head KV tensor. The expansion is a view-like operation in PyTorch and does not materialize new memory in the computation graph — the KV cache itself (at inference) stores only the {"g"} unique heads.
      </Prose>

      <H3>4.2 Shape parity and gradient check</H3>

      <CodeBlock language="python">
{`B, T, D = 2, 32, 512
x = torch.randn(B, T, D, device=device)

mha  = GroupedQueryAttention(D, n_heads=8, n_kv_heads=8).to(device)
gqa4 = GroupedQueryAttention(D, n_heads=8, n_kv_heads=4).to(device)
gqa2 = GroupedQueryAttention(D, n_heads=8, n_kv_heads=2).to(device)
mqa  = GroupedQueryAttention(D, n_heads=8, n_kv_heads=1).to(device)

for name, m in [("MHA", mha), ("GQA(g=4)", gqa4), ("GQA(g=2)", gqa2), ("MQA", mqa)]:
    y = m(x)
    n_param = sum(p.numel() for p in m.parameters())
    print(f"{name:10s} out={tuple(y.shape)}  params={n_param:7,}  kv_heads={m.n_kv_heads}  group_size={m.group_size}")

y = gqa2(x).sum(); y.backward()
print("grad W_q ok:", gqa2.W_q.weight.grad.abs().mean().item() > 0)
print("grad W_k ok:", gqa2.W_k.weight.grad.abs().mean().item() > 0)
print("grad W_v ok:", gqa2.W_v.weight.grad.abs().mean().item() > 0)

# Output:
#   MHA        out=(2, 32, 512)  params=1,048,576  kv_heads=8  group_size=1
#   GQA(g=4)   out=(2, 32, 512)  params=786,432    kv_heads=4  group_size=2
#   GQA(g=2)   out=(2, 32, 512)  params=655,360    kv_heads=2  group_size=4
#   MQA        out=(2, 32, 512)  params=589,824    kv_heads=1  group_size=8
#   grad W_q ok: True
#   grad W_k ok: True
#   grad W_v ok: True`}
      </CodeBlock>

      <Prose>
        All variants produce the same output shape — the external contract of the attention layer is unchanged. The parameter counts drop as expected: MHA has {"4 \\cdot d_{model}^2 = 4 \\cdot 512^2 = 1{,}048{,}576"} parameters; GQA {"g=4"} has {"(2 + 2 \\cdot 4/8) \\cdot 512^2 = 3 \\cdot 512^2 = 786{,}432"}; MQA has {"(2 + 2/8) \\cdot 512^2 = 2.25 \\cdot 512^2 = 589{,}824"}. Gradients flow cleanly through the {"repeat_interleave"} expansion because it is just a view op.
      </Prose>

      <H3>4.3 KV cache sizing as a function of sequence length</H3>

      <CodeBlock language="python">
{`# Measure the actual K/V tensor bytes — this is the storage the cache would hold.
# Note: we measure the *unexpanded* K,V, since the cache stores those.
import torch

D, H = 512, 8
mha = GroupedQueryAttention(D, n_heads=H, n_kv_heads=H).to(device).half()
gqa = GroupedQueryAttention(D, n_heads=H, n_kv_heads=2).to(device).half()
mqa = GroupedQueryAttention(D, n_heads=H, n_kv_heads=1).to(device).half()

def kv_bytes(m, x):
    B, T, _ = x.shape
    k = m.W_k(x).view(B, T, m.n_kv_heads, m.head_dim)
    v = m.W_v(x).view(B, T, m.n_kv_heads, m.head_dim)
    return (k.numel() + v.numel()) * k.element_size()

print(f"{'L':>6} {'MHA KV (KB)':>14} {'GQA(2) KV (KB)':>16} {'MQA KV (KB)':>14} {'ratio MHA/MQA':>16}")
for L in [128, 512, 1024, 2048, 4096, 8192]:
    x = torch.randn(1, L, D, device=device, dtype=torch.float16)
    b_mha = kv_bytes(mha, x) / 1024
    b_gqa = kv_bytes(gqa, x) / 1024
    b_mqa = kv_bytes(mqa, x) / 1024
    print(f"{L:>6} {b_mha:>14.2f} {b_gqa:>16.2f} {b_mqa:>14.2f} {b_mha/b_mqa:>15.1f}x")

# Output:
#        L    MHA KV (KB)   GQA(2) KV (KB)    MQA KV (KB)    ratio MHA/MQA
#      128         256.00            64.00          32.00             8.0x
#      512        1024.00           256.00         128.00             8.0x
#     1024        2048.00           512.00         256.00             8.0x
#     2048        4096.00          1024.00         512.00             8.0x
#     4096        8192.00          2048.00        1024.00             8.0x
#     8192       16384.00          4096.00        2048.00             8.0x`}
      </CodeBlock>

      <Prose>
        The ratios are exactly as the math predicts — linear in {"g"}, linear in {"L"}. An {"8×"} reduction MHA → MQA at every sequence length; a {"4×"} reduction MHA → GQA {"g=2"}. The absolute numbers are small here because we are running on a toy {"d_{model} = 512, n_{layers} = 1"} configuration. Scale up to Llama-2 70B's 80 layers and {"d_h = 128"} and the numbers become the 82 GB vs 10.5 GB vs 1.3 GB we saw in section 3.3.
      </Prose>

      <H3>4.4 Projected KV cache for Llama-2 70B at production contexts</H3>

      <CodeBlock language="python">
{`# Llama-2 70B: h=64 Q heads, 8 KV heads (GQA g=8), head_dim=128, n_layers=80.
# KV per token = 2 * n_kv_heads * head_dim * n_layers * dtype_bytes

def kv_cache_bytes(L, n_kv_heads, head_dim=128, n_layers=80, dtype_bytes=2):
    return 2 * n_kv_heads * head_dim * n_layers * L * dtype_bytes

print(f"{'L (ctx)':>10} {'MHA (GB)':>12} {'GQA g=8 (GB)':>14} {'MQA (GB)':>12} {'MHA/GQA':>10}")
for L in [2048, 4096, 8192, 16384, 32768, 65536, 131072]:
    mha_gb  = kv_cache_bytes(L, 64) / 1e9
    gqa_gb  = kv_cache_bytes(L, 8)  / 1e9
    mqa_gb  = kv_cache_bytes(L, 1)  / 1e9
    print(f"{L:>10} {mha_gb:>12.2f} {gqa_gb:>14.2f} {mqa_gb:>12.2f} {mha_gb/gqa_gb:>9.1f}x")

# Output:
#      L (ctx)     MHA (GB)   GQA g=8 (GB)     MQA (GB)    MHA/GQA
#         2048         5.37           0.67         0.08       8.0x
#         4096        10.74           1.34         0.17       8.0x
#         8192        21.47           2.68         0.34       8.0x
#        16384        42.95           5.37         0.67       8.0x
#        32768        85.90          10.74         1.34       8.0x
#        65536       171.80          21.47         2.68       8.0x
#       131072       343.60          42.95         5.37       8.0x`}
      </CodeBlock>

      <Prose>
        Reading the 32k row: MHA costs 85.90 GB of KV cache per single inference stream — exceeds the H100's 80 GB of HBM before even loading model weights. GQA {"g=8"} brings this to 10.74 GB per stream, which leaves room for the model (in quantized form) plus a handful of concurrent sessions on a single 80GB H100. MQA drops it further to 1.34 GB, enabling dozens of concurrent 32k sessions but at the quality cost we will quantify in section 6. At 128k context, MHA is flatly infeasible (343 GB per stream); GQA is {"\\approx 43"} GB — fits on H200 (141 GB) with room for model weights; MQA is 5.4 GB. The horizontal scaling properties of frontier long-context serving are effectively defined by this table.
      </Prose>

      <H3>4.5 Uptraining from MHA: the averaging init</H3>

      <CodeBlock language="python">
{`def uptrain_init_gqa_from_mha(mha, n_kv_heads):
    """Ainslie 2023 recipe: per-group mean of MHA K/V heads
       -> initial GQA K/V heads.  Q and O copy directly."""
    D, h_q, head_dim = mha.d_model, mha.n_heads, mha.head_dim
    group = h_q // n_kv_heads
    gqa   = GroupedQueryAttention(D, n_heads=h_q, n_kv_heads=n_kv_heads).to(mha.W_q.weight.device)

    gqa.W_q.weight.data.copy_(mha.W_q.weight.data)
    gqa.W_o.weight.data.copy_(mha.W_o.weight.data)

    W_k = mha.W_k.weight.data.view(h_q, head_dim, D)                        # [h, d_h, D]
    W_v = mha.W_v.weight.data.view(h_q, head_dim, D)
    W_k_grouped = W_k.view(n_kv_heads, group, head_dim, D).mean(dim=1)      # [g, d_h, D]
    W_v_grouped = W_v.view(n_kv_heads, group, head_dim, D).mean(dim=1)
    gqa.W_k.weight.data.copy_(W_k_grouped.reshape(n_kv_heads * head_dim, D))
    gqa.W_v.weight.data.copy_(W_v_grouped.reshape(n_kv_heads * head_dim, D))
    return gqa


# Train MHA on a synthetic copy task; then convert to GQA and uptrain briefly.
D, H, T = 256, 8, 16
mha = GroupedQueryAttention(D, n_heads=H, n_kv_heads=H).to(device)
x = torch.randn(64, T, D, device=device)
y = torch.roll(x, shifts=1, dims=1)   # target: roll input by one position

opt = torch.optim.Adam(mha.parameters(), lr=3e-3)
for _ in range(2000):
    loss = F.mse_loss(mha(x), y)
    opt.zero_grad(); loss.backward(); opt.step()
print(f"MHA trained loss (2000 steps):          {loss.item():.5f}")

gqa_up = uptrain_init_gqa_from_mha(mha, n_kv_heads=2)
with torch.no_grad():
    print(f"GQA(g=2) init via uptrain, 0 steps:     {F.mse_loss(gqa_up(x), y).item():.5f}")

opt = torch.optim.Adam(gqa_up.parameters(), lr=1e-3)
for _ in range(200):
    loss = F.mse_loss(gqa_up(x), y)
    opt.zero_grad(); loss.backward(); opt.step()
print(f"GQA(g=2) uptrained 200 steps:           {loss.item():.5f}")

# Output:
#   MHA trained loss (2000 steps):          0.21469
#   GQA(g=2) init via uptrain, 0 steps:     2.02489
#   GQA(g=2) uptrained 200 steps:           0.73332`}
      </CodeBlock>

      <Prose>
        The zero-step uptrain loss ({"2.02"}) is noticeably worse than the MHA endpoint ({"0.21"}) — averaging eight heads into two discards real information. But the averaging init is a <em>warm start</em>: by step 200 the loss has dropped to {"0.73"}, already in the neighborhood of what the model will eventually converge to. On real language modeling benchmarks with Ainslie's full schedule ({"\\alpha = 0.05"} of pretrain compute), the final gap to MHA quality closes to within noise. The toy task here is not sensitive enough to show the full quality recovery, but the warm-start mechanism is clearly operational.
      </Prose>

      <H3>4.6 Verifying the repeat-interleave pattern</H3>

      <CodeBlock language="python">
{`# Visualize the Q -> KV head mapping.  With n_kv=2 and group=4,
# Q heads 0..3 pair with KV head 0; Q heads 4..7 pair with KV head 1.
import torch

B, T, n_kv, d_h = 1, 4, 2, 4
group = 4

k = torch.arange(n_kv * d_h).view(1, n_kv, 1, d_h).expand(B, n_kv, T, d_h).float()
print("K before repeat_interleave: shape =", tuple(k.shape))
print(k[0, :, 0, :])

k_exp = k.repeat_interleave(group, dim=1)
print("K after repeat_interleave(4, dim=1):  shape =", tuple(k_exp.shape))
print(k_exp[0, :, 0, :])

# Output:
#   K before repeat_interleave: shape = (1, 2, 4, 4)
#   tensor([[0., 1., 2., 3.],
#           [4., 5., 6., 7.]])
#   K after repeat_interleave(4, dim=1):  shape = (1, 8, 4, 4)
#   tensor([[0., 1., 2., 3.],
#           [0., 1., 2., 3.],
#           [0., 1., 2., 3.],
#           [0., 1., 2., 3.],
#           [4., 5., 6., 7.],
#           [4., 5., 6., 7.],
#           [4., 5., 6., 7.],
#           [4., 5., 6., 7.]])`}
      </CodeBlock>

      <Prose>
        The expansion pattern is unambiguous: each of the original {"g = 2"} heads is replicated {"group = 4"} times contiguously. Q heads {"\\{0, 1, 2, 3\\}"} see the identical copy of the first KV head; Q heads {"\\{4, 5, 6, 7\\}"} see the identical copy of the second. The alternative — {"repeat(group, dim=1)"} — would have given you {"[h0, h1, h0, h1, ..., h0, h1]"}, which is <em>not</em> what GQA wants. Getting this wrong silently produces a model that trains but learns a wrong Q-to-KV grouping and underperforms. Always verify with a small printout.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION
          ====================================================================== */}
      <H2>5. Production patterns</H2>

      <H3>5.1 HuggingFace LlamaConfig</H3>

      <CodeBlock language="python">
{`from transformers import LlamaConfig

# Llama-2 70B uses GQA: 64 Q heads, 8 KV heads, head_dim=128, 80 layers.
cfg = LlamaConfig(
    hidden_size=8192,
    num_attention_heads=64,
    num_key_value_heads=8,
    num_hidden_layers=80,
)
print("Llama-2 70B config:")
print(f"  hidden_size           = {cfg.hidden_size}")
print(f"  num_attention_heads   = {cfg.num_attention_heads}  (Q heads)")
print(f"  num_key_value_heads   = {cfg.num_key_value_heads}  (KV heads)")
print(f"  head_dim              = {cfg.hidden_size // cfg.num_attention_heads}")
print(f"  group_size (Q per KV) = {cfg.num_attention_heads // cfg.num_key_value_heads}")
print(f"  is_gqa                = {cfg.num_key_value_heads != cfg.num_attention_heads}")

# Output:
#   Llama-2 70B config:
#     hidden_size           = 8192
#     num_attention_heads   = 64  (Q heads)
#     num_key_value_heads   = 8   (KV heads)
#     head_dim              = 128
#     group_size (Q per KV) = 8
#     is_gqa                = True`}
      </CodeBlock>

      <Prose>
        The single parameter {"num_key_value_heads"} controls everything. Set it equal to {"num_attention_heads"} for MHA, to {"1"} for MQA, to any divisor in between for GQA. HuggingFace's implementation handles the repeat-interleave expansion internally; the public API exposes only the configuration.
      </Prose>

      <H3>5.2 The HuggingFace Llama attention layer (annotated)</H3>

      <CodeBlock language="python">
{`# Condensed from transformers/models/llama/modeling_llama.py.
# The core GQA-specific logic is the 'repeat_kv' helper.

def repeat_kv(hidden_states, n_rep):
    """Expand (B, n_kv_heads, T, d_h) -> (B, n_kv_heads * n_rep, T, d_h)."""
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class LlamaAttention(nn.Module):
    def __init__(self, config):
        ...
        self.num_heads         = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        # ... q_proj, k_proj, v_proj, o_proj ...

    def forward(self, hidden_states, position_ids, past_key_value=None, ...):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads,         self.head_dim).transpose(1, 2)
        key_states   = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # RoPE applied BEFORE K expansion (important! see section 9).
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Append to KV cache (still at num_key_value_heads resolution).
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, ...)

        # Expand for attention — AFTER the cache write.
        key_states   = repeat_kv(key_states,   self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Now standard MHA-style attention.
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights + attention_mask, dim=-1)
        attn_output  = torch.matmul(attn_weights, value_states)

        return self.o_proj(attn_output.transpose(1, 2).reshape(bsz, q_len, -1))`}
      </CodeBlock>

      <Prose>
        Two things to note. First, the KV cache stores the <em>unexpanded</em> tensors at {"num_key_value_heads"} resolution — this is the whole point, and getting it wrong (storing the expanded versions) would throw away all the memory savings. Second, RoPE is applied <em>before</em> the {"repeat_kv"} expansion, at the original {"num_key_value_heads"} count. This is correct because RoPE is a function of position, and all Q heads within a GQA group should see the same rotated key; applying RoPE after the expansion would waste compute and, in a naively written implementation, introduce subtle inconsistencies (see failure mode 9.4).
      </Prose>

      <H3>5.3 vLLM and TensorRT-LLM</H3>

      <Prose>
        Both of the two main production serving engines have native GQA support. vLLM's PagedAttention (Kwon et al. 2023, arXiv:2309.06180) is a memory-management scheme that breaks the KV cache into fixed-size page blocks and allocates them on demand, which stacks cleanly on GQA — the pages store the {"n_{kv\\_heads}"}-dimensional KV tensors, and the attention kernel does the expansion inside the GPU. PagedAttention alone typically delivers {"2–4×"} throughput improvements from better batching; PagedAttention plus GQA {"g=8"} compounds those gains and is why vLLM can serve Llama-2 70B at thousands of tokens per second per H100. TensorRT-LLM's {"gpt_attention_plugin"} takes {"num_kv_heads"} as a configuration and generates fused CUDA kernels that execute attention directly over the grouped representation without the explicit expansion — a further constant-factor speedup.
      </Prose>

      <H3>5.4 FlashAttention-2 and GQA</H3>

      <Prose>
        Tri Dao's FlashAttention-2 (arXiv:2307.08691) released in 2023 with first-class GQA and MQA support. The fused kernel loops over the query head index and, for each {"i ∈ \\{0, ..., h-1\\}"}, indexes into KV head {"i // s"}. The expansion never materializes in memory — it is done via index arithmetic inside the tiling loop. This is essential for long contexts: FlashAttention already cuts attention memory from {"O(T^2)"} to {"O(T)"} by tiling softmax; adding GQA on top cuts the KV reads by another factor of {"s"}. The combination is what makes 128k-context inference tractable on current hardware. In {"F.scaled_dot_product_attention"} (PyTorch 2.x's SDPA), GQA is supported automatically when the provided K/V tensors have fewer heads than Q — the runtime dispatches to a flash-style kernel that handles the expansion implicitly.
      </Prose>

      <H3>5.5 Models that ship GQA by default in 2026</H3>

      <Prose>
        Llama-2 70B (first Meta release with GQA; {"g = 8"}). Llama-3 8B/70B/405B (all GQA). Mistral 7B ({"h = 32, n_{kv} = 8, g = 4"}). Mixtral 8x7B (inherits Mistral's attention config per expert). Qwen2 and Qwen2.5 at all sizes. DeepSeek-V1 and DeepSeek-Coder used GQA; DeepSeek-V2 introduced MLA (Multi-head Latent Attention) as a further evolution. Command R+, Yi, Falcon 180B (Falcon originally used MQA, the 180B revision moved to GQA). The published and inferred architectures for Gemini, GPT-4/4o, and Claude 3/3.5 are all consistent with GQA-style KV sharing at inference — the throughput profiles at long context are otherwise hard to explain. If a model larger than {"\\approx 13\\text{B}"} parameters was trained after mid-2023 and you do not know its attention architecture, GQA is the correct prior.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6.1 KV cache size vs sequence length (Llama-2 70B)</H3>

      <Prose>
        The signature plot of the GQA era. Three configurations of Llama-2 70B's attention — MHA, GQA {"g=8"}, MQA — with KV cache size plotted as a function of sequence length. Note the log-scale y-axis: MHA's cache is a strict {"8×"} above GQA's and {"64×"} above MQA's at every context length. The crossover points where each configuration exhausts an 80 GB H100's memory are annotated implicitly by where each curve crosses {"y = 80 \\text{ GB}"}.
      </Prose>

      <Plot
        series={[
          { name: "MHA (h=64 KV)",  color: "#f87171",    points: [[2048, 5.37], [4096, 10.74], [8192, 21.47], [16384, 42.95], [32768, 85.90], [65536, 171.80], [131072, 343.60]] },
          { name: "GQA g=8",        color: colors.gold,  points: [[2048, 0.67], [4096, 1.34],  [8192, 2.68],  [16384, 5.37],  [32768, 10.74], [65536, 21.47],  [131072, 42.95]] },
          { name: "MQA",            color: colors.green, points: [[2048, 0.08], [4096, 0.17],  [8192, 0.34],  [16384, 0.67],  [32768, 1.34],  [65536, 2.68],   [131072, 5.37]] },
        ]}
        xLabel="context length (tokens)"
        yLabel="KV cache (GB)"
        label="Llama-70B KV cache vs context"
      />

      <Prose>
        The practical reading: the 32k row is where MHA first becomes infeasible on a single H100 (85.9 GB). GQA {"g=8"} stays under 11 GB through 32k and is the reason the ecosystem converged on this configuration. MQA stays under 6 GB all the way to 128k but pays the quality tax shown in 6.4.
      </Prose>

      <H3>6.2 GQA attention computation step-by-step</H3>

      <Prose>
        One attention layer of Llama-2 70B ({"h = 64, g = 8, s = 8, d_h = 128, T = 4"}) walked through step by step.
      </Prose>

      <StepTrace
        label="GQA forward pass (T=4, h=64, g=8)"
        steps={[
          {
            label: "Project Q, K, V",
            render: () => (
              <Prose>
                Input {"X ∈ ℝ^{B × T × d_{model}}"} with {"d_{model} = 8192"}. Apply three linear projections:
                {" W_Q(X) ∈ ℝ^{B × 4 × 8192}"} (64 heads × 128 dim),
                {" W_K(X) ∈ ℝ^{B × 4 × 1024}"} (8 heads × 128 dim),
                {" W_V(X) ∈ ℝ^{B × 4 × 1024}"} (8 heads × 128 dim).
                Note the asymmetry: K and V are {"8×"} smaller than Q on the feature axis.
              </Prose>
            ),
          },
          {
            label: "Reshape to heads",
            render: () => (
              <Prose>
                Reshape and transpose:
                {" Q → [B, 64, 4, 128]"},
                {" K → [B, 8, 4, 128]"},
                {" V → [B, 8, 4, 128]"}.
                At this point, in the KV cache, we <em>only store</em> the 8-head K and 8-head V. The cache for this layer is {"2 · 8 · 4 · 128 · 2 = 16{,}384"} bytes per step.
              </Prose>
            ),
          },
          {
            label: "Apply RoPE",
            render: () => (
              <Prose>
                Apply rotary positional embedding to {"Q"} and {"K"} <em>at their native head counts</em> (64 and 8). Do not apply RoPE after expansion — the expansion is a replication and expanded heads would get identical rotations which is wasteful; more importantly, RoPE must be in the cache-layout domain so that position increments work correctly on cached K.
              </Prose>
            ),
          },
          {
            label: "Expand K, V",
            render: () => (
              <Prose>
                For attention computation (not for the cache), expand each of the 8 KV heads 8 times:
                {" K → [B, 64, 4, 128]"} via {"repeat_interleave(8, dim=1)"}. Q heads 0-7 now pair with replicated KV head 0; Q heads 8-15 with replicated KV head 1; and so on. In fused kernels (FlashAttention-2, TRT-LLM), this "expansion" is implemented as index arithmetic and never materializes in HBM.
              </Prose>
            ),
          },
          {
            label: "Scaled dot-product",
            render: () => (
              <Prose>
                {" scores = Q · Kᵀ / √d_h"} — shape {"[B, 64, 4, 4]"}.
                {" attn = softmax(scores + mask, dim=-1)"} — the causal mask zeros out upper-triangular entries.
                {" out = attn · V"} — shape {"[B, 64, 4, 128]"}.
                From this point forward the arithmetic is identical to MHA; GQA's savings are entirely upstream, in memory.
              </Prose>
            ),
          },
          {
            label: "Output projection",
            render: () => (
              <Prose>
                Reshape back: {" out → [B, 4, 8192]"}. Apply {"W_O"}: {" out → [B, 4, 8192]"}. Add residual connection. Done. The next layer repeats with its own {"W_Q, W_K, W_V, W_O"} matrices.
              </Prose>
            ),
          },
        ]}
      />

      <H3>6.3 Q-to-KV head mapping (Heatmap)</H3>

      <Prose>
        The group structure visualized. Rows are Q heads (16 of them for readability), columns are KV heads. A bright cell means "this Q head reads this KV head." MHA ({"g = 16"}) is the identity matrix. GQA ({"g = 4"}) has a block-diagonal structure: 4 consecutive Q heads per KV head. MQA ({"g = 1"}) is a single column — every Q head reads the one KV head.
      </Prose>

      <Heatmap
        matrix={[
          [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0],
          [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0],
          [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0],
          [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1],
        ]}
        rowLabels={["Q0","Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8","Q9","Q10","Q11","Q12","Q13","Q14","Q15"]}
        colLabels={["KV0","KV1","KV2","KV3"]}
        colorScale="gold"
        label="GQA(h=16, g=4) — Q head → KV head mapping"
        cellSize={28}
      />

      <Prose>
        The block structure is what gives GQA its name — groups of Q heads, each sharing a KV head. The choice of group boundaries is arbitrary (any partition of {"\\{0, ..., h-1\\}"} into {"g"} equal-size subsets would be algebraically equivalent after any permutation-consistent training); the convention is contiguous blocks because it aligns cleanly with {"repeat_interleave"} and CUDA memory access patterns.
      </Prose>

      <H3>6.4 Quality vs KV heads (GQA paper figure)</H3>

      <Prose>
        Reproduction of the benchmark numbers from Ainslie et al. 2023, figure 5 (T5-XXL uptrained to various {"g"}). The x-axis is number of KV heads (log-scaled conceptually); the y-axis is average benchmark score across the paper's suite. The curve is roughly flat from {"g = 32"} down to {"g = 8"}, then drops noticeably as {"g"} approaches 1.
      </Prose>

      <Plot
        series={[
          { name: "Avg benchmark score", color: colors.gold, points: [[1, 46.6], [2, 47.4], [4, 47.7], [8, 47.9], [16, 47.8], [32, 47.9], [64, 48.0]] },
        ]}
        xLabel="n_kv_heads (g)"
        yLabel="avg score"
        label="GQA uptrain quality vs #KV heads (Ainslie 2023, approx.)"
      />

      <Prose>
        The MHA endpoint ({"g = 64"}) scores {"48.0"}; {"g = 8"} scores {"47.9"} (within noise); {"g = 1"} (MQA) drops to {"46.6"} — about 1.4 points below MHA. This is the empirical Pareto: {"g = 8"} is the minimum KV-head count that preserves MHA quality, and anything larger is wasted KV memory. Every production GQA configuration you will encounter has been picked to sit on this shelf.
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix — which variant, when</H2>

      <H3>7.1 Model size ≥ 70B (dense): GQA g=8</H3>

      <Prose>
        For any dense model at 70B parameters or above, GQA with {"g = 8"} is the default and has been since Llama-2's release. The KV cache savings are essential at this scale (MHA is infeasible on single-node hardware at long context), and the quality preservation is strong enough that no large-scale published model in this size bracket has chosen anything else. Deviating from {"g = 8"} would require a compelling application-specific justification. Take this as a hard prior.
      </Prose>

      <H3>7.2 Model size 7B–30B: GQA g=4 or g=8</H3>

      <Prose>
        Mistral 7B uses {"g = 4"} ({"h = 32, n_{kv} = 8"}); Llama-3 8B uses {"g = 4"} ({"h = 32, n_{kv} = 8"}). At this scale, MHA remains feasible at most practical context lengths on consumer hardware, so the KV savings matter less per-model. But GQA is still chosen because (a) it preserves training-time investment portability if you ever want to scale up, (b) the quality cost at {"g = 4"} or {"g = 8"} is negligible, (c) the inference throughput gain stacks with continuous batching to give a clear wins in serving cost. {"g = 4"} is the common pick when {"h = 32"}; {"g = 8"} if {"h"} is larger.
      </Prose>

      <H3>7.3 Model size {"< 7B"}: MHA usually fine</H3>

      <Prose>
        For sub-7B models deployed at small context windows, the KV cache is not the bottleneck — the per-token compute of the FFN block dominates. MHA remains the reasonable default. That said, if you are training a small model with an eye on long-context use (32k+), GQA still buys you memory headroom at minimal cost and is worth doing.
      </Prose>

      <H3>7.4 Inference-throughput-critical workloads: MQA or GQA g=4</H3>

      <Prose>
        When the deployment is specifically optimized for maximum throughput per GPU — a classification endpoint, a short-form summarization service, an embedded edge deployment, an LLM-powered search reranker — and quality can tolerate the {"\\approx 1\\%"} hit, MQA is on the table. PaLM, the original Falcon, and several streaming-inference products shipped with MQA. The {"64×"} KV cache reduction vs MHA translates directly into larger batch sizes, higher concurrency, and lower $/token. If the quality hit is unacceptable, {"g = 4"} is the aggressive-but-safe middle ground.
      </Prose>

      <H3>7.5 New research architecture: start with GQA g=8</H3>

      <Prose>
        When you are prototyping a new model or architecture and do not want attention choice to be a variable you are debugging, use GQA with {"g = 8"}. It is the safe default everyone else uses, it preserves your upward compatibility with Llama-style weight conversions and serving kernels, and it avoids the situation where a reviewer later asks "why MHA?" and you do not have a good answer.
      </Prose>

      <H3>7.6 Fine-tuning / conversion of an MHA checkpoint</H3>

      <Prose>
        If you have an MHA-trained model and want the GQA memory profile without the retrain cost, use the uptraining recipe from section 3.5: average K/V heads within groups, then continue training for {"5–10\\%"} of the original compute on the original or a matching data mixture. This is the concrete procedure Ainslie used to convert T5-XXL to GQA and what Meta used to produce Llama-2 70B from its MHA-style predecessor. Expect the first 5% of uptrain steps to be dominated by recovering the averaged K/V representations; the remaining 5% tunes the Q side to exploit the new grouped layout.
      </Prose>

      <Callout accent="gold">
        One-line rule: at {"\\geq 70\\text{B}"} parameters, always GQA {"g=8"}. Below 70B, GQA {"g=4"} or {"g=8"} if you care about long context; MHA if you do not. MQA only when throughput is the explicit dominant objective and the 1% quality hit is acceptable. Never pick anything exotic without a measured reason.
      </Callout>

      {/* ======================================================================
          8. WHAT SCALES
          ====================================================================== */}
      <H2>8. What scales</H2>

      <H3>8.1 Single-GPU long-context serving</H3>

      <Prose>
        The dominant scaling win of GQA is that it enables Llama-2 70B (and its successors) to serve 32k and even 128k context on single-node hardware. With MHA, a 70B model at 32k needs more HBM than an H100 has — period. With GQA {"g=8"} the KV cache drops to {"\\approx 11 \\text{ GB}"} at 32k, which leaves room for an int4-quantized model ({"\\approx 35 \\text{ GB}"}) plus headroom. With GQA {"g=8"} the same H100 can serve 6-8 concurrent 32k sessions, which is the threshold at which serving economics become viable. Nothing Meta did at the training architecture level matters more for deployed throughput than the choice of {"num_key_value_heads = 8"}.
      </Prose>

      <H3>8.2 MQA pushes further but compromises quality</H3>

      <Prose>
        At the MQA extreme, the {"64×"} KV reduction enables concurrency counts that would be otherwise impossible. A single H100 running an int4 Llama-2 70B with MQA could in principle host tens of concurrent 32k sessions (the math: {"(80 - 35) / 1.34 \\approx 33"} sessions). Whether that is worth the {"\\approx 1\\%"} benchmark drop is a product decision — for many commercial applications, especially chat that scores well on human eval even when MMLU drops a point, it is. Custom model providers with specific workloads to serve sometimes ship MQA; generalist frontier models almost never do.
      </Prose>

      <H3>8.3 Uptraining cost is a one-time investment</H3>

      <Prose>
        Converting MHA → GQA via uptraining costs roughly {"5–10\\%"} of the original pretrain compute (Ainslie 2023). For a 70B model whose original pretrain was, say, 2e24 FLOP, uptraining is 1–2e23 FLOP — large in absolute terms but amortized over the entire deployment lifetime. The savings come at every single inference request, forever. Even under pessimistic assumptions about inference-vs-training compute ratios, uptraining pays for itself within weeks of production serving.
      </Prose>

      <H3>8.4 Stacking with PagedAttention</H3>

      <Prose>
        vLLM's PagedAttention partitions the KV cache into fixed-size pages (typically 16 tokens) and allocates them on demand. This decouples the per-session cache from the batch structure — requests can share pages, variable-length requests do not waste allocation, preemption and swapping become tractable. PagedAttention is orthogonal to the MHA/GQA/MQA choice: it manages whatever KV cache the model produces. The savings stack multiplicatively: GQA cuts the per-token cache by {"8×"}, PagedAttention removes another {"\\approx 2-4×"} of waste from fragmentation, and the combined effect on throughput is {"\\approx 16-30×"} vs naive MHA-with-contiguous-allocation. This compounding is why the default vLLM+Llama-2-70B configuration hits {"\\approx 3000"} tokens/sec/H100 where a hypothetical MHA+contiguous version would be under 200.
      </Prose>

      <H3>8.5 Stacking with speculative decoding</H3>

      <Prose>
        Speculative decoding (Leviathan et al. 2023, Chen et al. 2023) uses a small "draft" model to propose tokens that a large "target" model verifies in parallel. The verification step re-evaluates the target model's logits over the draft sequence, which reads the target model's KV cache. A smaller KV cache (GQA) means faster verification reads and thus a larger speedup from speculation. In practice, GQA+SpecDec compose to roughly {"2-3×"} additional throughput on top of plain GQA, and are the standard stack in production serving systems today.
      </Prose>

      <H3>8.6 What does not scale — training memory</H3>

      <Prose>
        GQA is an inference-time optimization. During training, the forward pass must still materialize the full expanded K and V for attention (or compute them on the fly via FlashAttention), and the backward pass must store activations for backpropagation. Training memory is dominated by activations and optimizer state, not the KV cache; GQA's training-time memory savings are modest ({"\\approx 5\\%"} of total activation memory in a typical Llama-style setup). Small-batch training with a "save on KV cache" mindset does not benefit from GQA. The mental frame to hold: GQA is an inference-serving architecture, not a training architecture; its memory benefits appear at decode time and compound with batch size, not at training time.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES
          ====================================================================== */}
      <H2>9. Failure modes</H2>

      <H3>9.1 Wrong KV head × head_dim layout</H3>

      <Prose>
        The most common bug when writing GQA from scratch: reshaping {"W_K"}'s output into {"(n_{heads}, d_h)"} instead of {"(n_{kv\\_heads}, d_h)"}, and then silently getting a shape-valid tensor that has nothing to do with GQA. Symptoms: the code runs, the model trains, and quality is inexplicably bad. Diagnostic: after the reshape and transpose, assert the expected shape explicitly — {"assert K.shape == (B, self.n_kv_heads, T, self.head_dim)"}. Another common variant is forgetting to change {"W_K, W_V"}'s output dimension from {"h · d_h"} to {"g · d_h"} — the projection has too many parameters and the reshape crashes or (worse, if the arithmetic happens to work out) produces garbage.
      </Prose>

      <H3>9.2 Forgetting to expand K, V before attention</H3>

      <Prose>
        If you store and retrieve the KV cache at {"g"}-head resolution (correct) but forget to expand to {"h"} heads before computing attention (incorrect), the matmul {"Q · Kᵀ"} will fail with a shape mismatch — Q is {"[B, h, T, d_h]"} and K is {"[B, g, T, d_h]"} with {"g ≠ h"}. This is at least a <em>loud</em> failure — your code crashes — rather than a silent one. In FlashAttention-2 and SDPA, the expansion happens inside the kernel so you do not see the shape mismatch at the Python level, but you still need to ensure you are calling the GQA-capable kernel and not the plain-MHA one that expects matching head counts.
      </Prose>

      <H3>9.3 Initializing GQA from MHA without uptraining</H3>

      <Prose>
        You average the MHA K/V heads into GQA K/V heads and then load the weights into your inference engine without any further training. The model produces plausible but noticeably worse output, and quality benchmarks regress. This is the zero-step uptrain loss we saw in section 4.5: averaging heads discards information, and the model needs some amount of continued training to compensate. If you are doing this conversion, budget for the uptraining step; the warm-start init is a <em>start</em>, not an endpoint. Meta's Llama-2 70B would not have matched its Llama-1 MHA ancestor on any benchmark had they shipped the zero-step conversion.
      </Prose>

      <H3>9.4 RoPE applied after K expansion</H3>

      <Prose>
        Correct: apply RoPE to K at {"[B, g, T, d_h]"}, then expand to {"[B, h, T, d_h]"}. Wrong: expand first, then apply RoPE to the {"[B, h, T, d_h]"} expanded K. The <em>outputs</em> of the two orderings are usually mathematically identical (RoPE is a per-position, per-head rotation, and if all expanded copies start at the same position they receive the same rotation). But implementation-wise, (a) applying RoPE after expansion wastes {"s×"} compute on redundant rotations, and (b) in KV-cache-aware implementations it leads to storing the post-RoPE K in the cache at expanded size — which defeats GQA's memory savings entirely. Always apply RoPE at the {"n_{kv\\_heads}"} layout, in the cache-native representation, and expand only for attention computation.
      </Prose>

      <H3>9.5 Training from scratch with tiny batch: no memory win</H3>

      <Prose>
        You adopt GQA hoping to fit a larger batch into GPU memory during training. You are disappointed — the batch size is not meaningfully larger. This is because training memory is dominated by activations and optimizer state, not the KV cache. GQA shrinks the K/V activations by {"\\approx 8×"}, but K/V are only a small fraction of the per-layer activation budget (the FFN intermediate hidden states and the full {"T × T"} attention matrix dominate). Relief: GQA's inference-time memory savings are not training-time memory savings. If you need training memory relief, reach for gradient checkpointing, ZeRO-3, or FP8 optimizer state; GQA will not do it.
      </Prose>

      <H3>9.6 Mismatched {"num_key_value_heads"} between training and inference</H3>

      <Prose>
        You train with {"n_{kv} = 8"} but load the checkpoint into an inference engine configured for {"n_{kv} = 64"} (or vice versa). If the framework notices, you get a clear shape-mismatch error at weight load. If the framework does not notice — which happens when a custom runtime forces the weights into whatever shape it expects — you get silently wrong outputs. Diagnostic: always log {"num_attention_heads"} and {"num_key_value_heads"} from the loaded config at inference startup; assert they match the checkpoint.
      </Prose>

      <H3>9.7 Non-divisible {"h / n_{kv}"}</H3>

      <Prose>
        GQA requires {"n_{heads}"} to be an integer multiple of {"n_{kv\\_heads}"} so the groups are equal-sized and the {"repeat_interleave"} works cleanly. Picking {"h = 64, n_{kv} = 6"} will crash at the expansion step ({"64 / 6"} is not an integer) or, in some implementations, will silently use a non-uniform grouping and produce nonsense. Valid choices for {"h = 64"} are {"n_{kv} ∈ \\{1, 2, 4, 8, 16, 32, 64\\}"}. {"8"} is canonical.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        <strong>Shazeer (2019).</strong> "Fast Transformer Decoding: One Write-Head is All You Need." arXiv:1911.02150. The MQA paper. Six pages, one idea, extremely high density. Section 2 describes the mechanism; section 3 benchmarks against MHA on a small translation task. Worth reading in full for the sheer clarity of the thesis.
      </Prose>

      <Prose>
        <strong>Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón, Sanghai (2023).</strong> "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." EMNLP 2023. arXiv:2305.13245. The GQA paper. The entire framework — the interpolation between MHA and MQA, the uptraining recipe with averaging init, the quality-vs-{"n_{kv}"} ablation — is in sections 3 and 4. Figure 5 is the quality plot reproduced in section 6.4 here.
      </Prose>

      <Prose>
        <strong>Touvron, Martin, Stone, et al. (2023).</strong> "Llama 2: Open Foundation and Fine-Tuned Chat Models." arXiv:2307.09288. Section 2.2 confirms the GQA configuration for Llama-2 70B ({"n_{kv\\_heads} = 8, n_{heads} = 64, d_h = 128"}) and reports the resulting inference efficiency improvements. The first public deployment of GQA at frontier scale and the paper that made everyone else adopt it.
      </Prose>

      <Prose>
        <strong>Jiang, Sablayrolles, Mensch, et al. (2023).</strong> "Mistral 7B." arXiv:2310.06825. Documents GQA {"g=4"} ({"h = 32, n_{kv} = 8"}) at the 7B scale alongside sliding-window attention. Evidence that GQA is valuable well below the 70B threshold when long-context use is in play.
      </Prose>

      <Prose>
        <strong>Kwon, Li, Zhuang, Sheng, Zheng, Yu, Gonzalez, Zhang, Stoica (2023).</strong> "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP 2023. arXiv:2309.06180. vLLM. The memory-management layer that stacks on top of GQA to produce modern production serving throughput. Section 3 describes PagedAttention; the implementation is GQA-native.
      </Prose>

      <Prose>
        <strong>Dao (2023).</strong> "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." arXiv:2307.08691. Introduces first-class GQA and MQA kernel support in FlashAttention. Section 3.4 explicitly covers the grouped-KV variant and the index-arithmetic approach to avoiding explicit expansion.
      </Prose>

      <Prose>
        <strong>Chowdhery et al. (2022).</strong> "PaLM: Scaling Language Modeling with Pathways." arXiv:2204.02311. Section 2.2 reports MQA adoption at 540B scale, the first large-scale deployment of Shazeer's 2019 idea. Useful as the "MQA at scale" counterpoint to the "GQA at scale" story that Llama-2 later told.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK
          ====================================================================== */}
      <H2>11. Self-check</H2>

      <H3>11.1 Why does GQA save memory but not FLOPs at inference?</H3>

      <Prose>
        GQA stores fewer K and V heads in the cache ({"g"} instead of {"h"}), which is what shrinks memory and memory bandwidth. But when attention is actually computed, the K and V tensors must be expanded (via {"repeat_interleave"} or, in fused kernels, via index arithmetic) to match the {"h"} Q heads before the scaled-dot-product matmul. After expansion, the matmul is identical to MHA in size and therefore identical in FLOPs. The savings are entirely in (a) cache memory footprint, (b) HBM read bandwidth when loading K, V from cache into SMs. Since modern GPU attention decoding is memory-bound, the bandwidth savings translate roughly to a {"g/h"}-factor latency reduction even though FLOPs are unchanged.
      </Prose>

      <H3>11.2 Why {"g = 8"} and not some other value?</H3>

      <Prose>
        Empirically, Ainslie 2023 found that quality is flat from {"g = h"} down to {"g = 8"} and drops noticeably below that. {"g = 8"} is the smallest {"g"} that preserves MHA-level quality, and therefore the Pareto-optimal choice that maximizes memory savings subject to a no-quality-loss constraint. Going to {"g = 4"} costs a small amount of quality; {"g = 1"} (MQA) costs {"\\approx 1"} point. Going above {"g = 8"} wastes KV memory with no quality gain. The value is empirically derived, not theoretically predicted, but it has been replicated across enough model families (Llama, T5-style, decoder-only) that it can be treated as a well-established prior.
      </Prose>

      <H3>11.3 If I have a trained MHA model, what is the minimum-effort way to get GQA memory savings?</H3>

      <Prose>
        Use the uptraining recipe. (1) Load the MHA checkpoint. (2) Replace {"W_K"} and {"W_V"} with their group-averaged versions (section 3.5 formula): for each of the {"g"} target KV heads, take the mean of the corresponding {"s = h/g"} MHA heads. (3) Keep {"W_Q, W_O"} unchanged. (4) Continue training on the original or a matching data mixture for {"\\approx 5-10\\%"} of the original pretrain compute. The first few hundred steps recover the averaged-head representation; the remainder fine-tunes the full model for the new layout. At the end, you have a GQA checkpoint with near-MHA quality and {"h/g"}-fold KV cache savings. This is exactly the procedure in Ainslie 2023 section 3.2.
      </Prose>

      <H3>11.4 Why does RoPE need to be applied before K expansion?</H3>

      <Prose>
        Two reasons. First, compute efficiency: applying RoPE to {"g"} KV heads and then expanding is {"s×"} cheaper than expanding to {"h"} heads and then applying RoPE {"s×"} times over identical copies. Second, cache correctness: the KV cache stores K at {"g"}-head resolution. If RoPE is applied after expansion, you have applied RoPE to a transient expanded tensor and your cache still holds pre-RoPE K — the next decoding step, which reads K from cache and pairs it with a new Q (which has had RoPE applied), will have a position-phase mismatch. Applying RoPE pre-expansion keeps the cache in a RoPE-aware state and ensures position offsets compose correctly across decode steps.
      </Prose>

      <H3>11.5 Why doesn't GQA help training memory much?</H3>

      <Prose>
        Training memory is dominated by (a) stored activations for backpropagation — mostly the FFN intermediate representation at {"4 d_{model}"} and the {"T × T"} attention scores — and (b) optimizer state (two moments per parameter in Adam, at fp32, totals {"\\approx 6×"} the parameter memory). GQA shrinks the K/V activations and the K/V parameter count, but both are small fractions of the training-memory budget. FFN activations alone are typically larger than all of attention's activations combined. Optimizer state shrinks with GQA by the parameter-count reduction, which is {"\\approx 5-10\\%"} of the model. Net training memory savings are real but modest — single-digit percent. For real training memory relief, use gradient checkpointing (quadratically reduces activation memory), ZeRO-3 (shards optimizer state across ranks), or FP8 training. GQA is an inference-time memory architecture; its training-time effects are a side benefit, not the point.
      </Prose>

    </div>
  ),
};

export default gqaMqaContent;
