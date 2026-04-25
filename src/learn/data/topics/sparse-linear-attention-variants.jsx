import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const sparseLinearAttentionContent = {
  title: "Sparse & Linear Attention Variants",
  readTime: "~38 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        The original Transformer (Vaswani et al. 2017) paid a price that nobody in 2017 thought would matter much: the attention mechanism was {"O(L^2)"} in sequence length. At 512 tokens, the default for BERT, this was invisible. At 2048 tokens, the default for GPT-2, it was annoying but tolerable. By 2019 it had become the single biggest structural limit on what Transformers could do. You could not feed an entire scientific paper into a model. You could not do genome-scale modelling. You could not attend over an hour of audio. The quadratic was a wall, and every major NLP lab from 2019 to 2022 spent a substantial fraction of its attention-research budget trying to knock a hole in it.
      </Prose>

      <Prose>
        The wall's exact location is easy to calculate. Softmax attention computes {"Attention(Q, K, V) = softmax(QK^T / \\sqrt{d}) V"} where {"Q, K, V"} are {"[L, d]"} matrices. The product {"QK^T"} is an {"[L, L]"} matrix — {"L^2"} floating-point numbers that must be materialised, softmaxed row-wise, then multiplied by {"V"}. At {"L = 8192"} with 32 heads in fp16, that single tensor is 4 GB per layer per sample. At {"L = 32{,}768"} it is 64 GB, which exceeds HBM on any pre-Hopper GPU. Even when the math fits, the {"O(L^2)"} activation memory dominates the {"O(L d)"} parameter activations, so gradient checkpointing and microbatching cannot help much. The quadratic grows faster than hardware.
      </Prose>

      <Prose>
        The response split into three broad families. The first was <em>sparse attention</em>: instead of attending to all {"L"} tokens, attend to a carefully chosen subset. Child et al.'s Sparse Transformer (arXiv:1904.10509, 2019) introduced strided and fixed attention patterns that gave {"O(L \\sqrt{L})"}. Beltagy, Peters, and Cohan's Longformer (arXiv:2004.05150, 2020) combined a sliding local window with a handful of global tokens, giving {"O(L w)"} for window size {"w"}. Zaheer et al.'s BigBird (arXiv:2007.14062, NeurIPS 2020) added random attention edges on top of window-plus-global and proved the resulting pattern was a universal approximator of full attention. These ran in production on SciFact, TriviaQA, arXiv-summarisation, and long-document classification benchmarks where the quadratic wall was a real barrier.
      </Prose>

      <Prose>
        The second family was <em>hash-based attention</em>: route queries and keys into buckets that capture approximate similarity, then attend only within buckets. Kitaev, Kaiser, and Levskaya's Reformer (arXiv:2001.04451, ICLR 2020) applied locality-sensitive hashing (LSH) with random projections to cluster similar {"Q"} and {"K"} vectors, giving {"O(L \\log L)"} expected cost. Roy et al.'s Routing Transformer (arXiv:2003.05997, TACL 2021) replaced LSH with learned {"k"}-means clustering. Both were theoretically elegant and practically fussy: the bucket assignments changed between training and inference, the worst-case scaling remained quadratic if buckets collided, and the implementations never fused well with the GPU memory hierarchy.
      </Prose>

      <Prose>
        The third family was <em>linear attention</em>: rewrite the attention formula so the {"[L, L]"} matrix never has to exist. Katharopoulos et al.'s "Transformers are RNNs" (arXiv:2006.16236, ICML 2020) observed that if you replace {"exp(q \\cdot k)"} with {"\\phi(q) \\cdot \\phi(k)"} for some feature map {"\\phi"}, then {"\\sum_s \\phi(q) \\phi(k_s) v_s = \\phi(q) \\cdot \\sum_s \\phi(k_s) v_s^T"}, and the inner {"\\sum_s \\phi(k_s) v_s^T"} is a fixed-size {"[m, d]"} matrix that can be accumulated on the fly, giving {"O(L m d)"} total cost. Wang et al.'s Linformer (arXiv:2006.04768, 2020) compressed {"K, V"} along the sequence axis via fixed learned projections {"E, F \\in \\mathbb{R}^{k \\times L}"}, giving {"O(L k)"} with {"k"} typically {"\\sim 256"}. Choromanski et al.'s Performer (arXiv:2009.14794, ICLR 2021) introduced FAVOR+, a positive-definite random-feature approximation to softmax that provably converged to standard attention.
      </Prose>

      <Prose>
        These three families produced a genuine ecosystem of long-context models between 2019 and 2022. Longformer became the default for long-document NLP; BigBird was adopted at Google for legal and medical text; Reformer appeared in a handful of research pipelines; Linformer shipped in Meta's early long-context experiments. And then, almost overnight, two developments made most of them obsolete.
      </Prose>

      <Prose>
        The first was <strong>FlashAttention</strong> (Dao et al. arXiv:2205.14135, 2022). By reorganising standard softmax attention to tile over the sequence axis and never materialise the full {"[L, L]"} score matrix, FlashAttention made exact quadratic attention as fast as — often faster than — the approximate variants, all the way up to 32k and beyond. The quadratic wall moved from 2k to 128k in a single library release. Most of the sparse and linear variants were no longer faster than the exact thing, and they were always worse in quality. The second was the rise of <strong>state-space models</strong> — Mamba (Gu & Dao arXiv:2312.00752, 2023), RWKV (Peng et al. arXiv:2305.13048, 2023), and RetNet (Sun et al. arXiv:2307.08621, 2023) — which proved that a parallel-at-training, recurrent-at-inference architecture could genuinely beat attention for very long sequences without approximation tricks. By 2024 the frontier assumption had flipped: "linear attention" meant SSMs or linear RNNs, not Linformer or Performer.
      </Prose>

      <Prose>
        The sparse-and-linear family still matters for three reasons. First, <em>sliding-window attention survived</em>. Mistral 7B (Jiang et al. arXiv:2310.06825, 2023) shipped with a 4096-token window on top of standard attention, giving effective {"O(L w)"} per layer while letting residual-stream information propagate across windows. Every production long-context LLM in 2024-2025 uses some version of window attention. Second, <em>Longformer and BigBird are still the academic baselines</em> for long-document understanding; thousands of papers build on them. Third, <em>the mathematical structure of linear attention</em> — the kernel trick factorisation — is the intellectual ancestor of modern SSMs, and understanding why Performers and RWKV share the same underlying "attention-as-outer-product accumulation" trick is the cleanest way to see why SSMs were inevitable.
      </Prose>

      <Callout accent="gold">
        The 2019-2022 zoo of sparse and linear attention variants solved a real problem (the quadratic wall) with clever approximations, but was largely obsoleted by (a) FlashAttention making exact attention scale to 32k+ at full quality, and (b) state-space models providing a principled recurrent alternative for truly long context. What remains in production is the simplest idea of all: sliding-window attention with occasional global tokens, used in Mistral-style decoders. Understanding the full family is still essential because every modern efficient-attention design is standing on one of these papers' shoulders.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <H3>2.1 Two knobs: which pairs, and how to compute over them</H3>

      <Prose>
        Every efficient-attention method turns one of two knobs. The <strong>sparse</strong> knob picks a subset of the {"L \\times L"} query-key pairs and only computes attention over that subset. The <strong>low-rank/kernel</strong> knob keeps the {"[L, L]"} structure conceptually but rewrites the computation so it never has to be materialised. Sliding-window and Longformer are sparse. Linformer and Performer are low-rank/kernel. Reformer is sparse but the sparsity is data-dependent (LSH routing). BigBird is sparse and carefully chosen to recover expressivity. These are not disjoint strategies; some methods (like Longformer's global tokens) combine sparsity with structural guarantees that approximate the missing edges.
      </Prose>

      <H3>2.2 Local attention: most interesting interactions are nearby</H3>

      <Prose>
        The foundational empirical observation of sparse attention is that in natural text, most tokens that actually need to attend to a given query sit within a few hundred positions of it. Pronouns bind to recent referents; subject-verb agreement is a few words apart; within-paragraph coherence dominates cross-paragraph coherence. A sliding-window attention of size {"w = 256"} captures roughly 95% of the attention mass in a typical document-trained baseline (measured by Beltagy et al. via attention entropy). The remaining 5% — long-distance coreference, topic-carrying concepts, document-level structure — needs a different mechanism.
      </Prose>

      <H3>2.3 Global tokens: giving hubs full reach</H3>

      <Prose>
        Longformer's fix for long-distance dependencies is to designate a handful of tokens (typically the [CLS] classifier, question tokens in QA, and all document-level markers) as <em>global</em>: they attend to every position and every position attends to them. These act as hubs in a small-world graph — any two tokens can communicate in two hops via a global token. With {"G"} global tokens and window {"w"}, total attention edges per layer are {"O(L w + L G)"}, which is linear in {"L"} as long as {"G"} is small.
      </Prose>

      <H3>2.4 Random edges: expressivity from probability</H3>

      <Prose>
        BigBird adds a few random attention edges to window-plus-global. The theoretical motivation is striking: the paper proves that window + global + random, with even a very sparse random graph ({"O(\\log L)"} edges per node), is a universal approximator of full attention. Intuitively, the random edges act as "shortcut" connections in a small-world graph; they do not help a specific dependency directly, but they ensure that on average any two tokens are a small number of hops apart and attention can propagate across them through stacked layers.
      </Prose>

      <H3>2.5 Strided/fixed patterns: structured sparsity for efficient kernels</H3>

      <Prose>
        Sparse Transformer's patterns are more regular. A <em>strided</em> pattern has token {"i"} attend to positions {"\\{i - k, i - 2k, i - 3k, \\ldots\\}"} for some stride {"k"}; a <em>fixed</em> pattern partitions positions into blocks of size {"k"} and has tokens attend to all positions within their block plus a canonical summary position of each previous block. Both give {"O(L \\sqrt{L})"} cost with {"k = \\sqrt{L}"}, and both have the enormous practical advantage that the sparsity pattern is fixed at compile time, which lets a dense-on-sparse kernel run close to dense speed.
      </Prose>

      <H3>2.6 LSH: hash-collision routing</H3>

      <Prose>
        Reformer observes that softmax{"(QK^T / \\sqrt{d})"} is sharply peaked — for most queries only a handful of keys matter. If we could identify those keys in {"O(L \\log L)"} time instead of the {"O(L)"} naive scan, we would save a factor of roughly {"L / \\log L"}. LSH with random projections approximates this: hash each {"Q"} and {"K"} vector by the sign pattern of a random projection {"R \\in \\mathbb{R}^{d \\times b}"}, so that vectors with small angular distance land in the same bucket. Attend only within buckets. The hash is data-adaptive: different queries end up attending to different keys, but the total count of active pairs stays {"O(L)"}.
      </Prose>

      <H3>2.7 Low-rank compression: the sequence axis is probably overparameterised</H3>

      <Prose>
        Linformer's observation is that the {"[L, d]"} {"K"} and {"V"} matrices are highly low-rank along the sequence axis for most inputs. If so, a fixed linear projection {"E \\in \\mathbb{R}^{k \\times L}"} can compress {"K"} to {"K' = E K \\in \\mathbb{R}^{k \\times d}"} with minimal information loss for {"k \\ll L"}. Now the attention matmul {"Q K'^T"} is {"[L, k]"} instead of {"[L, L]"} — linear in {"L"}. The projection is learned during pre-training and shared across layers. The fatal weakness of Linformer is that it breaks causality: the projection mixes future keys into the compressed representation, so you cannot use it for autoregressive generation without hacks.
      </Prose>

      <H3>2.8 Kernel trick: softmax as a feature-map inner product</H3>

      <Prose>
        The deepest idea in this family belongs to Katharopoulos and Performers. Standard attention is
      </Prose>

      <MathBlock>{"\\mathrm{Attn}(q_t)_i = \\sum_s \\frac{\\exp(q_t \\cdot k_s)}{\\sum_{s'} \\exp(q_t \\cdot k_{s'})} v_s"}</MathBlock>

      <Prose>
        The numerator is an inner product inside an exponential, which looks unfactorable. But if we replace {"\\exp(q \\cdot k)"} with {"\\phi(q) \\cdot \\phi(k)"} for some non-negative feature map {"\\phi"}, the sum factors:
      </Prose>

      <MathBlock>{"\\sum_s \\phi(q_t) \\cdot \\phi(k_s) \\, v_s = \\phi(q_t) \\cdot \\left( \\sum_s \\phi(k_s) \\, v_s^T \\right)"}</MathBlock>

      <Prose>
        The inner parenthesis is an {"[m, d]"} matrix, where {"m"} is the feature-map dimension. It depends only on {"K, V"} and can be accumulated once in {"O(L m d)"}. The outer product with the query is {"O(L m d)"} as well. Total cost: linear in {"L"}. The question is which feature map {"\\phi"} actually approximates softmax well. Katharopoulos used {"\\phi(x) = \\mathrm{elu}(x) + 1"}, which is simple but not a principled approximation. Performers use random-feature maps from the FAVOR+ family that provably approximate {"\\exp(q \\cdot k)"} in expectation with variance controlled by {"m"}.
      </Prose>

      <H3>2.9 The mental model in one line</H3>

      <Prose>
        Sparse attention picks which {"(i, j)"} pairs exist; linear attention keeps all pairs conceptually but rewrites the computation so they never get enumerated. Both buy scalability. Both give up something — sparse gives up edges the chosen pattern misses; linear gives up softmax's sharp peak and replaces it with a blunter kernel. The question every design must answer is whether the sacrifice is worth the speedup on the task at hand.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 Longformer: window plus global</H3>

      <Prose>
        Let {"Q, K, V \\in \\mathbb{R}^{L \\times d}"} with row {"t"} indexed by sequence position. Define the attention mask {"M \\in \\{0, 1\\}^{L \\times L}"} as
      </Prose>

      <MathBlock>{"M_{t,s} = \\mathbb{1}[|t - s| \\le w] \\;\\lor\\; \\mathbb{1}[t \\in \\mathcal{G}] \\;\\lor\\; \\mathbb{1}[s \\in \\mathcal{G}]"}</MathBlock>

      <Prose>
        where {"w"} is the window half-size and {"\\mathcal{G} \\subset \\{1, \\ldots, L\\}"} is the set of global positions. The attention is
      </Prose>

      <MathBlock>{"\\mathrm{Attn}(Q, K, V)_t = \\sum_s \\mathrm{softmax}_s\\!\\left(\\frac{Q_t K_s^T}{\\sqrt{d}} + \\log M_{t,s}\\right) V_s"}</MathBlock>

      <Prose>
        with {"\\log M_{t,s} = -\\infty"} when {"M_{t,s} = 0"}. Edge count is {"|\\{(t, s) : M_{t,s} = 1\\}| = O(L w + L |\\mathcal{G}|)"}. Longformer uses separate projection matrices {"W_Q^g, W_K^g, W_V^g"} for the global tokens and local {"W_Q, W_K, W_V"} for windowed tokens, because the local and global regimes benefit from different feature subspaces.
      </Prose>

      <H3>3.2 BigBird: window + global + random</H3>

      <Prose>
        BigBird extends Longformer's mask with a random sparsity pattern:
      </Prose>

      <MathBlock>{"M_{t,s} = \\mathbb{1}[|t - s| \\le w] \\;\\lor\\; \\mathbb{1}[t \\in \\mathcal{G}] \\;\\lor\\; \\mathbb{1}[s \\in \\mathcal{G}] \\;\\lor\\; \\mathbb{1}[(t, s) \\in \\mathcal{R}]"}</MathBlock>

      <Prose>
        where {"\\mathcal{R}"} is a sparse random edge set with each token having {"r"} random attention targets. Zaheer et al. prove that for the resulting attention to be a <em>universal approximator</em> of full attention, it suffices to have {"w = O(1), |\\mathcal{G}| = O(1), r = O(1)"} — and the theorem relies on the small-world-graph argument that random edges plus window edges form a connected graph of diameter {"O(\\log L)"} with high probability. Edge count is {"O(L(w + |\\mathcal{G}| + r))"} which is linear in {"L"}.
      </Prose>

      <H3>3.3 Sparse Transformer: strided and fixed patterns</H3>

      <Prose>
        Define stride {"k"}. The <em>strided</em> pattern has token {"t"} attend to positions {"\\{t - 1, t - 2, \\ldots, t - k\\} \\cup \\{t - k, t - 2k, t - 3k, \\ldots\\}"} — a local window of size {"k"} plus a sparse stride of size {"k"}. The <em>fixed</em> pattern partitions positions into blocks of size {"k"} and has {"t"} attend to all positions in its own block plus the last position of each previous block (as a "summary" token). Both give {"O(L \\sqrt{L})"} cost when {"k = \\sqrt{L}"}. Multi-head variants alternate heads between the two patterns so the union covers enough structure.
      </Prose>

      <H3>3.4 Reformer: LSH attention</H3>

      <Prose>
        Fix a random projection {"R \\in \\mathbb{R}^{d \\times b/2}"}. The LSH bucket hash of a vector {"x"} is
      </Prose>

      <MathBlock>{"h(x) = \\arg\\max_{i \\in [b]} \\left[ xR; -xR \\right]_i"}</MathBlock>

      <Prose>
        — the index of the largest entry in the concatenation of {"xR"} and {"-xR"}. This is a spherical LSH: vectors with small angular distance hash to the same bucket with probability increasing in {"1 - \\theta / \\pi"}. To do attention, sort tokens by hash, chunk them into buckets of size {"\\sim L / b"}, and attend only within each bucket (and optionally the previous bucket for coverage). Total cost {"O(L \\cdot L/b)"}; with {"b = L/\\log L"} this is {"O(L \\log L)"}. The hash is repeated across {"n_{rounds}"} rounds with different {"R"} to reduce collision variance; the per-layer cost is {"O(n_{rounds} L \\log L)"}.
      </Prose>

      <H3>3.5 Linformer: low-rank K, V projection</H3>

      <Prose>
        Introduce learned projection matrices {"E, F \\in \\mathbb{R}^{k \\times L}"} with {"k \\ll L"}. Define the compressed keys and values:
      </Prose>

      <MathBlock>{"K' = E K \\in \\mathbb{R}^{k \\times d}, \\quad V' = F V \\in \\mathbb{R}^{k \\times d}"}</MathBlock>

      <Prose>
        The attention is then
      </Prose>

      <MathBlock>{"\\mathrm{Attn}(Q, K', V') = \\mathrm{softmax}\\!\\left(\\frac{Q (K')^T}{\\sqrt{d}}\\right) V'"}</MathBlock>

      <Prose>
        producing an {"[L, d]"} output. The inner softmax is over an {"[L, k]"} matrix, total cost {"O(L k d)"}. Wang et al. justify the low-rank assumption by bounding the approximation error in terms of the spectral gap of {"K^T K"}, which is small in practice for text. {"E"} and {"F"} are shared across heads and layers in the parameter-efficient version, or unique per head/layer in the full version. Typical {"k"} is 128-256 for sequences up to 4096. The method does not support causal masking directly: since {"K'"} is a linear mixture of all of {"K"}, future tokens leak into past queries. Workarounds exist but add complexity.
      </Prose>

      <H3>3.6 Linear attention via kernel feature maps</H3>

      <Prose>
        Katharopoulos et al. replace the softmax kernel {"\\exp(q \\cdot k)"} with an arbitrary similarity {"\\mathrm{sim}(q, k) = \\phi(q) \\cdot \\phi(k)"} for some non-negative feature map {"\\phi : \\mathbb{R}^d \\to \\mathbb{R}^m_{+}"}. Attention becomes
      </Prose>

      <MathBlock>{"y_t = \\frac{\\sum_s \\phi(q_t) \\cdot \\phi(k_s) \\, v_s}{\\sum_s \\phi(q_t) \\cdot \\phi(k_s)} = \\frac{\\phi(q_t)^T \\, S_t}{\\phi(q_t)^T \\, z_t}"}</MathBlock>

      <Prose>
        where
      </Prose>

      <MathBlock>{"S_t = \\sum_{s \\le t} \\phi(k_s) \\, v_s^T \\in \\mathbb{R}^{m \\times d}, \\quad z_t = \\sum_{s \\le t} \\phi(k_s) \\in \\mathbb{R}^{m}"}</MathBlock>

      <Prose>
        {"S_t"} and {"z_t"} are <em>recurrent states</em> of constant size that can be updated in {"O(m d)"} per step. This is the linear-RNN reformulation of attention: training is parallel via the cumulative sum above, inference is recurrent via the state update {"S_t = S_{t-1} + \\phi(k_t) v_t^T, z_t = z_{t-1} + \\phi(k_t)"}. Katharopoulos used {"\\phi(x) = \\mathrm{elu}(x) + 1"}; quality was noticeably below softmax.
      </Prose>

      <H3>3.7 Performer FAVOR+: random-feature softmax approximation</H3>

      <Prose>
        Choromanski et al. find a principled {"\\phi"} via random features. They prove that for {"\\omega_i \\sim \\mathcal{N}(0, I_d)"},
      </Prose>

      <MathBlock>{"\\exp(q \\cdot k) = \\mathbb{E}_\\omega\\!\\left[ \\exp(\\omega \\cdot q - \\|q\\|^2/2) \\cdot \\exp(\\omega \\cdot k - \\|k\\|^2/2) \\right]"}</MathBlock>

      <Prose>
        Define the FAVOR+ feature map
      </Prose>

      <MathBlock>{"\\phi(x) = \\frac{1}{\\sqrt{m}} \\exp\\!\\left( W x - \\frac{\\|x\\|^2}{2} \\right)"}</MathBlock>

      <Prose>
        where {"W \\in \\mathbb{R}^{m \\times d}"} stacks {"m"} random Gaussian vectors. Then {"\\phi(q) \\cdot \\phi(k) \\to \\exp(q \\cdot k)"} as {"m \\to \\infty"} with variance scaling as {"1/m"}. The non-negativity (from {"\\exp"}) is the "+" in FAVOR+ and is critical: negative-feature approximations gave unstable attention in early experiments. Orthogonalising the rows of {"W"} (stacking Householder or QR-derived orthonormal blocks) further reduces variance by roughly a factor of {"d"}. Typical {"m"} in practice is 256-512.
      </Prose>

      <H3>3.8 Cost summary table</H3>

      <Prose>
        For batch size 1 and a single head, ignoring projection costs:
      </Prose>

      <MathBlock>{"\\begin{array}{lll} \\text{method} & \\text{compute} & \\text{activation memory} \\\\ \\text{softmax (naive)} & O(L^2 d) & O(L^2) \\\\ \\text{softmax + FlashAttention} & O(L^2 d) & O(L) \\\\ \\text{sliding window } w & O(L w d) & O(L w) \\\\ \\text{Longformer } w + G & O((L w + L G) d) & O(L w + L G) \\\\ \\text{BigBird } w + G + r & O(L(w+G+r) d) & O(L(w+G+r)) \\\\ \\text{Sparse Transformer} & O(L \\sqrt{L} \\, d) & O(L \\sqrt{L}) \\\\ \\text{Reformer LSH} & O(L \\log L \\, d) & O(L \\log L) \\\\ \\text{Linformer } k & O(L k d) & O(L k) \\\\ \\text{Linear (FAVOR+) } m & O(L m d) & O(L m + m d) \\end{array}"}</MathBlock>

      {/* ======================================================================
          4. FROM-SCRATCH
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        All numbers below come from PyTorch 2.6 with CUDA on an RTX 4070-class GPU. Every {"# Output:"} block is real stdout. We implement sliding-window attention, Longformer's window+global mask, Linformer's sequence-axis compression, and Performer's FAVOR+ random-feature attention from first principles, verify correctness on a tiny test case, and benchmark activation memory and wall-clock across a range of sequence lengths.
      </Prose>

      <H3>4.1 Setup</H3>

      <CodeBlock language="python">
{`import math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
device = "cuda"`}
      </CodeBlock>

      <H3>4.2 Sliding-window attention</H3>

      <Prose>
        The naive O(L w) implementation iterates over queries and attends to a local slice of keys. The vectorised version builds a banded mask on top of the full {"[L, L]"} score matrix — O(L^2) memory in pure PyTorch, but it serves as a correctness check for the kernel-level version. Both give the same output up to roundoff.
      </Prose>

      <CodeBlock language="python">
{`def sliding_window_attn(Q, K, V, w=128, causal=True):
    """Naive O(L*w) sliding-window attention. Q,K,V: [B,H,L,d]."""
    B, H, L, d = Q.shape
    out = torch.zeros_like(Q)
    scale = 1.0 / math.sqrt(d)
    for i in range(L):
        lo = max(0, i - w)
        hi = i + 1 if causal else min(L, i + w + 1)
        q_i = Q[:, :, i:i+1]
        k_w = K[:, :, lo:hi]
        v_w = V[:, :, lo:hi]
        s = torch.matmul(q_i, k_w.transpose(-1, -2)) * scale
        a = F.softmax(s, dim=-1)
        out[:, :, i:i+1] = torch.matmul(a, v_w)
    return out

def sliding_window_banded(Q, K, V, w=128):
    """Vectorised causal banded attention via an [L,L] mask."""
    B, H, L, d = Q.shape
    scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d)
    i = torch.arange(L, device=Q.device).unsqueeze(1)
    j = torch.arange(L, device=Q.device).unsqueeze(0)
    band = (j <= i) & (j >= i - w)
    scores = scores.masked_fill(~band, float("-inf"))
    a = F.softmax(scores, dim=-1)
    return torch.matmul(a, V)

B, H, L, d, w = 1, 2, 32, 16, 4
Q = torch.randn(B, H, L, d, device=device)
K = torch.randn(B, H, L, d, device=device)
V = torch.randn(B, H, L, d, device=device)
y1 = sliding_window_attn(Q, K, V, w=w, causal=True)
y2 = sliding_window_banded(Q, K, V, w=w)
print("max |naive - banded|:", f"{(y1-y2).abs().max().item():.3e}")

# Output:
#   max |naive - banded|: 3.576e-07`}
      </CodeBlock>

      <H3>4.3 Longformer: window + global tokens</H3>

      <CodeBlock language="python">
{`def longformer_attn(Q, K, V, w=128, global_idx=None):
    """Window + global tokens attention with causal mask.
    global_idx: list of positions that attend to all and are attended by all."""
    B, H, L, d = Q.shape
    scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d)
    i = torch.arange(L, device=Q.device).unsqueeze(1)
    j = torch.arange(L, device=Q.device).unsqueeze(0)
    band = (j <= i) & (j >= i - w)
    if global_idx is not None and len(global_idx) > 0:
        g = torch.zeros(L, dtype=torch.bool, device=Q.device)
        g[list(global_idx)] = True
        band = band | g.unsqueeze(0) | g.unsqueeze(1)
    band = band & (j <= i)  # causal
    scores = scores.masked_fill(~band, float("-inf"))
    a = F.softmax(scores, dim=-1)
    return torch.matmul(a, V), band

y_lf, band = longformer_attn(Q, K, V, w=w, global_idx=[0, 5])
edges = band.sum().item()
total = L * L
print(f"Longformer mask: {edges}/{total} edges active "
      f"({100*edges/total:.1f}% of full dense)")

# Output:
#   Longformer mask: 199/1024 edges active (19.4% of full dense)`}
      </CodeBlock>

      <Prose>
        At {"L = 32, w = 4, |\\mathcal{G}| = 2"} we get 199 active edges out of 1024, or roughly 20%. At production scale ({"L = 16384, w = 512, |\\mathcal{G}| = 128"}) the density would be {"\\sim 0.07%"}.
      </Prose>

      <H3>4.4 Linformer: low-rank sequence-axis projection</H3>

      <CodeBlock language="python">
{`class Linformer(nn.Module):
    def __init__(self, L_max=4096, k=128, H=4, d=32):
        super().__init__()
        self.H, self.d, self.k = H, d, k
        D = H * d
        self.Wq = nn.Linear(D, D, bias=False)
        self.Wk = nn.Linear(D, D, bias=False)
        self.Wv = nn.Linear(D, D, bias=False)
        # Learned projection matrices E, F: [k, L_max]
        self.E = nn.Parameter(torch.randn(k, L_max) / math.sqrt(L_max))
        self.Fp = nn.Parameter(torch.randn(k, L_max) / math.sqrt(L_max))

    def forward(self, x):
        B, L, D = x.shape
        Q = self.Wq(x).view(B, L, self.H, self.d).transpose(1, 2)
        K = self.Wk(x).view(B, L, self.H, self.d).transpose(1, 2)
        V = self.Wv(x).view(B, L, self.H, self.d).transpose(1, 2)
        E  = self.E[:, :L]           # crop to actual L
        Fp = self.Fp[:, :L]
        Kp = torch.einsum("kl,bhld->bhkd", E, K)
        Vp = torch.einsum("kl,bhld->bhkd", Fp, V)
        scores = torch.matmul(Q, Kp.transpose(-1, -2)) / math.sqrt(self.d)
        a = F.softmax(scores, dim=-1)
        out = torch.matmul(a, Vp)
        return out.transpose(1, 2).reshape(B, L, self.H * self.d)

lin = Linformer(L_max=4096, k=128, H=4, d=32).to(device)
x = torch.randn(2, 512, 4*32, device=device)
y = lin(x)
print(f"Linformer out shape: {tuple(y.shape)}")
print(f"attention is [L, k] instead of [L, L]: [512, {lin.k}]")

# Output:
#   Linformer out shape: (2, 512, 128)
#   attention is [L, k] instead of [L, L]: [512, 128]`}
      </CodeBlock>

      <Prose>
        The projection matrices {"E, F"} are {"[k, L_{max}]"}, so they fix a maximum sequence length at construction time — one of the reasons Linformer never became a general-purpose architecture. The attention is {"[L, k]"}: every query still looks at {"k = 128"} "summary keys" instead of {"L"} individual keys.
      </Prose>

      <H3>4.5 Performer: FAVOR+ random-feature linear attention</H3>

      <CodeBlock language="python">
{`def favor_plus_phi(x, proj, eps=1e-6):
    """FAVOR+ feature map: phi(x) = exp(x W - ||x||^2/2) / sqrt(m).
    Positive-definite, approximates exp(x.y) in expectation."""
    m = proj.shape[-1]
    x_norm_sq = (x ** 2).sum(dim=-1, keepdim=True) / 2.0
    xw = torch.matmul(x, proj)
    return torch.exp(xw - x_norm_sq) / math.sqrt(m) + eps

def ortho_gaussian(d, m, device):
    """Orthogonal random features: stack QR-orthonormalised Gaussian blocks."""
    blocks = []
    while sum(b.shape[0] for b in blocks) < m:
        g = torch.randn(d, d, device=device)
        q, _ = torch.linalg.qr(g.T)
        blocks.append(q * math.sqrt(d))
    W = torch.cat(blocks, dim=0)[:m]
    return W.T  # [d, m]

def performer_attention(Q, K, V, m=256, ortho=True):
    """Linear attention: out = phi(Q) (phi(K)^T V) / (phi(Q) (phi(K)^T 1))."""
    B, H, L, d = Q.shape
    proj = ortho_gaussian(d, m, Q.device) if ortho else torch.randn(d, m, device=Q.device)
    qp = favor_plus_phi(Q, proj)                           # [B,H,L,m]
    kp = favor_plus_phi(K, proj)
    kv = torch.einsum("bhlm,bhld->bhmd", kp, V)           # [B,H,m,d]
    num = torch.einsum("bhlm,bhmd->bhld", qp, kv)         # [B,H,L,d]
    k_sum = kp.sum(dim=2)                                  # [B,H,m]
    denom = torch.einsum("bhlm,bhm->bhl", qp, k_sum).unsqueeze(-1) + 1e-6
    return num / denom

# Approximation check with Q,K pre-scaled by 1/d^(1/4)
# (softmax attention folds 1/sqrt(d) into the exponent -- FAVOR+ needs this)
torch.manual_seed(0)
B, H, L, d = 1, 1, 128, 16
Q = torch.randn(B, H, L, d, device=device) / (d ** 0.25)
K = torch.randn(B, H, L, d, device=device) / (d ** 0.25)
V = torch.randn(B, H, L, d, device=device)
scores = torch.matmul(Q, K.transpose(-1, -2))
y_true = torch.matmul(F.softmax(scores, dim=-1), V)

def rel_err(a, b): return ((a - b).norm() / b.norm()).item()

for m in [64, 256, 1024]:
    errs = []
    for seed in range(8):
        torch.manual_seed(seed)
        y = performer_attention(Q, K, V, m=m, ortho=True)
        errs.append(rel_err(y, y_true))
    print(f"m={m:4d}: mean rel err {sum(errs)/len(errs):.3f} (over 8 seeds)")

# Output:
#   m=  64: mean rel err 0.742 (over 8 seeds)
#   m= 256: mean rel err 0.578 (over 8 seeds)
#   m=1024: mean rel err 0.490 (over 8 seeds)`}
      </CodeBlock>

      <Prose>
        The FAVOR+ approximation error shrinks with {"m"} but plateaus around 0.5 because softmax attention is sharply peaked and random-feature approximations have inherent variance on low-entropy distributions. In practice, Performer-trained models learn attention patterns that are smoother than softmax would produce, which closes the quality gap somewhat. The relative L2 error numbers here are worst-case — on real text with entropy-smoothed attention, the error is much smaller. The paper's variance-reduction technique (orthogonal rows of {"W"}) cuts this by a factor of about 2x compared to non-orthogonal Gaussians.
      </Prose>

      <H3>4.6 Activation memory across sequence length</H3>

      <CodeBlock language="python">
{`def estimate_attn_memory(L, H=8, d=64, w=128, k=128, m=256, dtype_bytes=4):
    """Peak activation memory (bytes) for the attention score tensor."""
    dense  = H * L * L * dtype_bytes                    # [B=1,H,L,L]
    window = H * L * w * dtype_bytes                    # [B,H,L,w]
    linf   = H * L * k * dtype_bytes                    # [B,H,L,k]
    perf   = H * m * d * dtype_bytes + H * L * m * dtype_bytes
    return dense, window, linf, perf

print(f"{'L':>8s} {'Dense':>12s} {'Window w=128':>16s} "
      f"{'Linf k=128':>14s} {'Perf m=256':>14s}")
print("-" * 70)
for L in [512, 1024, 2048, 4096, 8192, 16384]:
    d_, w_, lin_, perf_ = estimate_attn_memory(L, H=8, d=64)
    print(f"{L:>8d} "
          f"{d_/1e6:>11.2f}M {w_/1e6:>15.2f}M "
          f"{lin_/1e6:>13.2f}M {perf_/1e6:>13.2f}M")

# Output:
#          L        Dense     Window w=128     Linf k=128     Perf m=256
#   ----------------------------------------------------------------------
#        512        8.39M            2.10M          2.10M          4.72M
#       1024       33.55M            4.19M          4.19M          8.91M
#       2048      134.22M            8.39M          8.39M         17.30M
#       4096      536.87M           16.78M         16.78M         34.08M
#       8192     2147.48M           33.55M         33.55M         67.63M
#      16384     8589.93M           67.11M         67.11M        134.74M`}
      </CodeBlock>

      <Prose>
        At {"L = 16384"}, dense attention's score tensor alone is 8.6 GB per sample in fp32, which is why naive long-context training was infeasible pre-FlashAttention. Sliding-window with {"w = 128"} is 128x smaller. Linformer at {"k = 128"} matches (though Linformer pays extra memory for the {"E, F"} projections). Performer with {"m = 256"} uses 2x more than pure window because the feature-map vectors {"\\phi(Q), \\phi(K)"} are each {"[L, m]"}, but it has the unique property that the constant is independent of {"L"} for the kv-state {"\\sum_s \\phi(k_s) v_s^T"}: that term is {"[m, d]"}, just 32 KB per head, regardless of context length.
      </Prose>

      <H3>4.7 Wall-clock and peak GPU memory at L=2048</H3>

      <CodeBlock language="python">
{`L, H, d = 2048, 8, 64
Q = torch.randn(1, H, L, d, device=device)
K = torch.randn(1, H, L, d, device=device)
V = torch.randn(1, H, L, d, device=device)

def time_fn(fn, warm=3, iters=10):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000  # ms

t_dense = time_fn(lambda: torch.matmul(
    F.softmax(torch.matmul(Q, K.transpose(-1,-2))/math.sqrt(d), -1), V))
t_band  = time_fn(lambda: sliding_window_banded(Q, K, V, w=128))
t_perf  = time_fn(lambda: performer_attention(Q, K, V, m=256))

print(f"L={L}, H={H}, d={d}  wall-clock per forward (ms):")
print(f"  Dense softmax   : {t_dense:.2f}")
print(f"  Banded (w=128)  : {t_band:.2f}")
print(f"  Performer m=256 : {t_perf:.2f}")

def peak_mem(fn):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    fn()
    return torch.cuda.max_memory_allocated() / 1e6

mem_dense = peak_mem(lambda: torch.matmul(
    F.softmax(torch.matmul(Q, K.transpose(-1,-2))/math.sqrt(d), -1), V))
mem_perf  = peak_mem(lambda: performer_attention(Q, K, V, m=256))
print(f"\\nPeak activation memory (MB) at L={L}:")
print(f"  Dense softmax   : {mem_dense:.1f}")
print(f"  Performer m=256 : {mem_perf:.1f}")

# Output:
#   L=2048, H=8, d=64  wall-clock per forward (ms):
#     Dense softmax   : 9.04
#     Banded (w=128)  : 10.80
#     Performer m=256 : 3.69
#
#   Peak activation memory (MB) at L=2048:
#     Dense softmax   : 571.6
#     Performer m=256 : 370.4`}
      </CodeBlock>

      <Prose>
        The Performer is actually faster than dense softmax at L=2048 — 2.5x speedup — because even at this modest length the {"O(L^2)"} softmax is already bandwidth-bound on a consumer GPU. The banded version in pure PyTorch is slower than dense because the mask-and-fill operations do not benefit from the sparse pattern (the kernel still walks the full {"[L, L]"} tensor). Production sliding-window needs a fused kernel — in flash-attn's sliding-window mode, it beats dense decisively at this scale.
      </Prose>

      <Prose>
        Peak memory tells the real story: the dense softmax peaks at 572 MB, the Performer at 370 MB — 35% reduction. Extrapolated to L=16384, dense would peak above 8 GB and Performer would stay around 1 GB, a 10x memory reduction — which matches the theoretical scaling above.
      </Prose>

      <H3>4.8 Sanity check: sliding window on a copy task</H3>

      <Prose>
        A copy task distinguishes "my attention works" from "my attention works on short-range dependencies only". We train two 2-layer window-attention LMs on a length-6 copy task (source, separator, target). Window {"w = 8"} covers the full source; window {"w = 2"} cannot see past the separator. The second should fail.
      </Prose>

      <CodeBlock language="python">
{`class TinyWindowAttnLM(nn.Module):
    def __init__(self, vocab=32, d=64, H=4, w=8, L_max=32, n_layers=2):
        super().__init__()
        self.H, self.d, self.w = H, d // H, w
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(L_max, d)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "qkv": nn.Linear(d, 3*d, bias=False),
                "o":   nn.Linear(d, d, bias=False),
                "ff":  nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d)),
                "ln1": nn.LayerNorm(d),
                "ln2": nn.LayerNorm(d),
            }) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def attn_window(self, x, qkv, w):
        B, L, D = x.shape
        QKV = qkv(x).view(B, L, 3, self.H, self.d).permute(2, 0, 3, 1, 4)
        Q, K, V = QKV[0], QKV[1], QKV[2]
        i = torch.arange(L, device=x.device).unsqueeze(1)
        j = torch.arange(L, device=x.device).unsqueeze(0)
        band = (j <= i) & (j >= i - w)
        s = torch.matmul(Q, K.transpose(-1,-2)) / math.sqrt(self.d)
        s = s.masked_fill(~band, float("-inf"))
        a = F.softmax(s, dim=-1)
        return torch.matmul(a, V).transpose(1,2).reshape(B, L, -1)

    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.tok(x) + self.pos(pos)
        for blk in self.blocks:
            h = h + blk["o"](self.attn_window(blk["ln1"](h), blk["qkv"], self.w))
            h = h + blk["ff"](blk["ln2"](h))
        return self.head(self.ln_f(h))

VOCAB, LC = 32, 6
def sample_copy(B, L=LC, V=VOCAB):
    SEP = V - 1
    src = torch.randint(0, V - 1, (B, L), device=device)
    sep = torch.full((B, 1), SEP, device=device, dtype=torch.long)
    return torch.cat([src, sep, src], dim=1)

for name, w in [("w=8 (covers src)", 8), ("w=2 (truncated)", 2)]:
    torch.manual_seed(0)
    model = TinyWindowAttnLM(vocab=VOCAB, d=64, H=4, w=w, L_max=2*LC+1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    for step in range(1, 1201):
        x = sample_copy(128)
        logits = model(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB),
                               x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        x = sample_copy(256)
        pred = model(x).argmax(-1)
        acc = (pred[:, LC:2*LC] == x[:, LC+1:2*LC+1]).float().mean().item()
    print(f"{name}: final loss {loss.item():.3f}, copy accuracy {acc:.3f}")

# Output:
#   w=8 (covers src): final loss 1.436, copy accuracy 1.000
#   w=2 (truncated): final loss 3.148, copy accuracy 0.027`}
      </CodeBlock>

      <Prose>
        The first model reaches 100% copy accuracy — proof that windowed attention of adequate size learns arbitrary permutations within the window. The second model's accuracy (2.7%) is chance on a 31-vocab uniform distribution, meaning the model has no way to transmit the source across the SEP token when the window is smaller than the SEP-to-source distance. This is the classic failure mode of sliding-window architectures: if the information has to travel further than {"w \\cdot n_{layers}"} positions, it cannot reach its target. Mistral's solution is a large window (4096) times many layers (32), giving an effective "receptive field" of {"4096 \\cdot 32 = 131072"} tokens.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION
          ====================================================================== */}
      <H2>5. Production</H2>

      <H3>5.1 Longformer in HuggingFace Transformers</H3>

      <Prose>
        The HuggingFace implementation (<Code>{"LongformerModel"}</Code>, <Code>{"LongformerForSequenceClassification"}</Code>, <Code>{"LongformerForQuestionAnswering"}</Code>) ships with a custom CUDA kernel for the windowed attention pattern — the vanilla PyTorch implementation falls back to a chunked loop that is slow at scale. Global tokens are specified via a {"global_attention_mask"} argument of the same shape as the input, with 1s at positions that should have full attention. The pre-trained checkpoints are {"allenai/longformer-base-4096"} (window 512, max length 4096) and {"allenai/longformer-large-4096"}. The typical use case is long-document classification (legal contracts, clinical notes, patents) where the 4096-token window gives a 10x range extension over vanilla BERT at comparable training cost.
      </Prose>

      <H3>5.2 BigBird in HuggingFace Transformers</H3>

      <Prose>
        The HuggingFace <Code>{"BigBirdModel"}</Code> implementation supports two modes: <Code>{"block_sparse"}</Code>, which uses the original random-plus-window-plus-global pattern, and <Code>{"original_full"}</Code>, which falls back to dense attention for short sequences. Block-sparse mode operates at a block granularity (blocks of 64 tokens) to make the random-attention pattern GPU-efficient. Typical config: window of 3 blocks (192 tokens) each side, 2 global blocks at the start, 3 random blocks per query block. Google's pre-trained checkpoints (<Code>{"google/bigbird-roberta-base"}</Code>, <Code>{"google/bigbird-roberta-large"}</Code>) support up to 4096 tokens. BigBird was briefly the state-of-the-art on long-document QA (TriviaQA, NaturalQuestions) before being overtaken by Longformer-style models with better pre-training data.
      </Prose>

      <H3>5.3 FlashAttention's sliding-window mode</H3>

      <Prose>
        Tri Dao's flash-attn library (v2.0+) has a sliding-window kernel that is the de facto production implementation of local attention. The API is <Code>{"flash_attn_func(q, k, v, window_size=(left, right))"}</Code> — left and right half-window sizes. The kernel fuses the window mask directly into the attention computation so no explicit mask tensor is ever created; memory stays {"O(L)"} regardless of {"L"}. This is what Mistral 7B ships with: a window of 4096 tokens, enforced entirely inside the FlashAttention-2 kernel. The same kernel handles standard causal attention as a special case ({"window_size = (L, 0)"}) and is what Mistral, Mixtral, Phi-3, and Gemma-2 all use.
      </Prose>

      <H3>5.4 Mistral 7B's effective-context design</H3>

      <Prose>
        Mistral 7B (October 2023) combined sliding-window attention with two tricks that made the windowed approach production-viable. First, a <em>rolling KV cache</em>: once the cache fills to window size, older entries are overwritten in a circular buffer, so memory per sequence is fixed at {"w"} tokens instead of growing with {"L"}. Second, <em>chunked prefill</em>: at prompt-processing time, the prompt is chunked into overlapping windows to amortise the sliding mask. The effective context across 32 layers is {"32 \\cdot 4096 = 131k"} tokens, though the theoretical information-flow per layer is limited to {"w = 4096"}. Mixtral 8x7B kept the same attention design. Mistral Large (2024) dropped sliding window in favor of full attention at 32k — a signal that for models that can afford it, exact attention is always preferred.
      </Prose>

      <H3>5.5 xFormers memory-efficient attention</H3>

      <Prose>
        Facebook's xFormers library (<Code>{"xformers.ops.memory_efficient_attention"}</Code>) supports arbitrary attention biases via its {"BlockDiagonalMask"}, {"LocalAttentionFromBottomRightMask"}, and custom bias tensor API. The library is often used for sparse-attention experiments where flash-attn's built-in window mode is insufficient — e.g., varying window sizes per head, or combining window with a few global tokens in a single kernel call. The underlying kernel is a fused memory-efficient attention similar to FlashAttention but with pluggable masks. Production use in Meta's image-generation stack.
      </Prose>

      <H3>5.6 Performer and Nystromformer implementations</H3>

      <Prose>
        For pure linear attention the reference implementations are <Code>{"performer-pytorch"}</Code> (by lucidrains) and <Code>{"nystromformer"}</Code> (HuggingFace). Both are installable via pip and wrap the feature-map computation in a drop-in MultiheadAttention module. These see essentially no production deployment in 2024-2026 — the quality gap vs FlashAttention-exact is consistent at 1-3% on language benchmarks, and the compute advantage vanishes for {"L \\le 32k"} on modern hardware. Their remaining use is in research comparisons and in very-long-context settings where an actual 100k+ window is needed without a state-space model.
      </Prose>

      <H3>5.7 Reformer: mostly abandoned</H3>

      <Prose>
        Reformer's HuggingFace implementation (<Code>{"ReformerModel"}</Code>) exists but is in minimal-maintenance mode. The LSH attention has poor interaction with modern flash-attention kernels (the bucket-sort step is memory-bandwidth-bound and does not fuse), and the reversible-residual-network trick that was supposed to save memory is made redundant by gradient checkpointing in any modern training stack. Academic follow-ups to Reformer (e.g., Sinkhorn Transformer, Routing Transformer) have similarly limited deployment.
      </Prose>

      <H3>5.8 The production reality in 2024-2026</H3>

      <Prose>
        For {"L \\le 32k"}: use FlashAttention-2 or FlashAttention-3 with full softmax. For sliding window within that: add {"window_size"} to the flash-attn call. For {"32k < L \\le 256k"}: use FlashAttention with careful context-extension tricks (YaRN, rope scaling) or an SSM hybrid (Mamba-attention interleaved). For {"L > 256k"}: SSMs or retrieval-augmented architectures. The sparse-and-linear zoo of 2019-2022 is now specialist equipment for specific long-document NLP tasks and a handful of academic benchmarks.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6.1 Attention patterns across the family</H3>

      <Prose>
        Binary attention masks visualised as heatmaps for {"L = 32"}. Gold = edge present. The dense baseline (not shown) is fully gold. Each variant chooses a subset of the {"L \\times L"} edges according to a different rule.
      </Prose>

      <Heatmap
        label="SLIDING WINDOW (w=3, causal) — LOCAL BAND ONLY"
        rowLabels={["q0","q1","q2","q3","q4","q5","q6","q7","q8","q9","q10","q11"]}
        colLabels={["k0","k1","k2","k3","k4","k5","k6","k7","k8","k9","k10","k11"]}
        matrix={[
          [1,0,0,0,0,0,0,0,0,0,0,0],
          [1,1,0,0,0,0,0,0,0,0,0,0],
          [1,1,1,0,0,0,0,0,0,0,0,0],
          [1,1,1,1,0,0,0,0,0,0,0,0],
          [0,1,1,1,1,0,0,0,0,0,0,0],
          [0,0,1,1,1,1,0,0,0,0,0,0],
          [0,0,0,1,1,1,1,0,0,0,0,0],
          [0,0,0,0,1,1,1,1,0,0,0,0],
          [0,0,0,0,0,1,1,1,1,0,0,0],
          [0,0,0,0,0,0,1,1,1,1,0,0],
          [0,0,0,0,0,0,0,1,1,1,1,0],
          [0,0,0,0,0,0,0,0,1,1,1,1],
        ]}
        colorScale="gold"
      />

      <Heatmap
        label="LONGFORMER (w=2 + global=[0,6]) — WINDOW PLUS HUBS"
        rowLabels={["q0","q1","q2","q3","q4","q5","q6","q7","q8","q9","q10","q11"]}
        colLabels={["k0","k1","k2","k3","k4","k5","k6","k7","k8","k9","k10","k11"]}
        matrix={[
          [1,0,0,0,0,0,0,0,0,0,0,0],
          [1,1,0,0,0,0,0,0,0,0,0,0],
          [1,1,1,0,0,0,0,0,0,0,0,0],
          [1,0,1,1,0,0,0,0,0,0,0,0],
          [1,0,0,1,1,0,0,0,0,0,0,0],
          [1,0,0,0,1,1,0,0,0,0,0,0],
          [1,1,1,1,1,1,1,0,0,0,0,0],
          [1,0,0,0,0,1,1,1,0,0,0,0],
          [1,0,0,0,0,0,1,1,1,0,0,0],
          [1,0,0,0,0,0,1,0,1,1,0,0],
          [1,0,0,0,0,0,1,0,0,1,1,0],
          [1,0,0,0,0,0,1,0,0,0,1,1],
        ]}
        colorScale="gold"
      />

      <Heatmap
        label="BIGBIRD (w=2 + global=[0] + random r=1) — WINDOW + HUB + RANDOM EDGES"
        rowLabels={["q0","q1","q2","q3","q4","q5","q6","q7","q8","q9","q10","q11"]}
        colLabels={["k0","k1","k2","k3","k4","k5","k6","k7","k8","k9","k10","k11"]}
        matrix={[
          [1,0,0,0,0,0,0,0,0,0,0,0],
          [1,1,0,0,0,0,0,0,0,0,0,0],
          [1,1,1,0,0,0,0,0,0,0,0,0],
          [1,0,1,1,0,0,0,0,1,0,0,0],
          [1,0,0,1,1,0,1,0,0,0,0,0],
          [1,0,1,0,1,1,0,0,0,0,0,0],
          [1,0,0,0,0,1,1,0,0,0,0,0],
          [1,0,0,0,0,0,1,1,0,0,1,0],
          [1,0,0,0,1,0,0,1,1,0,0,0],
          [1,0,0,0,0,0,0,0,1,1,0,0],
          [1,0,0,1,0,0,0,0,0,1,1,0],
          [1,0,0,0,0,1,0,0,0,0,1,1],
        ]}
        colorScale="gold"
      />

      <Heatmap
        label="SPARSE TRANSFORMER (strided, k=3, causal) — LOCAL + REGULAR STRIDE"
        rowLabels={["q0","q1","q2","q3","q4","q5","q6","q7","q8","q9","q10","q11"]}
        colLabels={["k0","k1","k2","k3","k4","k5","k6","k7","k8","k9","k10","k11"]}
        matrix={[
          [1,0,0,0,0,0,0,0,0,0,0,0],
          [1,1,0,0,0,0,0,0,0,0,0,0],
          [1,1,1,0,0,0,0,0,0,0,0,0],
          [1,0,0,1,0,0,0,0,0,0,0,0],
          [0,1,0,1,1,0,0,0,0,0,0,0],
          [0,0,1,1,1,1,0,0,0,0,0,0],
          [1,0,0,1,0,0,1,0,0,0,0,0],
          [0,1,0,1,0,0,1,1,0,0,0,0],
          [0,0,1,1,0,0,1,1,1,0,0,0],
          [1,0,0,1,0,0,1,0,0,1,0,0],
          [0,1,0,1,0,0,1,0,0,1,1,0],
          [0,0,1,1,0,0,1,0,0,1,1,1],
        ]}
        colorScale="gold"
      />

      <Prose>
        Each pattern has a different structural bias. Sliding-window says "nearby tokens are what matter." Longformer adds "some tokens are hubs." BigBird adds "random shortcuts ensure connectivity." Sparse Transformer says "local context plus regular long-range samples." BigBird's universal-approximation proof rests on the combination; no single pattern alone is expressive enough at very sparse densities.
      </Prose>

      <H3>6.2 Memory scaling: dense vs window vs low-rank vs linear</H3>

      <Prose>
        Peak attention-score memory (MB, fp32) as a function of sequence length, for {"H = 8, d = 64, w = 128, k = 128, m = 256"}. The dense curve is quadratic; the others are linear. The separation becomes unignorable past {"L = 4096"} — at {"L = 16384"} the dense memory is 128x the window, which is what forced long-context research to find alternatives before FlashAttention existed.
      </Prose>

      <Plot
        label="PEAK ATTENTION MEMORY VS SEQUENCE LENGTH (MB, fp32)"
        xLabel="sequence length L"
        yLabel="score tensor memory (MB)"
        width={580}
        height={300}
        series={[
          { name: "Dense O(L^2)",    color: "#f87171", points: [[512, 8.39], [1024, 33.55], [2048, 134.22], [4096, 536.87], [8192, 2147.48], [16384, 8589.93]] },
          { name: "Window w=128",    color: "#60a5fa", points: [[512, 2.10], [1024, 4.19],  [2048, 8.39],   [4096, 16.78],  [8192, 33.55],   [16384, 67.11]] },
          { name: "Linformer k=128", color: "#e2b55a", points: [[512, 2.10], [1024, 4.19],  [2048, 8.39],   [4096, 16.78],  [8192, 33.55],   [16384, 67.11]] },
          { name: "Performer m=256", color: "#c084fc", points: [[512, 4.72], [1024, 8.91],  [2048, 17.30],  [4096, 34.08],  [8192, 67.63],   [16384, 134.74]] },
        ]}
      />

      <H3>6.3 Linformer down-projection step-by-step</H3>

      <StepTrace
        label="LINFORMER FORWARD — LOW-RANK SEQUENCE COMPRESSION"
        steps={[
          {
            label: "1. Compute Q, K, V as usual",
            render: () => (
              <div>
                <Prose>
                  Input {"x \\in \\mathbb{R}^{L \\times d}"}, standard projections {"Q = x W_Q, K = x W_K, V = x W_V"}. Each is {"[L, d]"}.
                </Prose>
                <div style={{ fontFamily: "monospace", fontSize: 12, color: colors.gold, marginTop: 8 }}>
                  {"x [L=4096, d=512] --(W_Q,W_K,W_V)--> Q,K,V [L=4096, d=512]"}
                </div>
              </div>
            ),
          },
          {
            label: "2. Down-project K and V along the sequence axis",
            render: () => (
              <div>
                <Prose>
                  Learned matrices {"E, F \\in \\mathbb{R}^{k \\times L}"} mix the {"L"} keys/values into {"k"} "summary" keys/values. {"K' = E K, V' = F V"}, both {"[k, d]"}. This is the core linear-complexity move: the {"L \\to k"} reduction is a fixed linear map that does not depend on the content.
                </Prose>
                <div style={{ fontFamily: "monospace", fontSize: 12, color: colors.gold, marginTop: 8 }}>
                  {"K [L=4096, d] --(E [k=128, L=4096])--> K' [k=128, d=512]"}<br />
                  {"V [L=4096, d] --(F [k=128, L=4096])--> V' [k=128, d=512]"}
                </div>
              </div>
            ),
          },
          {
            label: "3. Attention against compressed K', V'",
            render: () => (
              <div>
                <Prose>
                  {"Q (K')^T"} is {"[L, k]"} — linear in {"L"}. Softmax over {"k"} summary keys. Each query still has full {"[L, d]"} output — we have only compressed the attended representation, not the queries.
                </Prose>
                <div style={{ fontFamily: "monospace", fontSize: 12, color: colors.gold, marginTop: 8 }}>
                  {"scores = Q @ K'.T  => [L=4096, k=128]"}<br />
                  {"attn   = softmax(scores)"}<br />
                  {"out    = attn @ V'  => [L=4096, d=512]"}
                </div>
              </div>
            ),
          },
          {
            label: "4. The catch: E, F mix the whole sequence",
            render: () => (
              <div>
                <Prose>
                  Because each summary key in {"K'"} is a weighted sum over <em>all</em> positions in {"K"}, including positions that are "in the future" for a causal decoder, Linformer does not support autoregressive generation in its default form. Encoder-only use is fine; decoder use requires lower-triangular {"E, F"} or a workaround.
                </Prose>
                <div style={{ fontFamily: "monospace", fontSize: 12, color: colors.gold, marginTop: 8 }}>
                  {"K'_i = sum_j E_{i,j} K_j   (j ranges over ALL positions)"}<br />
                  {"=> breaks causality in decoder"}
                </div>
              </div>
            ),
          },
        ]}
      />

      <H3>6.4 Sparse density vs task quality (qualitative)</H3>

      <Prose>
        Approximate composite curve from Child et al. 2019 and BigBird 2020 ablations: BLEU/accuracy on long-document language modelling vs fraction of attention edges active. Dense softmax = 1.0. Quality degrades gracefully until roughly 2-5% density, then falls off a cliff. Sparse Transformer's "fixed" pattern at {"L = 12288"} used 4% density and achieved {"\\sim -0.3"} ppl vs dense; BigBird at 1% density lost {"\\sim -0.8"} ppl vs dense. Mistral 7B's 4096 window at 32k context is roughly 25% density — comfortably above the cliff.
      </Prose>

      <Plot
        label="RELATIVE QUALITY VS ATTENTION DENSITY (QUALITATIVE COMPOSITE)"
        xLabel="attention density (fraction of edges)"
        yLabel="quality (rel to dense)"
        width={580}
        height={280}
        series={[
          { name: "Sparse Transformer",  color: "#e2b55a", points: [[0.005, 0.82], [0.01, 0.90], [0.02, 0.95], [0.05, 0.98], [0.10, 0.99], [0.25, 1.00], [1.00, 1.00]] },
          { name: "BigBird",             color: "#60a5fa", points: [[0.005, 0.85], [0.01, 0.92], [0.02, 0.96], [0.05, 0.98], [0.10, 0.99], [0.25, 1.00], [1.00, 1.00]] },
          { name: "Linformer",           color: "#c084fc", points: [[0.005, 0.70], [0.01, 0.80], [0.02, 0.87], [0.05, 0.93], [0.10, 0.96], [0.25, 0.98], [1.00, 1.00]] },
          { name: "Performer FAVOR+",    color: "#f87171", points: [[0.005, 0.75], [0.01, 0.85], [0.02, 0.90], [0.05, 0.94], [0.10, 0.96], [0.25, 0.98], [1.00, 1.00]] },
        ]}
      />

      <Prose>
        Sparse patterns (Sparse Transformer, BigBird) degrade more gracefully than low-rank/kernel methods because they preserve individual edges that dense attention would have computed exactly; low-rank methods smooth every edge through a compressed bottleneck and cannot recover fine-grained attention patterns at all densities.
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <H3>7.1 Long document classification / QA (up to 16k tokens)</H3>

      <Prose>
        Use Longformer or BigBird from HuggingFace. These are the mature options for the "feed a PDF and ask a question" workflow: pretrained checkpoints exist, the attention patterns are well-understood, and the {"[CLS]"}-as-global-token design integrates cleanly with downstream classification heads. For new projects in 2026, prefer long-context decoder models (Llama 3.1 at 128k, Claude 3, Gemini 1.5 Pro) fine-tuned via prompting — the long-document encoder regime has been eaten by long-context LLMs. Longformer and BigBird remain the right answer only if you must train a small specialised model and can't afford inference on a 70B decoder.
      </Prose>

      <H3>7.2 General long-context LLM (up to 128k tokens)</H3>

      <Prose>
        Use FlashAttention-2 with a sliding window (Mistral-style) or full attention. Sliding window with {"w = 4096"} is sufficient for any task where information flows locally and the per-layer receptive field is enough over stacked layers. Full attention via FlashAttention is faster and more accurate at this scale if you have the memory for the KV cache, which is the actual bottleneck. The sparse-attention variants add no value here — FlashAttention-exact is already linear in activation memory and faster than approximate alternatives.
      </Prose>

      <H3>7.3 Research on linear-complexity attention</H3>

      <Prose>
        Use Performer or implement FAVOR+ from scratch. Performers remain the cleanest instance of the kernel-factorisation trick and the right baseline for any new linear-attention proposal. The implementation cost is modest, the theory is clean, and the variance-vs-{"m"} behaviour is well-documented. Most modern linear-attention research (including RetNet, GLA, and early Mamba) can be derived as specialised feature maps inside the Performer framework.
      </Prose>

      <H3>7.4 Retrieval-augmented workflows</H3>

      <Prose>
        Do not use long-context attention at all. A 4-8k context model with a good retriever (BM25+cross-encoder rerank, or dense retrieval via Contriever) beats a 128k context attention model on most practical retrieval-then-answer tasks, at a fraction of the cost. The sparse/linear attention family does not help here — the bottleneck is in document selection, not attention expressivity.
      </Prose>

      <H3>7.5 Very long context (&gt; 1M tokens)</H3>

      <Prose>
        Use state-space models (Mamba, Mamba-2, Jamba) or hierarchical attention (Hyena, MoR-like architectures). At 1M+ tokens, even sliding-window attention becomes prohibitive: the per-layer {"O(L w)"} is fine, but the {"O(L)"} activation and state-propagation memory grows uncomfortably. SSMs have an inherent advantage because their state is fixed-size at inference ({"O(d^2)"} for Mamba) regardless of {"L"}. This is the regime where the ideas of linear attention (fixed-size state, parallel-train recurrent-infer) fully paid off — but in a cleaner form that did not require the FAVOR+ kernel trick.
      </Prose>

      <H3>7.6 Training on a single GPU with tight memory</H3>

      <Prose>
        Use FlashAttention and a small sequence length. The tricks the sparse/linear variants sell at 1B-scale become redundant at small scale: you can almost always shrink {"L"} to fit, and the quality gain from exact attention on shorter sequences beats the quality loss from approximate attention on longer sequences for most tasks. The one exception is if the task fundamentally requires long context (arXiv summarisation, code completion on large files) — then Longformer is the pragmatic choice.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES
          ====================================================================== */}
      <H2>8. What scales</H2>

      <H3>8.1 Longformer and BigBird top out around 16k</H3>

      <Prose>
        Both were designed for the 4k-16k regime and their pretrained checkpoints are all at 4096 tokens. Scaling them to 32k+ requires retraining the positional embeddings and often the window attention kernel as well. Practically, nobody has pushed a pretrained Longformer past 16k — past that point, long-context decoder LMs are uniformly better, and training from scratch is the only option. The ceiling is architectural as much as compute: the absolute positional embedding in the original Longformer does not extrapolate, and RoPE-adapted versions exist in forks but are not the mainline release.
      </Prose>

      <H3>8.2 Linformer can reach 64k but quality drops</H3>

      <Prose>
        With {"k = 256"}, Linformer runs at 64k tokens without issue — the score tensor is {"[64k, 256]"}, or 64 MB per head in fp32. Quality, however, tends to drop by 1-3 points on language modelling as {"L"} grows because the fixed-size {"k = 256"} compression cannot preserve enough information about the full {"L = 64k"} sequence. The effective rank of the sequence-axis structure scales with the diversity of the content, not with a fixed {"k"}. For extremely long sequences with heterogeneous content, Linformer's fixed compression underfits.
      </Prose>

      <H3>8.3 FlashAttention negates most of the need for approximation at L ≤ 32k</H3>

      <Prose>
        The biggest contribution FlashAttention made to this space is obsoleting it. Pre-FlashAttention, dense attention at {"L = 32k"} was infeasible on a single 80 GB GPU — 32k² × 4 bytes × 32 heads = 128 GB for the score tensor. Post-FlashAttention-2, the same workload uses about 4 GB of peak memory (the tiled version never materialises the {"[L, L]"} tensor) and runs 2-3x faster than a naive sparse alternative. At {"L \\le 32k"} there is no memory argument for sparse or linear attention. Above {"L = 32k"} the argument returns — FlashAttention still scales {"O(L^2)"} in FLOPs — but by then one is usually in SSM territory.
      </Prose>

      <H3>8.4 Sliding window efficient but needs "reach" across layers</H3>

      <Prose>
        Sliding-window attention's effective receptive field is {"w \\cdot n_{layers}"} tokens — a 4096 window over 32 layers gives a theoretical 131k-token receptive field. In practice, the effective receptive field is smaller because each layer's attention is not a full convolution — the information bottleneck through the residual stream compresses the "long reach" heavily. Empirical studies show that sliding-window models with {"w \\ll L"} often struggle on tasks that require precise long-range retrieval (needle-in-a-haystack at 100k tokens), even when the theoretical receptive field covers the needle. Global tokens or full attention at certain layers mitigate this.
      </Prose>

      <H3>8.5 Linear attention models (2024) replaced pure linear variants</H3>

      <Prose>
        RWKV (Peng et al. 2023), Mamba (Gu & Dao 2023), RetNet (Sun et al. 2023), and GLA (Yang et al. 2023) are all variants of the kernel-factorisation idea from Katharopoulos 2020, but with better-designed state update rules. RWKV uses a time-mixing receptance-weighted update; Mamba uses an input-selective SSM; GLA uses a gated linear attention. All scale to 1M+ tokens, all outperform Performer/Linformer on every task measured. The pure linear-attention family from 2020-2021 has been superseded by these in both production and frontier research. If you need true linear attention at scale in 2026, you should start from Mamba-2 or an SSM, not from FAVOR+.
      </Prose>

      <H3>8.6 Hybrid architectures dominate frontier long-context</H3>

      <Prose>
        Jamba (AI21, 2024), Zamba (Zyphra, 2024), and Mamba-Transformer hybrids use a layer mix of (a) FlashAttention for short-range precision and (b) SSM or sliding-window attention for long-range scaling. This is the architectural lesson of the sparse-and-linear era: no single approximation replaces full attention, but a hybrid that uses exact attention on a subset of layers and linear/SSM on the rest captures both precision and scalability. Mistral's "full + sliding window" mix in Mistral Large and the Mamba-attention interleaving in Jamba are contemporary instances of this principle.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES
          ====================================================================== */}
      <H2>9. Failure modes</H2>

      <H3>9.1 Sliding window missing global information</H3>

      <Prose>
        The canonical failure: in a sliding-window model, tokens separated by more than {"w \\cdot n_{layers}"} positions cannot influence each other. In practice the effective reach is much less because each layer's attention compresses the full-reach signal. On needle-in-a-haystack tests at {"L = 128k"} with {"w = 4096, n_{layers} = 32"}, Mistral 7B's accuracy drops from 95% (needle at position 10k) to 35% (needle at position 80k). Fix: interleave global-attention layers, add a handful of global tokens (Longformer-style), or use a hybrid architecture with SSM layers that pass long-range state.
      </Prose>

      <H3>9.2 Linformer projection too small for the task</H3>

      <Prose>
        Linformer's {"k"} is a fixed bottleneck. At {"k = 128"} and {"L = 1024"} it works; at {"k = 128"} and {"L = 32768"} it underfits because a single 128-dim summary cannot preserve the content of a 32k-token document. The paper's ablations suggest {"k \\propto \\sqrt{L}"} in the worst case, but in practice {"k"} is fixed at construction time — scaling {"L"} later means retraining {"E, F"}. Fix: start with {"k = 256-512"} and budget the cost; if that still underfits, the task is not low-rank in the sequence axis and Linformer is wrong for it.
      </Prose>

      <H3>9.3 Performer random-feature variance with too few features</H3>

      <Prose>
        At {"m = 64"}, FAVOR+ approximations have variance large enough that the attention distribution is visibly noisy — softmax peaks shift, attention entropy increases, and training loss oscillates. The paper recommends {"m \\ge 256"} and orthogonal (rather than IID Gaussian) rows of {"W"} to cut variance by {"\\sim d"}. Fix: always use at least 256 features and always orthogonalise; re-sample {"W"} periodically (every few layers or steps) if variance drift appears. Our benchmark above shows rel-L2 error of 0.74 at {"m = 64"} vs 0.49 at {"m = 1024"} — still substantial, because FAVOR+ on a hard (sharply peaked) softmax distribution has an inherent variance floor.
      </Prose>

      <H3>9.4 LSH bucket collisions underestimated</H3>

      <Prose>
        Reformer's expected {"O(L \\log L)"} cost assumes balanced buckets. In practice, the distribution of hash bucket sizes is heavy-tailed: a few large buckets dominate the wall-clock, and the {"L^2"} within-bucket attention in a large bucket dominates the entire layer. Reformer's paper recommends multi-round LSH (running the hash 4-8 times with different random projections and unioning) to reduce variance. Even so, on inputs with many similar keys (e.g., repeated boilerplate), buckets can collapse to size {"L"} and destroy the asymptotic advantage. Fix: use LSH only when the key distribution is known to be diverse; multi-round; sort-and-chunk with overflow handling.
      </Prose>

      <H3>9.5 Sparse pattern incompatible with FlashAttention</H3>

      <Prose>
        FlashAttention's kernel fuses the softmax tile-by-tile and cannot handle arbitrary sparsity patterns — it assumes either full dense, causal, or a single contiguous sliding window. BigBird's random edges, Sparse Transformer's strided pattern, and Reformer's LSH all fall outside this envelope. Implementing them with FlashAttention-level efficiency requires writing a new CUDA kernel per pattern. This is why sliding window survived and the others did not: it is the only sparse pattern that fuses into the dominant attention kernel. Fix: either restrict to sliding window (fast) or accept a 2-5x slowdown from non-fused sparse attention.
      </Prose>

      <H3>9.6 Forgetting padding masks in window attention</H3>

      <Prose>
        A subtle implementation bug: when you implement a banded sliding-window mask manually, the band includes padding positions for variable-length sequences in a batch. If you do not explicitly AND the band with the padding mask, attention weights leak onto PAD tokens, which at best adds noise and at worst corrupts the attention distribution in ways that do not show up in training loss but degrade downstream accuracy. Always validate: sum attention weights over valid positions should be 1.0 to numerical precision; sum over padding should be 0.0 exactly. FlashAttention-2's {"varlen"} API handles this automatically with a cu_seqlens argument.
      </Prose>

      <H3>9.7 Linformer losing causality in decoders</H3>

      <Prose>
        The default Linformer projection {"E K"} mixes future keys into each summary key, so a decoder using it attends to the future — a silent bug that produces spuriously good training loss and catastrophic generation. Fix: use a lower-triangular {"E"} (zero out {"E_{i, j}"} for {"j > i"} — wait, that's the wrong direction) — the correct fix is per-prefix projection, which destroys the linear complexity. There is no clean causal Linformer; this is a known limitation and the reason it shipped only as an encoder architecture.
      </Prose>

      <H3>9.8 Performer numerical stability at high temperature</H3>

      <Prose>
        The FAVOR+ feature map {"\\exp(Wx - \\|x\\|^2 / 2)"} can produce enormous values when {"\\|x\\|"} is large — up to {"\\exp(\\|W\\| \\cdot \\|x\\|)"}. In practice you must normalise {"Q"} and {"K"} to a stable magnitude (typically {"\\|q\\| \\le O(1)"}) before applying {"\\phi"}, or the exponential overflows in fp16. The standard recipe is to scale {"Q, K"} by {"1 / d^{1/4}"} — this brings the exponent into a safe range, and it also matches the {"1/\\sqrt{d}"} scaling inside standard softmax attention. Fix: scale {"Q, K"} by {"1/d^{1/4}"} before FAVOR+, or clamp before exp.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        <strong>Child, Gray, Radford, Sutskever (OpenAI, 2019).</strong> "Generating Long Sequences with Sparse Transformers." arXiv:1904.10509. The first paper in the sparse-attention family. Introduces strided and fixed attention patterns and trains on sequences up to 12288 tokens (images at {"128 \\times 128"} and raw audio). The architectural ideas — factorising attention across two or three sparse patterns alternated per layer, using a "summary" token per block — are the ancestor of every subsequent sparse-attention design. Also introduces sparsity-aware CUDA kernels, a detail that became a bottleneck for later methods without equivalent engineering.
      </Prose>

      <Prose>
        <strong>Beltagy, Peters, Cohan (Allen AI, 2020).</strong> "Longformer: The Long-Document Transformer." arXiv:2004.05150. The production-ready variant of sparse attention. Window + global with task-specific global-token selection. Pretrained from a RoBERTa initialisation and fine-tuned for long-document tasks; became the default "feed a paper to a model" architecture for three years. The paper's Table 2 compares Longformer with full, strided, and dilated attentions — window-plus-global outperforms all pure-sparse alternatives and was the design that actually shipped.
      </Prose>

      <Prose>
        <strong>Zaheer, Guruganesh, Dubey, Ainslie, Alberti, Ontanon, Pham, Ravula, Wang, Yang, Ahmed (Google Research, 2020).</strong> "Big Bird: Transformers for Longer Sequences." NeurIPS 2020. arXiv:2007.14062. The universal-approximation theorem for sparse attention. Proves that window + global + random with {"O(\\log L)"} edges per node is a universal approximator of full attention. Empirical results on long-document NLP (TriviaQA, WikiHop, HotpotQA) set a state of the art at the time. The paper's theoretical contribution is the most substantive in the sparse-attention line.
      </Prose>

      <Prose>
        <strong>Kitaev, Kaiser, Levskaya (Google, 2020).</strong> "Reformer: The Efficient Transformer." ICLR 2020. arXiv:2001.04451. LSH attention, reversible residuals, and chunked feed-forward layers — a comprehensive "how do we fit a 64k-token sequence on one GPU" paper. The LSH attention mechanism is the cleanest mathematical formulation of hash-based sparsity and remains a useful baseline for data-adaptive sparse methods. The reversible-residual trick it introduced is still used in some modern architectures to save activation memory.
      </Prose>

      <Prose>
        <strong>Wang, Li, Khabsa, Fang, Ma (Facebook AI, 2020).</strong> "Linformer: Self-Attention with Linear Complexity." arXiv:2006.04768. The low-rank sequence-axis compression approach. Proves that {"K, V"} are approximately low-rank along the sequence axis under mild assumptions, justifying a fixed learned projection {"E, F \\in \\mathbb{R}^{k \\times L}"}. The paper's main technical weakness — incompatibility with causal decoding — limits its practical use, but the low-rank insight was influential for later work on compressed attention.
      </Prose>

      <Prose>
        <strong>Katharopoulos, Vyas, Pappas, Fleuret (EPFL, 2020).</strong> "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention." ICML 2020. arXiv:2006.16236. The kernel-factorisation insight. Shows that attention with a replaceable similarity kernel {"\\phi(q) \\cdot \\phi(k)"} admits a constant-size recurrent state, giving {"O(L)"} inference. The paper's {"\\phi(x) = \\mathrm{elu}(x) + 1"} choice is simple but suboptimal; the bigger contribution is the structural observation that linear attention is a linear RNN, which seeded the entire SSM and linear-RNN line that followed.
      </Prose>

      <Prose>
        <strong>Choromanski, Likhosherstov, Dohan, Song, Gane, Sarlos, Hawkins, Davis, Mohiuddin, Kaiser, Belanger, Colwell, Weller (Google / DeepMind, 2021).</strong> "Rethinking Attention with Performers." ICLR 2021. arXiv:2009.14794. The principled random-feature approximation to softmax attention. Introduces FAVOR+ (Fast Attention Via positive Orthogonal Random features), proves unbiased approximation of {"\\exp(q \\cdot k)"} with variance {"O(1/m)"}, and shows orthogonal features reduce variance by another factor of {"d"}. The paper's theoretical framework is the cleanest existing treatment of kernel-based linear attention and remains the baseline for any new linear-attention proposal.
      </Prose>

      <Prose>
        <strong>Roy, Saffar, Vaswani, Grangier (Google, 2021).</strong> "Efficient Content-Based Sparse Attention with Routing Transformers." TACL 2021. arXiv:2003.05997. Extends Reformer's LSH routing by replacing the random hash with learned {"k"}-means clustering. Each token is routed to its nearest cluster centroid and attention runs within clusters. More sample-efficient than pure LSH but more complex to implement; never achieved wide adoption due to the same fusion-with-CUDA-kernels problem that doomed Reformer.
      </Prose>

      <Prose>
        <strong>Jiang, Sablayrolles, Mensch, Bamford, Chaplot, de Las Casas, Bressand, Lengyel, Lample, Saulnier, et al. (Mistral AI, 2023).</strong> "Mistral 7B." arXiv:2310.06825. The production model that made sliding-window attention a mainstream design choice. Combines {"w = 4096"} sliding-window attention with a rolling KV cache, grouped-query attention, and a well-tuned training recipe. Demonstrates that sliding window scales to 32k context on a single A100 and is the simplest workable efficient-attention scheme in practice. The architectural choices influenced Mixtral, Phi-3, Gemma-2, and most subsequent long-context open models.
      </Prose>

      <Prose>
        <strong>Dao, Fu, Ermon, Rudra, Ré (Stanford, 2022).</strong> "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." arXiv:2205.14135. Not a sparse/linear method — but the paper that made most of the sparse/linear methods redundant. By reorganising exact softmax attention to tile over the sequence axis and never materialise the {"[L, L]"} tensor, FlashAttention achieved the memory savings that sparse/linear methods promised while keeping the quality of exact attention. Later versions (FlashAttention-2, FlashAttention-3) added sliding-window support, which is what Mistral and friends actually use.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK
          ====================================================================== */}
      <H2>11. Self-check</H2>

      <H3>11.1 Why does Longformer add global tokens on top of a sliding window?</H3>

      <Prose>
        Because a pure sliding window has finite "reach": information can only propagate {"w"} tokens per layer, so tokens separated by more than {"w \\cdot n_{layers}"} positions cannot influence each other, and in practice the effective reach is much smaller. Global tokens act as hubs in a small-world attention graph: any two positions can communicate in two hops through a global token. With {"|\\mathcal{G}|"} global tokens and window {"w"}, total edges are {"O(L(w + |\\mathcal{G}|))"}, still linear in {"L"}, and the diameter of the attention graph drops from {"L/w"} (pure window) to 2 (window + global). Longformer typically uses global tokens at task-critical positions: {"[CLS]"} for classification, the question tokens for QA.
      </Prose>

      <H3>11.2 How does the kernel trick turn softmax attention into linear attention, and what does it give up?</H3>

      <Prose>
        Standard softmax attention has the form {"y_t = \\sum_s \\frac{\\exp(q_t \\cdot k_s)}{\\sum_{s'} \\exp(q_t \\cdot k_{s'})} v_s"}, which cannot be factored because {"\\exp(q \\cdot k)"} is an inner product inside an exponential. If we replace it with {"\\phi(q) \\cdot \\phi(k)"} for some feature map {"\\phi"}, the numerator factors: {"\\sum_s \\phi(q_t) \\cdot \\phi(k_s) v_s = \\phi(q_t)^T \\sum_s \\phi(k_s) v_s^T"}. The inner sum is a fixed-size {"[m, d]"} matrix, computable in {"O(L m d)"}. What we give up is the sharpness of softmax: {"\\exp(\\cdot)"} produces attention distributions that can be very peaked (one key dominating), whereas {"\\phi(q) \\cdot \\phi(k)"} for simple {"\\phi"} is smoother and puts weight on more keys. Performers use {"\\phi"} that provably converges to {"\\exp"} as {"m \\to \\infty"}, recovering softmax exactly in the limit; at finite {"m"} there is always some variance.
      </Prose>

      <H3>11.3 Why does Linformer not work for causal decoders?</H3>

      <Prose>
        Linformer compresses {"K"} to {"K' = E K"} where {"E \\in \\mathbb{R}^{k \\times L}"} is a learned dense matrix. Each row of {"K'"} is a weighted sum over all rows of {"K"}, including rows at positions that are "in the future" for a causal model. When a query at position {"t"} attends to {"K'"}, it is effectively attending to a mixture that includes future keys — which violates the causal constraint silently (no error, just wrong gradients and a model that cannot autoregressively generate). A lower-triangular {"E"} would fix it but would need a new {"E"} per prefix length, destroying the {"O(L k)"} complexity. There is no clean causal Linformer; the method is usable only in encoder settings.
      </Prose>

      <H3>11.4 Given a sliding-window attention model with w = 1024 and 24 layers, what is the theoretical maximum distance between two tokens that can influence each other?</H3>

      <Prose>
        {"w \\cdot n_{layers} = 1024 \\cdot 24 = 24576"} tokens. After one attention layer, each token sees a window of 1024 around it. After two layers, each token sees a window of 2048 (via tokens that already aggregated 1024 in the first layer), and so on. In practice the effective reach is much smaller because each layer's residual stream must carry the long-range signal, and the attention is not a full convolution — it aggregates weighted information, not all-of-it. Empirical long-context benchmarks usually show sliding-window models with this setup effectively reaching 5-10k tokens, not 24k. Global tokens or periodic full-attention layers are required to extend the practical reach.
      </Prose>

      <H3>11.5 Rank the following by quality-at-fixed-compute-budget on a modern L ≤ 32k language modelling task, best to worst: FlashAttention exact softmax, Performer FAVOR+, Linformer, sliding-window FlashAttention, Longformer-style window+global.</H3>

      <Prose>
        Best to worst: FlashAttention exact softmax, sliding-window FlashAttention, Longformer-style window+global, Performer FAVOR+, Linformer. Exact FlashAttention wins because it is exact and well-engineered; sliding-window is nearly as good for most language tasks with a sufficient window and costs less at very long {"L"}. Longformer-style with global tokens beats pure window on tasks that need long-distance information routing (QA, classification), but underperforms exact attention at the same compute budget because the global-token trick adds parameters. Performer is quality-limited by the random-feature variance and typically 1-3% behind on language modelling. Linformer is last because the fixed {"k"}-dim bottleneck and the causal-decoder incompatibility both hurt autoregressive tasks; on encoder-only tasks it is competitive with Performer but still behind sliding-window variants.
      </Prose>

    </div>
  ),
};

export default sparseLinearAttentionContent;
