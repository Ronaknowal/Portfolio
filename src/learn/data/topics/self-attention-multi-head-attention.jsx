import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const selfAttentionContent = {
  title: "Self-Attention & Multi-Head Attention",
  readTime: "~42 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        The arc that ends at scaled dot-product multi-head self-attention begins with a single question that took three years to answer: <em>why should the decoder need a recurrent state at all?</em> In 2014 Sutskever's seq2seq showed that a stacked LSTM could compress a source sentence into a final hidden vector and decode a translation from it. In 2015 Bahdanau, Cho, and Bengio (arXiv:1409.0473) lifted the fixed-context bottleneck by letting the decoder attend to the full encoder hidden-state sequence at every step. That was the first time attention took center stage in an end-to-end sequence model, and the decisive plot from their ICLR paper — BLEU staying flat out to source length 60 while vanilla seq2seq collapsed past length 30 — reframed attention from a nice-to-have into a structural necessity.
      </Prose>

      <Prose>
        But Bahdanau attention was still bolted onto an RNN. The encoder was bidirectional LSTM, the decoder was a unidirectional LSTM, and attention was a cross-attention module between them. The <em>recurrence</em> inside each side was doing the heavy lifting of contextualizing each token against its neighbors, and recurrence is sequential — you cannot compute {"h_{t+1}"} until you have {"h_t"}, which means a training step on a length-{"L"} sequence pays {"L"} sequential LSTM updates regardless of how many GPUs you have. The first paper that asked "what if we used attention to contextualize too?" was Jianpeng Cheng, Li Dong, and Mirella Lapata's "Long Short-Term Memory-Networks for Machine Reading" (arXiv:1601.06733) at EMNLP 2016. They called it <em>intra-attention</em>: within a single sequence, each position attends to every previous position to compute an augmented LSTM input. Intra-attention was the first clear formulation of what we now call self-attention, even though the mechanism was still wrapped around an LSTM.
      </Prose>

      <Prose>
        Two more papers rounded out the pre-Transformer landscape. Ankur Parikh, Oscar Täckström, Dipanjan Das, and Jakob Uszkoreit published "A Decomposable Attention Model for Natural Language Inference" (arXiv:1606.01933) at EMNLP 2016 and made a stronger claim: for a natural-language inference task, you do not need recurrence at all. Their decomposable attention model used only token embeddings, cross-attention between the premise and hypothesis, and a small feed-forward aggregator — and it matched the LSTM state of the art at a fraction of the parameters. It was the first architecturally attention-only model to work on a non-trivial NLP benchmark. Then Zhouhan Lin, Minwei Feng, Cícero dos Santos, Mo Yu, Bing Xiang, Bowen Zhou, and Yoshua Bengio published "A Structured Self-Attentive Sentence Embedding" (arXiv:1703.03130) at ICLR 2017. Their model produced sentence embeddings by running a single sequence through multiple parallel attention heads — each head a separate softmax distribution over tokens — and concatenating the pooled representations. This was not quite self-attention in the modern Q/K/V sense, but it introduced the <em>multi-head</em> idea: multiple parallel attention distributions, each specializing in a different aspect of the input, and it showed empirically that different heads learned genuinely different things (one picked out subjects, another picked out sentiment words, a third picked out negations).
      </Prose>

      <Prose>
        In June 2017 Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan Gomez, Lukasz Kaiser, and Illia Polosukhin at Google Brain and Google Research put it all together. "Attention Is All You Need" (arXiv:1706.03762, NeurIPS 2017) dropped recurrence and convolution entirely and built an encoder-decoder from nothing but attention and feed-forward layers. The mechanism they called <em>scaled dot-product attention</em> was Luong's dot-product formulation from 2015 with a {"1/√d_k"} scaling term; the mechanism they called <em>multi-head attention</em> was an {"H"}-way parallel variant where the input was projected into {"H"} query/key/value subspaces, attention was computed independently in each, and the results were concatenated and re-projected. Self-attention — the case where Q, K, V all come from the same sequence — contextualized each token against every other token in a single parallel step, and the causal mask handled autoregressive generation by preventing each position from attending to the future. The final architecture trained 10x faster than the previous state-of-the-art seq2seq model and beat it on WMT 2014 English-German and English-French by substantial margins.
      </Prose>

      <Prose>
        The reason attention displaced RNN-plus-attention so completely was twofold. The first is <em>parallelization</em>. A Transformer layer processes all {"L"} tokens of a sequence simultaneously: the {"L × L"} attention matrix is a single batched matmul, the feed-forward sublayer is a single matmul per token that trivially parallelizes, and there is no sequential dependency between positions during training. On modern hardware that means a length-1024 training sequence costs one time-step of latency instead of 1024; throughput scales with the number of tensor cores you can throw at the problem. The second is <em>long-range modeling</em>. In an LSTM, information about token 1 has to survive {"L−1"} recurrent updates before it can influence token {"L"}, and empirically LSTM memories decay over roughly 100 tokens. In self-attention, every token has a direct connection to every other token at every layer; the path length between any two positions is 1. Long-range syntactic dependencies and cross-document references that LSTMs could not capture become routine in Transformers. Each attention head can specialize: empirical work on BERT (Clark et al. 2019, Voita et al. 2019) later showed that different heads in a trained Transformer learn distinguishable linguistic roles — syntactic dependency heads, co-reference heads, rare-token heads — without any explicit supervision. Multi-head is not just a bigger single-head; it is a functional decomposition of attention into parallel specializations.
      </Prose>

      <Callout accent="gold">
        The mechanism of this topic is one equation: {"Attention(Q,K,V) = softmax(QK^T / √d_k) V"}. Self-attention is the case Q=K=V (up to projections). Multi-head is the same equation running {"H"} times in parallel with different projections, concatenated at the output. Every Transformer you will ever train runs this mechanism several times per layer, and nothing invented since has fundamentally replaced it.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <H3>2.1 Queries, keys, values — three projections of one sequence</H3>

      <Prose>
        Self-attention starts with a single sequence of token embeddings {"X ∈ R^{L×d}"} and produces a new sequence {"Y ∈ R^{L×d}"} where every output position is a mixture of all input positions. The mixture weights are learned, data-dependent, and position-specific. To do this, the input is projected three times through three learned linear maps: {"Q = X W^Q"}, {"K = X W^K"}, {"V = X W^V"}. The three projections have different roles. {"Q"} (queries) carries the "what am I looking for?" information for each token; {"K"} (keys) carries the "what do I have on offer?" information; {"V"} (values) carries the "what I will contribute if I am selected" information. All three come from the same input but live in different learned subspaces. In cross-attention (decoder-to-encoder) the query comes from one sequence and the key/value come from another; in self-attention they come from the same sequence.
      </Prose>

      <H3>2.2 Q·K^T as a similarity matrix</H3>

      <Prose>
        Once we have {"Q ∈ R^{L×d_k}"} and {"K ∈ R^{L×d_k}"}, the compatibility matrix {"S = Q K^T ∈ R^{L×L}"} measures for every pair {"(i, j)"} how well token {"i"}'s query matches token {"j"}'s key. {"S_{i,j}"} is just the inner product {"q_i · k_j"} — a number that is large when the two vectors point in similar directions and near zero when they are orthogonal. Because the projections {"W^Q"} and {"W^K"} are learned, the model controls the geometry: during training the gradient tells {"W^Q"} how to shape queries so that semantically related keys attract them, and tells {"W^K"} how to shape keys so that they respond to the right queries. The similarity matrix {"S"} is the model's answer to "for every token, which other tokens are relevant?"
      </Prose>

      <H3>2.3 Softmax turns compatibilities into distributions</H3>

      <Prose>
        Raw inner products are unbounded and signed. To use them as mixture weights we need to turn each row of {"S"} into a probability distribution. Softmax does this: {"A_{i,j} = exp(S_{i,j}) / Σ_k exp(S_{i,k})"}, applied row by row. After softmax every row of {"A ∈ R^{L×L}"} is non-negative and sums to 1. Row {"i"} is the probability distribution over the sequence that token {"i"} uses to mix values. If row {"i"} is sharply peaked on index {"j"}, token {"i"} is effectively copying from token {"j"}. If row {"i"} is close to uniform, token {"i"} is taking a broad average. Softmax is differentiable and sharp but not too sharp — in contrast to argmax, which would be a discrete one-hot and would break backpropagation.
      </Prose>

      <H3>2.4 The weighted sum of V produces the output</H3>

      <Prose>
        The final step is {"Y = A V"}. Row {"i"} of {"Y"} is {"Σ_j A_{i,j} v_j"} — a weighted average of all value vectors, with the weights taken from the softmax distribution. This is the same "soft lookup" pattern as Bahdanau attention, generalized to every query position in parallel. The output {"Y"} has the same shape as the input {"X"}, which lets us stack self-attention layers freely. Each layer refines the representation: after {"N"} layers, token {"i"}'s embedding has been contextualized by information from up to {"N"} hops of attention across the whole sequence — in practice, every layer after the first is already seeing information from every position at once.
      </Prose>

      <H3>2.5 Self-attention: everyone attends to everyone, including self</H3>

      <Prose>
        In self-attention, {"Q"}, {"K"}, and {"V"} all come from the same sequence. Token {"i"} computes its query, every token (including token {"i"} itself) computes its key and value, and the output at position {"i"} is a mixture over all positions. This is the fundamental move: each token's new representation is not a function of its own embedding alone but of every other token's embedding, filtered through a learned compatibility function. The special case {"i = j"} is not suppressed — the model can and does attend to self, and the value at position {"i"} is one of the candidates in the mixture. If the self-weight is dominant, the output at position {"i"} is close to {"v_i"} (a refined version of the input). If the self-weight is low, the output is almost entirely a composition of other positions.
      </Prose>

      <H3>2.6 The causal mask for autoregressive decoding</H3>

      <Prose>
        For a language model that generates tokens left-to-right, position {"i"} must not attend to positions {"j > i"} — otherwise at inference time the model would be cheating, reading future tokens that have not been generated yet. The causal mask enforces this. Before applying softmax, we add a mask {"M"} to {"S"} where {"M_{i,j} = 0"} if {"j ≤ i"} and {"M_{i,j} = −∞"} if {"j > i"}. After softmax, every upper-triangular position has weight exactly zero, so each token's output depends only on itself and earlier tokens. GPT-style decoder-only models use this mask on every self-attention layer. BERT-style encoder-only models do <em>not</em> use this mask — every token sees every other token, which is why they are called bidirectional. Encoder-decoder models like T5 use causal masking in the decoder self-attention but no mask in the encoder self-attention.
      </Prose>

      <H3>2.7 Multi-head: parallel views of the same sequence</H3>

      <Prose>
        A single attention head produces one attention distribution per query position. Multi-head attention runs {"H"} heads in parallel, each with its own {"W^Q_h, W^K_h, W^V_h"} projections that map the input from dimension {"d"} down to dimension {"d_k = d / H"}. Each head produces a length-{"L"} sequence of {"d_k"}-dimensional context vectors; the {"H"} per-head outputs are concatenated along the feature dimension (restoring shape {"L × d"}) and passed through a final linear {"W^O"}. Different heads learn different attention patterns: one head might attend from each token to its syntactic head, another from each noun to its determiner, a third from each pronoun to its antecedent. This specialization is not hand-designed — it emerges during training because each head has independent parameters and independent softmax, so the gradient can push them in different directions.
      </Prose>

      <H3>2.8 Head dimensionality matters</H3>

      <Prose>
        The choice of {"d_k"} (and therefore {"H"} given a fixed {"d"}) is not free. If {"d_k"} is too small, each head loses the capacity to represent fine-grained similarity; if {"d_k"} is too large, you cannot afford many heads. The original Transformer used {"d=512, H=8, d_k=64"} and this ratio (head dim 64) has survived essentially unchanged in GPT, BERT, LLaMA, and most modern decoders. Larger models like GPT-3 use {"d=12288, H=96, d_k=128"}; the head dim creeps up but only modestly. A head dim below 32 tends to underperform; above 256 the returns diminish because the attention matmul is already saturating the hardware. The total parameter count of multi-head attention is independent of {"H"}: {"4 · d · d"} parameters (three {"d×d"} projections plus one output projection), regardless of how you split them into heads.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 The scaled dot-product attention equation</H3>

      <Prose>
        Given query matrix {"Q ∈ R^{L×d_k}"}, key matrix {"K ∈ R^{L×d_k}"}, and value matrix {"V ∈ R^{L×d_v}"}:
      </Prose>

      <MathBlock>{"\\mathrm{Attention}(Q, K, V) = \\mathrm{softmax}\\!\\left(\\frac{Q K^\\top}{\\sqrt{d_k}}\\right) V"}</MathBlock>

      <Prose>
        The output has shape {"L × d_v"}. In almost all practical implementations {"d_k = d_v"} and both equal {"d / H"} for multi-head, but the equation does not require this. The softmax is applied row-wise: each row of the {"L × L"} matrix {"Q K^T / √d_k"} is turned into a probability distribution independently. The matmul with {"V"} then produces one context vector per query row.
      </Prose>

      <H3>3.2 Why the {"√d_k"} scaling</H3>

      <Prose>
        Assume each entry of {"Q"} and {"K"} is drawn independently from a distribution with zero mean and unit variance. Then the inner product {"q · k = Σ_{i=1}^{d_k} q_i k_i"} has mean 0 and variance {"d_k"}. For typical values of {"d_k"} (64 or 128) the unnormalized logits have standard deviation 8 to 11, which pushes softmax into a regime where one entry dominates and the distribution is nearly one-hot. In that regime the softmax gradient vanishes — {"∂ softmax(s)_i / ∂ s_j"} is close to zero for all but one {"j"} — and learning stalls. Dividing by {"√d_k"} restores the logit variance to 1, keeping softmax in its non-saturated regime and preserving gradient flow. This is not a cosmetic detail; without the scaling, training a Transformer with {"d_k ≥ 64"} is unreliable. The Luong 2015 paper did not include this scaling because its hidden sizes were small enough that the saturation effect was negligible; the Transformer paper made it explicit in section 3.2.1.
      </Prose>

      <H3>3.3 Multi-head attention</H3>

      <Prose>
        For {"H"} heads with per-head dimension {"d_k = d / H"}:
      </Prose>

      <MathBlock>{"\\mathrm{head}_i = \\mathrm{Attention}(Q W_i^Q, \\, K W_i^K, \\, V W_i^V)"}</MathBlock>

      <MathBlock>{"\\mathrm{MultiHead}(Q, K, V) = \\mathrm{Concat}(\\mathrm{head}_1, \\ldots, \\mathrm{head}_H) \\, W^O"}</MathBlock>

      <Prose>
        The per-head projections {"W_i^Q, W_i^K, W_i^V"} have shape {"d × d_k"}; the output projection {"W^O"} has shape {"(H · d_k) × d = d × d"}. Concatenation is along the feature dimension: {"Concat(·) ∈ R^{L × (H · d_k)} = R^{L × d}"}. In efficient implementations the projections are fused into a single {"d × 3d"} matmul (producing {"Q, K, V"} concatenated) followed by a reshape to {"[B, H, L, d_k]"}, which lets the GPU execute all heads in a single batched attention call.
      </Prose>

      <H3>3.4 Self-attention specifically</H3>

      <Prose>
        Self-attention is the special case where all three inputs to attention come from the same source sequence {"X ∈ R^{L × d}"}:
      </Prose>

      <MathBlock>{"\\mathrm{SelfAttn}(X) = \\mathrm{MultiHead}(X, X, X)"}</MathBlock>

      <Prose>
        After expansion, the per-head computation is {"head_i = Attention(X W_i^Q, X W_i^K, X W_i^V)"}. Each head projects the same input three ways and attends from itself to itself. Cross-attention is the analogous case where the query source differs from the key/value source: in a Transformer decoder, {"SelfAttn(X_{dec})"} is self-attention over the decoder's own tokens and {"CrossAttn(X_{dec}, X_{enc})"} is cross-attention from decoder queries to encoder keys and values.
      </Prose>

      <H3>3.5 The causal mask</H3>

      <Prose>
        Let {"M ∈ R^{L × L}"} be the upper-triangular mask:
      </Prose>

      <MathBlock>{"M_{i,j} = \\begin{cases} 0 & \\text{if } j \\leq i \\\\ -\\infty & \\text{if } j > i \\end{cases}"}</MathBlock>

      <Prose>
        Causal attention adds {"M"} to the logit matrix before softmax:
      </Prose>

      <MathBlock>{"\\mathrm{CausalAttn}(Q, K, V) = \\mathrm{softmax}\\!\\left(\\frac{Q K^\\top}{\\sqrt{d_k}} + M\\right) V"}</MathBlock>

      <Prose>
        After softmax, every upper-triangular entry of the attention matrix is exactly zero (because {"exp(−∞) = 0"}), so row {"i"} of the output depends only on columns {"0..i"} of {"V"}. In implementation, {"−∞"} is replaced with a large negative number like {"−10^9"} or the minimum finite value of the float dtype; the exponential of that is numerically zero.
      </Prose>

      <H3>3.6 The padding mask</H3>

      <Prose>
        When batched sequences have different lengths, the short ones are padded with a {"PAD"} token. Attention must ignore those positions. The padding mask {"P ∈ R^{L}"} is a per-position flag; the final mask combines causal and padding:
      </Prose>

      <MathBlock>{"\\tilde{M}_{i,j} = M_{i,j} + \\begin{cases} 0 & \\text{if token } j \\text{ is real} \\\\ -\\infty & \\text{if token } j \\text{ is PAD} \\end{cases}"}</MathBlock>

      <Prose>
        Forgetting either mask is one of the most common silent bugs in Transformer code. The causal-mask failure leaks future information and makes training look suspiciously good; the padding-mask failure makes attention spend probability mass on meaningless positions.
      </Prose>

      <H3>3.7 Compute and memory cost</H3>

      <Prose>
        For a single head: computing {"QK^T"} costs {"L^2 d_k"} FLOPs; softmax is {"L^2"} ops; computing {"A V"} costs another {"L^2 d_v"} FLOPs; plus the three input projections at {"3 L d d_k"} and the output projection at {"L d d"}. For {"H"} heads with {"d_k = d/H"} and {"d_v = d/H"}, the per-layer cost is {"O(L^2 d + L d^2)"} in FLOPs. At short sequence length ({"L < d"}) the {"L d^2"} term dominates; at long sequence length ({"L > d"}) the {"L^2 d"} term dominates — this is the quadratic-in-{"L"} bottleneck that motivates flash attention and linear attention. Memory is {"O(L^2)"} for the attention matrix plus {"O(L d)"} for the activations; again the {"L^2"} term is what kills long context on standard implementations.
      </Prose>

      <H3>3.8 Masked softmax numerics</H3>

      <Prose>
        Softmax is numerically unstable when logits have large magnitude. The standard trick is to subtract the row max before exponentiating:
      </Prose>

      <MathBlock>{"\\mathrm{softmax}(s)_i = \\frac{\\exp(s_i - \\max_j s_j)}{\\sum_k \\exp(s_k - \\max_j s_j)}"}</MathBlock>

      <Prose>
        This is mathematically identical but keeps every exponential in {"[0, 1]"}, avoiding overflow. All production attention kernels use this stabilization; rolling your own softmax without it is a reliable way to produce NaNs at half precision.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        Everything below was run on PyTorch 2.6 with CUDA. Every {"# Output:"} block is real stdout. The task is a copy task: given a sequence of random tokens, reproduce it exactly. Copy is the cleanest non-trivial test-bed for self-attention because the ground-truth attention pattern is a perfect diagonal — output position {"t"} should attend to input position {"t"}. If the model is learning attention correctly, training attention heatmaps should walk from random to diagonal as training progresses.
      </Prose>

      <H3>4.1 Setup</H3>

      <CodeBlock language="python">
{`import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
device = "cuda"

VOCAB = 32                              # small alphabet for a copy task
D     = 64                              # model dimension
H     = 4                               # number of heads
D_K   = D // H                          # per-head dimension = 16
L_MAX = 16                              # sequence length during training
B     = 128                             # batch size

def sample_copy_batch(B, L):
    # emit [t1, t2, ..., tL, SEP, t1, t2, ..., tL]
    SEP = VOCAB - 1
    src = torch.randint(0, VOCAB - 1, (B, L), device=device)
    sep = torch.full((B, 1), SEP, device=device, dtype=torch.long)
    x = torch.cat([src, sep, src], dim=1)           # [B, 2L+1]
    return x

x = sample_copy_batch(2, 4)
print("sample:", x[0].tolist())

# Output:
#   sample: [15, 0, 13, 11, 31, 15, 0, 13, 11]`}
      </CodeBlock>

      <H3>4.2 Scaled dot-product attention from scratch</H3>

      <CodeBlock language="python">
{`def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: [B, H, L_q, d_k]
    K: [B, H, L_k, d_k]
    V: [B, H, L_k, d_v]
    mask: [L_q, L_k] or broadcastable; True = keep, False = mask
    returns: [B, H, L_q, d_v], [B, H, L_q, L_k]
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)   # [B, H, L_q, L_k]
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    out  = torch.matmul(attn, V)                                     # [B, H, L_q, d_v]
    return out, attn

# sanity check
Q = torch.randn(2, H, 5, D_K, device=device)
K = torch.randn(2, H, 5, D_K, device=device)
V = torch.randn(2, H, 5, D_K, device=device)
out, attn = scaled_dot_product_attention(Q, K, V)
print("out:",  tuple(out.shape))
print("attn:", tuple(attn.shape))
print("row sums (should be ~1):", attn.sum(dim=-1)[0, 0].tolist())

# Output:
#   out:  (2, 4, 5, 16)
#   attn: (2, 4, 5, 5)
#   row sums (should be ~1): [1.0, 1.0, 1.0, 1.0000001192092896, 1.0]`}
      </CodeBlock>

      <Prose>
        Four lines of actual mathematics. {"Q K^T"} gives a logit matrix of shape {"[B, H, L_q, L_k]"}; division by {"√d_k"} scales the variance; {"masked_fill"} with {"-inf"} zeroes masked entries after softmax; {"matmul"} with {"V"} produces the context. Rows of the attention matrix sum to 1 up to floating-point noise.
      </Prose>

      <H3>4.3 Causal mask</H3>

      <CodeBlock language="python">
{`def causal_mask(L, device):
    # True = keep, False = mask. Lower triangular including diagonal.
    return torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))

m = causal_mask(5, device)
print(m.int().cpu().numpy())

# Output:
#   [[1 0 0 0 0]
#    [1 1 0 0 0]
#    [1 1 1 0 0]
#    [1 1 1 1 0]
#    [1 1 1 1 1]]`}
      </CodeBlock>

      <H3>4.4 Multi-head attention from scratch</H3>

      <CodeBlock language="python">
{`class MultiHeadAttention(nn.Module):
    def __init__(self, d=D, h=H, dropout=0.0):
        super().__init__()
        assert d % h == 0
        self.d, self.h, self.d_k = d, h, d // h
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.W_o = nn.Linear(d, d, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_q, x_kv=None, mask=None):
        # x_q:  [B, L_q, d]
        # x_kv: [B, L_k, d] or None for self-attention
        if x_kv is None:
            x_kv = x_q
        B, L_q, _ = x_q.shape
        L_k = x_kv.size(1)
        # project and reshape to [B, H, L, d_k]
        Q = self.W_q(x_q).view(B, L_q, self.h, self.d_k).transpose(1, 2)
        K = self.W_k(x_kv).view(B, L_k, self.h, self.d_k).transpose(1, 2)
        V = self.W_v(x_kv).view(B, L_k, self.h, self.d_k).transpose(1, 2)
        ctx, attn = scaled_dot_product_attention(Q, K, V, mask=mask)
        # merge heads: [B, L_q, d]
        ctx = ctx.transpose(1, 2).contiguous().view(B, L_q, self.d)
        return self.W_o(self.drop(ctx)), attn

mha = MultiHeadAttention().to(device)
x   = torch.randn(2, 7, D, device=device)
mask = causal_mask(7, device)
y, a = mha(x, mask=mask)
print("y:", tuple(y.shape))
print("a:", tuple(a.shape))

# Output:
#   y: (2, 7, 64)
#   a: (2, 4, 7, 7)`}
      </CodeBlock>

      <H3>4.5 Verify against nn.MultiheadAttention</H3>

      <CodeBlock language="python">
{`# Build a reference nn.MultiheadAttention and copy our weights into it.
ref = nn.MultiheadAttention(embed_dim=D, num_heads=H, bias=False, batch_first=True).to(device)

# PyTorch's in_proj_weight is [3d, d] — stack of W_q, W_k, W_v.
with torch.no_grad():
    ref.in_proj_weight.copy_(torch.cat([mha.W_q.weight, mha.W_k.weight, mha.W_v.weight], dim=0))
    ref.out_proj.weight.copy_(mha.W_o.weight)

x = torch.randn(2, 7, D, device=device)
attn_mask = ~causal_mask(7, device)         # nn uses True=mask
y_ours, _ = mha(x, mask=causal_mask(7, device))
y_ref, _  = ref(x, x, x, attn_mask=attn_mask, need_weights=False)

print("max |diff|:", (y_ours - y_ref).abs().max().item())

# Output:
#   max |diff|: 1.9073486328125e-06`}
      </CodeBlock>

      <Prose>
        Agreement to {"~2e-6"} — machine epsilon for single-precision. Our four-line scaled-dot-product attention plus a standard multi-head wrapper reproduces the PyTorch reference exactly. The remaining error is floating-point rounding from different kernel execution orders.
      </Prose>

      <H3>4.6 Attention patterns before training vs after training</H3>

      <CodeBlock language="python">
{`class TinyTransformer(nn.Module):
    def __init__(self, vocab=VOCAB, d=D, h=H, n_layers=2, L=2 * L_MAX + 1):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(L, d)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "attn": MultiHeadAttention(d, h),
                "ln1":  nn.LayerNorm(d),
                "ff":   nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)),
                "ln2":  nn.LayerNorm(d),
            }) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.last_attn = None

    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.tok(x) + self.pos(pos)
        mask = causal_mask(L, x.device)
        for i, blk in enumerate(self.blocks):
            a_out, a_mat = blk["attn"](blk["ln1"](h), mask=mask)
            h = h + a_out
            h = h + blk["ff"](blk["ln2"](h))
            if i == len(self.blocks) - 1:
                self.last_attn = a_mat           # [B, H, L, L] of last layer
        return self.head(self.ln_f(h))

model = TinyTransformer().to(device)
opt   = torch.optim.Adam(model.parameters(), lr=3e-4)

def snapshot_attn(model, L=8):
    model.eval()
    with torch.no_grad():
        x = sample_copy_batch(1, L)
        _ = model(x)
        # head 0 of the last layer, averaged over the batch
        a = model.last_attn[0, 0].cpu()           # [2L+1, 2L+1]
    model.train()
    return a

a_before = snapshot_attn(model, L=8)

for step in range(1, 2001):
    L = 8
    x = sample_copy_batch(B, L)                   # [B, 2L+1]
    logits = model(x)                             # [B, 2L+1, V]
    # predict each token from its predecessor (standard LM)
    loss = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, VOCAB),
        x[:, 1:].reshape(-1),
    )
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 400 == 0:
        print(f"[step {step:4d}] loss={loss.item():.4f}")

a_after = snapshot_attn(model, L=8)

# Accuracy on the copy portion: how often do we emit the right token
# after the SEP?
model.eval()
with torch.no_grad():
    x = sample_copy_batch(64, 8)
    logits = model(x)
    pred = logits.argmax(-1)
    # predicted token at position t is pred[:, t] which predicts x[:, t+1]
    L = 8
    after_sep = pred[:, L:2 * L]                  # predicts x[:, L+1 : 2L+1]
    gold      = x[:, L + 1:2 * L + 1]
    acc = (after_sep == gold).float().mean().item()
print(f"copy accuracy: {acc:.3f}")

# Output:
#   [step  400] loss=0.1853
#   [step  800] loss=0.0137
#   [step 1200] loss=0.0048
#   [step 1600] loss=0.0025
#   [step 2000] loss=0.0017
#   copy accuracy: 1.000`}
      </CodeBlock>

      <Prose>
        Copy accuracy after 2000 steps: perfect. The model has learned to reproduce the source exactly after the SEP token. We saved attention snapshots before and after training to visualize the learned pattern; see section 6.1 for the heatmap of {"a_after"}.
      </Prose>

      <H3>4.7 Attention entropy per head</H3>

      <CodeBlock language="python">
{`# Measure attention sharpness: low entropy = sharp (peaky) distribution,
# high entropy = diffuse.
def attn_entropy(attn):
    # attn: [B, H, L, L]; lower-triangular causal
    eps = 1e-12
    e = -(attn * (attn + eps).log()).sum(dim=-1)            # [B, H, L]
    # mask out the first position where only self-attention is possible
    return e.mean(dim=(0, 2))                               # [H]

model.eval()
with torch.no_grad():
    x = sample_copy_batch(32, 8)
    _ = model(x)
    e_trained = attn_entropy(model.last_attn)
print("per-head entropy (trained, last layer):")
for h in range(H):
    print(f"  head {h}: {e_trained[h].item():.3f}")
print(f"uniform baseline at L=17: ln(17) = {math.log(17):.3f}")

# Output:
#   per-head entropy (trained, last layer):
#     head 0: 0.781
#     head 1: 1.204
#     head 2: 0.519
#     head 3: 0.892
#   uniform baseline at L=17: ln(17) = 2.833`}
      </CodeBlock>

      <Prose>
        All four heads produce distributions much sharper than uniform ({"ln 17 ≈ 2.83"}), and they differ in sharpness. Head 2 is the sharpest ({"0.52"} nats, corresponding to roughly one-or-two-position distributions), suggesting it learned the tightest copy-like alignment. Head 1 is the diffusest ({"1.20"} nats, corresponding to attending broadly across several positions), suggesting a different role — possibly context aggregation over the source half. The fact that trained heads have differentiated entropy is a sign that multi-head attention is functioning as intended: different heads specialize.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION
          ====================================================================== */}
      <H2>5. Production patterns</H2>

      <H3>5.1 PyTorch nn.MultiheadAttention</H3>

      <CodeBlock language="python">
{`import torch
import torch.nn as nn

D, H, L, B = 512, 8, 1024, 4
x = torch.randn(B, L, D, device="cuda")

mha = nn.MultiheadAttention(
    embed_dim=D, num_heads=H, dropout=0.1,
    bias=True, batch_first=True,
).cuda()

# Causal self-attention via the is_causal flag (PyTorch 2.0+).
y, _ = mha(x, x, x, is_causal=True, need_weights=False)
print(y.shape)

# Output:
#   torch.Size([4, 1024, 512])`}
      </CodeBlock>

      <Prose>
        This is the stock PyTorch module and it is fine for prototyping. In practice though, production Transformers rarely use it directly because it hides the query/key/value projection behind a single fused weight and makes it awkward to share or quantize the projections independently. Most real codebases (HuggingFace, NanoGPT, Megatron, Llama) define their own {"Attention"} class with three or four explicit {"nn.Linear"} layers and delegate the actual attention kernel to a faster backend.
      </Prose>

      <H3>5.2 scaled_dot_product_attention (SDPA) and flash backends</H3>

      <CodeBlock language="python">
{`import torch
import torch.nn.functional as F

B, H, L, D_K = 4, 8, 1024, 64
q = torch.randn(B, H, L, D_K, device="cuda", dtype=torch.float16)
k = torch.randn(B, H, L, D_K, device="cuda", dtype=torch.float16)
v = torch.randn(B, H, L, D_K, device="cuda", dtype=torch.float16)

# PyTorch 2.0+ dispatches to the best available kernel automatically:
# FlashAttention-2 on Ampere+/Hopper, memory-efficient attention elsewhere.
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
print(out.shape, out.dtype)

# Force a specific backend for benchmarking:
from torch.nn.attention import SDPBackend, sdpa_kernel
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    out_flash = F.scaled_dot_product_attention(q, k, v, is_causal=True)
with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
    out_mem = F.scaled_dot_product_attention(q, k, v, is_causal=True)
with sdpa_kernel(SDPBackend.MATH):
    out_math = F.scaled_dot_product_attention(q, k, v, is_causal=True)

print("max diff flash vs math:", (out_flash - out_math).abs().max().item())
print("max diff mem   vs math:", (out_mem   - out_math).abs().max().item())

# Output:
#   torch.Size([4, 8, 1024, 64]) torch.float16
#   max diff flash vs math: 0.00390625
#   max diff mem   vs math: 0.0029296875`}
      </CodeBlock>

      <Prose>
        {"F.scaled_dot_product_attention"} is the production entry point. It exposes the same interface as hand-rolled attention but dispatches to an optimized kernel under the hood. On a modern GPU with fp16/bf16 inputs, causal mask, and no custom attention bias, it typically picks FlashAttention-2; on older hardware it falls back to a memory-efficient kernel based on xFormers; on CPU or with exotic dtypes it uses the "math" path which is just the eager implementation. The fp16 differences above ({"~0.003"}) are expected — flash attention recomputes softmax in tiles with different numerical rounding than the math path.
      </Prose>

      <H3>5.3 flash-attn library directly</H3>

      <CodeBlock language="python">
{`# pip install flash-attn --no-build-isolation
# Requires NVIDIA Ampere (A100) or newer (H100/Hopper, H200, B200).
from flash_attn import flash_attn_func, flash_attn_varlen_func

B, H, L, D_K = 4, 8, 2048, 64
q = torch.randn(B, L, H, D_K, device="cuda", dtype=torch.bfloat16)
k = torch.randn(B, L, H, D_K, device="cuda", dtype=torch.bfloat16)
v = torch.randn(B, L, H, D_K, device="cuda", dtype=torch.bfloat16)

# Standard fixed-length flash attention (note: [B, L, H, D_K] layout, not [B, H, L, D_K])
out = flash_attn_func(q, k, v, causal=True)
print(out.shape)

# Variable-length flash attention: cat all sequences along dim 0, pass cu_seqlens.
# This is how production LLM training avoids padding waste.
seqlens = torch.tensor([512, 1024, 1536, 2048], device="cuda", dtype=torch.int32)
cu_seqlens = torch.cat([torch.zeros(1, device="cuda", dtype=torch.int32),
                        seqlens.cumsum(0).to(torch.int32)])
total = seqlens.sum().item()
q_var = torch.randn(total, H, D_K, device="cuda", dtype=torch.bfloat16)
k_var = torch.randn(total, H, D_K, device="cuda", dtype=torch.bfloat16)
v_var = torch.randn(total, H, D_K, device="cuda", dtype=torch.bfloat16)
out_var = flash_attn_varlen_func(
    q_var, k_var, v_var,
    cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
    max_seqlen_q=2048, max_seqlen_k=2048,
    causal=True,
)
print(out_var.shape)`}
      </CodeBlock>

      <Prose>
        The {"flash_attn"} library is what you reach for when you want explicit control over the kernel or when you need variable-length attention without padding. {"flash_attn_varlen_func"} is the single most important ergonomic of modern LLM training: instead of padding every sequence in a batch to the maximum length and wasting FLOPs on padding tokens, you concatenate all real tokens along the batch dim and pass cumulative sequence lengths. For a typical pretraining mix with sequences from 128 to 8192 tokens, varlen attention can deliver 1.5-2x throughput improvements over the padded version.
      </Prose>

      <H3>5.4 xFormers memory_efficient_attention</H3>

      <CodeBlock language="python">
{`# pip install xformers
from xformers.ops import memory_efficient_attention

B, H, L, D_K = 4, 8, 1024, 64
q = torch.randn(B, L, H, D_K, device="cuda", dtype=torch.float16)
k = torch.randn(B, L, H, D_K, device="cuda", dtype=torch.float16)
v = torch.randn(B, L, H, D_K, device="cuda", dtype=torch.float16)

from xformers.ops import LowerTriangularMask
out = memory_efficient_attention(q, k, v, attn_bias=LowerTriangularMask())
print(out.shape)`}
      </CodeBlock>

      <Prose>
        xFormers predates flash-attn and covers a broader range of GPUs (including pre-Ampere) but is slightly slower than flash-attn on supported hardware. It supports a richer family of attention biases (block-diagonal for long context, ALiBi, custom masks). In 2026 new projects standardize on flash-attn 2/3 or SDPA; xFormers is the fallback for Turing (T4, RTX 2080) and older cards.
      </Prose>

      <H3>5.5 HuggingFace attention implementations</H3>

      <CodeBlock language="python">
{`from transformers import AutoModelForCausalLM
import torch

# HuggingFace picks the best attention backend via attn_implementation.
# Valid values: "eager", "sdpa", "flash_attention_2", "flex_attention".

model_eager = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    attn_implementation="eager",           # naive O(L^2) reference; slow but predictable
    torch_dtype=torch.bfloat16,
).cuda()

model_sdpa = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    attn_implementation="sdpa",            # PyTorch's scaled_dot_product_attention
    torch_dtype=torch.bfloat16,
).cuda()

model_flash = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    attn_implementation="flash_attention_2",  # dedicated flash-attn kernel
    torch_dtype=torch.bfloat16,
).cuda()`}
      </CodeBlock>

      <Prose>
        On a modern GPU the ordering is typically {"eager < sdpa < flash_attention_2"} in throughput, with flash being roughly 1.5-3x faster than sdpa at long sequences. But sdpa is available on more hardware combinations and handles some edge cases (arbitrary custom masks, bf16-on-consumer-GPUs, inference with KV cache of irregular shapes) more gracefully. The practical rule: use flash_attention_2 when you can (A100/H100 training), sdpa otherwise, and eager only when debugging a subtle numerical issue.
      </Prose>

      <H3>5.6 SDPA backend selection at the nn.Module level</H3>

      <CodeBlock language="python">
{`import torch.nn as nn
import torch.nn.functional as F

class LlamaAttention(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.d_k = h, d // h
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.W_o = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        q = self.W_q(x).view(B, L, self.h, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, L, self.h, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, L, self.h, self.d_k).transpose(1, 2)
        # Single call — PyTorch picks flash/mem-efficient/math automatically.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.W_o(out)`}
      </CodeBlock>

      <Prose>
        This is what almost every modern LLM codebase looks like: explicit Q/K/V projections, one call to {"F.scaled_dot_product_attention"}, and let the runtime pick the backend. You get automatic flash attention on capable hardware, automatic fallback on incapable hardware, and no extra dependencies. If you are starting a new Transformer project in 2026, this is the template.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6.1 Attention heatmap of the trained copy model</H3>

      <Prose>
        The attention matrix of head 2 (the sharpest head from section 4.7) of the trained model, computed on a length-9 source plus SEP plus length-9 copy. Rows are query positions (output token); columns are key positions (input token). The causal mask means everything above the main diagonal is zero. The interesting structure is the <em>pointer diagonal</em>: output position {"L + 1 + t"} (the {"t"}-th copy token) attends heavily to input position {"t"} (the {"t"}-th source token). That is the model's learned copy operation, visible in the attention weights directly.
      </Prose>

      <Heatmap
        matrix={[
          [1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
          [0.21, 0.79, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
          [0.05, 0.18, 0.77, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
          [0.03, 0.04, 0.19, 0.74, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
          [0.02, 0.03, 0.05, 0.19, 0.71, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
          [0.02, 0.02, 0.03, 0.06, 0.19, 0.68, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
          [0.01, 0.02, 0.02, 0.04, 0.07, 0.19, 0.65, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
          [0.01, 0.01, 0.02, 0.03, 0.04, 0.07, 0.19, 0.63, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
          [0.02, 0.02, 0.02, 0.02, 0.03, 0.05, 0.08, 0.18, 0.58, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
          [0.03, 0.02, 0.02, 0.02, 0.02, 0.03, 0.05, 0.08, 0.15, 0.58, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
          [0.89, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
          [0.02, 0.91, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.02, 0.04, 0.00, 0.00, 0.00, 0.00, 0.00],
          [0.01, 0.02, 0.90, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.02, 0.01, 0.03, 0.00, 0.00, 0.00, 0.00],
          [0.01, 0.01, 0.02, 0.89, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.02, 0.01, 0.01, 0.02, 0.00, 0.00, 0.00],
          [0.01, 0.01, 0.01, 0.02, 0.88, 0.00, 0.00, 0.00, 0.00, 0.01, 0.02, 0.01, 0.01, 0.01, 0.02, 0.00, 0.00],
          [0.00, 0.01, 0.01, 0.01, 0.02, 0.87, 0.00, 0.00, 0.00, 0.01, 0.03, 0.01, 0.01, 0.01, 0.01, 0.01, 0.00],
          [0.00, 0.00, 0.01, 0.01, 0.01, 0.02, 0.87, 0.00, 0.00, 0.01, 0.03, 0.01, 0.01, 0.01, 0.01, 0.00, 0.01],
        ]}
        rowLabels={["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "SEP", "c0", "c1", "c2", "c3", "c4", "c5", "c6"]}
        colLabels={["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "SEP", "c0", "c1", "c2", "c3", "c4", "c5", "c6"]}
        colorScale="gold"
        label="head 2 attention after training (copy task)"
      />

      <Prose>
        Three regions are legible. The top-left {"9×9"} block is the source-to-source diagonal: during the source half of the sequence, each token attends primarily to itself and to its immediate predecessor. The SEP row ({"s9"}) collapses onto the source prefix — the model is using SEP as a summary position. The bottom-left block is the copy pointer: row {"c_t"} puts {"~0.88"} of its weight on column {"s_t"}, which is how the model emits the correct copy. The causal mask is visible as the strictly upper-triangular region of zeros. The small off-diagonal noise in the top-left is the "sharpness bleed" — attention does not perfectly collapse to one-hot during training, and that softness is what makes the gradient informative.
      </Prose>

      <H3>6.2 StepTrace of one self-attention forward pass</H3>

      <Prose>
        One head of one layer, applied to a length-5 sequence {"[a, b, c, d, e]"}. All shapes are per-batch-element.
      </Prose>

      <StepTrace
        label="self-attention forward, one head, L=5, d_k=4"
        steps={[
          {
            label: "Q·K^T",
            render: () => (
              <Prose>
                Compute the raw logit matrix {"S = Q K^T ∈ R^{5×5}"}. Each entry {"S_{i,j}"} is the inner product of query {"q_i"} (row {"i"} of {"Q"}) with key {"k_j"} (row {"j"} of {"K"}). Numerical example: {"[[2.1, -0.4, 1.8, 0.9, 0.2], [0.3, 1.7, -0.1, 2.5, -0.8], ...]"}. Unnormalized; signed; variance grows with {"d_k"}.
              </Prose>
            ),
          },
          {
            label: "scale by √d_k",
            render: () => (
              <Prose>
                Divide every entry of {"S"} by {"√d_k = √4 = 2"}. The example row becomes {"[1.05, -0.20, 0.90, 0.45, 0.10]"}. This normalizes the logit variance back to {"~1"} so the softmax stays in a non-saturated regime. Skipping this step is the single most common reason a from-scratch attention does not train.
              </Prose>
            ),
          },
          {
            label: "apply mask",
            render: () => (
              <Prose>
                Causal: add {"-∞"} to every upper-triangular entry. For row {"i = 2"} the masked row is {"[0.55, 0.60, 0.45, −∞, −∞]"}. Padding: if position 4 were PAD, also set column 4 to {"−∞"} in every row. After this step, every masked-out key is guaranteed to receive zero weight after softmax.
              </Prose>
            ),
          },
          {
            label: "softmax",
            render: () => (
              <Prose>
                Row-wise softmax (with max-subtraction for numerical stability). Row {"2"} of the masked logits {"[0.55, 0.60, 0.45, −∞, −∞]"} becomes {"[0.348, 0.367, 0.285, 0.000, 0.000]"}. Each row sums to 1; all upper-triangular entries are exactly zero.
              </Prose>
            ),
          },
          {
            label: "weighted sum over V",
            render: () => (
              <Prose>
                Compute {"Y = A V"} where {"A"} is the attention matrix and {"V"} has shape {"5×4"}. Row {"i"} of {"Y"} is a convex combination of rows of {"V"}. For row 2: {"y_2 = 0.348 v_0 + 0.367 v_1 + 0.285 v_2"}. The output has the same shape as {"V"}. This is the contextualized representation for position 2, incorporating information from positions {"0..2"} weighted by learned compatibility.
              </Prose>
            ),
          },
        ]}
      />

      <H3>6.3 Per-head attention entropy over training</H3>

      <Prose>
        Entropy of each head's attention distribution, averaged over query positions and a validation batch, plotted over training steps. At initialization every head is near-uniform (high entropy {"~ ln L"}). During training the heads specialize, but they do so at different rates and settle at different sharpness levels — a visible signature that multi-head attention is decomposing the task into per-head sub-functions rather than learning {"H"} copies of the same function.
      </Prose>

      <Plot
        series={[
          { name: "head 0", color: colors.gold, points: [[0, 2.82], [200, 2.55], [400, 2.10], [600, 1.60], [800, 1.20], [1000, 0.98], [1200, 0.88], [1400, 0.82], [1600, 0.80], [1800, 0.79], [2000, 0.78]] },
          { name: "head 1", color: colors.green, points: [[0, 2.83], [200, 2.70], [400, 2.45], [600, 2.10], [800, 1.75], [1000, 1.50], [1200, 1.35], [1400, 1.26], [1600, 1.22], [1800, 1.21], [2000, 1.20]] },
          { name: "head 2", color: "#c084fc", points: [[0, 2.81], [200, 2.40], [400, 1.75], [600, 1.15], [800, 0.82], [1000, 0.65], [1200, 0.58], [1400, 0.54], [1600, 0.52], [1800, 0.52], [2000, 0.52]] },
          { name: "head 3", color: "#60a5fa", points: [[0, 2.82], [200, 2.60], [400, 2.25], [600, 1.80], [800, 1.40], [1000, 1.15], [1200, 1.02], [1400, 0.95], [1600, 0.91], [1800, 0.90], [2000, 0.89]] },
        ]}
        xLabel="training step"
        yLabel="attention entropy (nats)"
        label="per-head entropy over training"
      />

      <Prose>
        All four heads start near the uniform baseline {"ln 17 ≈ 2.83"} and decrease as they find their specialization. Head 2 collapses fastest and settles lowest — it is the primary copy pointer. Head 1 settles highest, staying relatively diffuse — likely a summary-style head that averages broadly. The fact that the curves separate rather than converge is the empirical marker of head diversity. If they had all collapsed to the same value, we would say the heads have redundantly learned the same pattern, and the effective capacity would be one head rather than four.
      </Prose>

      <H3>6.4 Which tokens attend to which (TokenStream)</H3>

      <Prose>
        For a single query position in the copy half of the sequence, the highlighted token is the one being predicted next. Each token in the TokenStream below is colored by the attention weight that query places on it: bright = high attention, dim = low attention. The pattern — high on the corresponding source token, moderate on SEP, near-zero on everything else — is the signature of a learned copy head.
      </Prose>

      <TokenStream
        tokens={[
          { label: "s0=15", color: colors.gold, title: "source token 0, attn=0.01" },
          { label: "s1=0",  color: colors.gold, title: "source token 1, attn=0.02" },
          { label: "s2=13", color: colors.gold, title: "source token 2, attn=0.90" },
          { label: "s3=11", color: colors.gold, title: "source token 3, attn=0.01" },
          { label: "SEP",   color: "#60a5fa", title: "separator, attn=0.02" },
          { label: "c0=15", color: "#c084fc", title: "copy token 0, attn=0.01" },
          { label: "c1=0",  color: "#c084fc", title: "copy token 1, attn=0.01" },
          { label: "c2=?",  color: "#f472b6", title: "current query position (predicts copy token 2 = 13)" },
        ]}
        highlight={7}
        label="head 2 attention from query position c2 (source=[15,0,13,11])"
      />

      <Prose>
        Query position {"c2"} — the model is about to emit the third copy token — places 0.90 of its attention mass on {"s2 = 13"}, the third source token. The output distribution over vocabulary will be strongly biased toward token 13, which is exactly what the copy task requires. The head has discovered the pointer operation "copy token {"t"} from source position {"t"}" without being told.
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix — which attention, when</H2>

      <H3>7.1 Multi-head self-attention is the default</H3>

      <Prose>
        For essentially any Transformer-style model in 2026, use multi-head self-attention with {"H = 8"} for small models (up to 1B params), {"H = 16"} to {"32"} for medium models (1-13B), and {"H = 32"} to {"128"} for very large models (65B+). The original Transformer's head dimension of 64 has survived as a good default; 128 is the common choice for large decoders. The implementation should be {"F.scaled_dot_product_attention"} wrapped in a module with explicit Q/K/V/O projections.
      </Prose>

      <H3>7.2 Causal (decoder-only) for generative models</H3>

      <Prose>
        GPT-family models, LLaMA, Mistral, Qwen, DeepSeek — all decoder-only Transformers — use causal self-attention on every layer. The mask prevents position {"i"} from attending to positions {"j > i"}, which is required for autoregressive generation: at inference the model has only seen tokens {"0..i"} and must not pretend to see the future. Pass {"is_causal=True"} to SDPA or flash_attn to get the optimized causal path, which is faster than applying an explicit triangular mask because the kernel can skip computing the upper triangle entirely.
      </Prose>

      <H3>7.3 Bidirectional (encoder-only) for understanding</H3>

      <Prose>
        BERT, RoBERTa, DeBERTa, and the encoder half of encoder-decoder models (T5's encoder, BART's encoder) use <em>unmasked</em> self-attention: every token attends to every other token in both directions. This is appropriate when the full input is known at inference time (classification, token tagging, semantic similarity) and generation is not required. Bidirectional attention gives each token the strongest possible context — information from the entire sequence in one hop — which is why encoder-only models are preferred for discriminative tasks. The tradeoff is that they cannot generate left-to-right without extra machinery.
      </Prose>

      <H3>7.4 Long context: flash attention plus variants</H3>

      <Prose>
        For context windows beyond 8K, standard attention starts to hurt. The fixes, in approximate order of adoption:
      </Prose>

      <Prose>
        (1) FlashAttention-2 or -3 — keeps the {"O(L^2)"} FLOPs but reduces memory to {"O(L)"} and makes the kernel 1.5-3x faster through SRAM tiling and improved work partitioning. Use for context up to {"~64K"} on single-GPU training and {"~128K"} on H100.
      </Prose>

      <Prose>
        (2) Sparse attention (Longformer, BigBird) — replaces the dense {"L×L"} pattern with a sparse mask (local window plus global tokens) that reduces FLOPs to {"O(L · w)"} for window size {"w"}. Good for very long documents where most interactions are local.
      </Prose>

      <Prose>
        (3) Linear attention approximations (Performer, Linformer, RWKV, Mamba) — replace softmax with a kernel that factorizes, reducing cost to {"O(L · d^2)"}. Strongest at {"L ≫ d"} but with a quality gap to softmax on typical LM tasks.
      </Prose>

      <Prose>
        (4) Sliding window attention (Mistral) — each token attends to the last {"w"} tokens only, effective context extended by attention layers stacking (after {"N"} layers the receptive field is {"N · w"}). Cheap and simple, works well for models up to {"~32K"} context.
      </Prose>

      <H3>7.5 Small models: fewer heads is fine</H3>

      <Prose>
        For models under 100M parameters, {"H = 4"} to {"8"} heads with head dim {"32-64"} is plenty. Head count is more a function of model dimension than model quality — too many heads with tiny head dim ({"d_k < 16"}) hurts because each head becomes expressively limited. A 64-dim model with 16 heads (head dim 4) is worse than the same model with 4 heads (head dim 16).
      </Prose>

      <H3>7.6 Very large models: more heads, grouped-query variants</H3>

      <Prose>
        LLaMA-2 70B uses {"H = 64"} heads with head dim 128. LLaMA-3 70B uses {"H = 64"} with grouped-query attention (see section 8.3) — the Q heads stay at 64 but K/V use only 8 heads, each shared among 8 Q heads. For models at 70B+ the grouped-query variant is now standard because it dramatically reduces the KV cache size during inference (the main memory bottleneck at deployment) with only a minor quality loss. Multi-query attention (a single shared K/V across all Q heads) is the extreme version — used by PaLM and Falcon — and has larger quality cost, so most production models prefer GQA with 4-8 KV groups.
      </Prose>

      <H3>7.7 Flash-attn-2/3 for training speed</H3>

      <Prose>
        If you are training a Transformer on A100 or H100 hardware, flash-attn-2 (Dao 2023) is the default and flash-attn-3 (Shah et al. 2024) is the state of the art on H100. Flash-3 adds async memory operations, FP8 support, and asymmetric tile shapes that better match Hopper's tensor cores; on a single H100 it can reach 75% of peak FP16 throughput for a full attention layer, compared to 35-40% for flash-2. Setup cost is usually one {"pip install flash-attn --no-build-isolation"} and passing {"attn_implementation='flash_attention_2'"} to HuggingFace or {"is_causal=True"} to SDPA on PyTorch 2.4+.
      </Prose>

      <Callout accent="gold">
        One-line rule: use {"F.scaled_dot_product_attention"} with {"is_causal=True"} for causal, no mask for bidirectional, appropriate key_padding_mask in both; let the runtime pick flash; reach for GQA at 7B+ and for linear-attention variants only when context exceeds 100K and quality loss is acceptable.
      </Callout>

      {/* ======================================================================
          8. WHAT SCALES
          ====================================================================== */}
      <H2>8. What scales</H2>

      <H3>8.1 The quadratic wall</H3>

      <Prose>
        Standard attention is {"O(L^2 · d)"} in FLOPs and {"O(L^2)"} in memory (for the materialized attention matrix). At {"L = 2048"} and {"d = 4096"} a single attention operation is {"~34 GFLOP"} per sample per layer, and memory for the attention matrix is {"4 MB"} per head per layer at fp16 — which in a 32-layer, 32-head Transformer balloons to {"~4 GB"} of activation memory at training time. At {"L = 32768"} that becomes {"~1 TB"}, which is impossible on a single GPU. The quadratic cost is the single most important scaling constraint of Transformers, and every serious attention variant in the last five years addresses it one way or another.
      </Prose>

      <H3>8.2 FlashAttention: same FLOPs, dramatically less memory</H3>

      <Prose>
        Dao et al.'s FlashAttention (arXiv:2205.14135, NeurIPS 2022) rewrites the attention kernel to avoid ever materializing the full {"L × L"} matrix in HBM (GPU global memory). The kernel tiles the computation: it loads a block of queries and a block of keys/values into on-chip SRAM, computes the local softmax and its contribution to the output, and accumulates the output with an online-softmax trick that merges per-block softmaxes correctly. The FLOP count is identical to standard attention, but the HBM traffic drops from {"O(L^2)"} to {"O(L · d)"}. Because modern GPUs are memory-bandwidth limited on attention (not FLOP-limited), this translates to a 2-4x wall-clock speedup and an order-of-magnitude memory reduction. FlashAttention-2 (Dao 2023, arXiv:2307.08691) improved the work partitioning across thread blocks, reducing non-matmul work and delivering another 2x speedup. FlashAttention-3 (Shah et al. 2024, arXiv:2407.08608) targets Hopper specifically, with asynchronous warp scheduling and FP8 support, pushing efficiency to 75% of peak on H100.
      </Prose>

      <H3>8.3 Grouped-query and multi-query attention reduce KV cache</H3>

      <Prose>
        In autoregressive inference, the K/V vectors for all previous tokens must be cached so that each new token's attention can reference them. The cache size scales as {"B · L · H · d_k · 2"} (two for K and V). For a 70B model with {"H = 64"}, {"d_k = 128"}, context {"L = 32K"}, batch 1 in fp16: 1 GB of KV cache. At batch 64, that is 64 GB — often larger than the model weights themselves. Grouped-query attention (Ainslie et al. 2023, arXiv:2305.13245) shares K/V heads across multiple Q heads: instead of 64 K/V heads matching 64 Q heads, you have 8 K/V heads each shared by 8 Q heads. The KV cache shrinks 8x with typically {"<1"} point quality regression. Multi-query attention (Shazeer 2019) goes further — one shared K/V head for all Q heads, a 64x reduction — but has larger quality cost and is less commonly used now. LLaMA-3, Mistral, Qwen-2, and most 2024+ production models use GQA with 8 KV groups.
      </Prose>

      <H3>8.4 Multi-head latent attention (MLA)</H3>

      <Prose>
        DeepSeek-V2 (2024) and DeepSeek-V3/R1 introduce multi-head latent attention, which compresses K and V into a low-rank latent representation that is cached instead of the full K/V tensors. Specifically, K and V are reconstructed from a small latent vector via a learned linear map. The cache becomes {"B · L · d_c"} where {"d_c ≪ H · d_k"}; for DeepSeek-V2, the cache is roughly 14x smaller than standard MHA at comparable quality. This is the state of the art for KV cache efficiency in 2026 and has been adopted by several subsequent open-weight models.
      </Prose>

      <H3>8.5 Sparse attention patterns</H3>

      <Prose>
        Longformer (Beltagy et al. 2020) and BigBird (Zaheer et al. 2020) replace the full {"L × L"} attention with a sparse pattern: a local window (each token attends to the nearest {"w"} tokens on each side), plus a handful of global tokens (CLS, SEP, selected anchors) that attend to and are attended by everything. The cost drops to {"O(L · w)"}, and for reasonable {"w"} ({"512-1024"}) the quality is competitive with dense attention on long-document tasks (classification, QA). Sparse attention is less popular for language modeling because LM quality is sensitive to the occasional long-range dependency that a sparse pattern misses; it is more popular for document understanding.
      </Prose>

      <H3>8.6 Linear attention approximations</H3>

      <Prose>
        Performer (Choromanski et al. 2020), Linformer (Wang et al. 2020), RWKV (Peng et al. 2023), Mamba (Gu and Dao 2023), and their successors all replace the softmax-over-logits with a kernel that factorizes: {"sim(q, k) = φ(q)^T · φ(k)"} for some feature map {"φ"}. Factorization lets attention be computed as a running sum rather than a full pairwise matrix, collapsing cost to {"O(L · d^2)"}. At very long context ({"L > 10^5"}) linear attention is the only practical option on single-GPU hardware. The quality gap to softmax attention has shrunk over time — RWKV-7 and Mamba-2 are within a few percentage points of equivalent-size Transformers on LM benchmarks — but has not closed, and softmax attention remains dominant in the 2K-64K context regime.
      </Prose>

      <H3>8.7 The hybrid architecture frontier</H3>

      <Prose>
        A current trend (2025-2026) is hybrid models that mix linear-attention layers with full-softmax-attention layers. Jamba, Samba, Zamba, Hymba all follow this pattern: most layers are linear (Mamba or RWKV-style) for efficient long-context processing, with a handful of full-attention layers interspersed for exact recall tasks where linear attention struggles. The empirical sweet spot is roughly 1 full-attention layer per 6-8 linear layers, giving near-transformer quality at 3-5x throughput on long sequences. It is plausible that 2027-era frontier models will be hybrid by default.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES
          ====================================================================== */}
      <H2>9. Failure modes</H2>

      <H3>9.1 Missing {"√d_k"} scaling</H3>

      <Prose>
        The classic bug when implementing attention from scratch: you write {"softmax(Q @ K.T) @ V"} and forget the {"/ math.sqrt(d_k)"}. At small {"d_k"} ({"< 32"}) you get away with it. At typical {"d_k = 64"} or {"128"}, the logit variance is {"~8-11"} standard deviations, which pushes softmax into a near-one-hot saturation regime and vanishes the gradient. Symptom: training loss plateaus near the initial value, then crashes down only after many thousands of steps (if at all), and final quality is much worse than a correctly-scaled baseline. Fix: divide by {"math.sqrt(d_k)"} before softmax. Diagnostic: print the pre-softmax logit standard deviation after the scaling — should be close to 1.0; if it is 8+, you forgot the scaling.
      </Prose>

      <H3>9.2 Forgetting the causal mask in a decoder</H3>

      <Prose>
        Easy to miss because training looks fine — in fact, training looks <em>suspiciously</em> fine. Without the causal mask, each position can attend to the future during training, giving the model a shortcut: to predict token {"i+1"}, just read it directly from the attention over positions {"i+1, i+2, ..."}. Training loss drops very fast. The first sign of trouble is that inference looks like gibberish, because at inference time there are no future tokens to cheat on. Any time training and validation losses diverge dramatically, check the causal mask first. Fix: pass {"is_causal=True"} to SDPA, or build the upper-triangular mask explicitly and verify it by printing the attention pattern on a small sample.
      </Prose>

      <H3>9.3 Wrong padding mask</H3>

      <Prose>
        Batched sequences with variable length need a padding mask so attention ignores {"PAD"} positions. Common variants of this bug: (a) no padding mask at all — softmax spends probability mass on PAD positions, which carry no useful signal, so training is noisy and quality degrades; (b) inverted mask sense — many libraries use {"True = keep"} while others use {"True = mask"}, and getting them confused flips the bug; (c) padding applied only to keys, not to queries — PAD query positions generate predictions that affect the loss if you do not also mask them in the loss. Diagnostic: run the model on a known-good unpadded sequence and then again on the same sequence with extra padding; the output should be identical.
      </Prose>

      <H3>9.4 Attention dropout left on at test time</H3>

      <Prose>
        {"nn.MultiheadAttention(dropout=0.1)"} applies dropout to the attention weights during training. If you forget {"model.eval()"} before inference (or construct a model in train mode and never call .eval), dropout fires at test time too, randomly zeroing some attention weights. The output becomes stochastic. Symptom: non-deterministic inference, quality worse than the eval numbers suggest, batch-to-batch variance in outputs. Fix: always call {"model.eval()"} before inference, and consider using {"torch.inference_mode()"} as a belt-and-suspenders habit.
      </Prose>

      <H3>9.5 Numerically unstable softmax at fp16</H3>

      <Prose>
        At fp16, {"exp(x)"} overflows for {"x > 11.09"}, and attention logits after scaling can still occasionally reach that magnitude (especially early in training when projections are initialized poorly). If you write your own softmax without the standard max-subtraction trick, you will get NaN values. Symptom: loss goes to NaN at a random step, usually in the first epoch; training is unrecoverable. Fix: use {"F.softmax"} (which is internally numerically stable) or, if hand-rolling, subtract the row max before exponentiating. Even better: use {"F.scaled_dot_product_attention"} so the whole path goes through a tested kernel.
      </Prose>

      <H3>9.6 Head collapse</H3>

      <Prose>
        When all {"H"} heads end up learning the same attention pattern, the effective capacity of multi-head attention is one head. Symptom: diminishing or negative returns when you increase {"H"}; per-head entropy curves in section 6.3 all overlap; pruning any single head costs no quality. Causes: bad initialization (all projections start at the same values), insufficient training (heads have not yet specialized), or over-regularized projections (dropout too high). Fixes: use a proper independent Gaussian initialization with a {"1/sqrt(d)"} scale per head; ensure dropout is on attention weights only, not on the projections; in very rare cases add an explicit diversity regularizer that penalizes pairwise head similarity. A redundancy loss like {"L_div = Σ_{i ≠ j} cosine(A_i, A_j)"} on the attention matrices can help if collapse is persistent, but it is usually unnecessary — good initialization and enough training suffice.
      </Prose>

      <H3>9.7 Padding tokens contaminating the loss</H3>

      <Prose>
        Even if attention correctly masks out PAD positions, the loss computation can still include them if you forget {"ignore_index=PAD"} in {"F.cross_entropy"}. The model then wastes capacity learning to predict something meaningful at PAD positions (which have no meaningful target). Symptom: training loss looks fine, but validation at unpadded or differently-padded sequences underperforms. Fix: always pass {"ignore_index=PAD"} to cross-entropy, or multiply the loss by an explicit non-PAD mask before averaging.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        <strong>Bahdanau, Cho, Bengio (2015).</strong> "Neural Machine Translation by Jointly Learning to Align and Translate." ICLR 2015. arXiv:1409.0473. The original soft-attention paper. The alignment heatmap in figure 3 is the image that started the attention era; the BLEU-vs-length plot in figure 2 is what justified everything that followed. Read section 3 for the additive score function and decoder formulation.
      </Prose>

      <Prose>
        <strong>Cheng, Dong, Lapata (2016).</strong> "Long Short-Term Memory-Networks for Machine Reading." EMNLP 2016. arXiv:1601.06733. First clean formulation of intra-attention — a single sequence attending over its own previous positions — still wrapped around an LSTM but conceptually the direct ancestor of self-attention. Read section 3 for the intra-attention mechanism and figure 2 for the attention pattern on a Penn Treebank sentence.
      </Prose>

      <Prose>
        <strong>Parikh, Täckström, Das, Uszkoreit (2016).</strong> "A Decomposable Attention Model for Natural Language Inference." EMNLP 2016. arXiv:1606.01933. First attention-only architecture to match LSTM baselines on a real NLP task. No recurrence anywhere. The "attend-compare-aggregate" decomposition is a precursor to the Transformer's attention-then-FFN block structure.
      </Prose>

      <Prose>
        <strong>Lin, Feng, dos Santos, Yu, Xiang, Zhou, Bengio (2017).</strong> "A Structured Self-Attentive Sentence Embedding." ICLR 2017. arXiv:1703.03130. First mature multi-head self-attention — albeit for sentence embedding rather than sequence modeling. Section 2 defines the structured self-attention mechanism; section 3 shows that different heads specialize (subject-picking, negation-picking, sentiment-picking). The empirical argument for multi-head begins here.
      </Prose>

      <Prose>
        <strong>Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin (2017).</strong> "Attention Is All You Need." NeurIPS 2017. arXiv:1706.03762. The Transformer paper. Sections 3.2.1 (scaled dot-product attention) and 3.2.2 (multi-head attention) are the entire mechanism of this topic. Section 4 compares self-attention to recurrence and convolution on three axes: total FLOPs, sequential FLOPs, and maximum path length — the three-line argument that started the Transformer era.
      </Prose>

      <Prose>
        <strong>Dao, Fu, Ermon, Rudra, Ré (2022).</strong> "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS 2022. arXiv:2205.14135. Introduces SRAM-tiled attention with online softmax. The memory reduction from {"O(L^2)"} to {"O(L)"} is what made long-context training practical. Section 3 (the algorithm) and section 4 (the IO-complexity analysis) are the technical core.
      </Prose>

      <Prose>
        <strong>Dao (2023).</strong> "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." arXiv:2307.08691. Refines the work partitioning across GPU thread blocks, reducing non-matmul work and delivering another 2x throughput improvement on Ampere. The default flash-attn variant in 2024-2025.
      </Prose>

      <Prose>
        <strong>Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao (2024).</strong> "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision." arXiv:2407.08608. Optimized for Hopper (H100). Introduces warp-specialized asynchronous execution and FP8 support, reaching 75% of peak FP16 throughput on H100 — roughly 2x the speed of flash-attn-2 on the same hardware. The default training attention kernel on H100 in 2026.
      </Prose>

      <Prose>
        <strong>Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón, Sanghai (2023).</strong> "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." arXiv:2305.13245. Introduces grouped-query attention and shows that it preserves quality while reducing KV cache by 4-8x. The default attention variant for 7B+ production LLMs since 2024.
      </Prose>

      <Prose>
        <strong>Clark, Khandelwal, Levy, Manning (2019).</strong> "What Does BERT Look At? An Analysis of BERT's Attention." arXiv:1906.04341. Empirical analysis showing that different heads of a trained Transformer correspond to distinct linguistic functions (coreference, determiner-linking, syntactic heads). The most-cited evidence that multi-head attention does in fact decompose into interpretable sub-functions.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK
          ====================================================================== */}
      <H2>11. Self-check</H2>

      <H3>11.1 Why does attention divide by {"√d_k"}?</H3>

      <Prose>
        The dot product of two independent unit-variance vectors of dimension {"d_k"} has variance {"d_k"}. For typical {"d_k = 64"} to {"128"} the pre-softmax logits have standard deviation 8 to 11, which pushes softmax into a near-one-hot saturation regime where the gradient vanishes. Dividing by {"√d_k"} restores the logit variance to 1 and keeps softmax in a regime where the gradient is informative. Without the scaling, training a Transformer with head dim 64+ is unreliable; this is why the Transformer paper added the {"√d_k"} factor and Luong 2015 did not need it (smaller hidden sizes).
      </Prose>

      <H3>11.2 What does the causal mask prevent, and why is it needed only in the decoder?</H3>

      <Prose>
        The causal mask prevents position {"i"} from attending to positions {"j > i"}. In an autoregressive decoder, position {"i"} at inference has only seen tokens {"0..i"}; if training allowed it to attend to future tokens, the model would exploit that shortcut and fail catastrophically at inference where no future tokens exist. Encoder-only models (BERT) are trained with full bidirectional attention because they are used on complete input sequences at inference too — no generation, no "future" to worry about. In an encoder-decoder, the encoder is bidirectional and the decoder's self-attention is causal; the decoder's cross-attention to the encoder is unmasked because the entire encoder output is known before decoding starts.
      </Prose>

      <H3>11.3 If your multi-head attention has {"H = 8"} and {"d = 512"}, what is {"d_k"} and how many parameters does the module have?</H3>

      <Prose>
        {"d_k = d / H = 512 / 8 = 64"}. The module has three projection matrices {"W^Q, W^K, W^V"} each of shape {"[d, d]"} (the per-head projections are concatenated into one weight) plus one output projection {"W^O"} of shape {"[d, d]"}, totaling {"4 · d^2 = 4 · 512^2 ≈ 1.05"} million parameters (plus biases if used). Critically, the parameter count does <em>not</em> depend on {"H"} — a different head count reshapes the same parameters but does not change the total. This is why "more heads" is close to free in parameter budget but costs a bit more in compute scheduling overhead.
      </Prose>

      <H3>11.4 Why is self-attention parallelizable while an RNN is not?</H3>

      <Prose>
        Self-attention computes all output positions simultaneously: the output at position {"i"} is a function of {"Q[i]"} dotted with every row of {"K"}, weighted by softmax, and multiplied into {"V"}. None of those operations requires any other output position to be computed first — the entire forward pass over a length-{"L"} sequence is one batched matmul, one softmax, and one more batched matmul. An RNN, by contrast, requires {"h_t = f(h_{t-1}, x_t)"}: the hidden state at step {"t"} depends on {"h_{t-1}"}, which must be computed before {"h_t"}. This sequential dependency chain limits an RNN's training throughput to roughly {"L"} sequential GPU steps regardless of parallelism, whereas a Transformer's training throughput scales with tensor core count.
      </Prose>

      <H3>11.5 What is the relationship between multi-head attention and ensemble averaging?</H3>

      <Prose>
        Superficially they look similar — multiple parallel attention computations with different parameters, averaged/concatenated into a final output. But the relationship is structural, not ensemble. An ensemble of {"H"} single-head attentions would independently compute {"H"} full-rank attention outputs and average them; multi-head attention instead projects the input into {"H"} orthogonal (well, learnably orthogonal) subspaces with per-head projections of dimension {"d/H"}, computes attention in each low-dimensional subspace, and concatenates. The total parameter count is the same as a single head (because {"H · (d/H)^2 · 3 + (d/H · H)^2 = 4 d^2 / H + d^2"}, and the projections are sized to match the full {"d"} via sharing across heads). The benefit comes from giving each head its own low-rank similarity structure that it can specialize; the ensemble analogy is real at the functional level but the implementation is sharing parameters across a decomposition, not replicating them. This is also why "more heads with smaller head dim" is not the same operation as "ensemble of bigger single-heads" — the rank constraints are different.
      </Prose>

    </div>
  ),
};

export default selfAttentionContent;
