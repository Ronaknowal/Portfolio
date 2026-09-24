import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const interleavedCrossAttentionContent = {
  title: "Interleaved / Cross-Attention Architectures",
  readTime: "~38 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        The original Transformer (Vaswani et al., "Attention Is All You Need," arXiv:1706.03762, 2017) was an encoder-decoder: an encoder ingested the source sentence and a decoder generated the target sentence one token at a time. Self-attention ran inside each stack, but the bridge between them was <em>cross-attention</em> — a mechanism in which the queries came from the decoder's hidden states and the keys and values came from the encoder's final representations. For seven years after Vaswani, as the field pivoted to decoder-only language models (GPT-2, GPT-3, LLaMA), cross-attention faded from the default architecture. Self-attention over a single sequence was enough. You did not need two streams.
      </Prose>

      <Prose>
        Cross-attention came back when the field started stitching pretrained LLMs to pretrained vision encoders. The core question of multimodal learning in 2022-2024 was: how do you inject a {"224 × 224"} image — which, as a ViT patch grid, is 256 to 4096 visual tokens — into a language model that has already been trained for trillions of tokens on pure text? Three answers emerged, each with a different use of attention. Jaegle et al.'s Perceiver (arXiv:2103.03206, 2021) proposed a <em>latent-bottleneck</em> architecture in which a small fixed-size latent array (typically 256 to 1024 latents) cross-attends to an arbitrarily long input (10,000+ tokens) once, then runs self-attention on the latents. This decouples compute from input length and was originally motivated by perception tasks with multi-modal inputs (images, audio, point clouds). Perceiver IO (arXiv:2107.14795, 2022) generalised the output side with a symmetric decoder cross-attention.
      </Prose>

      <Prose>
        Alayrac et al.'s Flamingo (arXiv:2204.14198, 2022) took a different path. Start with a frozen Chinchilla-70B language model. Insert new <em>gated cross-attention</em> layers at regular intervals between the pretrained transformer blocks. The new layers cross-attend from the LM's hidden states to visual features produced by a frozen vision encoder (NFNet-F6) and passed through a small Perceiver Resampler. The key design choice was the <Code>{"tanh(α)"}</Code> gate on each new cross-attention and FFN sub-layer, initialised to zero. At step 0 the gate is exactly a no-op, so the frozen LLM's behavior is identical to its pretrained state. Training then opens the gates gradually, letting visual information flow into the LM without the early-training catastrophe that would happen if vision features were injected at full strength from the start. Flamingo is the canonical example of adding a new modality to a frozen LLM with cross-attention.
      </Prose>

      <Prose>
        Liu et al.'s LLaVA (arXiv:2304.08485, 2023) flipped the design once more. Rather than adding new cross-attention layers, LLaVA's recipe is almost embarrassingly simple: run CLIP-ViT-L/14 to get 256 image patch features, pass them through a single linear projection (later versions use a 2-layer MLP) to map from the 1024-d CLIP space to the LLM's 4096-d embedding space, and <em>concatenate</em> the projected visual tokens with the text tokens into one sequence. Run the LLM's native self-attention over the joined sequence. No new attention layers, no gates — just treat visual tokens like text tokens. This is the <em>interleaved</em> or <em>early-fusion</em> design. It works shockingly well: LLaVA-1.5 was trained with a 7B LLaMA-2 backbone, a trivial projector, and roughly 1M image-instruction pairs, and it matched or beat more elaborate VLMs on most VQA benchmarks (the original spec says ~1M image-instruction pairs).
      </Prose>

      <Prose>
        Li et al.'s BLIP-2 (arXiv:2301.12597, 2023) lived between the two extremes. The Q-Former is a small transformer (roughly 100M parameters) with a set of {"K"} learnable query tokens (32 in BLIP-2) that cross-attend to the vision encoder's output, then cross-attend to text, repeating for several layers. The output is {"K"} visual tokens in the LLM's embedding space, which are prepended to the text prompt (interleaved style, but with a learned compression stage before concatenation). The advantage is that the Q-Former decouples vision encoder and LLM training — you can swap either and keep the Q-Former as the adapter.
      </Prose>

      <Prose>
        By 2024 the field had converged on three dominant patterns. <em>Interleaved</em> (LLaVA, PaliGemma, Qwen-VL, Molmo, Gemma-3-VL): project visual tokens into the LM embedding space and concatenate. Simple, scales to many frames, requires long context. <em>Gated cross-attention</em> (Flamingo, Idefics, Idefics2): new cross-attention layers over a frozen or partially-frozen LLM. Good when you want to preserve language capabilities. <em>Query-compressed cross-attention</em> (BLIP-2, many follow-ups): a Q-Former or similar learned compressor reduces variable-length visual features to a fixed small set before they touch the LM. All frontier multimodal systems — GPT-4V, Claude 3.5 Sonnet, Gemini 1.5, PaLI-3 (Chen et al., arXiv:2310.09199, 2023) — are variants of these three patterns, often combined (Gemini uses interleaved with audio and video, PaLI uses joint vision-text encoder with cross-modal attention). Understanding interleaved and cross-attention architectures is understanding how every current VLM works.
      </Prose>

      <Callout accent="gold">
        Cross-attention is the mechanism that lets one sequence query another. Interleaving is the choice to not use a separate cross-attention at all — instead, make the two sequences one and let self-attention handle fusion. Gated cross-attention is the compromise: new cross-attention layers with a learned gate so the frozen LLM is preserved. Every multimodal LLM you have used in 2025 is one of these, or a combination.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <H3>2.1 Self-attention vs cross-attention</H3>

      <Prose>
        Self-attention computes {"Attn(Q, K, V)"} where {"Q, K, V"} are all linear projections of the <em>same</em> sequence. Every token in the sequence looks at every other token (subject to masking). Cross-attention computes {"Attn(Q, K, V)"} where {"Q"} comes from one sequence (the "query source") and {"K, V"} come from a different sequence (the "context"). The cardinality of Q and of K,V can differ: {"L_q"} queries attending to {"L_k"} context tokens produces an attention matrix of shape {"[L_q, L_k]"} and output of shape {"[L_q, d]"}. This is the Vaswani 2017 decoder block: each decoder position's query attends to the encoder's full sequence.
      </Prose>

      <Prose>
        The generalisation is that cross-attention is an <em>information bridge</em> between two spaces. If the two spaces are the same modality (encoder-decoder machine translation), cross-attention fuses source with target. If they are different modalities (visual features and language hidden states), cross-attention fuses vision with text. The same formula, different interpretation.
      </Prose>

      <H3>2.2 The interleaving alternative</H3>

      <Prose>
        There is a trivial way to avoid cross-attention: concatenate the two sequences and run self-attention on the whole thing. If you have 256 visual tokens {"v_1, ..., v_{256}"} and 100 text tokens {"t_1, ..., t_{100}"}, build the sequence {"[v_1, ..., v_{256}, t_1, ..., t_{100}]"} and apply causal self-attention over the 356-token sequence. Each text token's attention query naturally sees both earlier text and all visual tokens (because they are earlier in the sequence). Visual tokens attend to each other (they are before any text). This is the LLaVA design, and it has a lovely property: it requires no new parameters inside the LLM. The only new parameters are in the vision encoder and the small projector that maps vision features into the LLM's embedding space.
      </Prose>

      <Prose>
        The cost is sequence length. A frozen LLaMA-2-7B has a trained context of 4k tokens; adding 576 CLIP tokens per image eats a large chunk. LLaVA-NeXT (2024) pushes per-image tokens to 2880 by feeding multiple image crops, which multiplies the cost again. A single image with four {"336 × 336"} crops plus the original consumes roughly 2880 tokens. Video becomes prohibitive fast. The classical cross-attention pattern sidesteps this: visual features are held once, not copied into the autoregressive KV cache, and text tokens cross-attend to them by explicit reference.
      </Prose>

      <H3>2.3 The Perceiver latent bottleneck</H3>

      <Prose>
        Perceiver's move: if your input has {"N"} tokens ({"N"} large — 50k, 100k, millions for high-res images or long audio) and you want to run a deep transformer over it, the self-attention cost {"O(N^2)"} is infeasible. Instead, introduce a learned latent array {"L ∈ R^{M × d}"} with {"M ≪ N"} (typically {"M = 256"} or {"512"}). Run a single cross-attention in which the latents are the query source and the input is the K,V context. That costs {"O(M · N)"} — linear in the input. Now run {"D"} iterations of self-attention on the latents (cost {"O(M^2)"} each). The deep processing happens on the latents; the input is only touched once.
      </Prose>

      <Prose>
        The mental model: Perceiver replaces the input-length-sized scratchpad that self-attention normally builds with a fixed-size bottleneck. Information has to flow through that bottleneck, which forces the model to be selective. The design is not specific to any modality — Perceiver papers show the same architecture working on ImageNet, AudioSet, ModelNet (3D point clouds), and Kinetics (video), with almost no modality-specific code.
      </Prose>

      <H3>2.4 The Flamingo gated cross-attention</H3>

      <Prose>
        The Flamingo insight: if you want to add vision to a pretrained LLM without degrading its language ability, you must not disturb the pretrained weights at step 0. So freeze them. Then insert new trainable cross-attention layers between the existing transformer blocks. Each new layer computes {"x ← x + \\tanh(α_{xattn}) \\cdot CrossAttn(x, v)"} followed by {"x ← x + \\tanh(α_{ffn}) \\cdot FFN(x)"}, where {"α"} is a learnable scalar initialised to zero. Because {"\\tanh(0) = 0"}, the new layers are the identity at initialisation — the LLM behaves exactly as it did before. Training gradually opens the gates and lets vision features modulate the LM's hidden states.
      </Prose>

      <Prose>
        The gated-zero-init trick generalises: it is how you add any new capability to a frozen model without training instability. It shows up in ControlNet for diffusion models, in LoRA adapters (zero-init one of the two matrices), in LLaVA-style projectors (initialise the projector to produce small outputs), and in Adapter layers (Houlsby et al., 2019). The unifying idea is that if the new module starts as a no-op, gradient descent can safely explore non-zero values without catastrophic early-training drift.
      </Prose>

      <H3>2.5 Q-Former: a cross-attention compressor</H3>

      <Prose>
        BLIP-2's Q-Former is a small transformer whose inputs are {"K"} learnable "query" tokens and whose keys-and-values are the vision encoder's output features. The queries self-attend to each other and cross-attend to the vision features for several layers, producing {"K"} output vectors that summarise the image. These {"K"} vectors are then projected into the LLM's embedding space and prepended to the text — interleaved-style. The Q-Former is the learned compressor that sits between the two frozen modules (vision encoder, LLM). Its job is to turn variable-length visual features into a fixed-length representation the LLM can consume.
      </Prose>

      <Prose>
        Compared to LLaVA's linear projector, the Q-Former is more expressive (a full transformer vs one linear layer) and produces fewer tokens (32 vs 576 for a ViT-L grid), which saves LLM context but requires more training to learn good queries. Compared to Flamingo's Perceiver Resampler (a similar but smaller module), the Q-Former has an extra text branch in the vision-language pretraining stage that the Resampler does not. The choice between Q-Former, Perceiver Resampler, and linear projector is the core design axis of the VLM adapter.
      </Prose>

      <H3>2.6 The mental model</H3>

      <Prose>
        Cross-attention is a function {"(query_sequence, context_sequence) → output_sequence"} with the same length as the queries. Interleaving is the choice to put both sequences into the same self-attention by concatenating them. Gated cross-attention adds a zero-init scalar so a new cross-attention can be bolted onto a frozen backbone without initial disturbance. Perceiver uses cross-attention as a bottleneck from large input to small latent. Q-Former uses cross-attention as a compression stage before interleaving into an LLM. Four design patterns, one underlying mechanism.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 Standard cross-attention</H3>

      <Prose>
        Given a query sequence {"X_q ∈ R^{L_q × d_q}"} and a context sequence {"X_c ∈ R^{L_c × d_c}"}, with projection matrices {"W^Q ∈ R^{d_q × d}"}, {"W^K ∈ R^{d_c × d}"}, {"W^V ∈ R^{d_c × d}"}, define
      </Prose>

      <MathBlock>{"Q = X_q W^Q, \\quad K = X_c W^K, \\quad V = X_c W^V"}</MathBlock>

      <Prose>
        The attention score matrix is {"S ∈ R^{L_q × L_c}"}:
      </Prose>

      <MathBlock>{"S_{ij} = \\frac{Q_i \\cdot K_j}{\\sqrt{d_h}}, \\quad A = \\mathrm{softmax}_{j}(S), \\quad O = A V"}</MathBlock>

      <Prose>
        with multi-head extension: reshape {"Q, K, V"} to {"[L, H, d_h]"}, compute per-head attention independently, concatenate and project through {"W^O"}. Crucially, nothing requires {"L_q = L_c"} or {"d_q = d_c"} — the dimensions can be completely different, which is what lets cross-attention bridge modalities. The Vaswani 2017 decoder uses this with {"X_q"} = decoder hidden states and {"X_c"} = encoder output; Flamingo uses it with {"X_q"} = LM hidden states and {"X_c"} = Perceiver-Resampled visual features.
      </Prose>

      <H3>3.2 Interleaved self-attention</H3>

      <Prose>
        Interleaving is not a new operation — it is a choice of input. Given visual tokens {"V ∈ R^{L_v × d}"} (already projected into the LM embedding space by some projector) and text tokens {"T ∈ R^{L_t × d}"}, concatenate:
      </Prose>

      <MathBlock>{"X = [V; T] \\in \\mathbb{R}^{(L_v + L_t) \\times d}"}</MathBlock>

      <Prose>
        Run standard multi-head self-attention with a causal mask:
      </Prose>

      <MathBlock>{"\\mathrm{SelfAttn}(X) = \\mathrm{softmax}\\left(\\frac{X W^Q (X W^K)^\\top}{\\sqrt{d_h}} + M\\right) X W^V"}</MathBlock>

      <Prose>
        where {"M_{ij} = 0"} for {"i ≥ j"} and {"-\\infty"} otherwise. Visual tokens are at positions {"1, ..., L_v"} so they only attend to earlier visual tokens (and themselves). Text tokens at positions {"L_v + 1, ..., L_v + L_t"} attend to all visual tokens and earlier text. This is the LLaVA formulation; no new equations, just a particular concatenation.
      </Prose>

      <H3>3.3 Perceiver: iterated latent refinement</H3>

      <Prose>
        Let {"L^{(0)} ∈ R^{M × d}"} be the learned initial latent array. Given input {"X ∈ R^{N × d'}"} with {"N ≫ M"}:
      </Prose>

      <MathBlock>{"L^{(t+\\frac{1}{2})} = L^{(t)} + \\mathrm{CrossAttn}(L^{(t)}, X)"}</MathBlock>

      <MathBlock>{"L^{(t+1)} = L^{(t+\\frac{1}{2})} + \\mathrm{SelfAttn}(L^{(t+\\frac{1}{2})}) + \\mathrm{FFN}(\\cdot)"}</MathBlock>

      <Prose>
        iterated for {"t = 0, 1, ..., D-1"} rounds. The cross-attention cost at each round is {"O(M · N · d)"} — linear in input length. The self-attention cost is {"O(M^2 · d)"} — independent of input length. The original Perceiver paper shares weights across rounds to keep parameter count tight; Perceiver IO drops this to allow different processing depths. The output is either the final latents or an optional output cross-attention {"Y = \\mathrm{CrossAttn}(Y_{\\text{query}}, L^{(D)})"} where {"Y_{\\text{query}}"} is a per-task query array.
      </Prose>

      <H3>3.4 Flamingo gated cross-attention</H3>

      <Prose>
        For an LM hidden state {"x ∈ R^{L \\times d}"} and visual context {"v ∈ R^{L_v \\times d_v}"}, the gated cross-attention block is
      </Prose>

      <MathBlock>{"x \\leftarrow x + \\tanh(\\alpha_{\\text{xattn}}) \\cdot \\mathrm{CrossAttn}(\\mathrm{LN}(x), \\mathrm{LN}(v))"}</MathBlock>

      <MathBlock>{"x \\leftarrow x + \\tanh(\\alpha_{\\text{ffn}}) \\cdot \\mathrm{FFN}(\\mathrm{LN}(x))"}</MathBlock>

      <Prose>
        with {"α_{\\text{xattn}}, α_{\\text{ffn}} ∈ R"} learned scalars initialised to 0. At initialisation both gates are {"\\tanh(0) = 0"}, so the block is exactly the identity and the surrounding frozen LLM is untouched. The scalar gate is tiny (one float per layer) but essential: Flamingo's ablations show that removing it and using a normal residual causes the LLM's language scores to collapse in the first 100 training steps, because the randomly initialised cross-attention dumps noise into the hidden stream.
      </Prose>

      <Prose>
        A subtle variant: Flamingo uses a per-head gate in some versions and a single scalar per layer in others. The original paper (Alayrac 2022) uses scalar per sub-layer (one for cross-attn, one for FFN). Idefics2 (Laurençon et al. arXiv:2405.02246, 2024) uses scalar per sub-layer. Both work; per-head gives a small additional flexibility but is rarely worth the complexity.
      </Prose>

      <H3>3.5 BLIP-2 Q-Former</H3>

      <Prose>
        The Q-Former has {"K"} learned query tokens {"Q ∈ R^{K × d}"} (typically {"K = 32, d = 768"}). Given vision features {"V ∈ R^{L_v × d_v}"}:
      </Prose>

      <MathBlock>{"Q^{(t+\\frac{1}{3})} = Q^{(t)} + \\mathrm{SelfAttn}(\\mathrm{LN}(Q^{(t)}))"}</MathBlock>

      <MathBlock>{"Q^{(t+\\frac{2}{3})} = Q^{(t+\\frac{1}{3})} + \\mathrm{CrossAttn}(\\mathrm{LN}(Q^{(t+\\frac{1}{3})}), \\mathrm{LN}(V))"}</MathBlock>

      <MathBlock>{"Q^{(t+1)} = Q^{(t+\\frac{2}{3})} + \\mathrm{FFN}(\\mathrm{LN}(Q^{(t+\\frac{2}{3})}))"}</MathBlock>

      <Prose>
        for {"t = 0, ..., D-1"} layers (BLIP-2 uses {"D = 12"}). Every other layer has cross-attention; the rest are self-attention + FFN only, to limit parameter count. The output {"Q^{(D)} ∈ R^{K × d}"} is then projected into the LLM's embedding space via a single linear layer and prepended as {"K"} soft tokens to the text prompt.
      </Prose>

      <H3>3.6 Interleaved FLOPs vs cross-attention FLOPs</H3>

      <Prose>
        For a single forward pass through one transformer layer: interleaved self-attention over {"L_v + L_t"} tokens costs {"O((L_v + L_t)^2 · d)"} for the attention matrix. Cross-attention from {"L_t"} text queries to {"L_v"} visual K,V costs {"O(L_t · L_v · d)"} plus a separate self-attention over text at {"O(L_t^2 · d)"}. If {"L_v ≈ L_t"} the interleaved version is {"∼4 ×"} more expensive (because {"(L_v + L_t)^2 = L_v^2 + L_t^2 + 2 L_v L_t"}); if {"L_v ≫ L_t"} interleaved is dominated by {"L_v^2"} while cross-attention is dominated by {"L_v · L_t"}, a factor of {"L_v / L_t"} advantage to cross-attention.
      </Prose>

      <MathBlock>{"\\frac{\\mathrm{FLOPs}_{\\text{interleaved}}}{\\mathrm{FLOPs}_{\\text{cross-attn}}} \\approx \\frac{(L_v + L_t)^2}{L_t (L_v + L_t) + L_v \\cdot d_{\\text{share}}}"}</MathBlock>

      <Prose>
        The cache difference matters more. Interleaving requires the LLM's KV cache to hold entries for every visual token at every layer, because each text token's future attention will look back at them. Cross-attention holds visual features once (not replicated across layers, not entered into the text KV cache). For video with 16 frames at 576 tokens each, interleaving adds {"16 × 576 × 2 × n_{\\text{layers}} × d"} bytes to the KV cache, which at n_layers = 32, d = 4096, fp16 is 4.5 GB per video sample per sequence. Cross-attention holds the same 9216 visual tokens once, at {"9216 × d_{\\text{vis}} × 2"} bytes = 36 MB. Two orders of magnitude.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        Everything below was run on PyTorch 2.6. Every {"# Output:"} block is real stdout. We implement each of the four canonical patterns — standard cross-attention, Perceiver, gated cross-attention, and interleaved self-attention with modality tokens — and then train the last two on a toy copy task to verify that visual information actually reaches the text head.
      </Prose>

      <H3>4.1 Standard cross-attention (Vaswani encoder-decoder style)</H3>

      <CodeBlock language="python">
{`import math, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)

class CrossAttention(nn.Module):
    def __init__(self, d_model=64, n_heads=4):
        super().__init__()
        self.h  = n_heads
        self.dh = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x_dec, x_enc, mask=None):
        B, Ld, D = x_dec.shape
        _, Le, _ = x_enc.shape
        Q = self.Wq(x_dec).view(B, Ld, self.h, self.dh).transpose(1, 2)
        K = self.Wk(x_enc).view(B, Le, self.h, self.dh).transpose(1, 2)
        V = self.Wv(x_enc).view(B, Le, self.h, self.dh).transpose(1, 2)
        s = (Q @ K.transpose(-2, -1)) / math.sqrt(self.dh)
        if mask is not None:
            s = s.masked_fill(~mask, float('-inf'))
        a = s.softmax(-1)
        o = (a @ V).transpose(1, 2).reshape(B, Ld, D)
        return self.Wo(o), a

ca = CrossAttention(64, 4)
x_dec = torch.randn(2, 6, 64)
x_enc = torch.randn(2, 10, 64)
out, attn = ca(x_dec, x_enc)
print("x_dec:", tuple(x_dec.shape))
print("x_enc:", tuple(x_enc.shape))
print("out  :", tuple(out.shape))
print("attn :", tuple(attn.shape), "row sums to:", round(float(attn[0,0,0].sum()), 4))

# Output:
#   x_dec: (2, 6, 64)
#   x_enc: (2, 10, 64)
#   out  : (2, 6, 64)
#   attn : (2, 4, 6, 10) row sums to: 1.0`}
      </CodeBlock>

      <Prose>
        Note that Q is derived from {"x_{dec}"} (6 tokens) and K, V from {"x_{enc}"} (10 tokens). The attention matrix is {"[6 \\times 10]"} per head — rectangular because the two sequences have different lengths. Softmax is over the K axis, and each query row sums to 1. This is the Vaswani 2017 decoder's middle sub-layer.
      </Prose>

      <H3>4.2 Perceiver encoder with latent bottleneck</H3>

      <CodeBlock language="python">
{`def mha(Q, K, V, h):
    B, Lq, D = Q.shape
    _, Lk, _ = K.shape
    dh = D // h
    def split(x, L): return x.view(B, L, h, dh).transpose(1, 2)
    q, k, v = split(Q, Lq), split(K, Lk), split(V, Lk)
    s = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
    o = (s.softmax(-1) @ v).transpose(1, 2).reshape(B, Lq, D)
    return o

class PerceiverBlock(nn.Module):
    def __init__(self, d=64, h=4):
        super().__init__()
        self.h = h
        self.ln_l1, self.ln_l2, self.ln_i = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.Wq_c, self.Wk_c, self.Wv_c = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.Wo_c = nn.Linear(d, d, bias=False)
        self.Wq_s, self.Wk_s, self.Wv_s = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.Wo_s = nn.Linear(d, d, bias=False)
        self.mlp   = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        self.ln_mlp = nn.LayerNorm(d)

    def forward(self, latents, inputs):
        # cross-attn: latents (query) -> inputs (context)
        Ln = self.ln_l1(latents); In = self.ln_i(inputs)
        latents = latents + self.Wo_c(mha(self.Wq_c(Ln), self.Wk_c(In), self.Wv_c(In), self.h))
        # self-attn on latents
        Ln = self.ln_l2(latents)
        latents = latents + self.Wo_s(mha(self.Wq_s(Ln), self.Wk_s(Ln), self.Wv_s(Ln), self.h))
        latents = latents + self.mlp(self.ln_mlp(latents))
        return latents

class Perceiver(nn.Module):
    def __init__(self, d=64, h=4, n_latents=16, depth=3):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(n_latents, d) * 0.02)
        self.blocks  = nn.ModuleList([PerceiverBlock(d, h) for _ in range(depth)])
    def forward(self, inputs):
        B = inputs.shape[0]
        lat = self.latents.unsqueeze(0).expand(B, -1, -1).contiguous()
        for blk in self.blocks:
            lat = blk(lat, inputs)
        return lat

model = Perceiver(d=64, h=4, n_latents=16, depth=3)
inputs = torch.randn(2, 2048, 64)
out = model(inputs)
print("inputs :", tuple(inputs.shape), "  latents:", (2, 16, 64))
print("output :", tuple(out.shape))
print("params :", sum(p.numel() for p in model.parameters()))
print("cross-attn FLOPs ~ M*N =", 16*2048, "  self-attn would be N*N =", 2048*2048)

# Output:
#   inputs : (2, 2048, 64)   latents: (2, 16, 64)
#   output : (2, 16, 64)
#   params : 200128
#   cross-attn FLOPs ~ M*N = 32768   self-attn would be N*N = 4194304`}
      </CodeBlock>

      <Prose>
        2048 input tokens — representing a high-resolution image's patches or a long audio clip — are processed through 3 rounds of Perceiver blocks. Each round does one cross-attention (16 queries vs 2048 context) and one self-attention (16 vs 16). The output is 16 latent vectors in which information from the 2048 inputs has been distilled. The FLOP count is dominated by {"M \\cdot N = 32\\,768"} rather than {"N^2 = 4.2M"} — a 128x saving vs plain self-attention on the full input.
      </Prose>

      <H3>4.3 Flamingo-style gated cross-attention with zero-init</H3>

      <CodeBlock language="python">
{`class GatedCrossAttention(nn.Module):
    def __init__(self, d=64, h=4, d_vis=64):
        super().__init__()
        self.h = h
        self.ln_x = nn.LayerNorm(d)
        self.ln_v = nn.LayerNorm(d_vis)
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d_vis, d, bias=False)
        self.Wv = nn.Linear(d_vis, d, bias=False)
        self.Wo = nn.Linear(d, d, bias=False)
        self.ffn = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        # zero-init gates — tanh(0) = 0 so layer is identity at step 0
        self.alpha_xattn = nn.Parameter(torch.zeros(1))
        self.alpha_ffn   = nn.Parameter(torch.zeros(1))

    def forward(self, x, vis):
        xn = self.ln_x(x); vn = self.ln_v(vis)
        attn = self.Wo(mha(self.Wq(xn), self.Wk(vn), self.Wv(vn), self.h))
        x = x + torch.tanh(self.alpha_xattn) * attn
        x = x + torch.tanh(self.alpha_ffn) * self.ffn(self.ln_x(x))
        return x

layer = GatedCrossAttention(d=64, h=4, d_vis=64)
x   = torch.randn(2, 10, 64)       # LLM hidden states
vis = torch.randn(2,  4, 64)       # compressed visual features

y0 = layer(x, vis)
print("at init: out - x max diff =", round(float((y0 - x).abs().max()), 8))
print("         (tanh(0)=0  =>  layer is identity at step 0)")

layer.alpha_xattn.data.fill_(0.5); layer.alpha_ffn.data.fill_(0.5)
y1 = layer(x, vis)
print("after opening gates (alpha=0.5): out - x max diff =", round(float((y1 - x).abs().max()), 4))

# Output:
#   at init: out - x max diff = 0.0
#            (tanh(0)=0  =>  layer is identity at step 0)
#   after opening gates (alpha=0.5): out - x max diff = 0.4476`}
      </CodeBlock>

      <Prose>
        The first output confirms that at initialisation the gated block passes {"x"} through unchanged to numerical precision — the frozen LLM's behavior is identical to pre-insertion. Once the gate opens (here I set {"α = 0.5"} by hand to simulate a few training steps), visual information starts flowing into {"x"}. Flamingo's training drives the gate from zero to order-1 values over the first few hundred thousand training steps.
      </Prose>

      <H3>4.4 BLIP-2 Q-Former</H3>

      <CodeBlock language="python">
{`class QFormerBlock(nn.Module):
    def __init__(self, d=64, h=4, d_img=128):
        super().__init__()
        self.h = h
        self.Wqkv_self = nn.Linear(d, 3*d, bias=False)
        self.Wo_self   = nn.Linear(d, d, bias=False)
        self.ln_self   = nn.LayerNorm(d)
        self.Wq_cross  = nn.Linear(d, d, bias=False)
        self.Wk_cross  = nn.Linear(d_img, d, bias=False)
        self.Wv_cross  = nn.Linear(d_img, d, bias=False)
        self.Wo_cross  = nn.Linear(d, d, bias=False)
        self.ln_cross  = nn.LayerNorm(d)
        self.ln_img    = nn.LayerNorm(d_img)
        self.mlp       = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        self.ln_mlp    = nn.LayerNorm(d)

    def forward(self, q, img):
        qn = self.ln_self(q)
        Q, K, V = self.Wqkv_self(qn).chunk(3, -1)
        q = q + self.Wo_self(mha(Q, K, V, self.h))
        qn = self.ln_cross(q); inn = self.ln_img(img)
        q = q + self.Wo_cross(mha(self.Wq_cross(qn), self.Wk_cross(inn), self.Wv_cross(inn), self.h))
        q = q + self.mlp(self.ln_mlp(q))
        return q

class QFormer(nn.Module):
    def __init__(self, n_queries=32, d=64, h=4, depth=2, d_img=128):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_queries, d) * 0.02)
        self.blocks  = nn.ModuleList([QFormerBlock(d, h, d_img) for _ in range(depth)])
    def forward(self, img):
        B = img.shape[0]
        q = self.queries.unsqueeze(0).expand(B, -1, -1).contiguous()
        for b in self.blocks: q = b(q, img)
        return q

qf = QFormer(n_queries=32, d=64, h=4, depth=2, d_img=128)
img = torch.randn(2, 196, 128)      # 14x14 ViT patches, d_img=128
out = qf(img)
print("image feats :", tuple(img.shape), " (196 patches, 128-d)")
print("32 learnable queries ->", tuple(out.shape))

# Output:
#   image feats : (2, 196, 128)  (196 patches, 128-d)
#   32 learnable queries -> (2, 32, 64)`}
      </CodeBlock>

      <Prose>
        32 learnable queries compress 196 ViT-L/14 patch features into 32 soft tokens in the LLM's embedding dimension. In BLIP-2 those 32 tokens are then prepended to the text prompt, so the downstream LLM sees 32 "visual" tokens + the actual text. The Q-Former has two training stages in the paper: first a vision-language contrastive + matching + captioning loss (like BLIP-1), then a generative loss with the frozen LLM attached.
      </Prose>

      <H3>4.5 Interleaved VLM (LLaVA style)</H3>

      <CodeBlock language="python">
{`class CausalMHA(nn.Module):
    def __init__(self, d=64, h=4):
        super().__init__()
        self.h, self.dh = h, d // h
        self.Wqkv = nn.Linear(d, 3*d, bias=False)
        self.Wo   = nn.Linear(d, d, bias=False)
    def forward(self, x, mask):
        B, L, D = x.shape
        qkv = self.Wqkv(x).view(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        s = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        s = s.masked_fill(~mask, float('-inf'))
        return self.Wo((s.softmax(-1) @ v).transpose(1, 2).reshape(B, L, D))

class VisionProjector(nn.Module):
    def __init__(self, d_v=128, d=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_v, d), nn.GELU(), nn.Linear(d, d))
    def forward(self, v): return self.mlp(v)

class InterleavedVLM(nn.Module):
    def __init__(self, vocab=100, d=64, h=4, n_layers=2, d_v=128):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        # Learned modality boundary tokens, so the model can tell text from image
        self.img_begin = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.img_end   = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.proj      = VisionProjector(d_v, d)
        self.pos       = nn.Embedding(512, d)
        self.layers    = nn.ModuleList([CausalMHA(d, h) for _ in range(n_layers)])
        self.lns       = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.head      = nn.Linear(d, vocab, bias=False)

    def forward(self, text_ids, image_feats):
        B, Lt = text_ids.shape; _, Lv, _ = image_feats.shape
        img = self.proj(image_feats)
        beg = self.img_begin.expand(B, -1, -1); end = self.img_end.expand(B, -1, -1)
        txt = self.embed(text_ids)
        x = torch.cat([beg, img, end, txt], dim=1)
        L = x.shape[1]
        x = x + self.pos(torch.arange(L, device=x.device))
        mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=x.device))[None, None]
        for ln, lyr in zip(self.lns, self.layers):
            x = x + lyr(ln(x), mask)
        logits = self.head(x)
        return logits, (1 + Lv + 1)

model = InterleavedVLM(vocab=100, d=64, h=4, n_layers=2, d_v=128)
text = torch.randint(0, 100, (2, 8))
imgs = torch.randn(2, 16, 128)       # 16 image patches, CLIP-like
logits, text_offset = model(text, imgs)
print("text_ids    :", tuple(text.shape))
print("image_feats :", tuple(imgs.shape))
print("full seq len:", 1 + 16 + 1 + 8, "  (BOI + 16 patches + EOI + 8 text)")
print("logits      :", tuple(logits.shape))
print("text starts at position:", text_offset)
targets = torch.randint(0, 100, (2, 8))
text_logits = logits[:, text_offset:text_offset + 8]
loss = F.cross_entropy(text_logits.reshape(-1, 100), targets.reshape(-1))
loss.backward()
print("loss (random init):", round(float(loss), 4))
print("grad on projector last layer:", round(float(model.proj.mlp[-1].weight.grad.abs().mean()), 6))

# Output:
#   text_ids    : (2, 8)
#   image_feats : (2, 16, 128)
#   full seq len: 26   (BOI + 16 patches + EOI + 8 text)
#   logits      : (2, 26, 100)
#   text starts at position: 18
#   loss (random init): 5.0693
#   grad on projector last layer: 0.000767`}
      </CodeBlock>

      <Prose>
        The model takes 8 text tokens and 16 image patches, projects the patches, wraps them in learned boundary tokens {"[BOI]"} and {"[EOI]"}, and concatenates everything into a 26-token sequence. Standard causal self-attention runs over the whole thing. The loss is computed only on text positions (positions 18-25 in the concatenated sequence), and gradients do flow back through the projector — we see nonzero gradients on {"proj.mlp[-1]"}. This is LLaVA in miniature: the entire visual path fits in a dozen lines of PyTorch.
      </Prose>

      <H3>4.6 Train interleaved vs gated cross-attention on a copy task</H3>

      <CodeBlock language="python">
{`# Task: 4 image features encode a digit 0..9 (hot in dim = digit).
# Text target is to repeat the digit for 6 steps. Tests if visual signal reaches text head.

def make_batch(B=32, n_img=4, vocab=10, seq=6, d_v=32):
    digits = torch.randint(0, vocab, (B,))
    img = torch.zeros(B, n_img, d_v)
    for i, dg in enumerate(digits):
        img[i, :, :vocab] = 0.1 * torch.randn(n_img, vocab)
        img[i, :, int(dg)] += 2.0
    tgt = digits.unsqueeze(1).expand(B, seq).contiguous()
    inp = torch.full((B, seq), vocab)
    return img, inp, tgt

V, D, H, DV, N = 10, 32, 2, 32, 4

class Interleaved(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(V+1, D)
        self.proj  = nn.Linear(DV, D)
        self.Wqkv  = nn.Linear(D, 3*D, bias=False)
        self.Wo    = nn.Linear(D, D, bias=False)
        self.ln    = nn.LayerNorm(D)
        self.head  = nn.Linear(D, V, bias=False)
    def forward(self, img, inp):
        B, Lt = inp.shape
        x = torch.cat([self.proj(img), self.embed(inp)], 1)
        L = x.shape[1]; xn = self.ln(x)
        qkv = self.Wqkv(xn).view(B, L, 3, H, D//H).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        s = (q @ k.transpose(-2, -1)) / math.sqrt(D//H)
        m = torch.tril(torch.ones(L, L, dtype=torch.bool))[None, None]
        s = s.masked_fill(~m, float('-inf'))
        o = (s.softmax(-1) @ v).transpose(1, 2).reshape(B, L, D)
        x = x + self.Wo(o)
        return self.head(x[:, N:])

class GatedXAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(V+1, D)
        self.ln_v  = nn.LayerNorm(DV)
        self.Wq    = nn.Linear(D, D, bias=False)
        self.Wk    = nn.Linear(DV, D, bias=False)
        self.Wv    = nn.Linear(DV, D, bias=False)
        self.Wo    = nn.Linear(D, D, bias=False)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.head  = nn.Linear(D, V, bias=False)
    def forward(self, img, inp):
        B, Lt = inp.shape
        t = self.embed(inp); vn = self.ln_v(img)
        q = self.Wq(t).view(B, Lt, H, D//H).transpose(1, 2)
        k = self.Wk(vn).view(B, N, H, D//H).transpose(1, 2)
        v = self.Wv(vn).view(B, N, H, D//H).transpose(1, 2)
        s = (q @ k.transpose(-2, -1)) / math.sqrt(D//H)
        o = (s.softmax(-1) @ v).transpose(1, 2).reshape(B, Lt, D)
        t = t + torch.tanh(self.alpha) * self.Wo(o)
        return self.head(t)

def train(model, steps=400, lr=3e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr); hist = []
    for step in range(steps):
        img, inp, tgt = make_batch()
        loss = F.cross_entropy(model(img, inp).reshape(-1, V), tgt.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0 or step == steps - 1:
            hist.append((step, round(float(loss), 4)))
    return hist

torch.manual_seed(1); h_inter = train(Interleaved())
torch.manual_seed(1); h_gated = train(GatedXAttn())
print("interleaved (LLaVA-like):")
for s, l in h_inter: print(f"  step {s:>3}  loss {l}")
print("gated cross-attn (Flamingo-like):")
for s, l in h_gated: print(f"  step {s:>3}  loss {l}")

# Output:
#   interleaved (LLaVA-like):
#     step   0  loss 2.4789
#     step  50  loss 0.0054
#     step 100  loss 0.0016
#     step 150  loss 0.001
#     step 200  loss 0.0007
#     step 250  loss 0.0005
#     step 300  loss 0.0004
#     step 350  loss 0.0003
#     step 399  loss 0.0003
#   gated cross-attn (Flamingo-like):
#     step   0  loss 2.4533
#     step  50  loss 0.9625
#     step 100  loss 0.0042
#     step 150  loss 0.0021
#     step 200  loss 0.0013
#     step 250  loss 0.0009
#     step 300  loss 0.0007
#     step 350  loss 0.0005
#     step 399  loss 0.0005`}
      </CodeBlock>

      <Prose>
        Both models solve the copy task — visual information reaches the text head in both designs. The interleaved model converges slightly faster at first (by step 50 it is already near zero), because the full self-attention mechanism gives it direct access to visual tokens from the first layer. The gated cross-attention model lags at step 50 (loss 0.96) because the gate has to open first — you can see the gradient has to climb {"\\tanh(α)"} from zero before visual signal flows. Both end up equivalent by step 100. This is the training dynamic Flamingo papers discuss: gated architectures are slower to learn new information but safer for frozen backbones.
      </Prose>

      <H3>4.7 Memory budget comparison</H3>

      <CodeBlock language="python">
{`d = 4096; n_layers = 32; bytes_per = 2   # fp16 LLaMA-2-7B-ish
for patches in [64, 256, 576, 1024, 2048, 4096]:
    text_len = 512
    total = patches + text_len
    kv_inter = 2 * n_layers * d * total * bytes_per
    d_vis = 1024
    kv_xattn = 2 * n_layers * d * text_len * bytes_per + patches * d_vis * bytes_per
    print(f"patches={patches:>4}  interleave={kv_inter/1e6:7.1f} MB  x-attn={kv_xattn/1e6:7.1f} MB  ratio={kv_inter/kv_xattn:.2f}")

# Output:
#   patches=  64  interleave=  302.0 MB  x-attn=  268.6 MB  ratio=1.12
#   patches= 256  interleave=  402.7 MB  x-attn=  269.0 MB  ratio=1.50
#   patches= 576  interleave=  570.4 MB  x-attn=  269.6 MB  ratio=2.12
#   patches=1024  interleave=  805.3 MB  x-attn=  270.5 MB  ratio=2.98
#   patches=2048  interleave= 1342.2 MB  x-attn=  272.6 MB  ratio=4.92
#   patches=4096  interleave= 2415.9 MB  x-attn=  276.8 MB  ratio=8.73`}
      </CodeBlock>

      <Prose>
        For a LLaMA-2-7B-sized model, interleaved visual tokens add to the KV cache at every layer — 4096 patches balloon the cache to 2.4 GB per sample. Gated cross-attention holds the visual features once (outside the KV cache), so 4096 patches cost only 8 MB of additional memory vs text-only. The ratio grows from 1.1x at 64 patches to 8.7x at 4096 patches, reflecting the different scaling. For video and multi-image inputs, cross-attention's memory advantage becomes decisive; for a single image at 256-576 patches, interleaving is cheap enough to prefer.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION
          ====================================================================== */}
      <H2>5. Production</H2>

      <H3>5.1 HuggingFace Transformers VLM classes</H3>

      <Prose>
        The HuggingFace ecosystem organises multimodal models by architectural family. The three main classes, each corresponding to one of the patterns in Section 2:
      </Prose>

      <CodeBlock language="python">
{`# Interleaved (LLaVA): projector + concatenate into self-attn LLM
from transformers import LlavaForConditionalGeneration, AutoProcessor

model     = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

# Gated cross-attention (Flamingo-descendant)
from transformers import Idefics2ForConditionalGeneration, AutoProcessor as P2
model2 = Idefics2ForConditionalGeneration.from_pretrained("HuggingFaceM4/idefics2-8b")

# Q-Former compression (BLIP-2)
from transformers import Blip2ForConditionalGeneration, AutoProcessor as P3
model3 = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl")

# Interleaved multimodal (Qwen2-VL, native video support)
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor as P4
model4 = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")`}
      </CodeBlock>

      <Prose>
        <Code>{"LlavaForConditionalGeneration"}</Code> is the interleaved pattern: a CLIP-ViT vision encoder, a 2-layer MLP projector, and a LLaMA or Mistral LLM, all glued by concatenation. <Code>{"Idefics2ForConditionalGeneration"}</Code> is gated cross-attention in the Flamingo lineage, trained by HuggingFace as an open reproduction. <Code>{"Blip2ForConditionalGeneration"}</Code> uses a Q-Former between a frozen ViT-g and a Flan-T5 or OPT backbone. <Code>{"Qwen2VLForConditionalGeneration"}</Code> is the 2024 leading open VLM, interleaved with a vision encoder that handles arbitrary resolutions via "Naive Dynamic Resolution."
      </Prose>

      <H3>5.2 The canonical adapter recipe</H3>

      <Prose>
        The dominant recipe for training a new VLM in 2024-2026, seen in LLaVA-1.5, LLaVA-NeXT, Qwen-VL, Molmo, InternVL, and countless others, is:
      </Prose>

      <Prose>
        1. Take a frozen vision encoder (CLIP-ViT-L/14 at 336px, or SigLIP-SO400M, or InternViT-6B). Its output is a grid of patch features, typically 256-1024 tokens in dim 1024-1536.
      </Prose>

      <Prose>
        2. Train a small projector (1-2 layer MLP, sometimes with a depthwise convolution for spatial pooling) that maps the vision output dim to the LLM embedding dim. This projector is the only new parameters in stage 1.
      </Prose>

      <Prose>
        3. Stage 1: freeze everything except the projector. Train on ~500k to 1M image-caption pairs (LAION, COYO, Web-scale CC-3M). Loss is standard next-token prediction over the caption, conditioned on the (projected) visual tokens prepended. This teaches the projector to map vision into LLM-readable embeddings.
      </Prose>

      <Prose>
        4. Stage 2: unfreeze the LLM (sometimes also the vision encoder). Train on visual instruction data (LLaVA-665k, LAION-GPT4V, ShareGPT4V, etc.). Loss is next-token prediction over the assistant response. This teaches the whole stack to follow visual instructions.
      </Prose>

      <Prose>
        The whole process costs ~100-300 GPU-hours on 8×A100 for a 7B model and produces a VLM competitive with GPT-4V on most benchmarks. The critical thing is that stage 1 is cheap — most of the model is frozen, only the projector (40M-100M parameters) is training. Stage 2 is expensive but shorter because the projector already maps vision into the LLM's space.
      </Prose>

      <H3>5.3 Vision encoders as plug-ins</H3>

      <Prose>
        The frontier in 2024-2026 is swappable vision encoders. CLIP-ViT-L/14 (<Code>{"openai/clip-vit-large-patch14"}</Code>) is the baseline from 2021. SigLIP (<Code>{"google/siglip-so400m-patch14-384"}</Code>) uses a sigmoid contrastive loss instead of softmax and is better per parameter. DINOv2 (<Code>{"facebook/dinov2-large"}</Code>) is self-supervised and excels at dense prediction (segmentation-flavored tasks). InternViT-6B is the largest open vision encoder and is used by InternVL's 2024 series. The projector recipe is agnostic to the encoder; the same LLaVA architecture works with any of them by just changing the encoder and the projector input dim.
      </Prose>

      <H3>5.4 Perceiver in production</H3>

      <Prose>
        HuggingFace ships <Code>{"PerceiverModel"}</Code> (from <Code>{"transformers.models.perceiver"}</Code>) which includes the IO variant. In practice the pure Perceiver has not displaced transformers for language tasks — the latent bottleneck is too restrictive for autoregressive generation — but the pattern survives inside VLMs as the "Perceiver Resampler" used in Flamingo and Idefics2: a tiny 6-layer Perceiver that cross-attends learned latents (64 per image) against the vision encoder's output, producing a fixed-size visual token set for cross-attention. This is Perceiver as a <em>compression stage</em> rather than a full architecture, and it is the production-dominant use.
      </Prose>

      <H3>5.5 Serving considerations</H3>

      <Prose>
        Interleaved models like LLaVA have a single hot path: the LLM's native attention. vLLM, SGLang, and TensorRT-LLM all serve LLaVA variants without any VLM-specific code beyond the vision encoder prefill. The visual tokens are simply prepended to the input sequence, and the KV cache mechanism handles them like any other prefix.
      </Prose>

      <Prose>
        Gated cross-attention models need more care. The cross-attention layers reference external visual features at every step, and serving frameworks must know which layers are cross-attention and pass the visual features through. vLLM has explicit Flamingo/Idefics handling; SGLang does similarly. Because the visual features do not enter the KV cache, the per-request memory footprint is smaller, but the serving code must track them separately.
      </Prose>

      <Prose>
        Q-Former models run the Q-Former once per image at ingress (producing the 32 soft tokens), then serve like an interleaved model because those 32 tokens go into the LLM's prefix. This is the simplest to serve among the three patterns.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6.1 Flamingo gated cross-attention, step by step</H3>

      <StepTrace
        steps={[
          {
            label: "1. Frozen LM hidden stream arrives at the new layer",
            render: () => (
              <div>
                <Prose>
                  The pretrained LLM has been running normally. Its residual stream {"x"} arrives at a freshly inserted gated cross-attention block. Because the gate is zero-initialised, this block is supposed to be a no-op at step 0.
                </Prose>
                <div style={{ fontFamily: "monospace", fontSize: 12, color: colors.gold, marginTop: 8 }}>
                  {"x [B, L_text, 4096]  <- LLM residual stream"}
                </div>
              </div>
            ),
          },
          {
            label: "2. Visual features prepared once, outside the layer",
            render: () => (
              <div>
                <Prose>
                  The vision encoder ran earlier on the image and produced {"v_{raw} ∈ R^{L_v × d_v}"}. A Perceiver Resampler compressed this to {"v ∈ R^{64 × d}"}, 64 visual tokens in the LM's dim. These features are cached and passed to every cross-attention layer.
                </Prose>
                <div style={{ fontFamily: "monospace", fontSize: 12, color: colors.gold, marginTop: 8 }}>
                  {"image -> ViT -> v_raw [196, 1024] -> Perceiver-Resampler -> v [64, 4096]"}
                </div>
              </div>
            ),
          },
          {
            label: "3. Cross-attention: text queries attend to visual K,V",
            render: () => (
              <div>
                <Prose>
                  Compute {"Q = x W^Q, K = v W^K, V = v W^V"}. The attention score matrix is {"[L_{text}, 64]"} — each text position's query has 64 candidate keys. Softmax and weighted sum produce an {"[L_{text}, d]"} output — one visual-context vector per text position.
                </Prose>
                <div style={{ fontFamily: "monospace", fontSize: 12, color: colors.gold, marginTop: 8 }}>
                  {"attn = softmax(Q K^T / sqrt(d_h)) V   ∈ R^{L_text × 4096}"}
                </div>
              </div>
            ),
          },
          {
            label: "4. Gate the attention output: tanh(alpha) · attn",
            render: () => (
              <div>
                <Prose>
                  Multiply by {"\\tanh(α_{xattn})"}, where {"α_{xattn}"} is a learnable scalar. At step 0, {"α = 0 ⇒ \\tanh(α) = 0"}, so the attention output is zeroed out. The residual addition below is then {"x + 0 = x"} — the layer is the identity.
                </Prose>
                <div style={{ fontFamily: "monospace", fontSize: 12, color: colors.gold, marginTop: 8 }}>
                  {"gated = tanh(alpha_xattn) * attn   (= 0 at step 0)"}
                </div>
              </div>
            ),
          },
          {
            label: "5. Residual add + gated FFN",
            render: () => (
              <div>
                <Prose>
                  {"x ← x + gated"}. Then a gated FFN: {"x ← x + \\tanh(α_{ffn}) \\cdot FFN(x)"}. Both gates are independent; FFN gate also starts at zero. The output is exactly {"x"} at initialisation, perturbed once gates open during training.
                </Prose>
                <div style={{ fontFamily: "monospace", fontSize: 12, color: colors.gold, marginTop: 8 }}>
                  {"x = x + gated ;  x = x + tanh(alpha_ffn) * FFN(x)"}
                </div>
              </div>
            ),
          },
          {
            label: "6. Hidden stream continues to next frozen LLM block",
            render: () => (
              <div>
                <Prose>
                  The modified {"x"} is passed to the next <em>frozen</em> transformer block (a vanilla LLaMA layer). At step 0 the frozen block sees exactly what it would have seen without any gated cross-attention insertion. During training the gates open and visual signal starts to modulate the stream — slowly, safely, without destabilising the pretrained LLM.
                </Prose>
                <div style={{ fontFamily: "monospace", fontSize: 12, color: colors.gold, marginTop: 8 }}>
                  {"x -> next frozen LLM block -> ... -> head"}
                </div>
              </div>
            ),
          },
        ]}
      />

      <H3>6.2 Heatmap of cross-attention weights (text-to-image alignment)</H3>

      <Prose>
        A representative cross-attention matrix from a trained VLM answering "What is the person holding?" given an image with a person on the left and a dog on the right. The 6 text tokens ("what", "is", "the", "person", "holding", "?") cross-attend to 64 resampled visual tokens. Columns 0-15 roughly correspond to the image's left region (the person), 16-31 to center-left (handbag), 32-47 to center-right (dog), 48-63 to the right background. The "holding" query (row 4) concentrates attention on columns 16-31 (the handbag region) — the model has learned a text-to-region alignment. "Person" (row 3) attends to columns 0-15. Function words ("is", "the", "?") spread broadly — they get little task-relevant signal from the image.
      </Prose>

      <Heatmap
        label="CROSS-ATTENTION WEIGHTS: TEXT QUERY VS 64 VISUAL TOKENS"
        rowLabels={["what", "is", "the", "person", "holding", "?"]}
        colLabels={["vis0-15 (left)", "vis16-31 (cL)", "vis32-47 (cR)", "vis48-63 (right)"]}
        matrix={[
          [0.18, 0.31, 0.34, 0.17],
          [0.27, 0.24, 0.26, 0.23],
          [0.26, 0.25, 0.25, 0.24],
          [0.52, 0.22, 0.13, 0.13],
          [0.11, 0.64, 0.16, 0.09],
          [0.25, 0.28, 0.24, 0.23],
        ]}
        colorScale="gold"
      />

      <H3>6.3 Training curves: interleaved vs gated cross-attention</H3>

      <Prose>
        Loss over training steps for the two architectures on the toy copy task from Section 4.6. Interleaved converges almost immediately because self-attention can see visual tokens from layer 1. Gated cross-attention shows a characteristic lag — the gate has to open from zero before visual signal flows, so the first ~50 steps look nearly random. After the gate opens, both curves converge to the same loss. This is the fundamental trade-off: gated cross-attention is safer for frozen backbones but slower to start learning.
      </Prose>

      <Plot
        label="TRAINING LOSS — INTERLEAVED VS GATED CROSS-ATTENTION (COPY TASK)"
        xLabel="training step"
        yLabel="cross-entropy loss"
        width={560}
        height={280}
        series={[
          { name: "interleaved",         color: "#e2b55a", points: [[0, 2.48], [50, 0.005], [100, 0.002], [150, 0.001], [200, 0.0007], [250, 0.0005], [300, 0.0004], [350, 0.0003], [399, 0.0003]] },
          { name: "gated cross-attn",    color: "#60a5fa", points: [[0, 2.45], [50, 0.96],  [100, 0.004], [150, 0.002], [200, 0.0013], [250, 0.0009], [300, 0.0007], [350, 0.0005], [399, 0.0005]] },
        ]}
      />

      <H3>6.4 Perceiver latent size vs accuracy vs compute</H3>

      <Prose>
        Perceiver has one dominant hyperparameter: the latent count {"M"}. Increasing {"M"} gives more capacity at higher compute. The plot below is the qualitative trade-off seen across Perceiver and Perceiver IO ablations (reconstructed from the 2021 and 2022 paper figures): accuracy plateaus around {"M = 512-1024"} for ImageNet; compute grows linearly in {"M"}. The knee of the curve (most quality per FLOP) is around {"M = 256-512"} for vision and {"M = 64-256"} for the Perceiver Resampler in VLMs. Flamingo uses {"M = 64"}; Idefics2 uses {"M = 64"}; BLIP-2's Q-Former uses {"K = 32"} — all at the low-M end because downstream LLM compute dominates.
      </Prose>

      <Plot
        label="PERCEIVER: LATENTS M VS ACCURACY (ImageNet) VS GFLOPs"
        xLabel="M (num latents)"
        yLabel="relative quality / compute"
        width={560}
        height={280}
        series={[
          { name: "accuracy (rel)",  color: "#e2b55a", points: [[16, 0.55], [32, 0.62], [64, 0.70], [128, 0.78], [256, 0.84], [512, 0.88], [1024, 0.89]] },
          { name: "GFLOPs (rel)",    color: "#c084fc", points: [[16, 0.08], [32, 0.14], [64, 0.25], [128, 0.40], [256, 0.62], [512, 0.85], [1024, 1.00]] },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <H3>7.1 Frozen LLM, want to add vision</H3>

      <Prose>
        Two good choices. If you want the smallest engineering investment and the fastest working prototype, use LLaVA-style interleaved projection: run a linear projector into the LLM embedding space and concatenate. You'll need to unfreeze the LLM in stage 2 anyway (pure frozen-LLM LLaVA is worse on instruction following), but stage 1 with frozen LLM just trains the projector. If you must keep the LLM fully frozen — for compliance reasons, or because you're serving a licensed weights-only model — use Flamingo-style gated cross-attention. The zero-init gate lets you insert new layers without disturbing the pretrained weights.
      </Prose>

      <H3>7.2 Full multimodal pretraining from scratch</H3>

      <Prose>
        Use interleaved. PaLI, Gemini, Qwen-VL, Molmo, and the frontier closed models all use interleaved for full multimodal pretraining. The simplicity is a feature: one attention mechanism, one unified sequence, no gating or compression logic. You pay the sequence-length cost but you also get the maximum representational flexibility — every text token can attend to every visual token at every layer, and cross-modal interaction is not limited to specific cross-attention layers.
      </Prose>

      <H3>7.3 Very long input, fixed compute budget</H3>

      <Prose>
        Use Perceiver or a Perceiver Resampler. If your input is a 1M-token sequence (long audio, high-resolution point cloud, many-frame video), self-attention over the full input is infeasible and interleaving those tokens into an LLM is likewise infeasible. A Perceiver lets you cross-attend into {"M = 256"} or {"512"} latents once, then work in that latent space. The latent bottleneck is a constraint on expressiveness but a liberation on compute.
      </Prose>

      <H3>7.4 You want to decouple vision and language training</H3>

      <Prose>
        Use a Q-Former. The BLIP-2 philosophy is that the vision encoder and the LLM are expensive to train and expensive to change; the Q-Former is the small module that sits between them and can be retrained whenever you swap either side. If you're building a product that might change vision backbones (say, upgrading from CLIP to SigLIP to DINOv2 over time) or LLMs (7B to 13B to 70B) and you want the adapter to be cheap to re-train, the Q-Former's separation is worth the added complexity.
      </Prose>

      <H3>7.5 Simple and good enough</H3>

      <Prose>
        LLaVA-style interleaved projection. 2-layer MLP projector, no Q-Former, no gating, no Perceiver. Concatenate vision tokens with text and run self-attention. This is the winning simplicity-vs-capability point for ~80% of VLM use cases in 2026. LLaVA-1.5 and LLaVA-NeXT are the reference implementations; Qwen2-VL is the production-hardened descendant. Unless you have a specific reason (frozen LLM requirement, very long input, decoupled training) to pick one of the others, pick this.
      </Prose>

      <H3>7.6 Very small budget, tight latency</H3>

      <Prose>
        Interleaved with aggressive visual token reduction. Models like Phi-3-Vision and Gemma-3-4B-VL use only 64-256 visual tokens per image (via strong spatial pooling in the projector) and run on edge GPUs. The cross-attention alternative is fine but adds complexity that a small model does not need. Keep it interleaved, keep the token count low, and rely on the LLM's attention to do all the fusion work.
      </Prose>

      <H3>7.7 Video and multi-image</H3>

      <Prose>
        Interleaved is the default when context handles it (Qwen2-VL supports many video frames). Cross-attention is the fallback when it does not. For very long video ({">"}30 seconds at reasonable frame rate), neither works directly — you need a hierarchical scheme: interleaved within a short clip, cross-attention or retrieval across clips. Models like Video-ChatGPT and LLaVA-Video experiment with these hierarchies. No single pattern dominates yet for long video.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES
          ====================================================================== */}
      <H2>8. What scales</H2>

      <H3>8.1 Interleaved requires long context</H3>

      <Prose>
        Every image consumes tokens. A 336×336 CLIP-ViT-L/14 image is 576 tokens. LLaVA-NeXT uses 4-crop grids plus the original, so 2880 tokens per image. Qwen2-VL scales resolution dynamically — a 1024×1024 image becomes 4096 tokens. Five images in a conversation is 20k tokens. An 8-frame video at moderate resolution is similar. The base assumption of LLaVA and its descendants is that the underlying LLM has a long context window — 32k minimum, 128k preferred. Without long context, interleaving breaks on any but the simplest one-image queries.
      </Prose>

      <Prose>
        This has driven a feedback loop between context length research and VLM design. RoPE scaling, YaRN, and the 128k-context LLaMA-3 and 1M-context Gemini-1.5 are partly motivated by multimodal workloads. If context cost ever becomes prohibitive — whether from memory, compute, or attention quadratic scaling — interleaved VLMs would have to fall back to cross-attention or Perceiver-style compression.
      </Prose>

      <H3>8.2 Perceiver decouples input length from depth</H3>

      <Prose>
        Perceiver's scaling property is that input length {"N"} only enters compute through the {"M \\cdot N"} cross-attention, not through the {"M^2"} self-attention loops. Doubling input length doubles cross-attention cost; it does not affect the rest. This is the only pure-transformer architecture (as of 2026) with this scaling: state-space models like Mamba achieve linear-in-N by dropping attention entirely; Perceiver keeps attention but bottlenecks it. For applications with enormous but structured input — audio, video, scientific data (gravitational wave detection uses Perceiver IO) — this is the right trade-off.
      </Prose>

      <H3>8.3 Flamingo preserves LLM quality at scale</H3>

      <Prose>
        Flamingo's critical scaling property is quality preservation. Adding vision to a frozen Chinchilla-70B via gated cross-attention gave Flamingo 80B parameters total (80B because of the new cross-attention layers) with language-task quality within 1-2 points of pure Chinchilla, and vision-task quality on par with the best VLMs of the time. Compare to the alternative of full fine-tuning: Palm-E-like models that unfreeze the LLM suffer measurable regression on pure-text benchmarks. For deployments where both text-only and vision-grounded queries are served, Flamingo-style gating is the right scaling primitive.
      </Prose>

      <H3>8.4 LLaVA scales to video when context handles it</H3>

      <Prose>
        LLaVA-Video and similar models simply interleave video frame tokens. Each frame contributes its CLIP/SigLIP tokens, concatenated in time. A 16-frame video clip at 256 tokens/frame is 4096 tokens — tractable on any 8k+ LLM. At 64 frames and 512 tokens/frame it's 32k tokens, which needs a proper long-context LLM. The scaling is linear in frame count and linear in per-frame token count; both grow quickly, which caps practical video lengths at 1-2 minutes without hierarchical compression.
      </Prose>

      <H3>8.5 Multimodal scaling laws</H3>

      <Prose>
        Chen et al.'s PaLI-3 (arXiv:2310.09199, 2023) and follow-up scaling work show that multimodal models follow roughly the same scaling laws as text-only — loss is a power law in FLOPs and parameters, with modest modality-specific constants. The dominant trade-off is the ratio of vision to language compute. PaLI-3's finding: for joint multimodal tasks, vision encoder size matters more than LLM size up to a crossover point around 10B parameters, then LLM size dominates. PaLI-3's 5B total parameters (3B vision + 2B language) matches or beats much larger models on VQA tasks. The implication for architecture: interleaved is right when both streams are being scaled; cross-attention is right when the vision side is small and the language side is the scaling axis.
      </Prose>

      <H3>8.6 Quantisation and serving</H3>

      <Prose>
        Interleaved VLMs quantise like regular LLMs — visual tokens are ordinary hidden states by the time they enter the LLM, so int4/int8 quantisation of the LLM carries over. The projector is tiny and rarely needs quantisation. Cross-attention VLMs require quantisation of the new cross-attention layers and of the visual feature cache; this is well-supported in vLLM and SGLang but requires VLM-specific calibration. Q-Former models quantise simply because the Q-Former is small enough to keep at fp16 without memory impact.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES
          ====================================================================== */}
      <H2>9. Failure modes</H2>

      <H3>9.1 Forgetting the gate zero-init in Flamingo</H3>

      <Prose>
        If you initialise the gate at any nonzero value, the frozen LLM is immediately perturbed at step 0 by whatever the randomly-initialised cross-attention produces. The LLM's language performance collapses in the first few hundred steps — perplexity on text-only evaluation jumps by 2-5 points — and recovery can take tens of thousands of steps, sometimes never fully. Always initialise the gate to exactly zero. Verify on a sanity check: feed a text-only sequence (no image), run forward through the gated layer, confirm the output equals the input to numerical precision.
      </Prose>

      <H3>9.2 Cross-attending to too few context tokens</H3>

      <Prose>
        If the vision side produces only 8 or 16 resampled tokens, text queries cross-attending to them have very little to attend to — the attention matrix columns are few, the softmax is nearly uniform or nearly one-hot, and gradient signal is weak. Flamingo uses 64 Perceiver-Resampled tokens per image; Idefics2 uses 64; BLIP-2 uses 32. Below 32 tokens you start losing quality; above 256 you have diminishing returns. The sweet spot for production systems is 32-128 visual tokens when cross-attention is used.
      </Prose>

      <H3>9.3 Interleaving without modality boundary tokens</H3>

      <Prose>
        If you concatenate visual tokens and text tokens without explicit <Code>{"[IMG]"}</Code> / <Code>{"[/IMG]"}</Code> boundary markers (or similar), the model has to learn from the token content alone where images start and end. During inference this is fragile: on edge cases (tiny image, text that looks image-patch-like, unusual crop counts) the model confuses modalities. LLaVA from 1.5 onward uses explicit modality tokens. Qwen-VL uses <Code>{"<image>"}</Code> tokens. The marginal parameter cost is 2 embeddings; the robustness gain is large. Never skip modality boundaries.
      </Prose>

      <H3>9.4 Q-Former under-trained</H3>

      <Prose>
        The Q-Former has ~100M parameters and its training is the bottleneck of BLIP-2-style recipes. Under-training (too few steps, too small batch) produces queries that attend to noise or to a single patch — the 32 output tokens duplicate rather than covering the image. Diagnostic: look at the cross-attention weights of the final layer. Healthy Q-Formers show diverse, spatially-distributed attention patterns across queries; broken ones show the same pattern on every query. BLIP-2 trains the Q-Former for ~2M steps in stage 1 alone; shortcuts below 500k steps produce visibly degraded queries.
      </Prose>

      <H3>9.5 Position embeddings mismatched across modalities</H3>

      <Prose>
        Interleaved models use a single position embedding scheme over the concatenated sequence. Vision tokens at positions 0-575 receive RoPE rotations appropriate for positions 0-575 in the LLM's training distribution — but the LLM was trained with only text at those positions. The mismatch produces distribution shift that the stage-1 projector must absorb. Some VLMs (Idefics2, Chameleon) use modality-specific position embeddings for vision tokens (often 2D grid embeddings) added before the projection, helping the LLM see a clearer positional structure. Without this, VLMs can struggle with spatial reasoning ("what is on the left?") on high-resolution crops.
      </Prose>

      <H3>9.6 Visual tokens leaking into later text via causal mask</H3>

      <Prose>
        In a causal mask over {"[vision; text]"}, every text position can attend to every visual token — that is the design. But if vision tokens have positional embeddings that are in the same range as text tokens later in generation (positions 576+ are text, positions 0-575 are vision), the model may learn associations that are actually wrong: "position 100 is visual" during training, "position 100 is text" if a long generation loops back. In practice this is handled by extending context linearly and training with the full interleaved range from the start. Models that extend context with ad-hoc NTK scaling after training sometimes show VLM degradation on long-context visual tasks because the scaling disturbs the visual-token positions in ways not seen during training.
      </Prose>

      <H3>9.7 Vision encoder and LLM tokenisation drift</H3>

      <Prose>
        A subtle failure specific to cross-attention VLMs: the vision encoder produces feature vectors in its own space, and the cross-attention K,V projections learn to map that space into the LLM's attention space. If the vision encoder is updated (e.g., CLIP-ViT-L/14 replaced with SigLIP-SO400M) without retraining the cross-attention layers, the K,V projections produce garbage — the inputs no longer match what the layers were trained on. Always retrain at least the projection and cross-attention layers when the vision encoder changes. This is cheap (~1% of total training cost) and prevents silent quality collapse.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        <strong>Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin (Google, 2017).</strong> "Attention Is All You Need." arXiv:1706.03762. NeurIPS 2017. The paper that introduced cross-attention in its modern form — Section 3.2.3, "Applications of Attention in our Model," describes the decoder's cross-attention over the encoder output as one of three attention types. The equations {"Attn(Q, K, V) = \\mathrm{softmax}(Q K^\\top / \\sqrt{d_k}) V"} with Q from the decoder and K,V from the encoder are the ancestor of every subsequent cross-attention variant. Still the right first paper to read.
      </Prose>

      <Prose>
        <strong>Jaegle, Gimeno, Brock, Zisserman, Vinyals, Carreira (DeepMind, 2021).</strong> "Perceiver: General Perception with Iterative Attention." arXiv:2103.03206. ICML 2021. The paper that introduced the cross-attention-to-learned-latents architecture. Sections 3 (architecture) and 4 (positional encoding via Fourier features) are essential. The ImageNet, AudioSet, and ModelNet ablations show the same architecture winning or near-winning on three different modalities with the same code, which is what made the paper influential. Perceiver is the conceptual ancestor of Flamingo's Perceiver Resampler.
      </Prose>

      <Prose>
        <strong>Jaegle, Borgeaud, Alayrac, Doersch, Ionescu, Ding, et al. (DeepMind, 2022).</strong> "Perceiver IO: A General Architecture for Structured Inputs & Outputs." arXiv:2107.14795. ICLR 2022. Extends Perceiver with a symmetric decoder cross-attention for arbitrary output shapes. Table 1 covers tasks from optical flow to StarCraft units. The key insight is that if you can query the latents with a per-output-slot query, you can generate any output shape — the latent bottleneck is not a constraint on output dim. Foundation for later query-based decoders.
      </Prose>

      <Prose>
        <strong>Alayrac, Donahue, Luc, Miech, Barr, Hasson, et al. (DeepMind, 2022).</strong> "Flamingo: a Visual Language Model for Few-Shot Learning." arXiv:2204.14198. NeurIPS 2022. The paper that established gated cross-attention as a recipe. Sections 3 (method) and 4 (training setup) are the canonical reference. The zero-init {"\\tanh"} gate is defined in Equation 6; the interleaved training data format ("multimodal chain-of-text-and-images") is described in Section 4.1. The few-shot numbers (80B Flamingo matching or exceeding task-specific SOTA on many VQA tasks) were the demonstration that made industry take VLMs seriously.
      </Prose>

      <Prose>
        <strong>Liu, Li, Wu, Lee (UW, 2023).</strong> "Visual Instruction Tuning" (LLaVA). arXiv:2304.08485. NeurIPS 2023. The paper that introduced the interleaved projection recipe that now dominates VLM design. Section 3 describes the architecture: CLIP-ViT-L/14 vision encoder, linear projection, Vicuna-7B LLM, concatenate. Section 4 describes the two-stage training: stage 1 on image-caption pairs for feature alignment, stage 2 on ~158k GPT-4-generated multimodal instructions for instruction tuning. LLaVA-1.5 (arXiv:2310.03744) upgraded the projector to a 2-layer MLP and scaled the instruction data to ~665k; every modern interleaved VLM descends from this template.
      </Prose>

      <Prose>
        <strong>Li, Li, Savarese, Hoi (Salesforce, 2023).</strong> "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models." arXiv:2301.12597. ICML 2023. The Q-Former paper. Section 3 (Q-Former architecture) and Section 4 (two-stage pretraining: vision-language representation learning, then vision-to-language generative learning) are the canonical reference. The Q-Former has 32 learnable queries and ~188M parameters; the key design choice is the two-stage training with contrastive + matching + captioning losses in stage 1.
      </Prose>

      <Prose>
        <strong>Laurençon, Tronchon, Cord, Sanh (HuggingFace, 2024).</strong> "What matters when building vision-language models?" (Idefics2). arXiv:2405.02246. A systematic ablation of VLM design choices — architecture (gated cross-attn vs interleaved), vision encoder (CLIP vs SigLIP), pretraining data, instruction-tuning data. Table 2's comparison of gated cross-attention vs interleaved (both with SigLIP + Mistral-7B) is the best direct apples-to-apples comparison in the literature: interleaved with ~half the parameters achieves similar quality, at the cost of longer sequences. This paper's conclusions drove HuggingFace's move from Idefics1 (gated cross-attn) to Idefics2's hybrid and then to Idefics3 (interleaved).
      </Prose>

      <Prose>
        <strong>Chen, Wang, Dehghani, Salz, Pavetic, et al. (Google, 2024).</strong> "PaLI-3: Vision Language Models: Smaller, Faster, Stronger." arXiv:2310.09199. Shows that a 5B-parameter interleaved VLM (SigLIP-SO400M vision + UL2-3B language) can match or beat much larger models. Section 2 describes the contrastive+generative pretraining; Section 3 the multimodal co-training. Key finding (Section 5): the vision encoder choice (SigLIP vs CLIP) matters more than the language encoder choice at this scale, and interleaving with a joint cross-modal pretraining outperforms pipeline approaches. This paper is where the "vision encoder matters more than you think" consensus comes from.
      </Prose>

      <Prose>
        <strong>Reference implementations.</strong> The HuggingFace Transformers source files are the most readable production code: <Code>{"modeling_llava.py"}</Code> (interleaved), <Code>{"modeling_idefics2.py"}</Code> (gated cross-attn), <Code>{"modeling_blip_2.py"}</Code> (Q-Former), <Code>{"modeling_perceiver.py"}</Code> (Perceiver). The OpenFlamingo project (<Code>{"mlfoundations/open_flamingo"}</Code>) is the canonical open Flamingo reproduction with clear gated cross-attention code. LLaVA's original repo (<Code>{"haotian-liu/LLaVA"}</Code>) has the cleanest interleaved training code. Reading these alongside the papers is the fastest path to implementation.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK
          ====================================================================== */}
      <H2>11. Self-check</H2>

      <H3>11.1 What is the difference between self-attention and cross-attention, and when do you use each?</H3>

      <Prose>
        Self-attention computes {"Attn(Q, K, V)"} where all three are linear projections of the same sequence — every token looks at every other token in that sequence. Cross-attention computes {"Attn(Q, K, V)"} where Q comes from one sequence and K, V come from a different sequence (typically a different modality or an encoder output). You use self-attention when you want information to flow within a single sequence; you use cross-attention when you want one sequence to conditionally attend to another. The classic example is Vaswani 2017's decoder, which uses both: self-attention to relate decoder tokens to each other, and cross-attention to relate decoder tokens to encoder hidden states. In modern VLMs, cross-attention is the explicit bridge between vision and language (Flamingo, Q-Former), while interleaving sidesteps cross-attention by putting both modalities into a single self-attention.
      </Prose>

      <H3>11.2 Why does Flamingo initialise its gate at exactly zero?</H3>

      <Prose>
        The zero gate makes the newly-inserted cross-attention layer exactly the identity at step 0: {"x ← x + \\tanh(0) \\cdot \\mathrm{CrossAttn}(x, v) = x"}. Because the surrounding LLM is frozen and its pretrained behavior is valuable, any perturbation of the residual stream at step 0 would degrade the LLM's language performance — and recovery from that perturbation is slow and uncertain. By starting the gate at zero, training is guaranteed to begin from the pretrained LLM's exact behavior. Gradient descent then drives the gate toward nonzero values at whatever rate the loss demands, and the model learns to use vision without catastrophic early-training drift. Removing the gate or initialising it nonzero reliably produces an LLM that has forgotten its language abilities by step 1000 — a direct failure mode documented in Flamingo's ablations.
      </Prose>

      <H3>11.3 Compare the memory cost of interleaved vs cross-attention for a 16-frame video at 576 tokens per frame, LLaMA-2-7B scale.</H3>

      <Prose>
        16 frames × 576 tokens = 9216 visual tokens. Interleaved: these enter the KV cache at every layer. For LLaMA-2-7B (n_layers = 32, d = 4096, fp16), the cache cost is {"2 × 32 × 4096 × 9216 × 2"} bytes = 4.83 GB per sample, per sequence. Cross-attention: the 9216 visual features are held once, outside the KV cache, at their native dim (say 1024 for CLIP-ViT). Cost is {"9216 × 1024 × 2"} = 18.9 MB. The ratio is ~255x. For interactive serving where multiple video conversations are concurrent, this difference can be the line between feasible (cross-attn) and infeasible (interleaved) on a single 80 GB GPU. Interleaving is dominant for single-image workloads; for video it often must fall back to cross-attention or hierarchical compression.
      </Prose>

      <H3>11.4 What is the key architectural role of the Perceiver Resampler in Flamingo, and why is it not just a Q-Former?</H3>

      <Prose>
        The Perceiver Resampler is a small 6-layer Perceiver that takes the raw vision encoder output (196-576 patches at 1024-d) and compresses it to 64 visual tokens at the LM's embedding dim. It exists because cross-attention K,V cost scales with the context size, and 576 visual tokens is both expensive to cross-attend against at every layer and full of redundant low-level patch information that the LM doesn't need. The Resampler learns to extract 64 high-level visual summaries. The difference from a Q-Former: the Resampler is trained <em>end-to-end with the gated cross-attention layers</em> during VLM training (not pretrained separately), it has only self-attention plus cross-attention to the raw vision features (no text branch), and it uses the Perceiver weight-sharing scheme across its layers. The Q-Former is a more heavyweight, separately-pretrained compressor with its own vision-language contrastive stage. They solve the same problem (vision feature compression) with different training regimes.
      </Prose>

      <H3>11.5 Why does interleaving require explicit modality boundary tokens like [IMG] and [/IMG], and what happens without them?</H3>

      <Prose>
        In interleaved inputs, visual token embeddings and text token embeddings share the same residual stream dimensionality but come from different distributions — visual tokens are projected CLIP features, text tokens are learned word embeddings. Without explicit boundaries, the model has to infer "this is vision" vs "this is text" from the content of the embedding, which is fragile: on edge cases (rare images, unusual crops) the model can confuse modalities and produce nonsensical output. Modality boundary tokens are learned embeddings like <Code>{"<image>"}</Code> or <Code>{"[IMG]...[/IMG]"}</Code> that are inserted around visual spans. They give the model a clear signal about modality transitions, making attention patterns more robust. Production VLMs from LLaVA-1.5 onward universally use modality tokens; removing them in ablations produces 2-5 point drops on VQA benchmarks and much worse behavior on out-of-distribution inputs. The marginal cost is two embeddings per modality; the robustness gain is large.
      </Prose>

    </div>
  ),
};

export default interleavedCrossAttentionContent;
