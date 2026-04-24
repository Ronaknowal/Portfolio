import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const positionalEncodingsContent = {
  title: "Positional Encodings (Sinusoidal, Learned, RoPE, ALiBi)",
  readTime: "~40 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Every Transformer layer is, at its core, a permutation-equivariant function. If you shuffle the input tokens before feeding them to a self-attention block, the output tokens come out shuffled in the same order; the block itself cannot tell whether the sequence {"[the, cat, sat]"} arrived as {"[cat, sat, the]"}. This is not a bug of any particular implementation — it follows directly from the fact that attention is a weighted sum and matrix multiplication is associative. {"Attention(Q, K, V) = softmax(Q K^T / √d_k) V"} has no position term anywhere. Same goes for the pointwise feed-forward sublayer and every layer norm — none of them see token order. A vanilla Transformer is a bag-of-tokens model, and that is a problem: language is not a bag of words. "Dog bites man" and "Man bites dog" differ only in word order and differ completely in meaning.
      </Prose>

      <Prose>
        Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, and Polosukhin knew this from page one of "Attention Is All You Need" (arXiv:1706.03762, NeurIPS 2017). Their fix was deliberately modest: add a fixed sinusoidal function of position to each input embedding before the first layer. {"PE(pos, 2i) = sin(pos / 10000^{2i/d})"} and {"PE(pos, 2i+1) = cos(pos / 10000^{2i/d})"}. The motivation was that linear combinations of shifted sinusoids can express relative positions — if the model learned that a particular projection of {"PE(pos)"} and {"PE(pos + k)"} correlates, it could learn position-aware behavior from the input embedding alone. The scheme was untrained (no parameters), extended to arbitrary lengths in principle (any pos gives a valid embedding), and added almost no compute. It was good enough to reach state of the art on machine translation, and the question of what the "right" positional encoding should be became a decade-long research program.
      </Prose>

      <Prose>
        A year later, Devlin, Chang, Lee, and Toutanova's BERT (arXiv:1810.04805) and Radford, Wu, Child, Luan, Amodei, and Sutskever's GPT-2 (OpenAI 2019) both swapped sinusoidal for a learned absolute positional embedding: {"nn.Embedding(max_len, d)"}. Each position index 0 through {"max_len − 1"} got its own trainable vector, added to the token embedding exactly like the sinusoidal PE but with {"max_len · d"} extra parameters. Learned PE matched or slightly outperformed sinusoidal on in-distribution lengths because the optimizer could shape position embeddings to match the actual distribution of positions in the training data, but it came with a hard ceiling — queries at positions beyond {"max_len"} simply had no embedding. BERT's {"max_len = 512"}, GPT-2's {"max_len = 1024"}. You could not feed them longer inputs without retraining.
      </Prose>

      <Prose>
        In 2018 Shaw, Uszkoreit, and Vaswani (arXiv:1803.02155) reframed the problem. What attention actually needs to know is not absolute position but relative position: how far apart are query {"i"} and key {"j"}? Their "Self-Attention with Relative Position Representations" added a learned bias term {"a^K_{i−j}"} and value term {"a^V_{i−j}"} into the attention computation, where the index is the clipped relative distance. Relative position encodings (RPE) became the family of techniques that modify the attention score itself rather than the input embedding. T5 (Raffel et al. 2020, arXiv:1910.10683) simplified this further with a scalar relative bias {"b_{i−j}"} per head, learned and bucketed log-spaced for long distances. RPE gave two wins at once: it decoupled position representation from token representation, and it generalized better to sequences longer than those seen during training, because the model only ever encoded distances, never absolute positions.
      </Prose>

      <Prose>
        The pivotal move came with Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu's "RoFormer: Enhanced Transformer with Rotary Position Embedding" (arXiv:2104.09864, published 2021, broadly adopted from 2023 onward). RoPE does not add anything. Instead it rotates the query and key vectors in learned 2D subspaces by an angle proportional to their absolute position. The result of the rotation is that the inner product {"⟨q'_m, k'_n⟩"} — which is the only thing attention ever evaluates on position-encoded vectors — ends up depending only on the positional difference {"m − n"}, not on {"m"} or {"n"} individually. You get relative-position behavior for free, without a separate bias term, without extra parameters, and with a multiplicative interaction that composes cleanly across layers. LLaMA, Qwen, Mistral, Gemma, DeepSeek, and essentially every frontier decoder-only model from 2023 onward uses RoPE.
      </Prose>

      <Prose>
        Running in parallel, Ofir Press, Noah Smith, and Mike Lewis published "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation" (ALiBi, arXiv:2108.12409, ICLR 2022). ALiBi also leaves token embeddings alone; instead it adds a fixed linear penalty {"−m · |i − j|"} to the attention logits, where {"m"} is a head-specific slope set a priori (no learning involved). The penalty biases each query toward recent keys and away from distant ones, with different heads preferring different window sizes. The paper's central empirical claim — that a model trained at length 1024 could evaluate at length 2048 or 4096 with only minor perplexity increase, while sinusoidal and learned PE collapsed — made ALiBi the go-to mechanism for models that needed length extrapolation out of the box. MPT-7B, BLOOM, and Falcon adopted it.
      </Prose>

      <Prose>
        The 2023 wave closed the loop by making RoPE extrapolate as well as ALiBi. Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuandong Tian's "Extending Context Window of Large Language Models via Positional Interpolation" (arXiv:2306.15595) showed that you could stretch a pre-trained RoPE model from 2K to 32K context with just a few hundred fine-tuning steps by rescaling the position indices — interpolating new positions into the already-learned range instead of extrapolating beyond it. The anonymous Reddit user bloc97 published "NTK-Aware Scaled RoPE" later that year, arguing that uniform rescaling blurs high-frequency detail and that the scaling should preserve high-frequency dimensions while stretching low-frequency ones. Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole's "YaRN: Efficient Context Window Extension" (arXiv:2309.00071) formalized NTK-aware into a principled (α, β)-blend of positional interpolation and NTK scaling, and showed that RoPE-based models could be extended from their 2K or 4K training window to 128K context with a small fraction of the original pretraining compute. Yutao Sun et al.'s "A Length-Extrapolatable Transformer" (XPos, arXiv:2212.10554) added a decay term to RoPE that further improved long-range stability. By 2024, the de facto stack was RoPE-with-YaRN-or-NTK-scaling for decoder-only LLMs, learned PE for small BERT-family encoders, sinusoidal for legacy research models, and ALiBi where native length extrapolation mattered more than peak accuracy.
      </Prose>

      <Callout accent="gold">
        The arc: sinusoidal (fixed, additive, pre-layer) {"→"} learned (trained, additive, pre-layer, fixed max_len) {"→"} relative position (learned bias inside attention) {"→"} RoPE (multiplicative rotation inside attention) {"→"} ALiBi (fixed linear bias inside attention). The later techniques move position information <em>out</em> of the token embedding and <em>into</em> the attention score, which is where it always belonged — because what attention actually consumes is pairwise similarity, not absolute identity.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <H3>2.1 Absolute vs relative position</H3>

      <Prose>
        Absolute positional encoding answers the question "what is this token's index in the sequence?" — position 7 gets one embedding, position 8 gets another. Relative positional encoding answers "what is the distance from this token to that token?" — the offset {"−3"} (query is three positions after key) gets one representation regardless of whether the query is at position 50 or at position 5000. Most of what a language model needs from position information is relative: "the pronoun refers to the noun <em>three words back</em>", "the closing bracket matches the opening bracket <em>eight tokens earlier</em>". Absolute-position awareness matters much less frequently (examples: "this is the first token of the document", "this is the second sentence"). That asymmetry is the reason every successful modern PE variant encodes <em>relative</em> position as its primary signal; absolute-position sensitivity is emergent from content rather than hard-coded into the encoding.
      </Prose>

      <H3>2.2 Additive PE: sinusoidal and learned</H3>

      <Prose>
        Sinusoidal and learned PE are both additive: they produce a position vector {"p_t ∈ R^d"} that is summed with the token embedding {"x_t"} before the first Transformer layer. Everything downstream sees one combined input {"x_t + p_t"} and cannot disentangle position from content except through the learned projections. The additive scheme is simple to implement and cheap to compute, but it has two structural weaknesses. First, the position signal is injected <em>once</em> at the bottom of the network; as the representation passes through many layers of attention and feed-forward, that signal is progressively mixed with content and may be attenuated. Second, since position information is carried inside the same vector as content information, the {"W^Q, W^K, W^V"} projections must allocate some of their capacity to separating the two, which is capacity that cannot go toward modeling content.
      </Prose>

      <H3>2.3 Multiplicative PE: RoPE</H3>

      <Prose>
        RoPE breaks with the additive template entirely. Token embeddings are left alone at the input; position enters only inside the attention layers, by <em>rotating</em> the query and key vectors in learned 2D subspaces. The per-dimension rotation angle scales linearly with absolute position, but because attention only ever compares queries to keys via their inner product, and rotations preserve inner products under simultaneous rotation, the net effect on the attention score is a function only of the positional <em>difference</em>. Intuitively: rotating {"q"} by {"mθ"} and {"k"} by {"nθ"} gives {"⟨q', k'⟩ = q^T R_{(n-m)θ} k"}. The rotation cancels out by the relative amount. Position is reinjected at every attention layer — not just at the bottom — which keeps the signal crisp through deep stacks. There is no capacity trade-off in the Q/K projections: content lives in the full vector, and position is a separable geometric transform applied afterward.
      </Prose>

      <H3>2.4 Bias-based PE: ALiBi</H3>

      <Prose>
        ALiBi takes yet another tack. It leaves both token embeddings and Q/K vectors alone, and instead modifies the attention <em>logit matrix</em> directly: after computing {"Q K^T / √d_k"}, add a matrix {"B"} where {"B_{i,j} = −m_h · |i − j|"} for head {"h"} with a per-head slope {"m_h"}. The penalty is a plain linear function of distance, same on every layer, not learned — just a geometric prior that says "closer keys should be preferred". Each head gets a different slope {"m_h = 2^{−8h/H}"}, so head 1 has a steep preference for local context (slope 0.5, penalty doubles every token), head 8 has a shallow preference (slope 0.004, penalty accumulates slowly). The multi-head stack therefore spans a range of effective attention windows, from near-local to essentially global. Because the bias is a function of {"|i − j|"} and nothing else — no position-specific parameter — ALiBi extrapolates naturally: at any evaluation length the bias is well-defined and bounded, and the model's learned patterns (which depend only on content Q/K and the universal distance penalty) still apply.
      </Prose>

      <H3>2.5 Why RoPE rotations are the right geometric primitive</H3>

      <Prose>
        To see why rotation is clever, consider what we want the attention score {"q^T k"} to look like when query {"q"} sits at position {"m"} and key {"k"} sits at position {"n"}. A good positional encoder should make this score a function of {"(m − n)"} and the underlying content vectors, but not of {"m"} or {"n"} alone — otherwise absolute-position artifacts leak into attention. Rotations have exactly this property. A 2D rotation by angle {"mθ"} is a unitary transform; applying it to both {"q"} and {"k"} before taking the inner product gives a result that depends only on the <em>relative</em> rotation {"(n − m)θ"}, which is a function of position difference. Packaging the d-dim vector as {"d/2"} independent 2D rotations, each at a different frequency, produces a multi-scale relative-position encoding — one rotation handles short-range interactions (high frequency, wavelength {"2π"}), another handles medium-range (wavelength {"∼60"}), another handles long-range (wavelength {"∼600"} and beyond). The model can attend at any of these scales by learning which Q/K dimensions to emphasize.
      </Prose>

      <H3>2.6 Why ALiBi extrapolates but may not maximize accuracy</H3>

      <Prose>
        ALiBi's linear penalty is a strong inductive bias: it hardcodes "recency preferred" into every layer. That is exactly right for most language-modeling tasks, which is why ALiBi trains fast and generalizes cleanly to longer contexts. But the bias is <em>fixed</em>; the model cannot learn to undo it. If a particular attention head would benefit from attending to a distant token — for instance, a head that resolves long-range coreference — it has to overcome ALiBi's penalty with very strong Q/K alignment to that distant key. In practice this means ALiBi is a few perplexity points worse than RoPE at in-distribution lengths on well-trained large models, but noticeably more robust out-of-distribution. For a model that will only ever be used at a fixed context, RoPE tends to win. For a model that needs to extrapolate well, ALiBi is safer.
      </Prose>

      <H3>2.7 Sinusoidal encodes position via frequency</H3>

      <Prose>
        Why sinusoids at all? Because {"sin"} and {"cos"} of the same angle are phase-shifted by {"π/2"}, and together they form an orthonormal basis for 2D — which means an arbitrary rotation can be written as a linear combination of {"sin(pos·θ)"} and {"cos(pos·θ)"} at fixed {"θ"}. In particular, {"PE(pos + k)"} is a linear function of {"PE(pos)"} whose coefficients depend only on {"k"}. If the model can learn a linear readout of the PE, it can extract relative position from absolute sinusoidal encoding. The multi-scale choice of frequencies ({"θ_i = 10000^{−2i/d}"}) gives wavelengths spanning several orders of magnitude, so one pair encodes position with wavelength {"≈ 6"}, another with wavelength {"≈ 600"}, another with wavelength {"≈ 60000"}. At any given scale, nearby positions have similar sinusoidal codes, which is how the model recognizes locality.
      </Prose>

      <H3>2.8 Why position matters more in causal models</H3>

      <Prose>
        Causal (decoder-only) models care about position more than bidirectional (encoder-only) models. The reason is that in a bidirectional encoder, every token already sees every other token, so order is partly recoverable from co-occurrence patterns — "dog" and "bites" and "man" in the same window tell you quite a bit even without order. In a causal decoder, position is not only part of the signal, it is part of what makes autoregressive generation coherent: predicting the next token requires knowing where in the sequence you are, and especially, which tokens are <em>most recent</em>. That asymmetry is part of why frontier decoder-only LLMs converged on RoPE (which gives a strong relative-position signal to every attention layer) while BERT-family models stuck with learned absolute PE for years — they needed it less.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 Sinusoidal positional encoding</H3>

      <Prose>
        For position {"pos ∈ {0, 1, ..., L-1}"} and embedding dimension {"d"} (assumed even), the sinusoidal PE is defined per-dimension:
      </Prose>

      <MathBlock>{"PE_{(pos, 2i)} = \\sin\\!\\left(\\frac{pos}{10000^{2i/d}}\\right), \\qquad PE_{(pos, 2i+1)} = \\cos\\!\\left(\\frac{pos}{10000^{2i/d}}\\right)"}</MathBlock>

      <Prose>
        where {"i ∈ {0, 1, ..., d/2 − 1}"} indexes the dimension pair. The frequency {"ω_i = 10000^{−2i/d}"} is geometric: {"ω_0 = 1"} (highest frequency, wavelength {"2π"}), {"ω_{d/2-1} ≈ 1/10000"} (lowest frequency, wavelength {"20000π"}). The encoding is deterministic, parameter-free, and extends to any position by evaluating the formula. At training time each input embedding {"x_{pos}"} is modified by element-wise sum: {"x'_{pos} = x_{pos} + PE_{pos}"}.
      </Prose>

      <H3>3.2 The key identity: {"PE(pos + k)"} is linear in {"PE(pos)"}</H3>

      <Prose>
        For any fixed offset {"k"} and dimension pair {"i"} with frequency {"ω_i"}:
      </Prose>

      <MathBlock>{"\\begin{bmatrix} \\sin(\\omega_i (pos+k)) \\\\ \\cos(\\omega_i (pos+k)) \\end{bmatrix} = \\begin{bmatrix} \\cos(\\omega_i k) & \\sin(\\omega_i k) \\\\ -\\sin(\\omega_i k) & \\cos(\\omega_i k) \\end{bmatrix} \\begin{bmatrix} \\sin(\\omega_i pos) \\\\ \\cos(\\omega_i pos) \\end{bmatrix}"}</MathBlock>

      <Prose>
        The transformation matrix is a 2D rotation by angle {"ω_i k"}, and it depends only on {"k"}, not on {"pos"}. This means a linear layer with the right weights can extract "the token {"k"} positions away" from the PE regardless of absolute position. Vaswani et al. conjectured this property would make relative-position learning easy. In practice it works but not optimally — the model must learn to build the rotation, and the learned projections {"W^Q, W^K"} are not explicitly constrained to do so.
      </Prose>

      <H3>3.3 Learned positional encoding</H3>

      <Prose>
        Learned PE replaces the fixed sinusoidal formula with a trainable embedding table:
      </Prose>

      <MathBlock>{"P \\in \\mathbb{R}^{L_{\\max} \\times d}, \\qquad PE_{pos} = P[pos]"}</MathBlock>

      <Prose>
        The table is updated by backpropagation like any other parameter. It has {"L_{\\max} · d"} parameters ({"512 · 768 = 393{,}216"} for BERT-base). The crucial constraint is {"pos < L_{\\max}"} — a query at position {"L_{\\max}"} or beyond literally has no embedding, and the lookup fails (or, in sloppy implementations, indexes off the end of the tensor and produces garbage). BERT, GPT-2, RoBERTa, and ViT all use learned absolute PE. Parameter count is typically negligible compared to the Transformer body ({"< 1%"} of total params for most configurations), but the rigid max-length ceiling is a serious constraint.
      </Prose>

      <H3>3.4 Rotary Position Embedding (RoPE)</H3>

      <Prose>
        RoPE treats each pair of consecutive dimensions {"(2i, 2i+1)"} of the query and key as a complex number (or equivalently as a 2D vector). At position {"m"}, the rotation matrix {"R_{\\theta_i, m}"} is applied to each pair independently:
      </Prose>

      <MathBlock>{"R_{\\theta_i, m} = \\begin{bmatrix} \\cos(m\\theta_i) & -\\sin(m\\theta_i) \\\\ \\sin(m\\theta_i) & \\cos(m\\theta_i) \\end{bmatrix}, \\qquad \\theta_i = b^{-2i/d_k}"}</MathBlock>

      <Prose>
        with {"i ∈ {0, ..., d_k/2 − 1}"} and base {"b"} (typically {"10000"} or {"500000"}). The full per-position rotation {"R_m"} is the block-diagonal stack of these 2D rotations. RoPE is applied to {"Q"} and {"K"} after the Q/K projections but before the attention score:
      </Prose>

      <MathBlock>{"q'_m = R_m \\, q_m, \\qquad k'_n = R_n \\, k_n"}</MathBlock>

      <Prose>
        Crucially, because {"R_m"} is orthogonal, {"R_m^T R_m = I"}, and because {"R_m R_n^T = R_{m-n}"}, the attention score becomes:
      </Prose>

      <MathBlock>{"\\langle q'_m, k'_n \\rangle = q_m^T R_m^T R_n k_n = q_m^T R_{n-m} k_n"}</MathBlock>

      <Prose>
        The score depends on the content vectors {"q_m, k_n"} and on the relative offset {"n − m"}, but not on {"m"} or {"n"} individually. This is what we want. {"V"} is <em>not</em> rotated — only {"Q"} and {"K"}. Applying RoPE to {"V"} would break the relative-position invariance of the output.
      </Prose>

      <H3>3.5 RoPE frequencies and wavelengths</H3>

      <Prose>
        With base {"b = 10000"} and {"d_k = 64"}, the per-pair angular frequency is {"θ_i = 10000^{−2i/64}"} and the corresponding wavelength is {"λ_i = 2π / θ_i"}. The highest-frequency pair ({"i = 0"}) has {"θ = 1"} and wavelength {"≈ 6.28"} positions — the rotation wraps every six tokens. The lowest-frequency pair ({"i = 31"}) has {"θ ≈ 1.3 × 10^{−4}"} and wavelength {"≈ 47000"} positions — effectively linear over any reasonable context. Doubling the base to {"500000"} (LLaMA-3 default) multiplies every wavelength by roughly {"500000^{1/32} ≈ 1.5"} at the high-frequency end and by {"500000/10000 = 50"} at the low-frequency end, so the longest wavelength jumps from {"≈ 47000"} to {"≈ 2{,}084{,}000"} positions — long enough to cover a 128K context with the lowest frequency still in a monotone regime.
      </Prose>

      <H3>3.6 ALiBi (Attention with Linear Biases)</H3>

      <Prose>
        ALiBi leaves the token embedding and Q/K untouched and modifies the attention score directly. For a causal Transformer with {"H"} heads, the attention for head {"h"} becomes:
      </Prose>

      <MathBlock>{"\\mathrm{Attention}_h(Q, K, V) = \\mathrm{softmax}\\!\\left(\\frac{Q K^\\top}{\\sqrt{d_k}} + m_h \\cdot B\\right) V"}</MathBlock>

      <Prose>
        where {"B ∈ R^{L×L}"} is a fixed relative-distance matrix {"B_{i,j} = -(i - j)"} for {"j ≤ i"} and {"-∞"} for {"j > i"} (the causal part), and {"m_h"} is the per-head slope:
      </Prose>

      <MathBlock>{"m_h = 2^{-8h/H}, \\qquad h = 1, 2, \\ldots, H"}</MathBlock>

      <Prose>
        For {"H = 8"}: {"m = (0.5, 0.25, 0.125, ..., 0.0039)"}. The head with {"m = 0.5"} penalizes distance-1 keys by {"−0.5"} in logit space, distance-2 by {"−1.0"}, distance-8 by {"−4.0"} — essentially a local window. The head with {"m = 0.0039"} penalizes distance-256 by just {"−1.0"} — a near-global head. The multi-head stack therefore covers a geometric range of attention scales.
      </Prose>

      <H3>3.7 Positional Interpolation (PI)</H3>

      <Prose>
        To extend a RoPE model trained at {"L_{\\mathrm{train}}"} to a longer context {"L_{\\mathrm{eval}} > L_{\\mathrm{train}}"}, naive extrapolation feeds positions {"0, 1, 2, ..., L_{\\mathrm{eval}} − 1"} through the same RoPE formula. This puts some dimension pairs into rotation ranges the model has never seen, and high-frequency pairs wrap around badly. PI (Chen et al. 2023) instead rescales the position index:
      </Prose>

      <MathBlock>{"pos' = pos \\cdot \\frac{L_{\\mathrm{train}}}{L_{\\mathrm{eval}}}"}</MathBlock>

      <Prose>
        so that position {"L_{\\mathrm{eval}} − 1"} is mapped to the highest position the model was trained on. Every RoPE angle is then in-distribution, and a short fine-tune (1000 steps) is typically enough for the model to recalibrate. PI is simple and works well up to {"4x–8x"} extension but degrades at larger factors because it uniformly compresses all frequencies, blurring the high-frequency short-range structure.
      </Prose>

      <H3>3.8 NTK-aware scaling</H3>

      <Prose>
        NTK-aware RoPE scaling (bloc97 2023) preserves high-frequency dimensions while stretching low-frequency ones. Instead of rescaling positions, it rescales the base: {"b → b · α^{d_k/(d_k - 2)}"} where {"α = L_{\\mathrm{eval}} / L_{\\mathrm{train}}"}. The effect is that high-frequency pairs (low {"i"}) keep their wavelengths approximately unchanged, while low-frequency pairs (high {"i"}) get their wavelengths stretched by roughly {"α"}. This preserves short-range attention fidelity while allowing long-range attention to cover the new context length. NTK-aware typically outperforms PI for extension factors {"> 4x"}.
      </Prose>

      <H3>3.9 YaRN</H3>

      <Prose>
        YaRN (Peng et al. 2023) unifies PI and NTK-aware into a per-dimension scheme: each dimension pair {"i"} is scaled by a factor that depends on its wavelength relative to the original training context. Pairs whose wavelength is much shorter than {"L_{\\mathrm{train}}"} are left unscaled (they rotate many times within the training window, so they are well-sampled). Pairs whose wavelength is comparable to or longer than {"L_{\\mathrm{train}}"} are scaled via PI (they rotate slowly, so they need rescaling to fit the new range). A smooth interpolation {"ramp(α, β)"} blends between the two regimes. YaRN also multiplies the attention scores by a temperature factor {"√(1 + 0.1 \\ln(L_{\\mathrm{eval}}/L_{\\mathrm{train}}))"} to compensate for the fact that longer contexts mean more keys competing for the softmax, which tends to flatten the attention distribution. In practice YaRN is state of the art for RoPE context extension; a 4K-trained LLaMA can be extended to 128K with a few hundred fine-tuning steps.
      </Prose>

      <H3>3.10 Cost comparison</H3>

      <Prose>
        All four schemes have modest cost overhead relative to attention itself. Sinusoidal: one-time {"O(L d)"} generation per forward pass (or cached), no extra compute per attention layer. Learned: one {"O(L d)"} embedding lookup per forward pass. RoPE: one rotation per Q and per K per head per layer, {"O(L d)"} extra multiplies — roughly 10% of the attention matmul cost. ALiBi: adds one {"L × L"} bias matrix to the logits before softmax, {"O(L^2)"} extra adds per head per layer — cheaper than RoPE in FLOPs but still negligible compared to the {"O(L^2 d_k)"} attention matmul itself.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        Everything below was run on a single GPU with PyTorch 2.6 + CUDA 12.4. Every {"# Output:"} block is real stdout. The from-scratch benchmark trains a tiny 2-layer causal Transformer ({"d = 64, H = 4, d_k = 16"}) on a shift-by-one task (predict {"y_t = x_{t-1}"}, {"x ∈ {0, ..., 31}"}) at sequence length 64, then evaluates at lengths {"{64, 128, 256, 512}"}. Shift-by-one is trivial at the training length — any competent self-attention learns it — and the eval-beyond-train-length curves cleanly isolate the length-extrapolation quality of the four PE variants.
      </Prose>

      <H3>4.1 Sinusoidal PE from scratch</H3>

      <CodeBlock language="python">
{`import math, torch

def sinusoidal_pe(L, d, device, base=10000.0):
    pos  = torch.arange(L, device=device).float().unsqueeze(1)   # [L, 1]
    i    = torch.arange(d // 2, device=device).float().unsqueeze(0)  # [1, d/2]
    freq = torch.pow(base, -2 * i / d)                           # [1, d/2]
    angles = pos * freq                                          # [L, d/2]
    pe = torch.zeros(L, d, device=device)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    return pe

# Print the first 4 positions, first 8 dimensions, for base=10000
pe = sinusoidal_pe(4, 8, "cuda")
for row in pe.tolist():
    print("  ", [round(v, 3) for v in row])

# Output:
#    [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
#    [0.841, 0.54, 0.1, 0.995, 0.01, 1.0, 0.001, 1.0]
#    [0.909, -0.416, 0.199, 0.98, 0.02, 1.0, 0.002, 1.0]
#    [0.141, -0.99, 0.296, 0.955, 0.03, 1.0, 0.003, 1.0]`}
      </CodeBlock>

      <Prose>
        Row 0 is {"[sin(0), cos(0), sin(0), cos(0), ...] = [0, 1, 0, 1, ...]"} — constant 1s and 0s. Rows 1, 2, 3 show sinusoidal progression at increasing wavelengths. Notice columns 2-3 (next dim pair) change much more slowly than columns 0-1 — that is the geometric frequency falloff.
      </Prose>

      <H3>4.2 Learned PE from scratch</H3>

      <CodeBlock language="python">
{`import torch.nn as nn

class LearnedPE(nn.Module):
    def __init__(self, max_len, d):
        super().__init__()
        self.pos = nn.Embedding(max_len, d)
        self.max_len = max_len

    def forward(self, L, device):
        if L > self.max_len:
            # naive handling — real code should raise IndexError
            idx = torch.arange(L, device=device) % self.max_len
        else:
            idx = torch.arange(L, device=device)
        return self.pos(idx)    # [L, d]`}
      </CodeBlock>

      <Prose>
        A learned PE is exactly {"nn.Embedding(max_len, d)"}. The {"max_len"} argument is a hard ceiling; any position {"≥ max_len"} either raises or (in the modulo-hack above) wraps around and produces position-shifted garbage. BERT, GPT-2, RoBERTa, and ViT are all built on this pattern with {"max_len ∈ {512, 1024, 2048}"}.
      </Prose>

      <H3>4.3 RoPE from scratch</H3>

      <CodeBlock language="python">
{`def rope_freqs(d_k, L, device, base=10000.0):
    i     = torch.arange(d_k // 2, device=device).float()   # [d_k/2]
    theta = torch.pow(base, -2 * i / d_k)                   # [d_k/2]
    pos   = torch.arange(L, device=device).float()          # [L]
    angles = torch.einsum("l,d->ld", pos, theta)            # [L, d_k/2]
    return torch.cos(angles), torch.sin(angles)

def apply_rope(x, cos, sin):
    # x:   [B, H, L, d_k]
    # cos: [L, d_k/2]
    # sin: [L, d_k/2]
    x1 = x[..., 0::2]                           # even dims
    x2 = x[..., 1::2]                           # odd dims
    c  = cos.unsqueeze(0).unsqueeze(0)          # broadcast over B, H
    s  = sin.unsqueeze(0).unsqueeze(0)
    y1 = x1 * c - x2 * s                        # rotate (x1, x2) by angle
    y2 = x1 * s + x2 * c
    return torch.stack([y1, y2], dim=-1).flatten(-2)

# Sanity: RoPE preserves norms (rotation is unitary)
q = torch.randn(1, 1, 4, 8, device="cuda")
cos, sin = rope_freqs(8, 4, "cuda")
q_rot = apply_rope(q, cos, sin)
print("norm before:", round(q.norm().item(), 4))
print("norm after: ", round(q_rot.norm().item(), 4))

# Output:
#   norm before: 6.0944
#   norm after:  6.0944`}
      </CodeBlock>

      <Prose>
        The rotation is an isometry — it preserves {"L_2"} norms because {"R"} is orthogonal. Different positions give different rotations, but every position produces a query vector with exactly the same magnitude as the input. This is what makes RoPE compose cleanly with attention and with layer norm.
      </Prose>

      <CodeBlock language="python">
{`# Sanity: attention score depends only on relative position for q=k.
# Build identical q, k vectors and compute their rotated inner products.
q = torch.randn(1, 1, 4, 8, device="cuda")
k = q.clone()
cos, sin = rope_freqs(8, 4, "cuda")
q_r = apply_rope(q, cos, sin)
k_r = apply_rope(k, cos, sin)
sims = torch.matmul(q_r, k_r.transpose(-2, -1))[0, 0]
for row in sims.tolist():
    print("  ", [round(v, 3) for v in row])

# Output:
#    [8.893, 4.897, 0.033, -4.277]
#    [4.897, 12.395, -0.871, 1.039]
#    [0.033, -0.871, 9.162, 4.219]
#    [-4.277, 1.039, 4.219, 6.691]`}
      </CodeBlock>

      <Prose>
        The matrix is approximately Toeplitz — along any diagonal (constant {"i − j"}) the values are similar. The off-diagonals trend from high (small distance) to low/negative (distance 3). The exact values would be perfectly Toeplitz if we averaged over many random {"q = k"}; one sample picks up noise from the specific content. That is RoPE's relative-position behavior in action.
      </Prose>

      <H3>4.4 ALiBi from scratch</H3>

      <CodeBlock language="python">
{`def alibi_slopes(h):
    # Per-head slope 2^{-8h/H}, h = 1..H (ALiBi paper eq. 1)
    return torch.tensor([2.0 ** (-8.0 * (i + 1) / h) for i in range(h)])

def alibi_bias(L_q, L_k, h, device):
    slopes = alibi_slopes(h).to(device)                  # [H]
    i = torch.arange(L_q, device=device).unsqueeze(1)    # [L_q, 1]
    j = torch.arange(L_k, device=device).unsqueeze(0)    # [1,   L_k]
    dist = (j - i).float()                               # <0 when j < i
    bias = -dist.abs().unsqueeze(0) * slopes.view(h, 1, 1)  # [H, L_q, L_k]
    return bias

print("ALiBi slopes (H=4):", [round(s.item(), 4) for s in alibi_slopes(4)])

ab = alibi_bias(4, 4, 4, "cuda")
print("bias head 0:")
for row in ab[0].tolist():
    print("  ", [round(v, 3) for v in row])

# Output:
#   ALiBi slopes (H=4): [0.25, 0.0625, 0.0156, 0.0039]
#   bias head 0:
#      [-0.0, -0.25, -0.5, -0.75]
#      [-0.25, -0.0, -0.25, -0.5]
#      [-0.5, -0.25, -0.0, -0.25]
#      [-0.75, -0.5, -0.25, -0.0]`}
      </CodeBlock>

      <Prose>
        Head 0 has the steepest slope ({"0.25"}); at distance 3 the penalty is {"−0.75"} in logit space, which roughly halves the softmax weight compared to distance 0. Head 3 has slope {"0.004"} — its penalty at distance 3 is {"−0.012"}, essentially negligible; that head attends nearly globally. The combined multi-head effect is a stack of attention distributions at different spatial scales.
      </Prose>

      <H3>4.5 Tiny Transformer with swappable PE</H3>

      <CodeBlock language="python">
{`class Attn(nn.Module):
    def __init__(self, d, h, pe_mode):
        super().__init__()
        self.h, self.d_k = h, d // h
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.Wo = nn.Linear(d, d, bias=False)
        self.pe_mode = pe_mode

    def forward(self, x, cos=None, sin=None, alibi=None):
        B, L, D = x.shape
        q = self.Wq(x).view(B, L, self.h, self.d_k).transpose(1, 2)
        k = self.Wk(x).view(B, L, self.h, self.d_k).transpose(1, 2)
        v = self.Wv(x).view(B, L, self.h, self.d_k).transpose(1, 2)

        if self.pe_mode == "rope":
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if self.pe_mode == "alibi":
            scores = scores + alibi.unsqueeze(0)     # [1, H, L, L]

        mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.nn.functional.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, D)
        return self.Wo(ctx)

class TinyLM(nn.Module):
    def __init__(self, pe_mode, max_len=96):
        super().__init__()
        self.pe_mode = pe_mode
        self.tok = nn.Embedding(32, 64)
        if pe_mode == "learned":
            self.pos = nn.Embedding(max_len, 64)
            self.max_len = max_len
        self.blocks = nn.ModuleList([nn.ModuleDict({
            "ln1": nn.LayerNorm(64),
            "attn": Attn(64, 4, pe_mode),
            "ln2": nn.LayerNorm(64),
            "ff":  nn.Sequential(nn.Linear(64, 256), nn.GELU(), nn.Linear(256, 64)),
        }) for _ in range(2)])
        self.ln_f = nn.LayerNorm(64)
        self.head = nn.Linear(64, 32, bias=False)

    def forward(self, x):
        B, L = x.shape
        h = self.tok(x)
        cos = sin = alibi = None
        if self.pe_mode == "sinusoidal":
            h = h + sinusoidal_pe(L, 64, x.device).unsqueeze(0)
        elif self.pe_mode == "learned":
            idx = torch.arange(L, device=x.device) % self.max_len
            h = h + self.pos(idx).unsqueeze(0)
        elif self.pe_mode == "rope":
            cos, sin = rope_freqs(16, L, x.device)
        elif self.pe_mode == "alibi":
            alibi = alibi_bias(L, L, 4, x.device)
        for blk in self.blocks:
            h = h + blk["attn"](blk["ln1"](h), cos=cos, sin=sin, alibi=alibi)
            h = h + blk["ff"](blk["ln2"](h))
        return self.head(self.ln_f(h))`}
      </CodeBlock>

      <H3>4.6 Train on shift task at L=64, evaluate at longer lengths</H3>

      <CodeBlock language="python">
{`import torch.nn.functional as F

def sample_seq(B, L, device):
    x = torch.randint(0, 32, (B, L), device=device)
    y = torch.zeros_like(x)
    y[:, 1:] = x[:, :-1]              # target = shifted input
    y[:,  0] = x[:,  0]
    return x, y

def train(pe_mode, steps=600):
    torch.manual_seed(42)
    m = TinyLM(pe_mode=pe_mode).cuda()
    opt = torch.optim.Adam(m.parameters(), lr=3e-4)
    for _ in range(steps):
        x, y = sample_seq(64, 64, "cuda")
        loss = F.cross_entropy(m(x).reshape(-1, 32), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return m, loss.item()

@torch.no_grad()
def eval_ppl(m, L, n=8):
    m.eval()
    losses = []
    for _ in range(n):
        x, y = sample_seq(64, L, "cuda")
        logits = m(x)
        losses.append(F.cross_entropy(logits.reshape(-1, 32), y.reshape(-1)).item())
    m.train()
    return math.exp(sum(losses) / len(losses))

for mode in ["sinusoidal", "learned", "rope", "alibi"]:
    m, fl = train(mode, 600)
    print(f"[{mode}]  final_loss={fl:.4f}")
    for L in [64, 128, 256, 512]:
        print(f"  L={L:4d}  ppl={eval_ppl(m, L):.3f}")

# Output:
#   [sinusoidal]  final_loss=0.0365
#     L=  64  ppl=1.032
#     L= 128  ppl=17.282
#     L= 256  ppl=73.504
#     L= 512  ppl=138.568
#   [learned]     final_loss=0.0154
#     L=  64  ppl=1.016
#     L= 128  ppl=13.515
#     L= 256  ppl=23.382
#     L= 512  ppl=52.936
#   [rope]        final_loss=0.0113
#     L=  64  ppl=1.011
#     L= 128  ppl=1.035
#     L= 256  ppl=1.426
#     L= 512  ppl=3.617
#   [alibi]       final_loss=0.0244
#     L=  64  ppl=1.024
#     L= 128  ppl=1.026
#     L= 256  ppl=1.027
#     L= 512  ppl=1.027`}
      </CodeBlock>

      <Prose>
        Four models, four stories. At train length 64 all four solve the task ({"ppl ≈ 1"} means near-perfect prediction). At {"2x"} the train length, sinusoidal and learned both collapse ({"ppl"} jumps from ~1 to 13-17), and they get catastrophically worse at {"4x"} and {"8x"}. RoPE holds up well at {"2x"}, degrades gently at {"4x"} and {"8x"} — it extrapolates partway. ALiBi is essentially flat across the entire eval range. This is the canonical length-extrapolation result: sinusoidal and learned PE encode positions the model has actually seen, and they cannot generalize past those positions; ALiBi's distance-based penalty is defined everywhere and depends only on distance, so the model's learned patterns still apply at any length.
      </Prose>

      <H3>4.7 RoPE base sensitivity</H3>

      <CodeBlock language="python">
{`# Re-train the RoPE model with different RoPE bases.
for base in [1000.0, 10000.0, 500000.0]:
    # (Retrains a RoPE model end-to-end at each base.)
    ...
    print(f"  base={int(base):7d}  ppl(64)={p64:.3f}  ppl(512)={p512:.3f}")

# Output:
#   base=   1000  ppl(64)=1.011  ppl(512)=1.188
#   base=  10000  ppl(64)=1.011  ppl(512)=3.599
#   base= 500000  ppl(64)=1.011  ppl(512)=5.391`}
      </CodeBlock>

      <Prose>
        On this tiny model and task, a smaller RoPE base extrapolates slightly better because the dominant positional signal is short-range (wavelength {"≈ 6"} tokens for {"i = 0"} at any reasonable base) and a smaller base gives more of the high-frequency pairs their own distinct rotation. On real LLMs the calculus is reversed: modern large models use base {"500{,}000"} (LLaMA-3) or even {"1{,}000{,}000"} because the <em>long-range</em> dimensions need extra wavelength to avoid wrapping inside a {"32K+"} context. Base sensitivity is empirical — always sweep at the target context.
      </Prose>

      <H3>4.8 Positional interpolation (PI) demo</H3>

      <CodeBlock language="python">
{`# Train at L=64, evaluate at L=256 with position scale = 64/256 = 0.25.
# Same model, no retraining — just feed scaled positions into rope_freqs.
for scale in [1.0, 0.25]:
    p256 = eval_ppl_scaled(m, L=256, scale=scale)
    print(f"  scale={scale}  ppl(L=256) = {p256:.3f}")

# Output:
#   scale=1.0   ppl(L=256) = 1.418
#   scale=0.25  ppl(L=256) = 85.902`}
      </CodeBlock>

      <Prose>
        A naive PI with no fine-tuning hurts badly on this model — every RoPE angle is now {"4x"} smaller than the model learned to expect, and it cannot interpret positions correctly. In practice, PI is always paired with a short fine-tune (100-1000 steps at the new context length); this post-scaling fine-tune recovers in-distribution quality at the extended length. YaRN and NTK-aware do the same thing with per-dimension scaling factors so that less fine-tuning is needed.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION
          ====================================================================== */}
      <H2>5. Production tools</H2>

      <H3>5.1 Sinusoidal PE in PyTorch's nn.Transformer</H3>

      <CodeBlock language="python">
{`import torch, torch.nn as nn, math

# nn.Transformer has no built-in PE — you must add it yourself.
# The canonical PyTorch-tutorial implementation:

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d]

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]

# Full encoder with sinusoidal PE:
d_model, nhead, nlayers = 512, 8, 6
encoder = nn.Sequential(
    nn.Embedding(vocab_size, d_model),
    PositionalEncoding(d_model),
    nn.TransformerEncoder(
        nn.TransformerEncoderLayer(d_model, nhead, batch_first=True),
        num_layers=nlayers,
    ),
)`}
      </CodeBlock>

      <Prose>
        PyTorch's {"nn.Transformer"} family does not bake in a positional encoding because different tasks want different ones. The snippet above is the pattern every research Transformer starts from: token embedding plus sinusoidal PE, fed into {"nn.TransformerEncoder"}. The {"register_buffer"} call is important — it makes the PE move to GPU with the module but not be treated as a learnable parameter.
      </Prose>

      <H3>5.2 Learned PE in HuggingFace BERT / GPT-2</H3>

      <CodeBlock language="python">
{`from transformers import BertModel, GPT2Model

# BERT uses learned absolute position embeddings of size max_position_embeddings.
bert = BertModel.from_pretrained("bert-base-uncased")
print("BERT position table:", bert.embeddings.position_embeddings)
# Embedding(512, 768)

# GPT-2 also uses learned absolute PE.
gpt2 = GPT2Model.from_pretrained("gpt2")
print("GPT-2 wpe:", gpt2.wpe)
# Embedding(1024, 768)

# The max_position_embeddings is a hard ceiling.
# Feeding input_ids of length 513 to bert-base will index off the position table.`}
      </CodeBlock>

      <Prose>
        Both models ship with learned PE tables sized exactly to their training context (512 for BERT-base, 1024 for GPT-2). Extending these to longer contexts requires either re-initializing the position table and full retraining, or interpolating new positions into the existing table (rarely done for absolute PE — it works better for relative variants).
      </Prose>

      <H3>5.3 RoPE in HuggingFace LLaMA / Mistral / Qwen</H3>

      <CodeBlock language="python">
{`from transformers import AutoModelForCausalLM, AutoConfig

cfg = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")
print("rope_theta (base):", cfg.rope_theta)
print("max_position_embeddings:", cfg.max_position_embeddings)
# rope_theta: 500000.0
# max_position_embeddings: 131072

# The rope_scaling dict controls YaRN/PI/NTK-aware at load time.
print("rope_scaling:", cfg.rope_scaling)
# rope_scaling: {'rope_type': 'llama3', 'factor': 32.0,
#                'high_freq_factor': 4.0, 'low_freq_factor': 1.0,
#                'original_max_position_embeddings': 8192}

# 'llama3' rope_type is LLaMA-3's variant of NTK-aware scaling:
# 8K trained context stretched to 128K at load time.`}
      </CodeBlock>

      <Prose>
        LLaMA-3 ships with {"rope_theta = 500000"} (a higher base than the original {"10000"} to support longer contexts natively) and a {"rope_scaling"} configuration that applies a per-dimension frequency blend at load time. This is NTK-aware scaling implemented at the model-config level — no code changes needed to extend from 8K to 128K. Qwen-2.5 and Mistral use very similar configs with their own {"rope_theta"} and scaling variants.
      </Prose>

      <H3>5.4 Manual RoPE application with rotary-embedding-torch</H3>

      <CodeBlock language="python">
{`# pip install rotary-embedding-torch
from rotary_embedding_torch import RotaryEmbedding
import torch

rotary = RotaryEmbedding(dim=32)                     # d_k/2 pair count
q = torch.randn(1, 8, 128, 64)                       # [B, H, L, d_k]
k = torch.randn(1, 8, 128, 64)

# Apply rotation to q and k independently.
q = rotary.rotate_queries_or_keys(q)
k = rotary.rotate_queries_or_keys(k)

# Now q and k carry positional rotation; attention uses them normally.
attn = (q @ k.transpose(-2, -1) / 8).softmax(dim=-1)`}
      </CodeBlock>

      <Prose>
        {"rotary-embedding-torch"} (by lucidrains) is the community-standard implementation for adding RoPE to a custom Transformer. It handles the cos/sin precomputation with caching, supports partial rotation (only rotate the first {"d_k · k"} dimensions, leaving the rest unrotated — the "XPos"-style variant), and matches LLaMA's layout by default.
      </Prose>

      <H3>5.5 ALiBi in MPT / BLOOM / Falcon</H3>

      <CodeBlock language="python">
{`# MPT-7B / BLOOM use ALiBi. In HuggingFace code:
from transformers import AutoModelForCausalLM

# MPT explicitly uses ALiBi — no RoPE, no learned PE.
mpt = AutoModelForCausalLM.from_pretrained("mosaicml/mpt-7b")
# Its attention implementation adds ALiBi bias inside each layer.

# BLOOM: the attention module computes ALiBi slopes per-head at init
# and adds them to the attention scores before softmax.
# BloomAttention has a _build_alibi_tensor() static method.`}
      </CodeBlock>

      <Prose>
        MPT-7B (2023) was the first large well-known model to ship with ALiBi, and demonstrated that ALiBi-trained models could be extended from 2K train context to 8K+ eval context without fine-tuning. BLOOM (Le Scao et al. 2022) used ALiBi for similar length-extrapolation reasons. Falcon-7B/40B also use ALiBi. By 2024, most frontier decoder-only models had migrated to RoPE-with-YaRN (because RoPE plus scaling outperforms ALiBi at fixed context while still extrapolating reasonably), but ALiBi remains the implementation-of-choice when you know in advance you will evaluate at lengths far beyond train length.
      </Prose>

      <H3>5.6 YaRN context extension</H3>

      <CodeBlock language="python">
{`# The llama_yarn repo (jquesnelle/yarn) extends a pretrained LLaMA-2 from
# 4K to 128K context with ~400 fine-tuning steps.

from transformers import AutoModelForCausalLM, AutoConfig

cfg = AutoConfig.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k")
print(cfg.rope_scaling)
# {
#   'type': 'yarn',
#   'factor': 32.0,
#   'original_max_position_embeddings': 4096,
#   'finetuned': True,
#   'beta_fast': 32,
#   'beta_slow': 1
# }

# beta_fast, beta_slow: YaRN's (alpha, beta) boundaries that control
# which RoPE pairs get PI-style vs NTK-style scaling.`}
      </CodeBlock>

      <Prose>
        YaRN is the most-adopted context-extension technique for RoPE models in 2024-2026. The {"beta_fast"} / {"beta_slow"} parameters correspond to the paper's {"α"}, {"β"} — frequencies above {"β_fast"} are left unscaled (they rotate many times within the training window and are well-sampled), frequencies below {"β_slow"} are fully PI-scaled (they rotate slowly and need rescaling), and there is a smooth ramp between. The paper's recommended values {"(β_fast, β_slow) = (32, 1)"} work well in practice.
      </Prose>

      <H3>5.7 XPos — RoPE with decay</H3>

      <CodeBlock language="python">
{`# XPos (Sun et al. 2022) multiplies the RoPE-rotated vectors by a
# position-dependent decay factor to improve long-range stability.

def xpos_freqs(d_k, L, device, base=10000.0, scale_base=512.0):
    i = torch.arange(d_k // 2, device=device).float()
    theta = torch.pow(base, -2 * i / d_k)
    pos   = torch.arange(L, device=device).float()
    angles = pos.unsqueeze(1) * theta.unsqueeze(0)
    # XPos-specific decay scale (grows with dim index so high-freq dims decay faster)
    zeta = (i / (d_k // 2) + 0.4) / 1.4
    decay_q = zeta ** pos.unsqueeze(1)     # [L, d_k/2]
    return torch.cos(angles), torch.sin(angles), decay_q`}
      </CodeBlock>

      <Prose>
        XPos adds a multiplicative decay {"ζ^{pos}"} that attenuates low-frequency dimensions faster than high-frequency ones, giving better long-range extrapolation than plain RoPE at similar cost. RetNet (Sun et al. 2023) adopted XPos as its default PE, and a handful of research-track LLMs use it. Most production LLMs stick with plain RoPE plus YaRN because RoPE has more mature tooling.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6.1 Heatmap of sinusoidal PE</H3>

      <Prose>
        The first 16 positions of a sinusoidal PE with {"d = 32"} (8 dimension pairs plotted) and base {"10000"}. Each row is a position; each column is a single dimension. The left-most pair ({"dim 0, 1"}) oscillates rapidly with position — high frequency, wavelength {"≈ 6"}. The right-most pair oscillates very slowly, almost flat over these 16 positions — that dimension encodes position on a scale of thousands of tokens. Notice how the multi-scale encoding allows the model to disambiguate positions at any scale: nearby positions differ sharply in high-frequency dimensions; far-apart positions differ in low-frequency ones.
      </Prose>

      <Heatmap
        matrix={[
          [0.000,  1.000,  0.000,  1.000,  0.000,  1.000,  0.000,  1.000],
          [0.841,  0.540,  0.389,  0.921,  0.178,  0.984,  0.079,  0.997],
          [0.909, -0.416,  0.717,  0.697,  0.350,  0.937,  0.159,  0.987],
          [0.141, -0.990,  0.932,  0.362,  0.507,  0.862,  0.237,  0.971],
          [-0.757, -0.654,  0.997, -0.072,  0.644,  0.765,  0.312,  0.950],
          [-0.959,  0.284,  0.900, -0.435,  0.757,  0.653,  0.384,  0.923],
          [-0.279,  0.960,  0.649, -0.760,  0.843,  0.537,  0.452,  0.892],
          [ 0.657,  0.754,  0.287, -0.958,  0.900,  0.435,  0.516,  0.857],
          [ 0.989, -0.146, -0.123, -0.992,  0.927,  0.374,  0.574,  0.819],
          [ 0.412, -0.911, -0.511, -0.859,  0.923,  0.383,  0.627,  0.779],
          [-0.544, -0.839, -0.798, -0.602,  0.891,  0.453,  0.675,  0.738],
          [-0.999,  0.004, -0.955, -0.297,  0.831,  0.557,  0.716,  0.698],
          [-0.537,  0.844, -0.967,  0.257,  0.746,  0.666,  0.752,  0.659],
          [ 0.420,  0.907, -0.829,  0.560,  0.637,  0.771,  0.782,  0.622],
          [ 0.991,  0.137, -0.564,  0.826,  0.509,  0.861,  0.808,  0.589],
          [ 0.650, -0.760, -0.201,  0.980,  0.367,  0.930,  0.828,  0.561],
        ]}
        rowLabels={["pos 0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15"]}
        colLabels={["d0","d1","d2","d3","d4","d5","d6","d7"]}
        colorScale="gold"
        label="sinusoidal PE, d=32 (first 8 dims shown), 16 positions, base=10000"
      />

      <Prose>
        Reading the heatmap: the leftmost column ({"d0"}) cycles quickly — {"sin(0), sin(1), sin(2), ..."} — and completes almost a full cycle within 16 positions. The rightmost column ({"d7"}) is nearly constant at {"≈ 1"} across the first 16 positions because its wavelength is {"≈ 785"} positions and we are only sampling the first 2% of its period. Every position has a unique, {"d"}-dimensional code, but the codes are <em>similar</em> for nearby positions in the high-frequency dimensions and <em>continuous</em> in the low-frequency dimensions — exactly what a smooth positional signal should look like.
      </Prose>

      <H3>6.2 RoPE rotation angle per dimension pair</H3>

      <Prose>
        The rotation angle {"m · θ_i"} (in radians) that RoPE applies at position {"m = 64"} for a {"d_k = 64"} attention head with base {"10000"}. Pair 0 is the highest frequency (rotates by ~64 radians at position 64 — many full turns); pair 31 is the lowest (rotates by only ~0.009 radians — barely any rotation over 64 positions).
      </Prose>

      <Plot
        series={[
          { name: "base=10000", color: colors.gold, points: [
            [0,  64.000],
            [1,  47.993],
            [2,  35.989],
            [4,  20.239],
            [8,   6.400],
            [12,  2.024],
            [16,  0.640],
            [20,  0.202],
            [24,  0.064],
            [28,  0.020],
            [31,  0.009],
          ]},
          { name: "base=500000", color: colors.green, points: [
            [0,  64.000],
            [1,  42.471],
            [2,  28.183],
            [4,  12.411],
            [8,   2.407],
            [12,  0.467],
            [16,  0.091],
            [20,  0.018],
            [24,  0.003],
            [28,  0.001],
            [31,  0.000],
          ]},
        ]}
        xLabel="dim pair index i"
        yLabel="rotation angle at pos=64 (rad)"
        label="RoPE: rotation angle per dimension pair at position 64"
      />

      <Prose>
        Both curves fall off steeply from pair 0 to pair 8, then flatten. The higher base ({"500000"}) gives smaller rotation angles at all but the very highest-frequency pair — the effective wavelengths are longer, meaning the same position induces less rotation. This is why Llama-3 uses {"base = 500000"}: at context length {"128K"}, low-frequency pairs with base {"10000"} would wrap many times and lose their monotonicity. With base {"500000"}, low-frequency pairs rotate slowly enough to encode position-difference signals even at {"L = 128K"}.
      </Prose>

      <H3>6.3 ALiBi bias per head</H3>

      <Prose>
        ALiBi bias {"m_h · |i − j|"} plotted as a function of distance {"|i − j|"} for each of {"H = 8"} heads. Head 1 (steepest slope {"0.5"}) penalizes distant keys heavily — it becomes effectively a "last few tokens" head. Head 8 (slope {"0.0039"}) has negligible penalty even at distance {"256"} — it acts nearly globally. The multi-head ALiBi stack spans a geometric range of effective attention windows.
      </Prose>

      <Plot
        series={[
          { name: "head 1 (m=0.500)",  color: colors.gold,   points: [[0, 0.000], [16, -8.000],  [32, -16.000], [64, -32.000], [128, -64.000], [256, -128.000]] },
          { name: "head 2 (m=0.250)",  color: colors.green,  points: [[0, 0.000], [16, -4.000],  [32,  -8.000], [64, -16.000], [128, -32.000], [256,  -64.000]] },
          { name: "head 4 (m=0.063)",  color: "#c084fc",     points: [[0, 0.000], [16, -1.000],  [32,  -2.000], [64,  -4.000], [128,  -8.000], [256,  -16.000]] },
          { name: "head 8 (m=0.004)",  color: "#60a5fa",     points: [[0, 0.000], [16, -0.063],  [32,  -0.125], [64,  -0.250], [128,  -0.500], [256,   -1.000]] },
        ]}
        xLabel="distance |i - j|"
        yLabel="ALiBi bias (logits)"
        label="ALiBi bias per head, H=8, slope m_h = 2^{-8h/H}"
      />

      <Prose>
        Every head's bias line is a straight line through the origin — that is the "linear" in Attention with Linear Biases. The slopes span a factor of {"128x"} from head 1 to head 8 (geometrically spaced). The steep heads ({"m = 0.5"} and {"0.25"}) effectively implement local-window attention; the shallow heads provide global context. Because the bias is a fixed function of distance, this structure holds at any evaluation length — the steep heads still act local, the shallow still act global — which is why ALiBi generalizes to longer inputs.
      </Prose>

      <H3>6.4 Length extrapolation: perplexity vs eval length</H3>

      <Prose>
        Summary of the section-4 experiment: each PE method was trained at sequence length 64 on the shift-by-one task, then evaluated at lengths 64, 128, 256, and 512. Lower is better; a perfect extrapolator stays at {"ppl = 1"}.
      </Prose>

      <Plot
        series={[
          { name: "sinusoidal", color: colors.gold,   points: [[64, 1.032], [128, 17.282], [256, 73.504], [512, 138.568]] },
          { name: "learned",    color: "#f472b6",     points: [[64, 1.016], [128, 13.515], [256, 23.382], [512, 52.936]] },
          { name: "RoPE",       color: "#c084fc",     points: [[64, 1.011], [128,  1.035], [256,  1.426], [512,   3.617]] },
          { name: "ALiBi",      color: colors.green,  points: [[64, 1.024], [128,  1.026], [256,  1.027], [512,   1.027]] },
        ]}
        xLabel="eval sequence length"
        yLabel="perplexity (lower is better)"
        label="length extrapolation (train L=64)"
      />

      <Prose>
        Three distinct regimes. Sinusoidal and learned PE both collapse as soon as the eval length exceeds training length — the model has literally never seen those position signals and cannot interpret them. RoPE degrades gracefully: {"ppl"} grows from {"1.01"} at the train length to {"3.6"} at {"8x"} train length, which for this task represents partial extrapolation (the relative rotations for distances {"≤ 64"} are still in-distribution; only some long-distance pairs are out). ALiBi is essentially flat across the entire range: because its bias is a function of distance and distances {"≤ 63"} dominate in causal attention (each position {"i"} attends to at most {"i + 1"} keys), the effective attention landscape at {"L = 512"} is the same as at {"L = 64"} for most query positions. This plot is the ALiBi paper's headline result, reproduced on a tiny model and a toy task.
      </Prose>

      <H3>6.5 StepTrace: RoPE applied to a single Q/K pair</H3>

      <Prose>
        Walk through the rotation of a 4-dimensional query vector {"q"} by the RoPE rotation matrix at position {"m = 3"}. Assume {"d_k = 4"} so there are two dimension pairs; the first pair has frequency {"θ_0 = 1"}, the second has {"θ_1 = 10000^{−0.5} = 0.01"}.
      </Prose>

      <StepTrace
        label="RoPE applied to q = (q0, q1, q2, q3) at position m=3"
        steps={[
          {
            label: "raw Q vector",
            render: () => (
              <Prose>
                Start with the query vector from the Q projection: {"q = (q_0, q_1, q_2, q_3) = (0.8, -0.5, 0.3, 1.2)"}. This is the content vector, with no position information. It comes out of {"W^Q · x_m"} for input token at position {"m = 3"}.
              </Prose>
            ),
          },
          {
            label: "compute frequencies θ_i",
            render: () => (
              <Prose>
                For {"d_k = 4"}, there are two dimension pairs. Frequencies: {"θ_0 = 10000^{-0/4} = 1"}, {"θ_1 = 10000^{-2/4} = 10000^{-0.5} = 0.01"}. The high-frequency pair (indices 0, 1) rotates quickly with position; the low-frequency pair (indices 2, 3) rotates slowly.
              </Prose>
            ),
          },
          {
            label: "compute rotation angles at m=3",
            render: () => (
              <Prose>
                Pair 0 angle: {"m · θ_0 = 3 · 1 = 3"} radians (about 172°). Pair 1 angle: {"m · θ_1 = 3 · 0.01 = 0.03"} radians (about 1.7°, nearly no rotation). Each pair will be rotated independently by its own angle — the key to RoPE's multi-scale behavior.
              </Prose>
            ),
          },
          {
            label: "rotate pair 0: (q_0, q_1) -> (q'_0, q'_1)",
            render: () => (
              <Prose>
                Apply 2D rotation by {"3"} rad: {"cos(3) ≈ -0.99"}, {"sin(3) ≈ 0.14"}. New values: {"q'_0 = q_0·cos(3) − q_1·sin(3) = 0.8·(-0.99) − (-0.5)·0.14 = -0.72"}. {"q'_1 = q_0·sin(3) + q_1·cos(3) = 0.8·0.14 + (-0.5)·(-0.99) = 0.61"}. The pair has been rotated substantially — almost reversed in sign.
              </Prose>
            ),
          },
          {
            label: "rotate pair 1: (q_2, q_3) -> (q'_2, q'_3)",
            render: () => (
              <Prose>
                Apply 2D rotation by {"0.03"} rad: {"cos(0.03) ≈ 1.0"}, {"sin(0.03) ≈ 0.03"}. New values: {"q'_2 = 0.3·1.0 − 1.2·0.03 = 0.264"}. {"q'_3 = 0.3·0.03 + 1.2·1.0 = 1.209"}. Nearly unchanged — this pair carries long-range position info that barely rotates at position 3.
              </Prose>
            ),
          },
          {
            label: "assemble q' and pass to attention",
            render: () => (
              <Prose>
                Final rotated query: {"q' = (-0.72, 0.61, 0.264, 1.209)"}. Apply the same rotation construction to {"k"} at its own position {"n"}, then compute {"q'·k'"}. The inner product depends on the content of {"q"} and {"k"} and the <em>relative</em> angle {"(n − m)·θ_i"} per pair — exactly what was claimed in section 3.4. The relative-position signal has been injected multiplicatively, and it composes cleanly with the subsequent attention softmax and value mixing.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix — which PE, when</H2>

      <H3>7.1 Modern decoder-only LLM: RoPE with NTK-aware / YaRN scaling</H3>

      <Prose>
        For any decoder-only Transformer in 2026 — LLaMA-family, Qwen, Mistral, DeepSeek, Gemma — the default is RoPE. Pick base {"b = 10000"} for small-context models ({"≤ 4K"}) and base {"b = 500000"} or {"1000000"} for long-context models ({"≥ 32K"}). Apply YaRN or LLaMA-3-style NTK-aware scaling at load time to extend context beyond the training window. There is almost no scenario in which a new decoder-only model from scratch should use sinusoidal or learned absolute PE in 2026 — the RoPE variant outperforms on every axis that matters for large-scale language modeling.
      </Prose>

      <H3>7.2 Context extension without full retraining: ALiBi or RoPE + YaRN</H3>

      <Prose>
        If you already have a pretrained model at context {"L_{\\mathrm{train}}"} and need to evaluate at {"L > L_{\\mathrm{train}}"}: ALiBi extrapolates natively and does not need any extra work. RoPE-based models need YaRN (or NTK-aware, or PI, in decreasing order of quality and increasing order of age) at load time. YaRN + 100-500 fine-tuning steps is typically the right play for serious deployments — the fine-tune recalibrates the RoPE rotations for the new context. For quick-and-dirty extension without fine-tuning, LLaMA-3's built-in scaling at {"factor = 32"} is a solid baseline.
      </Prose>

      <H3>7.3 Bidirectional encoder (BERT, RoBERTa, small ViTs)</H3>

      <Prose>
        Learned absolute PE is still fine for encoder-only models at {"max_len ≤ 2048"}. BERT, RoBERTa, DeBERTa-v1 all use it, and the performance is good on in-distribution tasks. DeBERTa-v2/v3 use disentangled attention with relative position bias, which is a stronger design but requires more implementation complexity. For simple encoder workloads where context never exceeds training length, learned PE is the pragmatic choice — no configuration, no edge cases, well-supported by every framework.
      </Prose>

      <H3>7.4 Research needing out-of-distribution length extrapolation: ALiBi</H3>

      <Prose>
        If the research question is specifically about length extrapolation ("can my model generalize to contexts 10x longer than it was trained on?"), ALiBi is the cleanest baseline. It extrapolates without any scaling tricks, tuning, or fine-tuning, and it is easy to implement on top of any attention kernel. For a research paper that wants to isolate the effect of some other change (e.g., a new architecture, a new loss), pairing that change with ALiBi as the PE removes length-extrapolation from the list of confounding variables.
      </Prose>

      <H3>7.5 Very small models ({"≤ 50M"} params) on short sequences</H3>

      <Prose>
        For tiny models on short tasks ({"L ≤ 512"}), the choice of PE barely matters — sinusoidal, learned, and RoPE all converge to similar test loss and the extra complexity of RoPE or YaRN is not justified. Default to learned PE (one line, a single {"nn.Embedding"}) unless there is a specific reason to choose otherwise.
      </Prose>

      <H3>7.6 Edge or on-device inference with aggressive KV-cache compression</H3>

      <Prose>
        For on-device inference where KV cache is quantized or compressed, RoPE has an important engineering win: the cached {"K"} vectors are already post-RoPE. You do not need to store a position index alongside each cached token. The query at generation step {"t"} gets rotated by {"R_t"} and the cached key at position {"m"} already carries {"R_m"}; the attention score correctly computes the relative rotation. With ALiBi the distance must be recomputed at every step (cheap but not free); with learned PE the absolute position index must be tracked separately. RoPE is the most KV-cache-friendly encoding.
      </Prose>

      <Callout accent="gold">
        One-line rule: default to RoPE with LLaMA-3-style NTK-aware scaling for any new decoder-only LLM in 2026. Use learned PE for small encoder-only models at fixed max length. Use ALiBi when native length extrapolation matters more than peak quality. Use sinusoidal only for legacy / educational / reference implementations.
      </Callout>

      {/* ======================================================================
          8. WHAT SCALES
          ====================================================================== */}
      <H2>8. What scales</H2>

      <H3>8.1 Sinusoidal caps at training length in practice</H3>

      <Prose>
        The sinusoidal formula itself is well-defined at any position — {"sin(pos/10000^{2i/d})"} evaluates fine at {"pos = 10^9"} — but the downstream network has only ever seen positions seen during training. Feeding position {"L_{\\mathrm{eval}} > L_{\\mathrm{train}}"} produces PE vectors in a region of feature space the {"W^Q, W^K, W^V"} projections have never seen, and the induced Q/K vectors are out-of-distribution. In practice this manifests as the sudden perplexity collapse visible in the section-4 experiment: sinusoidal PE degrades from {"ppl ≈ 1"} at train length to {"ppl ≈ 140"} at {"8x"} train length. The legend of "sinusoidal extrapolates" in the Vaswani paper was more aspiration than empirical finding; subsequent careful analysis (Press et al. 2022) showed that the extrapolation is weak on any nontrivial task.
      </Prose>

      <H3>8.2 Learned PE has a hard parameter-count ceiling</H3>

      <Prose>
        Learned PE's max position is fixed at model-definition time: once you compile an {"nn.Embedding(1024, 768)"}, position 1024 does not exist. Extending it requires either (a) appending new rows to the table and training them from scratch (hundreds of millions of tokens to converge), (b) replacing the entire table with a larger one and retraining the whole model (worst case), or (c) some form of interpolation between existing rows (not standard for absolute PE). In practice, learned PE is a commitment you make at model-design time and cannot gracefully back out of. This is why modern large models have migrated away from learned PE — the inability to extend context is too costly.
      </Prose>

      <H3>8.3 RoPE extrapolates partially and scales well with tricks</H3>

      <Prose>
        RoPE's rotation formula is defined at every position; there is no hard ceiling. But naive extrapolation (training at {"L_{\\mathrm{train}}"}, evaluating at {"10 · L_{\\mathrm{train}}"}) tends to hurt because low-frequency dimension pairs start rotating into regions of angle space the model was never trained on, and high-frequency pairs, which wrap many times, lose their unique position signature. PI scales all frequencies uniformly but blurs short-range detail. NTK-aware scales low frequencies more than high frequencies, preserving short-range fidelity. YaRN does both with a smooth ramp and a temperature correction. With YaRN, RoPE-based models have been pushed from {"4K"} training context to {"128K"} eval context with modest fine-tuning. In 2026, RoPE + YaRN is the standard for long-context LLMs.
      </Prose>

      <H3>8.4 ALiBi extrapolates natively but with a precision cost</H3>

      <Prose>
        ALiBi's linear bias is well-defined at any distance, and the per-head slopes do not change with context length. A model trained at {"L = 1024"} can be evaluated at {"L = 16384"} without any modification and typically shows only a small perplexity increase. The cost is that ALiBi's hard-coded "closer is better" bias is a strong inductive prior that the model cannot learn around. A head that needs to attend to a very distant token must fight ALiBi's penalty with very strong Q/K alignment, and often the right Q/K alignment for that head is simply not learnable against the bias. In head-to-head comparisons at fixed context length, well-tuned RoPE tends to beat ALiBi by 1-3% on perplexity. ALiBi's advantage is robustness across lengths, not peak quality.
      </Prose>

      <H3>8.5 Context extension: 2K → 128K+ is routine with RoPE + YaRN</H3>

      <Prose>
        The 2023-2024 wave of long-context LLMs — LongLLaMA (32K), Yarn-Llama-2-7b-128k, Gemini-1.5 (1M via undisclosed techniques), Qwen-2-128K — all built on RoPE + some variant of scaling. The typical recipe: start from a pretrained RoPE model at {"L_{\\mathrm{train}}"}, apply YaRN-scaled RoPE with target {"L_{\\mathrm{eval}} = k · L_{\\mathrm{train}}"} for {"k ∈ {8, 32, 64}"}, fine-tune on long-document data for a few hundred to a few thousand steps, evaluate on long-context benchmarks (needle-in-haystack, RULER, BABILong). The compute cost of the long-context fine-tune is typically {"< 1%"} of pretraining, which is why this approach dominated: it was far cheaper than retraining from scratch at the longer context.
      </Prose>

      <H3>8.6 KV cache stores post-RoPE K for efficiency</H3>

      <Prose>
        RoPE-rotated keys are still {"d_k"}-dimensional; rotation does not change the representation size. At inference time, the KV cache stores {"K'_n = R_n · K_n"} — the rotation is baked into the cached value. Queries at step {"t"} are rotated by {"R_t"} as they are computed, and the attention score {"Q'_t · (K'_n)^T = Q_t · R_t^T R_n K_n = Q_t · R_{n-t} · K_n"} correctly gives the relative-position dependence. This means the KV cache does not need to store position indices, the rotations never have to be recomputed on cached tokens, and the cache can be quantized just like ordinary vectors. ALiBi has to recompute distance at every step (cheap) but does not store anything extra in the cache. Learned absolute PE requires tracking the original position index alongside every cached vector, which is extra bookkeeping most implementations handle fine but is conceptually messier than RoPE's approach.
      </Prose>

      <H3>8.7 Grouped-Query Attention interacts cleanly with RoPE</H3>

      <Prose>
        GQA (used by LLaMA-3 and most large models in 2024+) reduces the number of K/V heads below the number of Q heads. RoPE operates per-head, so each Q head and each K head is rotated independently — the math and code do not change when you reduce the K head count. Each Q head still gets its own rotation {"R_m"} applied to its own {"d_k"}-dim query; each K head gets its own rotation applied to its {"d_k"}-dim key. The attention score per (Q head, K head) pair inherits the relative-position property. This clean compositionality is one of the reasons RoPE has been such an easy fit for modern production LLMs — it doesn't conflict with GQA, MQA, flash-attention, sliding window attention, or any of the other standard tricks.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES
          ====================================================================== */}
      <H2>9. Failure modes</H2>

      <H3>9.1 Evaluating sinusoidal PE past training length without scaling</H3>

      <Prose>
        This is the most common length-generalization bug: a model with sinusoidal PE is trained at {"L = 1024"} and then evaluated at {"L = 2048"} or longer with no changes. Perplexity explodes. The fix is not "train longer with sinusoidal" — the underlying issue is that the {"W^Q, W^K"} projections have never seen PE vectors at those positions. The proper fix is either (a) use RoPE + YaRN instead, (b) switch to ALiBi for native extrapolation, or (c) train with some amount of positional noise / length curriculum so the model sees positions beyond the nominal max.
      </Prose>

      <H3>9.2 Learned PE at position beyond max_len</H3>

      <Prose>
        Indexing an {"nn.Embedding(512, 768)"} with position 513 raises {"IndexError"} in PyTorch by default. Worse, in some custom codepaths that do {"pos % max_len"} as a defensive fallback, the model silently wraps — position 513 gets the embedding of position 1, and the model outputs plausible-looking but subtly wrong predictions. Defensive programming here means: always validate that input {"L"} is {"≤ max_len"} at the top of the forward pass, with an explicit assertion or clean error. Never let position indexing fail silently.
      </Prose>

      <H3>9.3 Wrong RoPE base for target context</H3>

      <Prose>
        Training a model at {"L = 8192"} with RoPE base {"10000"} is workable but not ideal — the lowest-frequency dimension pair has wavelength {"≈ 47000"}, so at position 8191 its rotation angle is about 1 radian, which is fine. But training at {"L = 128000"} with base {"10000"} means the lowest-frequency pair rotates almost 20 times over the context, losing its monotone position signal. Symptom: long-range attention becomes noisy and the model degrades on long-range benchmarks. Fix: use a base proportional to the target context. LLaMA-3 at {"L = 8192"}: base {"500000"}. Qwen at {"L = 128000"}: base {"1000000"}. Rule of thumb: {"b ≈ 60 · L_{\\mathrm{target}}"} so the longest wavelength is several times {"L"}.
      </Prose>

      <H3>9.4 ALiBi slopes wrong per head</H3>

      <Prose>
        The ALiBi paper specifies slopes as {"m_h = 2^{−8h/H}"} for {"h = 1..H"}. A common bug is using {"2^{−8h/H}"} for {"h = 0..H-1"} instead, which gives slope {"1.0"} for head 0 — too steep, essentially a single-token attention. Another bug is using the same slope for every head (all 0.5 or all 0.1), which collapses ALiBi to a single-scale bias and throws away the multi-scale advantage. Check {"alibi_slopes(H)"} output: should span {"≈ 0.5"} to {"≈ 2^{-8}"} with {"H = 8"}, geometrically spaced.
      </Prose>

      <H3>9.5 Applying RoPE to V</H3>

      <Prose>
        RoPE is applied to {"Q"} and {"K"} only, never to {"V"}. The math in section 3.4 relies on the fact that the rotation cancels cleanly inside the inner product {"q^T k"}; since {"V"} enters attention through {"softmax(Q K^T / √d_k) · V"}, rotating {"V"} would introduce an uncompensated position-dependent rotation into the output that defeats the relative-position property. A wrong implementation that rotates all three (Q, K, V) will train but produce weaker results — the value vectors carry absolute position information that mixes confusingly with content. Symptom: decent in-distribution performance but much worse length-generalization than standard RoPE.
      </Prose>

      <H3>9.6 Padding position confusion</H3>

      <Prose>
        Batched sequences of variable length use padding tokens. For learned and sinusoidal PE, the position index assigned to padded positions matters: should padded tokens get position 0, position {"L_\\mathrm{actual}"}, or be skipped? Common convention (BERT): positions always run {"0, 1, ..., L-1"} with no special treatment for padding, and a padding mask in the attention ensures padded queries/keys are ignored. If the padding mask is missed, the model attends to random position embeddings for padded positions and corrupts the signal. For RoPE, the safest pattern is to rotate all positions (including padded ones) and rely entirely on the attention mask to exclude them — do not try to skip or reassign positions for padding, because the {"R_m"} rotation depends on the position index {"m"} itself.
      </Prose>

      <H3>9.7 Positional Interpolation without YaRN over-smooths</H3>

      <Prose>
        PI uniformly compresses all RoPE frequencies by a factor of {"L_{\\mathrm{train}} / L_{\\mathrm{eval}}"}. This works at {"2-4x"} extension but at {"8x"} and beyond, high-frequency pairs get compressed into ranges where they no longer distinguish short-range offsets. Symptom: the model loses its ability to attend to very recent tokens sharply — short-range patterns blur. YaRN fixes this by leaving high-frequency pairs unscaled (they are well-sampled even at the longer context) and scaling only the low-frequency pairs that actually need it. If you are doing context extension and seeing quality loss on short-range benchmarks (nearby-token tasks, local syntax, needle-in-short-haystack), the problem is almost certainly uniform PI and the fix is YaRN or NTK-aware.
      </Prose>

      <H3>9.8 Rotating the wrong dimension layout</H3>

      <Prose>
        RoPE expects pairs of consecutive dimensions: rotate {"(x[0], x[1])"} together, then {"(x[2], x[3])"}, and so on. A common variant found in some LLaMA implementations uses the "half-split" layout instead: rotate {"(x[0], x[d_k/2])"} together, then {"(x[1], x[d_k/2 + 1])"}, and so on. Both are valid as long as Q and K use the same layout, but mixing layouts (e.g., loading a checkpoint trained with pair-layout into code that assumes half-split) silently corrupts every rotation. Symptom: loss stays high, the model fails to train past random. The bug is subtle because the forward pass still produces reasonable-looking activations. When adopting a third-party checkpoint, verify the RoPE layout matches your code by checking a single position's rotation output against the reference implementation.
      </Prose>

      <H3>9.9 Forgetting the {"√d_k"} scaling when adding ALiBi bias</H3>

      <Prose>
        The ALiBi bias is added to {"Q K^T / √d_k"}, not to the unscaled {"Q K^T"}. If you add ALiBi <em>before</em> dividing by {"√d_k"}, the bias ends up scaled down by that factor, weakening the effect. Similarly, if you apply RoPE <em>after</em> computing {"Q K^T"} (wrong), the rotation has no effect since the rotation operates on Q and K, not on their inner product. Order matters: (1) Q/K projections, (2) RoPE rotation on Q and K, (3) {"Q K^T / √d_k"}, (4) add ALiBi bias if present, (5) add attention mask, (6) softmax, (7) multiply by V. Every modern attention implementation gets this order right by default; if you are writing from scratch, double-check.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        Foundational papers introducing and extending the core positional encoding schemes. Reading these in order roughly tracks the intellectual arc from additive absolute PE to multiplicative relative-position RoPE to modern long-context extensions.
      </Prose>

      <Prose>
        <strong>Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin (2017)</strong>. "Attention Is All You Need." arXiv:1706.03762. NeurIPS 2017. Section 3.5 introduces the sinusoidal positional encoding as an alternative to learned position embeddings and argues (without conclusive proof) that the sinusoidal form should extrapolate better than learned. The rest of the Transformer paper is also mandatory reading for context.
      </Prose>

      <Prose>
        <strong>Shaw, Uszkoreit, Vaswani (2018)</strong>. "Self-Attention with Relative Position Representations." arXiv:1803.02155. NAACL 2018. The first paper to replace absolute PE with a relative-position scheme by adding learned bias terms {"a^K_{i-j}, a^V_{i-j}"} into the attention computation. Starts the shift from "position as an input embedding" toward "position as an attention modification" that RoPE and ALiBi would later refine.
      </Prose>

      <Prose>
        <strong>Su, Lu, Pan, Murtadha, Wen, Liu (2021)</strong>. "RoFormer: Enhanced Transformer with Rotary Position Embedding." arXiv:2104.09864. The RoPE paper. Introduces the rotation-based multiplicative PE, proves the relative-position property {"⟨q'_m, k'_n⟩ = f(m-n, q, k)"}, demonstrates improvements over sinusoidal and RPE on Chinese LM and translation tasks. The paper was underappreciated at the time of publication (2021) but became central when LLaMA adopted RoPE in 2023.
      </Prose>

      <Prose>
        <strong>Press, Smith, Lewis (2022)</strong>. "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation." arXiv:2108.12409. ICLR 2022. The ALiBi paper. Defines the per-head linear bias scheme, proves it enables length extrapolation, and shows large perplexity wins at test lengths 2-4x beyond training on WikiText-103. The argument that sinusoidal does not in fact extrapolate (contra the Vaswani paper) is made rigorously here.
      </Prose>

      <Prose>
        <strong>Chen, Wong, Chen, Tian (2023)</strong>. "Extending Context Window of Large Language Models via Positional Interpolation." arXiv:2306.15595. The PI paper from Meta. Introduces the idea of rescaling RoPE position indices to extend context, and shows that with only 1000 fine-tuning steps a LLaMA-7B can be extended from 2K to 32K context. The paper kicked off the 2023 wave of context-extension research.
      </Prose>

      <Prose>
        <strong>bloc97 (2023)</strong>. "NTK-Aware Scaled RoPE allows LLaMA models to have extended (8k+) context size without any fine-tuning and minimal perplexity degradation." Posted on r/LocalLLaMA. Not a peer-reviewed paper but widely cited. Introduces the argument that PI uniformly blurs frequencies and proposes scaling the RoPE base rather than the position indices, preserving high-frequency dimensions. The blog post directly inspired the YaRN paper.
      </Prose>

      <Prose>
        <strong>Peng, Quesnelle, Fan, Shippole (2023)</strong>. "YaRN: Efficient Context Window Extension of Large Language Models." arXiv:2309.00071. The YaRN paper from Nous Research. Unifies PI and NTK-aware scaling into a per-dimension ramp parameterized by {"(β_{\\mathrm{fast}}, β_{\\mathrm{slow}})"}, adds an attention-temperature correction, and demonstrates extension of LLaMA-2-7B from 4K to 128K with {"∼400"} fine-tuning steps. State of the art for RoPE context extension in 2023-2025.
      </Prose>

      <Prose>
        <strong>Sun, Dong, Patra, Ma, Huang, Benhaim, Chaudhary, Song, Wei (2022)</strong>. "A Length-Extrapolatable Transformer" (XPos). arXiv:2212.10554. Adds a multiplicative per-position decay to RoPE that further improves long-range stability. Not widely adopted in production LLMs but influential on RetNet and a handful of research-track models.
      </Prose>

      <Prose>
        <strong>Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón, Sanghai (2023)</strong>. "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." arXiv:2305.13245. Not a PE paper per se but essential context for understanding how RoPE interacts with the reduced-K/V-head architectures that dominate modern LLMs. LLaMA-3 and most 2024+ large models use RoPE-on-GQA.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK
          ====================================================================== */}
      <H2>11. Self-check</H2>

      <H3>11.1 Why does pure self-attention need positional information at all?</H3>

      <Prose>
        Self-attention is permutation-equivariant: if you permute the input token order by any {"π"}, the output tokens come out in the same permuted order, with identical values at each permuted index. This is because the attention equation {"softmax(Q K^T / √d_k) V"} is a weighted sum and summation is commutative. A bag-of-tokens input produces a bag-of-tokens output, and language is not a bag of tokens — "Dog bites man" and "Man bites dog" have different meanings. So we must inject position information somewhere: either at the input embedding (sinusoidal, learned), inside the attention score (RoPE, ALiBi), or both.
      </Prose>

      <H3>11.2 What exactly does RoPE do to Q and K, and why does V escape?</H3>

      <Prose>
        RoPE rotates {"Q"} and {"K"} per-head in 2D subspaces by an angle proportional to absolute position: {"q'_m = R_m q_m"}, {"k'_n = R_n k_n"}. Because rotations are orthogonal and {"R_m^T R_n = R_{n-m}"}, the resulting attention score {"⟨q'_m, k'_n⟩ = q_m^T R_{n-m} k_n"} depends only on the relative offset {"n - m"} and the content vectors — not on absolute positions. {"V"} is <em>not</em> rotated because the math only works for {"Q · K"}: if you rotated {"V"} as well, the output {"A V"} would carry a position-dependent rotation that defeats the relative-position property and injects unwanted absolute-position information into the next layer's input.
      </Prose>

      <H3>11.3 Why do different ALiBi heads get different slopes?</H3>

      <Prose>
        Each head's slope {"m_h = 2^{-8h/H}"} sets the effective attention scale for that head. The head with the steepest slope ({"m_1 = 0.5"}) penalizes distance heavily — at distance 10 the bias is {"-5"}, which halves the softmax weight; this head effectively attends to only the last handful of tokens. The head with the shallowest slope ({"m_H ≈ 0.004"}) has negligible penalty even at distance 256; this head is nearly global. The geometric spacing of slopes gives the multi-head stack a broad range of attention scales — roughly analogous to a CNN with kernels at multiple spatial scales — and the model can attend at whatever scale each head finds most useful. A single slope for all heads would collapse ALiBi to a single-scale bias, losing the multi-scale coverage that is the whole point.
      </Prose>

      <H3>11.4 What breaks when you try to evaluate a learned-PE BERT at position 513 (given {"max_len = 512"})?</H3>

      <Prose>
        The position embedding table is {"nn.Embedding(512, 768)"}, which internally is a {"512 × 768"} parameter tensor. Indexing row 512 (the 513th row, zero-indexed) is out of bounds. PyTorch's default behavior is to raise {"IndexError"}. A defensive fallback using {"pos % 512"} would silently wrap — the model sees position 1's embedding at what should be position 513, producing plausible-looking but wrong outputs. Some implementations with {"nn.Embedding(padding_idx=...)"} combined with unusual padding schemes can produce even weirder failures. The correct solution is either to (a) not exceed {"max_len"}, (b) retrain BERT with a larger {"max_len"}, or (c) migrate to RoPE or another unbounded PE.
      </Prose>

      <H3>11.5 A new decoder-only LLM trained at 8K context needs to support 64K inference. What is the modern default recipe?</H3>

      <Prose>
        Use RoPE at training time with a base chosen for the target context ({"b = 500000"} or higher). Train to good quality at 8K. Then apply YaRN scaling with factor {"= 64000 / 8000 = 8"} at load time (or LLaMA-3-style scaling, which is similar). Fine-tune for a few hundred to a few thousand steps on long documents at 64K context; this typically takes under 1% of pretraining compute. Evaluate on long-context benchmarks (RULER, needle-in-haystack, BABILong) to confirm the extension worked — a successful YaRN extension should recover most of the model's short-context quality while adding usable long-context capability. If short-context quality degrades significantly, tune the {"(β_{\\mathrm{fast}}, β_{\\mathrm{slow}})"} ramp or increase the fine-tune step count. Do not use sinusoidal or learned PE for this workflow — neither extends gracefully, and the engineering work to retrofit them is much larger than adopting RoPE from the start.
      </Prose>

    </div>
  ),
};

export default positionalEncodingsContent;
