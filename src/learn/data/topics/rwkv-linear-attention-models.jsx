import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const rwkvLinearAttentionContent = {
  title: "RWKV & Linear Attention Models",
  readTime: "~40 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        By 2020 the transformer had become the default architecture for language, but its self-attention had a costly secret: the attention matrix is quadratic in sequence length. For a sequence of length <Code>L</Code>, you compute an <Code>{"L \\times L"}</Code> score matrix, softmax it, and multiply by values. Time and memory both scale as <Code>{"O(L^2 \\cdot d)"}</Code>. At <Code>L = 1024</Code> this is a nuisance; at <Code>L = 16{,}000</Code> it is the bottleneck; at <Code>L = 100{,}000</Code> it is untenable on ordinary hardware. The question that drove everything that follows: can we keep transformer quality while paying only linear cost in <Code>L</Code>?
      </Prose>

      <Prose>
        The first clean answer came in June 2020 from Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret at Idiap Research Institute. Their ICML paper "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention" (arXiv:2006.16236) observed that softmax attention is a specific choice of similarity function, and if you replace <Code>{"\\text{sim}(q, k) = \\exp(q \\cdot k / \\sqrt{d})"}</Code> with any non-negative kernel that factorizes as <Code>{"\\text{sim}(q, k) = \\phi(q) \\cdot \\phi(k)"}</Code>, you can rewrite attention as a running sum: the model becomes an RNN at inference with <Code>O(L)</Code> time and <Code>O(1)</Code> memory per step, while remaining parallelizable at training in <Code>{"O(L \\cdot d^2)"}</Code>. The accuracy gap to softmax was real — on language modeling, roughly a 5-10 percent perplexity regression at matched compute in 2020 — but the existence of a linear-cost attention was suddenly not speculative.
      </Prose>

      <Prose>
        Around the same time, two other threads attacked the quadratic wall from different angles. Sinong Wang and colleagues at Facebook AI published "Linformer: Self-Attention with Linear Complexity" (arXiv:2006.04768) in June 2020. Linformer made the empirical observation that the <Code>{"L \\times L"}</Code> attention matrix has low effective rank, then projected keys and values along the sequence axis to a fixed dimension <Code>k \\ll L</Code>. Cost: <Code>{"O(L \\cdot k \\cdot d)"}</Code>. The trick was real but restricted to fixed-length sequences, because the projection is learned per length. Linformer worked for encoder tasks like sentence-pair classification; it never solved autoregressive language modeling cleanly.
      </Prose>

      <Prose>
        In September 2020 Krzysztof Choromanski and colleagues at Google Brain published "Rethinking Attention with Performers" (arXiv:2009.14794, ICLR 2021). Performers addressed a deficiency in Katharopoulos's formulation: the <Code>{"\\phi(x) = \\text{elu}(x) + 1"}</Code> feature map is ad hoc, and the resulting kernel only loosely approximates softmax. Choromanski proved that <Code>{"\\exp(q \\cdot k)"}</Code> can be approximated unbiasedly by random features using Fourier methods, giving a kernel <Code>{"\\phi(x) = \\frac{1}{\\sqrt{m}} [\\cos(w_i \\cdot x), \\sin(w_i \\cdot x)]"}</Code> for random <Code>{"w_i"}</Code>. With enough random features <Code>m</Code>, Performers approximate standard softmax attention with low variance and linear cost. They were the first linear attention variant to come within 1 percent of softmax on large-scale language benchmarks. Performers used Fast Attention via positive Orthogonal Random Features (FAVOR+), which is still a reference for kernel-based linear attention.
      </Prose>

      <Prose>
        The next move was structural rather than approximation-theoretic. In May 2023, Bo Peng and a large open-source collective released "RWKV: Reinventing RNNs for the Transformer Era" (arXiv:2305.13048) at EMNLP 2023 Findings. RWKV (Receptance Weight Key Value) took the linear-attention-is-an-RNN observation and designed an architecture from scratch around it. The model is attention-free in the softmax sense — no <Code>{"Q \\cdot K^T"}</Code> at all. Instead, each layer is a time-mix block that computes a weighted sum of past values with exponentially decaying weights, plus a channel-mix block that mixes within a token using a gated MLP. The result: a model that is literally an RNN at inference (<Code>O(1)</Code> memory per token regardless of context length) and parallelizable at training (a custom CUDA kernel computes the recurrence as a prefix scan). RWKV-4 was released with models up to 14B parameters and came within a few percent of transformer-equivalent models on LM benchmarks.
      </Prose>

      <Prose>
        The architecture iterated quickly. RWKV-5 "Eagle" (late 2023) and RWKV-6 "Finch" (2024) were published by Peng et al. in "Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence" (arXiv:2404.05892). Eagle replaced the scalar time-mixing state with a matrix-valued state, dramatically increasing model capacity without changing training cost. Finch added dynamic (data-dependent) recurrence: the time-decay weights themselves depend on the input, closing some of the expressivity gap with softmax attention. RWKV-7 "Goose" (2025) introduced further architectural refinements including delta-rule updates on the hidden state. By early 2026, RWKV-7 Goose at 14B parameters matches or exceeds LLaMA-2-13B on many downstream benchmarks, though a quality gap remains on tasks that require long-range exact recall.
      </Prose>

      <Prose>
        Parallel to RWKV, Microsoft Research Asia explored a cousin architecture. In July 2023 Yutao Sun and colleagues published "Retentive Network: A Successor to Transformer for Large Language Models" (arXiv:2307.08621). RetNet introduced the "retention" mechanism: a linear attention where the key-value state is decayed by a fixed per-head exponential factor <Code>{"\\gamma^{i-j}"}</Code>, giving three equivalent computation forms — parallel (for training), recurrent (for inference), and chunkwise parallel (for long-context training). RetNet's key claim: competitive quality with transformers, linear inference, and constant memory. The paper was influential within research but did not ship a comparably strong production model family.
      </Prose>

      <Prose>
        In December 2023, Songlin Yang and colleagues published "Gated Linear Attention Transformers with Hardware-Efficient Training" (arXiv:2312.06635, ICML 2024). GLA generalized RetNet's retention mechanism to a data-dependent gate — each key-value contribution to the hidden state is modulated by a learned, input-conditional gating vector — and provided a hardware-aware training kernel. GLA closed the quality gap to transformers further on LM benchmarks at matched compute. Then in May 2024 Tri Dao and Albert Gu published "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality" (arXiv:2405.21060), which showed that a particular form of state-space model (Mamba-2 with scalar-identity structure) is formally equivalent to a masked linear attention. The "SSM-attention duality" unified the two previously separate research threads and gave us a common vocabulary: all of these models — linear attention, RWKV, RetNet, GLA, Mamba — are different parameterizations of a recurrence on a dxd hidden state, with different choices of how the state is updated and read out.
      </Prose>

      <Prose>
        By 2026 the picture is: the "RWKV-family" of architectures (including Mamba-2, GLA, RetNet, and RWKV-7) occupies a stable niche at very long context (100K+ tokens), where quadratic attention is simply impossible on affordable hardware. For the mainstream LLM regime of 2K-32K context, quadratic attention with FlashAttention optimizations remains dominant because its quality is marginally higher and the infrastructure is more mature. But the research trajectory has been unambiguous: every generation of linear-attention models has narrowed the quality gap, and hardware vendors are now building kernels specifically for these architectures. The future of sequence modeling is probably some hybrid where most layers are linear and a few are full attention — exactly the direction that models like Jamba and Samba already represent.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        The entire family of linear attention models — including RWKV — is a single mathematical idea applied in several syntactically different ways. The idea: softmax attention is a specific kernel; if you replace it with a kernel that factorizes as an inner product of feature maps, attention becomes a running sum, which is an RNN. Everything else is engineering on top of that one substitution.
      </Prose>

      <Prose>
        <strong>Standard attention is a weighted sum over past values.</strong> At position <Code>t</Code>, the attention output is <Code>{"O_t = \\sum_i \\alpha_{t,i} V_i"}</Code> where <Code>{"\\alpha_{t,i} = \\text{softmax}_i(Q_t \\cdot K_i^T / \\sqrt{d})"}</Code>. Reading the formula left to right: for each query position <Code>t</Code>, compare the query to every key in the past, softmax to get a probability distribution, and take the expectation of values under that distribution. The softmax normalization is what forces the cost to <Code>{"O(L^2)"}</Code>, because you need all pairwise <Code>{"Q_t \\cdot K_i"}</Code> scores to form the denominator <Code>{"\\sum_j \\exp(Q_t \\cdot K_j)"}</Code>.
      </Prose>

      <Prose>
        <strong>The decomposition trick.</strong> Suppose instead of <Code>{"\\exp(Q \\cdot K^T)"}</Code> we use a similarity of the form <Code>{"\\text{sim}(q, k) = \\phi(q) \\cdot \\phi(k)"}</Code> for some feature map <Code>{"\\phi: \\mathbb{R}^d \\to \\mathbb{R}^{d'}"}</Code> where <Code>{"\\phi(x) \\ge 0"}</Code> elementwise (non-negativity ensures the output is well-defined). Then:
      </Prose>

      <MathBlock>
        {"O_t = \\frac{\\sum_i \\phi(Q_t) \\cdot \\phi(K_i) \\cdot V_i}{\\sum_i \\phi(Q_t) \\cdot \\phi(K_i)} = \\frac{\\phi(Q_t) \\cdot \\left( \\sum_i \\phi(K_i) V_i^T \\right)}{\\phi(Q_t) \\cdot \\left( \\sum_i \\phi(K_i) \\right)}"}
      </MathBlock>

      <Prose>
        Look at the parenthesized sums. They are independent of the query <Code>{"Q_t"}</Code>. They are aggregates over the past alone. Call them the numerator state <Code>{"S_t = \\sum_{i \\le t} \\phi(K_i) V_i^T"}</Code> (shape <Code>{"d' \\times d_v"}</Code>) and the denominator state <Code>{"z_t = \\sum_{i \\le t} \\phi(K_i)"}</Code> (shape <Code>{"d'"}</Code>). Both states can be updated recurrently: <Code>{"S_t = S_{t-1} + \\phi(K_t) V_t^T"}</Code> and <Code>{"z_t = z_{t-1} + \\phi(K_t)"}</Code>. Output at step <Code>t</Code> is then one matrix-vector multiply: <Code>{"O_t = \\phi(Q_t) \\cdot S_t / (\\phi(Q_t) \\cdot z_t)"}</Code>. The entire attention mechanism is now an RNN with state size <Code>{"d' \\cdot d_v + d'"}</Code> independent of sequence length.
      </Prose>

      <Prose>
        <strong>This works if and only if the causal mask commutes with the feature map.</strong> The decomposition above uses <Code>{"\\sum_i"}</Code> over all positions, but for causal attention we need <Code>{"\\sum_{i \\le t}"}</Code>. The prefix sum is exactly the causal mask — it commutes with any feature map <Code>{"\\phi"}</Code> because the sum order does not matter. For softmax this is not true: <Code>{"\\exp(Q_t \\cdot K_i)"}</Code> has no finite-dimensional feature map <Code>{"\\phi"}</Code> that makes this factorization exact; you need an infinite-dimensional kernel, which is why random-features methods (Performers) provide only an unbiased <em>estimator</em>. The mathematical elegance of linear attention is that causal masking is <em>free</em>: a running sum is the causal mask.
      </Prose>

      <Prose>
        <strong>RWKV: attention-free from a different starting point.</strong> RWKV is not presented as "linear attention with feature map <Code>{"\\phi"}</Code>." Its derivation starts from a time-decay: at step <Code>t</Code>, aggregate past values weighted by <Code>{"\\exp(w \\cdot (t - i - 1) + k_i)"}</Code>, where <Code>w</Code> is a per-channel learned decay and <Code>{"k_i"}</Code> is the key for position <Code>i</Code>. This is almost identical to linear attention with feature map <Code>{"\\phi(k) = \\exp(k)"}</Code> plus an explicit time-decay factor on the state. The difference: RWKV treats the decay as a first-class architectural parameter, uses a custom softmax-style normalization for numerical stability, and gates the output with a learned "receptance" vector. The formula <Code>{"wkv_t = \\sum \\exp(w \\cdot (t - i - 1) + k_i) \\cdot v_i / \\sum \\exp(w \\cdot (t - i - 1) + k_i)"}</Code> is the WKV (Weight-Key-Value) core; the receptance <Code>r</Code> is applied at output: <Code>{"\\text{out}_t = r_t \\odot wkv_t"}</Code>. R-W-K-V, the four projections of the input token, give the architecture its name.
      </Prose>

      <Prose>
        <strong>Time-mix and channel-mix.</strong> RWKV organizes each layer as two halves. The <em>time-mix</em> block is the WKV recurrence — it moves information across time. The <em>channel-mix</em> block is a position-wise gated MLP — it mixes information within a single token across channels. This is exactly the transformer layout (attention + FFN) but with the attention replaced by a linear recurrence. The gates in both blocks use a token-shift operation: each input is a mix of the current token and the previous token, <Code>{"x_t' = \\mu \\odot x_t + (1 - \\mu) \\odot x_{t-1}"}</Code>, giving the network cheap access to 1-step lookback without full attention. This simple mechanism carries a surprising amount of weight in practice; ablating it drops quality measurably.
      </Prose>

      <Prose>
        <strong>RetNet: retention as principled linear attention.</strong> RetNet writes attention as <Code>{"Y_t = Q_t \\cdot \\sum_{i \\le t} \\gamma^{t - i} K_i^T V_i"}</Code> where <Code>{"\\gamma < 1"}</Code> is a fixed per-head decay. This is linear attention with feature map <Code>{"\\phi = \\text{identity}"}</Code> and an exponential time-decay. The key insight: RetNet proves this admits three equivalent forms — a parallel (transformer-like) form computed as <Code>{"Y = (QK^T \\odot D) V"}</Code> with a <Code>{"D_{ij} = \\gamma^{i-j}"}</Code> mask, a recurrent form with state <Code>{"S_t = \\gamma S_{t-1} + K_t V_t^T"}</Code>, and a chunkwise parallel form that interpolates between the two. All three compute the same output; training uses the parallel form, inference uses the recurrent form, long-context training uses chunkwise.
      </Prose>

      <Prose>
        <strong>Gated linear attention: data-dependent decay.</strong> GLA replaces the fixed <Code>{"\\gamma"}</Code> of RetNet with a learned, input-conditional gate <Code>{"G_t = \\text{sigmoid}(W_g x_t)"}</Code>, giving the recurrence <Code>{"S_t = G_t \\odot S_{t-1} + K_t V_t^T"}</Code>. The gate decides per-channel how much of the past to carry forward as a function of the current input. This is the missing expressivity: fixed decay is too restrictive (can't suppress noise conditionally); softmax is too expensive (full pairwise). GLA sits between them and is currently the leading non-softmax formulation.
      </Prose>

      <Callout accent="gold">
        Mental model: linear attention rewrites the <Code>{"L \\times L"}</Code> attention matrix as a product of two rank-<Code>d</Code> matrices, which lets causal masking become a prefix sum. Every variant in this family — Katharopoulos's elu+1, Performers' random features, RetNet's retention, RWKV's WKV, GLA's gated recurrence, Mamba-2's selective SSM — is a different choice of feature map, decay, and gate on the same underlying recurrence. The recurrence is an RNN at inference and a prefix scan at training. The trade-off is always: quality (better with expressive gates) vs. hardware efficiency (better with simple, fixed recurrences). RWKV chose attention-free-RNN-as-core-architecture rather than attention-replacement-inside-a-transformer, which is the cleanest demonstration of the idea but not necessarily the highest-quality realization.
      </Callout>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 Standard softmax attention</H3>

      <Prose>
        Given queries <Code>{"Q \\in \\mathbb{R}^{L \\times d}"}</Code>, keys <Code>{"K \\in \\mathbb{R}^{L \\times d}"}</Code>, values <Code>{"V \\in \\mathbb{R}^{L \\times d_v}"}</Code>, standard scaled dot-product attention computes:
      </Prose>

      <MathBlock>
        {"A_{t,i} = \\frac{\\exp(Q_t \\cdot K_i / \\sqrt{d})}{\\sum_{j} \\exp(Q_t \\cdot K_j / \\sqrt{d})}, \\qquad O_t = \\sum_i A_{t,i} V_i"}
      </MathBlock>

      <Prose>
        The attention matrix <Code>A</Code> has shape <Code>{"L \\times L"}</Code>; this is the quadratic cost in memory and time. The causal variant sets <Code>{"A_{t,i} = 0"}</Code> for <Code>{"i > t"}</Code>, equivalently adds <Code>{"-\\infty"}</Code> to the unnormalized scores in the upper triangle before softmax.
      </Prose>

      <H3>3.2 Linear attention via feature maps</H3>

      <Prose>
        Replace the softmax kernel with a factorizable kernel <Code>{"\\text{sim}(q, k) = \\phi(q)^T \\phi(k)"}</Code>. The attention output becomes:
      </Prose>

      <MathBlock>
        {"O_t = \\frac{\\sum_{i \\le t} \\phi(Q_t)^T \\phi(K_i) V_i}{\\sum_{i \\le t} \\phi(Q_t)^T \\phi(K_i)} = \\frac{\\phi(Q_t)^T \\left( \\sum_{i \\le t} \\phi(K_i) V_i^T \\right)}{\\phi(Q_t)^T \\left( \\sum_{i \\le t} \\phi(K_i) \\right)}"}
      </MathBlock>

      <Prose>
        Define the numerator state <Code>{"S_t = \\sum_{i \\le t} \\phi(K_i) V_i^T \\in \\mathbb{R}^{d' \\times d_v}"}</Code> and the denominator state <Code>{"z_t = \\sum_{i \\le t} \\phi(K_i) \\in \\mathbb{R}^{d'}"}</Code>. The recurrence is:
      </Prose>

      <MathBlock>
        {"S_t = S_{t-1} + \\phi(K_t) V_t^T, \\qquad z_t = z_{t-1} + \\phi(K_t), \\qquad O_t = \\frac{\\phi(Q_t)^T S_t}{\\phi(Q_t)^T z_t}"}
      </MathBlock>

      <Prose>
        Cost: <Code>{"O(d' \\cdot d_v)"}</Code> per step for the state update, <Code>{"O(d' \\cdot d_v + d')"}</Code> per step for the output. Over a sequence of length <Code>L</Code>, total cost is <Code>{"O(L \\cdot d' \\cdot d_v)"}</Code> — linear in <Code>L</Code>. For typical settings <Code>{"d' = d = d_v"}</Code>, this is <Code>{"O(L \\cdot d^2)"}</Code>, vs. softmax's <Code>{"O(L^2 \\cdot d)"}</Code>. The crossover at which linear becomes cheaper: <Code>{"L \\approx d"}</Code>. For <Code>{"d = 128"}</Code>, linear wins starting at <Code>{"L > 128"}</Code>. For modern LLMs with <Code>{"d = 4096"}</Code>, linear only wins at <Code>{"L > 4096"}</Code> — but that is exactly the regime where quadratic attention runs out of GPU memory.
      </Prose>

      <H3>3.3 Feature map choices</H3>

      <Prose>
        The simplest non-negative feature map is <Code>{"\\phi(x) = \\text{elu}(x) + 1"}</Code>, proposed by Katharopoulos et al. 2020. This guarantees elementwise non-negativity and is differentiable. It works but the kernel it induces is only weakly related to softmax.
      </Prose>

      <Prose>
        Performers (Choromanski et al. 2021) use random features to approximate the exact softmax kernel. The result <Code>{"\\exp(q \\cdot k) = \\mathbb{E}_{w}[\\exp(w \\cdot q - \\|q\\|^2 / 2) \\cdot \\exp(w \\cdot k - \\|k\\|^2 / 2)]"}</Code> gives the FAVOR+ feature map with <Code>m</Code> random directions <Code>{"w_i"}</Code>:
      </Prose>

      <MathBlock>
        {"\\phi(x) = \\frac{1}{\\sqrt{m}} \\exp\\left(-\\frac{\\|x\\|^2}{2}\\right) \\cdot \\left[\\exp(w_1 \\cdot x), \\ldots, \\exp(w_m \\cdot x)\\right]"}
      </MathBlock>

      <Prose>
        This is an unbiased estimator of <Code>{"\\exp(q \\cdot k)"}</Code> in expectation. Choromanski proved the approximation error is <Code>{"O(1/\\sqrt{m})"}</Code>, independent of sequence length. In practice <Code>{"m = 256"}</Code> gives very tight approximations to softmax, and the cost is then <Code>{"O(L \\cdot m \\cdot d)"}</Code> which is linear in <Code>L</Code>.
      </Prose>

      <H3>3.4 RWKV-4 time-mix (WKV)</H3>

      <Prose>
        The RWKV time-mix block computes, for a per-channel time decay <Code>w</Code> (which is learned and typically constrained to be non-positive) and a per-channel bonus <Code>u</Code>:
      </Prose>

      <MathBlock>
        {"wkv_t = \\frac{\\sum_{i < t} \\exp(w \\cdot (t - i - 1) + k_i) \\cdot v_i + \\exp(u + k_t) \\cdot v_t}{\\sum_{i < t} \\exp(w \\cdot (t - i - 1) + k_i) + \\exp(u + k_t)}"}
      </MathBlock>

      <Prose>
        The <Code>u</Code> bonus gives the current token a different weight than the past — the standard RWKV-4 choice bumps the current token to compensate for the missing time-decay penalty. The output of the time-mix block is then <Code>{"r_t \\odot wkv_t"}</Code>, where <Code>r</Code> is the receptance, a learned gate applied elementwise.
      </Prose>

      <Prose>
        This admits a compact recurrent form. Define running accumulators <Code>{"a_t = \\sum_{i \\le t} \\exp(w \\cdot (t - i) + k_i) \\cdot v_i"}</Code> and <Code>{"b_t = \\sum_{i \\le t} \\exp(w \\cdot (t - i) + k_i)"}</Code>. Then <Code>{"a_{t+1} = \\exp(w) \\cdot a_t + \\exp(k_{t+1}) \\cdot v_{t+1}"}</Code> and <Code>{"b_{t+1} = \\exp(w) \\cdot b_t + \\exp(k_{t+1})"}</Code>. Numerical stability requires log-space tracking: RWKV maintains a running per-channel maximum <Code>{"p_t"}</Code> and stores <Code>{"a_t, b_t"}</Code> relative to it. The custom CUDA kernel does exactly this.
      </Prose>

      <H3>3.5 RWKV channel-mix</H3>

      <Prose>
        The channel-mix block is a position-wise gated MLP. Given input <Code>{"x_t"}</Code>, the token-shifted input <Code>{"x_t' = \\mu_r \\odot x_t + (1 - \\mu_r) \\odot x_{t-1}"}</Code> is passed through:
      </Prose>

      <MathBlock>
        {"\\text{channel-mix}(x_t) = \\sigma(R \\cdot x_t') \\odot \\left( V \\cdot \\text{ReLU}^2(K \\cdot x_t' + b_k) \\right)"}
      </MathBlock>

      <Prose>
        Here <Code>{"\\sigma"}</Code> is sigmoid, <Code>{"\\text{ReLU}^2"}</Code> is squared ReLU (a nonlinearity chosen empirically), and <Code>R, K, V</Code> are learned projections. The <Code>{"\\sigma(R \\cdot x_t')"}</Code> factor is a receptance gate — it controls how much of the channel-mix output is admitted to the residual stream. The squared ReLU is similar to the GLU variants used in transformer FFNs but without an explicit gate variable.
      </Prose>

      <H3>3.6 RetNet retention</H3>

      <Prose>
        RetNet's retention mechanism is linear attention with an exponential time-decay. Define <Code>{"D_{i,j} = \\gamma^{i - j}"}</Code> for <Code>{"i \\ge j"}</Code> and <Code>{"D_{i,j} = 0"}</Code> otherwise. Then:
      </Prose>

      <MathBlock>
        {"Y = (Q K^T \\odot D) V \\quad \\text{(parallel form)}"}
      </MathBlock>

      <MathBlock>
        {"S_t = \\gamma S_{t-1} + K_t^T V_t, \\qquad Y_t = Q_t S_t \\quad \\text{(recurrent form)}"}
      </MathBlock>

      <Prose>
        The two forms produce identical outputs. The chunkwise form processes <Code>C</Code> tokens at a time with a quadratic within-chunk computation (efficient on GPU) and a decayed carry between chunks, giving a cost of <Code>{"O(L \\cdot C)"}</Code> for compute and <Code>{"O(C^2)"}</Code> for activation memory within a chunk. This is the core of every modern long-context training setup for linear-attention models: chunkwise parallel to get GPU-friendly BMMs, recurrent state carry to get unbounded context.
      </Prose>

      <H3>3.7 Gated linear attention</H3>

      <Prose>
        GLA generalizes retention by replacing the fixed scalar <Code>{"\\gamma"}</Code> with a learned, data-dependent gating matrix <Code>{"G_t \\in \\mathbb{R}^{d \\times d}"}</Code>. The recurrence is:
      </Prose>

      <MathBlock>
        {"S_t = G_t \\odot S_{t-1} + K_t^T V_t, \\qquad Y_t = Q_t S_t"}
      </MathBlock>

      <Prose>
        The gate is typically parameterized as <Code>{"G_t = \\text{sigmoid}(W_g x_t)"}</Code> (if broadcast to <Code>{"d \\times d"}</Code>) or as a low-rank decomposition <Code>{"G_t = \\alpha_t \\beta_t^T"}</Code> for efficiency. The key property: <Code>{"G_t"}</Code> depends on <Code>{"x_t"}</Code>, giving the model input-conditional "forgetting" of the state. This is precisely the mechanism that lets GLA selectively retain or discard information — the same role played by softmax's ability to weight past tokens arbitrarily.
      </Prose>

      <H3>3.8 Mamba-2 and the SSM-attention duality</H3>

      <Prose>
        Dao and Gu (2024) showed that state-space models of the form <Code>{"h_t = A_t h_{t-1} + B_t x_t"}</Code>, <Code>{"y_t = C_t h_t"}</Code>, when <Code>{"A_t = a_t I"}</Code> is a scalar-times-identity, are formally equivalent to a masked linear attention with a specific structure. The mapping: <Code>{"K = B, V = x, Q = C"}</Code>, and the diagonal decay is <Code>{"D_{i,j} = \\prod_{k=j+1}^{i} a_k"}</Code>. This means Mamba-2's "selective SSM" and GLA's "gated linear attention" are the same algorithm written in different notation; both are specific instantiations of a generalized linear attention with time-varying gates. The duality unified a previously-divided literature and gave vendors a common interface for hardware kernels.
      </Prose>

      <H3>3.9 Cost comparison</H3>

      <Prose>
        Training cost for a sequence of length <Code>L</Code> with model dimension <Code>d</Code>:
      </Prose>

      <MathBlock>
        {"\\text{softmax attention: } O(L^2 \\cdot d), \\quad \\text{linear attention: } O(L \\cdot d^2), \\quad \\text{chunkwise: } O(L \\cdot C \\cdot d + (L/C) \\cdot d^2)"}
      </MathBlock>

      <Prose>
        Crossover: softmax becomes more expensive than linear when <Code>{"L > d"}</Code>. For a 7B model with <Code>{"d = 4096"}</Code>, linear attention is cheaper in theory for <Code>{"L > 4096"}</Code>. In practice, softmax attention has heavily optimized GPU kernels (FlashAttention) with constants small enough that the crossover in wall time is closer to <Code>{"L \\approx 8 \\cdot d"}</Code>. At <Code>{"L = 100{,}000"}</Code>, linear attention is unambiguously faster — and also the only option that fits in memory.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The four snippets below were run locally on PyTorch 2.6. Outputs are verbatim stdout from those runs.
      </Prose>

      <H3>4a. Linear attention via elu+1 feature map vs softmax</H3>

      <Prose>
        This snippet implements causal linear attention with Katharopoulos's <Code>{"\\phi(x) = \\text{elu}(x) + 1"}</Code> feature map and compares its output to standard causal softmax attention on a toy sequence. The two agree in direction on most tokens but not exactly — linear attention is a lossy approximation to softmax, and the discrepancy shows what you pay for the asymptotic speed-up.
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn.functional as F

torch.manual_seed(0)

def softmax_attention(Q, K, V, causal=True):
    L, d = Q.shape
    scores = Q @ K.T / (d ** 0.5)
    if causal:
        mask = torch.triu(torch.ones(L, L), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))
    w = F.softmax(scores, dim=-1)
    return w @ V

def phi(x):
    return F.elu(x) + 1.0

def linear_attention_causal(Q, K, V):
    L, d = Q.shape
    dv = V.shape[1]
    phiQ = phi(Q); phiK = phi(K)
    S = torch.zeros(d, dv)          # numerator state
    z = torch.zeros(d)              # denominator state
    out = torch.zeros(L, dv)
    for t in range(L):
        S = S + torch.outer(phiK[t], V[t])
        z = z + phiK[t]
        num = phiQ[t] @ S
        den = phiQ[t] @ z + 1e-6
        out[t] = num / den
    return out

L, d, dv = 8, 16, 16
Q = torch.randn(L, d); K = torch.randn(L, d); V = torch.randn(L, dv)

out_softmax = softmax_attention(Q, K, V, causal=True)
out_linear  = linear_attention_causal(Q, K, V)

print("softmax causal attention vs linear attention (elu+1)")
print(f"  L={L}, d={d}, dv={dv}")
print(f"  softmax output norm: {out_softmax.norm().item():.4f}")
print(f"  linear  output norm: {out_linear.norm().item():.4f}")
print("  cosine similarity per token:")
for t in range(L):
    cs = F.cosine_similarity(out_softmax[t], out_linear[t], dim=0).item()
    print(f"    t={t}: cos_sim={cs:+.4f}")
avg = F.cosine_similarity(out_softmax, out_linear, dim=1).mean().item()
print(f"  mean cosine similarity: {avg:+.4f}")

# Output:
# softmax causal attention vs linear attention (elu+1)
#   L=8, d=16, dv=16
#   softmax output norm: 7.1675
#   linear  output norm: 6.5640
#   cosine similarity per token:
#     t=0: cos_sim=+1.0000
#     t=1: cos_sim=+0.9284
#     t=2: cos_sim=+0.9986
#     t=3: cos_sim=+0.9301
#     t=4: cos_sim=+0.8353
#     t=5: cos_sim=+0.8704
#     t=6: cos_sim=+0.9278
#     t=7: cos_sim=+0.7155
#   mean cosine similarity: +0.9008`}
      </CodeBlock>

      <Prose>
        Mean cosine similarity of 0.90 between softmax and linear attention outputs on the same <Code>Q, K, V</Code> is a realistic measurement of the approximation quality. Some tokens agree nearly exactly (position 0 is a forced agreement — the first token's attention has only one term); others drop to 0.72. The feature map <Code>{"\\text{elu}(x) + 1"}</Code> is not a faithful approximation to the softmax kernel; it is a <em>different</em> kernel that happens to be factorizable. Newer variants (Performers, RWKV, RetNet) close this gap dramatically via better feature maps, random features, or gating — but the basic tension is visible right here at <Code>L = 8</Code>. Note that at position <Code>t = 0</Code> the cosine is exactly 1: with a single past token, any attention reduces to "copy that token", and both methods give the same answer.
      </Prose>

      <H3>4b. RWKV WKV kernel: naive O(L^2) vs recurrent O(L)</H3>

      <Prose>
        The RWKV time-mix kernel can be evaluated two equivalent ways: as a direct double sum over all past tokens (which is quadratic and used for verification), or as a running recurrence (which is linear and used at inference). The recurrent form requires log-space numerical stability because <Code>{"\\exp(w \\cdot (t - i - 1) + k_i)"}</Code> can overflow for large <Code>{"t - i"}</Code>. This implementation tracks a per-channel running maximum to keep everything well-scaled.
      </Prose>

      <CodeBlock language="python">
{`import torch

torch.manual_seed(0)

def rwkv_wkv_naive(w, u, k, v):
    """O(L^2) reference — used only for verification."""
    L, C = k.shape
    out = torch.zeros(L, C)
    for t in range(L):
        num = torch.zeros(C); den = torch.zeros(C)
        for i in range(t):
            weight = torch.exp(w * (t - i - 1) + k[i])
            num = num + weight * v[i]
            den = den + weight
        cur = torch.exp(u + k[t])
        num = num + cur * v[t]
        den = den + cur
        out[t] = num / (den + 1e-6)
    return out

def rwkv_wkv_recurrent(w, u, k, v):
    """O(L) recurrent form with log-space numerical stability."""
    L, C = k.shape
    out = torch.zeros(L, C)
    a = torch.zeros(C)
    b = torch.zeros(C)
    p = torch.full((C,), -1e30)   # running max, for stability
    for t in range(L):
        # output using current-token bonus 'u'
        q = torch.maximum(p, u + k[t])
        e1 = torch.exp(p - q); e2 = torch.exp(u + k[t] - q)
        out[t] = (e1 * a + e2 * v[t]) / (e1 * b + e2 + 1e-6)
        # update decayed running state
        q2 = torch.maximum(w + p, k[t])
        e1 = torch.exp(w + p - q2); e2 = torch.exp(k[t] - q2)
        a = e1 * a + e2 * v[t]
        b = e1 * b + e2
        p = q2
    return out

L, C = 6, 8
w = -torch.exp(torch.randn(C) * 0.5)   # learned decay, constrained <= 0
u = torch.randn(C) * 0.3               # per-channel current-token bonus
k = torch.randn(L, C); v = torch.randn(L, C)

out_naive = rwkv_wkv_naive(w, u, k, v)
out_recur = rwkv_wkv_recurrent(w, u, k, v)

print("RWKV WKV: naive O(L^2) vs recurrent O(L)")
print(f"  L={L}, C={C}")
print(f"  naive     norm: {out_naive.norm().item():.6f}")
print(f"  recurrent norm: {out_recur.norm().item():.6f}")
diff = (out_naive - out_recur).abs().max().item()
print(f"  max abs diff:   {diff:.2e}")
print()
print("per-token output (first 3 channels):")
for t in range(L):
    n = out_naive[t, :3].tolist()
    r = out_recur[t, :3].tolist()
    print(f"  t={t}: naive={[f'{x:+.3f}' for x in n]}  recur={[f'{x:+.3f}' for x in r]}")

# Output:
# RWKV WKV: naive O(L^2) vs recurrent O(L)
#   L=6, C=8
#   naive     norm: 4.606184
#   recurrent norm: 4.606183
#   max abs diff:   8.94e-07
#
# per-token output (first 3 channels):
#   t=0: naive=['-0.093', '+0.687', '-0.838']  recur=['-0.093', '+0.687', '-0.838']
#   t=1: naive=['-0.181', '+1.745', '-1.366']  recur=['-0.181', '+1.745', '-1.366']
#   t=2: naive=['+0.326', '+0.525', '-0.264']  recur=['+0.326', '+0.525', '-0.264']
#   t=3: naive=['+0.827', '-0.157', '+0.135']  recur=['+0.827', '-0.157', '+0.135']
#   t=4: naive=['+0.130', '-0.976', '+0.119']  recur=['+0.130', '-0.976', '+0.119']
#   t=5: naive=['-0.240', '-1.424', '+0.199']  recur=['-0.240', '-1.424', '+0.199']`}
      </CodeBlock>

      <Prose>
        Max abs diff between the O(L^2) reference and the O(L) recurrent implementation is 9e-7 — float32 accumulation noise. The recurrent form is exact, not an approximation. This is the critical structural property: RWKV's training graph <em>is</em> a recurrence, and the loss is identical whether you unroll it step-by-step (slow) or use the prefix-scan CUDA kernel (fast). Most RNN-style architectures do not have this property — typical RNNs have a hidden-state-dependent nonlinearity that prevents parallel unrolling.
      </Prose>

      <H3>4c. Benchmark: softmax vs linear attention at growing sequence length</H3>

      <Prose>
        This benchmark compares vectorized softmax attention against vectorized (parallel-form) linear attention at sequence lengths 256, 512, 1024, 2048, 4096, 8192 on CPU. The expectation: at short <Code>L</Code>, softmax wins because its highly optimized matmul kernels beat the overhead of the linear-attention implementation. At long <Code>L</Code>, linear attention wins because the <Code>{"L^2"}</Code> cost of softmax catches up. The crossover is exactly what we predicted from the math: around <Code>{"L \\approx d"}</Code>, scaled by the implementation's constants.
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn.functional as F
import time

torch.manual_seed(0)
d = 64

def softmax_attn(Q, K, V):
    L, d = Q.shape
    scores = Q @ K.T / (d ** 0.5)
    mask = torch.triu(torch.ones(L, L), diagonal=1).bool()
    scores = scores.masked_fill(mask, float("-inf"))
    return F.softmax(scores, dim=-1) @ V

def linear_attn_parallel(Q, K, V):
    # Vectorized causal linear attention via cumulative sum of outer products.
    phiQ = F.elu(Q) + 1.0
    phiK = F.elu(K) + 1.0
    L, d = phiQ.shape
    dv = V.shape[1]
    outer_kv = phiK.unsqueeze(-1) * V.unsqueeze(-2)  # [L, d, dv]
    S = outer_kv.cumsum(dim=0)                        # running numerator state
    z = phiK.cumsum(dim=0)                            # running denominator
    num = torch.einsum("ld,ldv->lv", phiQ, S)
    den = (phiQ * z).sum(dim=-1, keepdim=True) + 1e-6
    return num / den

def bench(fn, L, n_warm=2, n_run=3):
    Q = torch.randn(L, d); K = torch.randn(L, d); V = torch.randn(L, d)
    for _ in range(n_warm): _ = fn(Q, K, V)
    t0 = time.perf_counter()
    for _ in range(n_run): _ = fn(Q, K, V)
    return (time.perf_counter() - t0) / n_run * 1e3

print(f"attention throughput (d={d}, CPU, ms per forward)")
print(f"{'L':>6}  {'softmax':>10}  {'linear':>10}  {'ratio':>8}")
for L in [256, 512, 1024, 2048, 4096, 8192]:
    t_sm = bench(softmax_attn, L)
    t_ln = bench(linear_attn_parallel, L)
    print(f"{L:>6}  {t_sm:>9.2f}ms  {t_ln:>9.2f}ms  {t_sm/t_ln:>7.2f}x")

print()
print("attention matrix / state memory (KB):")
print(f"{'L':>6}  {'softmax LxL':>14}  {'linear dxd':>14}")
for L in [256, 1024, 4096, 16384, 65536]:
    sm_kb = (L * L * 4) / 1024
    ln_kb = (d * d * 4) / 1024
    print(f"{L:>6}  {sm_kb:>14.1f}  {ln_kb:>14.1f}")

# Output:
# attention throughput (d=64, CPU, ms per forward)
#      L     softmax      linear     ratio
#    256       1.10ms       4.62ms    0.24x
#    512       2.58ms      12.74ms    0.20x
#   1024       9.11ms      34.45ms    0.26x
#   2048      29.69ms      85.75ms    0.35x
#   4096     110.07ms     222.85ms    0.49x
#   8192     630.96ms     410.04ms    1.54x
#
# attention matrix / state memory (KB):
#      L     softmax LxL      linear dxd
#    256           256.0            16.0
#   1024          4096.0            16.0
#   4096         65536.0            16.0
#  16384       1048576.0            16.0
#  65536      16777216.0            16.0`}
      </CodeBlock>

      <Prose>
        At <Code>L = 256</Code>, softmax is 4× faster than our unoptimized linear attention — the constant factor of torch matmul is hard to beat on small sizes. At <Code>L = 8192</Code>, linear wins by 1.54× and the gap continues to widen. More striking is the memory: the softmax attention matrix grows from 256KB at <Code>L = 256</Code> to 16GB at <Code>L = 65K</Code>, while the linear state stays at 16KB regardless of <Code>L</Code>. In practice no one actually materializes the <Code>L × L</Code> matrix for long contexts — FlashAttention streams it — but the linear-attention state is genuinely constant-memory, which is the property that matters for deploying long-context models on limited hardware. The linear cost advantage at sequence length also scales with <Code>d</Code>: for real LLMs with <Code>d = 4096</Code> and modern FlashAttention kernels, the wall-clock crossover is closer to <Code>{"L \\approx 8{,}000"}</Code> to <Code>{"L \\approx 16{,}000"}</Code>.
      </Prose>

      <H3>4d. Chunked parallel linear attention (GLA-style training kernel)</H3>

      <Prose>
        The pure recurrent form of linear attention is <Code>O(L)</Code> but slow in practice because each step is a matrix-vector operation that does not use GPU BMMs efficiently. The GLA paper (Yang 2024) introduced a hardware-aware training kernel: break the sequence into chunks of size <Code>C</Code>, do a quadratic within-chunk attention (friendly to BMMs), and carry a <Code>{"d \\times d"}</Code> state between chunks. Cost is <Code>{"O(L \\cdot C + (L/C) \\cdot d^2)"}</Code>; with <Code>{"C \\approx d"}</Code> this gives the theoretical minimum <Code>{"O(L \\cdot d)"}</Code>. This snippet verifies the chunked form matches the recurrent reference bit-exactly and measures the speedup.
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn.functional as F
import time

torch.manual_seed(0)
d = 64

def linear_attn_recurrent(Q, K, V):
    phiQ = F.elu(Q) + 1.0; phiK = F.elu(K) + 1.0
    L = Q.shape[0]; dv = V.shape[1]
    S = torch.zeros(d, dv); z = torch.zeros(d)
    out = torch.zeros(L, dv)
    for t in range(L):
        S = S + torch.outer(phiK[t], V[t])
        z = z + phiK[t]
        out[t] = (phiQ[t] @ S) / (phiQ[t] @ z + 1e-6)
    return out

def linear_attn_chunked(Q, K, V, chunk_size=64):
    phiQ = F.elu(Q) + 1.0; phiK = F.elu(K) + 1.0
    L, d = phiQ.shape; dv = V.shape[1]
    out = torch.zeros(L, dv)
    S = torch.zeros(d, dv)    # inter-chunk numerator state
    z = torch.zeros(d)        # inter-chunk denominator state
    for start in range(0, L, chunk_size):
        end = min(start + chunk_size, L)
        qc = phiQ[start:end]; kc = phiK[start:end]; vc = V[start:end]
        c = qc.shape[0]
        # within-chunk: quadratic BMM on cxc (cheap for small chunks)
        inner = qc @ kc.T
        mask = torch.triu(torch.ones(c, c), diagonal=1).bool()
        inner = inner.masked_fill(mask, 0.0)
        num_in = inner @ vc
        den_in = inner.sum(dim=-1, keepdim=True)
        # cross-chunk: use carried state S, z
        num_cross = qc @ S
        den_cross = (qc @ z).unsqueeze(-1)
        out[start:end] = (num_in + num_cross) / (den_in + den_cross + 1e-6)
        # update inter-chunk state by adding the chunk's full contribution
        S = S + kc.T @ vc
        z = z + kc.sum(dim=0)
    return out

# Verify chunked == recurrent
L = 256
Q = torch.randn(L, d); K = torch.randn(L, d); V = torch.randn(L, d)
out_rec  = linear_attn_recurrent(Q, K, V)
out_c16  = linear_attn_chunked(Q, K, V, chunk_size=16)
out_c64  = linear_attn_chunked(Q, K, V, chunk_size=64)
out_c256 = linear_attn_chunked(Q, K, V, chunk_size=256)
print("chunked parallel linear attention vs recurrent reference")
print(f"  chunk=16  max abs diff: {(out_rec - out_c16 ).abs().max().item():.2e}")
print(f"  chunk=64  max abs diff: {(out_rec - out_c64 ).abs().max().item():.2e}")
print(f"  chunk=256 max abs diff: {(out_rec - out_c256).abs().max().item():.2e}")

# Speed at L=4096
L = 4096
Q = torch.randn(L, d); K = torch.randn(L, d); V = torch.randn(L, d)
def bench(fn, n=3):
    for _ in range(2): _ = fn()
    t0 = time.perf_counter()
    for _ in range(n): _ = fn()
    return (time.perf_counter() - t0) / n * 1e3
t_rec  = bench(lambda: linear_attn_recurrent(Q, K, V))
t_c16  = bench(lambda: linear_attn_chunked(Q, K, V, 16))
t_c64  = bench(lambda: linear_attn_chunked(Q, K, V, 64))
t_c256 = bench(lambda: linear_attn_chunked(Q, K, V, 256))
print(f"\\nspeed at L={L} (ms per forward, CPU):")
print(f"  pure recurrent O(L):         {t_rec:>7.1f}")
print(f"  chunked parallel, chunk=16:  {t_c16:>7.1f}")
print(f"  chunked parallel, chunk=64:  {t_c64:>7.1f}")
print(f"  chunked parallel, chunk=256: {t_c256:>7.1f}")

# Output:
# chunked parallel linear attention vs recurrent reference
#   chunk=16  max abs diff: 4.77e-07
#   chunk=64  max abs diff: 4.77e-07
#   chunk=256 max abs diff: 4.77e-07
#
# speed at L=4096 (ms per forward, CPU):
#   pure recurrent O(L):           813.4
#   chunked parallel, chunk=16:    138.2
#   chunked parallel, chunk=64:     59.9
#   chunked parallel, chunk=256:    30.9`}
      </CodeBlock>

      <Prose>
        Chunked parallel is a 26× speedup over the pure recurrent form at <Code>L = 4096</Code>, with chunk size 256. Correctness is exact to float precision — not an approximation. The sweet spot of chunk size is architecture-specific: small chunks waste time on tiny BMMs; large chunks reduce the BMM parallelism advantage by growing the within-chunk <Code>c × c</Code> matrix. In production GLA and Mamba-2 kernels, chunk sizes between 64 and 256 are standard. This is the training trick that made linear-attention models actually fast: without chunked parallelism, linear attention is linear-time in theory but slower than softmax in wall-clock because of poor GPU utilization.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <H3>5.1 RWKV-4 and RWKV-5 checkpoints on HuggingFace</H3>

      <Prose>
        The RWKV project (led by Bo Peng at BlinkDL) publishes checkpoints on HuggingFace. The canonical "Raven" line is an instruction-tuned RWKV-4 model; the non-instruction-tuned base models follow the <Code>RWKV/rwkv-4-</Code>* naming scheme. Inference works through HuggingFace <Code>transformers</Code>:
      </Prose>

      <CodeBlock language="python">
{`from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# RWKV-4 Raven 14B: instruction-tuned
tok = AutoTokenizer.from_pretrained("RWKV/rwkv-raven-14b")
model = AutoModelForCausalLM.from_pretrained(
    "RWKV/rwkv-raven-14b",
    torch_dtype=torch.float16,
    device_map="auto",
)

prompt = "Explain linear attention in one paragraph:"
inputs = tok(prompt, return_tensors="pt").to(model.device)

# At inference, RWKV is literally an RNN: each new token is O(1) memory
# regardless of the context length built up before it.
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=200)

print(tok.decode(out[0], skip_special_tokens=True))`}
      </CodeBlock>

      <Prose>
        At inference, the HuggingFace implementation of RWKV uses the recurrent formulation — a learned <Code>state</Code> is carried from token to token and updated in <Code>O(C)</Code> time per step (where <Code>C</Code> is the channel dimension, not sequence length). Because there is no KV cache growing with context, the memory footprint is fixed; you can push context to arbitrary length without memory blowing up. This is the property that makes RWKV attractive for streaming applications and edge deployment. RWKV-5 Eagle and RWKV-6 Finch checkpoints are available under <Code>RWKV/rwkv-5-world-</Code>* and <Code>RWKV/v5-Eagle-</Code>* variants on HuggingFace; RWKV-7 Goose (2025) is distributed primarily through BlinkDL's own releases on HuggingFace with names like <Code>BlinkDL/rwkv-7-world</Code>.
      </Prose>

      <H3>5.2 Flash Linear Attention (fla-org) library</H3>

      <Prose>
        The <Code>fla-org/flash-linear-attention</Code> library (maintained by Songlin Yang and collaborators) provides CUDA-optimized kernels for the entire linear-attention family: GLA, RetNet, RWKV variants, Mamba-2, DeltaNet, and more. It is the de facto reference for high-performance linear-attention training and inference. Usage:
      </Prose>

      <CodeBlock language="python">
{`# pip install flash-linear-attention

from fla.layers import GatedLinearAttention, RetNet, RWKV6Attention
import torch

# GLA layer: data-dependent gating + linear attention
layer = GatedLinearAttention(
    hidden_size=1024,
    num_heads=4,
    expand_ratio=1.0,
).cuda().to(torch.bfloat16)

x = torch.randn(2, 4096, 1024, device="cuda", dtype=torch.bfloat16)
y = layer(x)[0]        # returns (output, cache); cache is the recurrent state
print(y.shape)         # [2, 4096, 1024]

# RetNet layer
retnet = RetNet(hidden_size=1024, num_heads=4).cuda().to(torch.bfloat16)
y = retnet(x)[0]

# RWKV-6 attention block
rwkv_attn = RWKV6Attention(hidden_size=1024, num_heads=4).cuda().to(torch.bfloat16)
y = rwkv_attn(x)[0]

# All three share the same interface — drop-in replacements for torch.nn.MultiheadAttention`}
      </CodeBlock>

      <Prose>
        The library uses Triton kernels internally, which makes it portable across GPU architectures. For the chunked parallel path it implements the three-form decomposition described in section 3.6 — training uses the chunkwise form, inference uses the recurrent form, and a small "parallel within chunk" kernel handles the BMM within each chunk. Benchmarks on A100 and H100 show the fla kernels within 2x of FlashAttention-2 for softmax attention at short sequence lengths, and significantly faster at <Code>L > 16K</Code>.
      </Prose>

      <H3>5.3 A minimal RWKV block in PyTorch</H3>

      <Prose>
        The block below is a readable RWKV-4 time-mix + channel-mix pair. Production RWKV uses a custom CUDA kernel for the time-mix recurrence, but the pure-PyTorch version shown here is correct and faithful to the published paper; it is what you would use for prototyping on CPU or for pedagogy.
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn as nn
import torch.nn.functional as F

class RWKVTimeMix(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        # token-shift mixers for R, K, V (elementwise interpolation weights)
        self.time_mix_r = nn.Parameter(torch.ones(n_embd) * 0.5)
        self.time_mix_k = nn.Parameter(torch.ones(n_embd) * 0.5)
        self.time_mix_v = nn.Parameter(torch.ones(n_embd) * 0.5)
        # per-channel time decay (learned; constrained to be non-positive in practice)
        self.time_decay = nn.Parameter(-torch.ones(n_embd))   # "w"
        # per-channel current-token bonus (learned)
        self.time_first = nn.Parameter(torch.zeros(n_embd))    # "u"
        # projections
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.key        = nn.Linear(n_embd, n_embd, bias=False)
        self.value      = nn.Linear(n_embd, n_embd, bias=False)
        self.output     = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x):
        # x : [B, L, C]
        # token shift: prev-token mix
        x_prev = F.pad(x[:, :-1, :], (0, 0, 1, 0))             # [B, L, C]
        xr = x * self.time_mix_r + x_prev * (1 - self.time_mix_r)
        xk = x * self.time_mix_k + x_prev * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + x_prev * (1 - self.time_mix_v)
        r = torch.sigmoid(self.receptance(xr))
        k = self.key(xk)
        v = self.value(xv)
        # WKV recurrence (recurrent form, per-batch scan)
        B, L, C = x.shape
        wkv = torch.zeros_like(x)
        a = torch.zeros(B, C, device=x.device)
        b = torch.zeros(B, C, device=x.device)
        p = torch.full((B, C), -1e30, device=x.device)
        w = self.time_decay; u = self.time_first
        for t in range(L):
            q = torch.maximum(p, u + k[:, t])
            e1 = torch.exp(p - q); e2 = torch.exp(u + k[:, t] - q)
            wkv[:, t] = (e1 * a + e2 * v[:, t]) / (e1 * b + e2 + 1e-6)
            q2 = torch.maximum(w + p, k[:, t])
            e1 = torch.exp(w + p - q2); e2 = torch.exp(k[:, t] - q2)
            a = e1 * a + e2 * v[:, t]; b = e1 * b + e2; p = q2
        return self.output(r * wkv)

class RWKVChannelMix(nn.Module):
    def __init__(self, n_embd, n_hidden=None):
        super().__init__()
        n_hidden = n_hidden or 4 * n_embd
        self.time_mix_r = nn.Parameter(torch.ones(n_embd) * 0.5)
        self.time_mix_k = nn.Parameter(torch.ones(n_embd) * 0.5)
        self.key        = nn.Linear(n_embd, n_hidden, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value      = nn.Linear(n_hidden, n_embd, bias=False)

    def forward(self, x):
        x_prev = F.pad(x[:, :-1, :], (0, 0, 1, 0))
        xr = x * self.time_mix_r + x_prev * (1 - self.time_mix_r)
        xk = x * self.time_mix_k + x_prev * (1 - self.time_mix_k)
        k  = F.relu(self.key(xk)).pow(2)        # ReLU-squared nonlinearity
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv

class RWKVBlock(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd); self.ln2 = nn.LayerNorm(n_embd)
        self.time_mix    = RWKVTimeMix(n_embd)
        self.channel_mix = RWKVChannelMix(n_embd)
    def forward(self, x):
        x = x + self.time_mix(self.ln1(x))
        x = x + self.channel_mix(self.ln2(x))
        return x`}
      </CodeBlock>

      <Prose>
        Three structural details are worth noting. First, <em>token shift</em> (the <Code>{"x_t' = \\mu \\odot x_t + (1 - \\mu) \\odot x_{t-1}"}</Code> mix) is applied before computing R, K, V — this gives the network a cheap one-step lookback without paying for full attention, and ablations in the RWKV-4 paper show this is critical (removing it loses 1-2 points of downstream accuracy). Second, the receptance gate <Code>{"\\sigma(r)"}</Code> multiplies the attention output before the final linear — it is the "admit to residual stream" gate, analogous to an LSTM's output gate. Third, the channel-mix uses <Code>{"\\text{ReLU}^2"}</Code> (squared ReLU) rather than GELU or SwiGLU; this was found empirically to work well but is not theoretically motivated.
      </Prose>

      <H3>5.4 Deployment considerations</H3>

      <Callout accent="gold">
        Production rules-of-thumb for 2026: (1) For inference on CPU or edge, RWKV is currently the best supported linear-attention family — there are <Code>rwkv.cpp</Code> and <Code>web-rwkv</Code> projects with int8/int4 quantization that run 7B+ models interactively on a phone. (2) For training and server inference, use <Code>flash-linear-attention</Code> (<Code>fla-org</Code>) — it has the most mature Triton kernels and covers GLA, RetNet, Mamba-2, and RWKV-6+. (3) For long-context tasks ({">"}32K tokens), hybrid architectures (a few full-attention layers plus many linear-attention layers) like Jamba or Samba give the best quality-for-cost. (4) RetNet checkpoints exist but the production model ecosystem is thin — RetNet is a research baseline more than a deployment target. (5) Never expect linear-attention models to match softmax on exact-recall tasks (passkey retrieval, needle-in-a-haystack past 32K); the compressive state has a fundamental capacity limit set by <Code>{"d \\times d"}</Code>, and large contexts will exhaust it.
      </Callout>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6a. RWKV time-mix forward pass — StepTrace</H3>

      <StepTrace
        label="RWKV-4 time-mix forward pass over 5 tokens"
        steps={[
          {
            label: "Step 1 — Token shift",
            render: () => (
              <Prose>
                {"At each position t, we mix the current input with the previous input: x_t' = mu * x_t + (1 - mu) * x_{t-1}. This is done independently for the three projections R, K, V, each with its own learned mix vector (time_mix_r, time_mix_k, time_mix_v). The purpose: give the R/K/V computation a small amount of one-step lookback without paying for full attention. At t=0 there is no previous token, so x_prev is zero-padded and the effective mix is x_0' = mu * x_0."}
              </Prose>
            ),
          },
          {
            label: "Step 2 — Compute R, K, V projections",
            render: () => (
              <Prose>
                {"r_t = sigmoid(W_R @ x_t') — the receptance gate, used at the end to scale the block's output. k_t = W_K @ x_t' — the key, a per-channel log-weight on how much this token should contribute to the running state. v_t = W_V @ x_t' — the value, what gets aggregated. The sigmoid on r is critical; without it, the output gating cannot suppress channels and quality drops."}
              </Prose>
            ),
          },
          {
            label: "Step 3 — Initialize state at t=0",
            render: () => (
              <Prose>
                {"The WKV recurrence maintains three accumulators per channel: a (weighted sum of values), b (sum of weights), and p (log-space running maximum for stability). Initial values: a = 0, b = 0, p = -infinity. At t=0 we compute the output using only the current-token bonus exp(u + k_0): wkv_0 = v_0 (since only one term). Then update the running state with decayed contributions."}
              </Prose>
            ),
          },
          {
            label: "Step 4 — Step t=1: current + decayed past",
            render: () => (
              <Prose>
                {"Compute q = max(p, u + k_1). Rescale: e1 = exp(p - q), e2 = exp(u + k_1 - q). Output: wkv_1 = (e1 * a + e2 * v_1) / (e1 * b + e2). This is the weighted sum of the current token (with bonus u) and the accumulated decayed past. Then update running state with time decay w: new a = exp(w) * old_a + exp(k_1) * v_1 (log-space rescaled). Each element of w is learned and typically constrained to be negative — closer to zero means longer memory, more negative means faster forgetting."}
              </Prose>
            ),
          },
          {
            label: "Step 5 — Decay compounds over t=2,3,4",
            render: () => (
              <Prose>
                {"At each subsequent step, the running state is multiplied by exp(w) and a new contribution exp(k_t) * v_t is added. After T steps, the contribution from position i has weight exp(w * (t - i - 1) + k_i). Channels with w close to 0 retain information for many steps; channels with very negative w forget almost immediately. The learned w is per-channel, so some channels specialize in short-range features and others in long-range. This is the 'time-decay' pattern that replaces softmax's pairwise attention — no O(L^2) computation, but the expressivity is limited by the fixed exponential schedule."}
              </Prose>
            ),
          },
          {
            label: "Step 6 — Apply receptance and output projection",
            render: () => (
              <Prose>
                {"At each step, the output of the time-mix block is out_t = W_out @ (r_t * wkv_t). The receptance r_t (a sigmoid gate) decides per-channel how much of the wkv signal to pass through. Channels where r_t is close to 0 are suppressed; channels where r_t is near 1 pass unaltered. The final projection W_out mixes across channels to produce the output that gets added to the residual stream. The block is complete — one token, one recurrence step, one O(d^2) update; no attention matrix was ever formed."}
              </Prose>
            ),
          },
        ]}
      />

      <H3>6b. Throughput vs sequence length</H3>

      <Prose>
        Approximate reported throughput (tokens per second) for three model families at a common 7B parameter scale on an A100 80GB, varying context length. Transformer numbers from FlashAttention-2 benchmarks; RWKV-6 from BlinkDL's reported numbers; Mamba-2 from the state-spaces paper. All figures are inference tokens/sec at batch size 1 — the regime where linear attention's constant-memory property matters most.
      </Prose>

      <Plot
        label="inference throughput vs sequence length (tokens/sec, 7B models on A100, higher is better)"
        xLabel="sequence length"
        yLabel="tokens / sec"
        series={[
          {
            name: "Transformer (FlashAttention-2)",
            color: "#f87171",
            points: [
              [1024, 180],
              [2048, 170],
              [4096, 140],
              [8192, 95],
              [16384, 55],
              [32768, 28],
              [65536, 12],
            ],
          },
          {
            name: "RWKV-6 Finch",
            color: colors.gold,
            points: [
              [1024, 150],
              [2048, 148],
              [4096, 147],
              [8192, 145],
              [16384, 143],
              [32768, 140],
              [65536, 138],
            ],
          },
          {
            name: "Mamba-2",
            color: colors.green,
            points: [
              [1024, 145],
              [2048, 145],
              [4096, 144],
              [8192, 143],
              [16384, 142],
              [32768, 140],
              [65536, 138],
            ],
          },
        ]}
      />

      <Prose>
        The transformer starts faster at short contexts (FlashAttention's optimized kernels beat the linear-attention constants up to about 4K), then decays rapidly as the KV cache grows and memory bandwidth becomes the bottleneck. At 64K tokens the transformer is at ~12 tok/sec — essentially unusable for interactive applications. RWKV-6 and Mamba-2 are nearly flat across the range: linear time and constant memory mean throughput barely depends on context length. The crossover sits around 6K-8K tokens. This is the regime where linear attention actually wins in wall-clock performance, not just asymptotically.
      </Prose>

      <H3>6c. Memory usage vs sequence length</H3>

      <Plot
        label="inference memory footprint vs sequence length (GB, 7B model, fp16)"
        xLabel="sequence length"
        yLabel="GB"
        series={[
          {
            name: "Transformer (KV cache)",
            color: "#f87171",
            points: [
              [1024, 14.5],
              [4096, 16.0],
              [16384, 22.0],
              [65536, 46.0],
              [262144, 142.0],
            ],
          },
          {
            name: "RWKV-6 (constant state)",
            color: colors.gold,
            points: [
              [1024, 14.0],
              [4096, 14.0],
              [16384, 14.0],
              [65536, 14.0],
              [262144, 14.0],
            ],
          },
        ]}
      />

      <Prose>
        Transformer memory grows linearly with context: the KV cache at each layer stores <Code>{"2 \\cdot L \\cdot d"}</Code> float16s, which at 32 layers and <Code>{"d = 4096"}</Code> is about 500KB per token. At 64K tokens the cache alone is 32GB; at 256K it is 128GB — more than a single A100 can hold. RWKV's memory is flat: the recurrent state is <Code>{"O(d^2)"}</Code> regardless of context, measured in MB, not GB. This is why RWKV and other linear-attention architectures are the natural choice for applications that need long context on limited hardware — streaming agents, on-device assistants, long-document analysis.
      </Prose>

      <H3>6d. RWKV time-decay heatmap (w over channels × time)</H3>

      <Prose>
        The heatmap below shows <Code>{"\\exp(w \\cdot \\Delta t)"}</Code> — the effective weight of a past contribution as a function of how far in the past it was (<Code>{"\\Delta t"}</Code>) and which channel we are looking at. Channels with <Code>w</Code> close to zero (top rows) retain contributions for many steps; channels with very negative <Code>w</Code> (bottom rows) forget within 2-3 steps. This separation is learned during training and is one of RWKV's key expressive resources — different channels handle different time scales.
      </Prose>

      <Heatmap
        label="rwkv time-decay weight exp(w*dt) for 8 channels (rows) vs lookback dt (cols). brighter = longer memory"
        rowLabels={[
          "chan 0 (slow)",
          "chan 1",
          "chan 2",
          "chan 3",
          "chan 4",
          "chan 5",
          "chan 6",
          "chan 7 (fast)",
        ]}
        colLabels={["dt=1", "dt=2", "dt=4", "dt=8", "dt=16", "dt=32", "dt=64", "dt=128"]}
        matrix={[
          [0.99, 0.98, 0.96, 0.92, 0.85, 0.72, 0.52, 0.27],
          [0.97, 0.94, 0.88, 0.78, 0.60, 0.36, 0.13, 0.02],
          [0.95, 0.90, 0.81, 0.66, 0.44, 0.19, 0.04, 0.00],
          [0.90, 0.81, 0.66, 0.43, 0.19, 0.04, 0.00, 0.00],
          [0.82, 0.67, 0.45, 0.20, 0.04, 0.00, 0.00, 0.00],
          [0.70, 0.49, 0.24, 0.06, 0.00, 0.00, 0.00, 0.00],
          [0.50, 0.25, 0.06, 0.00, 0.00, 0.00, 0.00, 0.00],
          [0.30, 0.09, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00],
        ]}
        colorScale="gold"
      />

      <Prose>
        Channel 0 (top row) has a very slow decay — exp(-0.01) per step, so a contribution from 128 steps ago still has weight 0.27. This channel specializes in long-range features (maintaining subject of a paragraph, tracking narrative context). Channel 7 (bottom row) has fast decay — exp(-1.2) per step, so a contribution from just 4 steps ago is down to 0.01. This channel specializes in short-range syntactic features (agreement, local word-order). The per-channel learnable decay is what lets RWKV handle mixed time-scale dependencies within a single recurrence — no softmax-style adaptive mechanism needed, just different learned <Code>w</Code> values per channel.
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <StepTrace
        label="when to choose linear attention / RWKV / transformer"
        steps={[
          {
            label: "Extreme long context (100K+ tokens) — linear attention family",
            render: () => (
              <Prose>
                {"If your task needs context beyond 64K tokens on a single GPU, standard transformers are not an option — the KV cache alone exceeds GPU memory. Use RWKV-6/7, Mamba-2, or a hybrid (Jamba, Samba) that mixes linear-attention layers with a few sparse full-attention layers. Quality will be 1-3 points below a full-attention model at the same parameter count, but the capability is real: you can process book-length inputs or large codebases without splitting. Passkey-retrieval and needle-in-a-haystack benchmarks favor attention-full architectures even at long context, but for tasks with dense rather than sparse information use (summarization, translation, chat over long documents) linear attention is often indistinguishable from full attention at the top end."}
              </Prose>
            ),
          },
          {
            label: "Mainstream LLM (2K-8K context, 1-70B params) — transformer with FlashAttention",
            render: () => (
              <Prose>
                {"For most production LLMs of 2026, context lengths sit in the 2K-32K range and compute is dominated by the matmul in the FFN, not the attention. Here the transformer still wins slightly on quality (1-2 points on MMLU/GSM8K benchmarks at matched parameters), the training infrastructure is more mature, and FlashAttention-2/3 make the attention cost negligible. Unless you have a specific long-context reason, use a transformer. The dominance of this regime is what has kept linear-attention adoption in research rather than production."}
              </Prose>
            ),
          },
          {
            label: "Embedded/edge real-time inference — RWKV",
            render: () => (
              <Prose>
                {"For on-device inference (smartphones, embedded AI chips, browser via WebAssembly), RWKV is currently the leading choice. The constant-memory recurrent inference fits in 4-8GB RAM for 7B models at int4 quantization. There are production deployments of RWKV for mobile chat assistants and real-time transcription captioning. Projects like rwkv.cpp (C++ inference) and web-rwkv (browser) have matured significantly. Transformers can run on-device too (LLaMA-2 7B at int4), but the KV cache grows with session length, which is a real limitation for agents with persistent memory."}
              </Prose>
            ),
          },
          {
            label: "Research: RNN-transformer hybrids — very active area",
            render: () => (
              <Prose>
                {"A large body of research in 2024-2026 studies architectures that interleave attention and SSM/linear-attention layers: Jamba (AI21), Samba (Microsoft), Hymba (NVIDIA), Granite-4 (IBM), Zamba (Zyphra). Empirically these hybrids capture most of the recall capability of full attention with most of the efficiency of linear attention. As of 2026 the typical recipe is 1 attention layer for every 4-7 SSM/linear layers, plus a global mixer layer. If you are starting a new language model project with efficiency concerns, a hybrid architecture is likely the right default by 2026."}
              </Prose>
            ),
          },
          {
            label: "Quality loss tolerance — when linear is 'good enough'",
            render: () => (
              <Prose>
                {"Rule of thumb: linear attention is a drop-in replacement when the acceptable quality loss is 1-3% and the sequence length is 16K+. Below 16K you are paying a quality tax without getting meaningful speedup. Above 16K the quality tax is fixed but the speedup compounds. For tasks where the input is routinely 50K+ tokens (e.g., code repositories, long documents, video transcripts), linear attention is nearly always the right answer. For chat with 4K-8K context it is almost never the right answer today, though the gap closes yearly."}
              </Prose>
            ),
          },
          {
            label: "Avoid — applying linear attention without long context or constant memory needs",
            render: () => (
              <Prose>
                {"Don't use linear attention just because it sounds modern. At 2K context with transformer infrastructure, softmax is faster, higher quality, and uses less memory (activations, not just weights). Linear attention is worth the quality hit only when (a) you need very long context, (b) you need constant memory inference, or (c) you are doing research on architecture. Applying it to a short-context task is a straightforward quality regression with no compensating benefit."}
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          8. WHAT SCALES
          ====================================================================== */}
      <H2>8. What scales</H2>

      <H3>8.1 Linear vs quadratic: the crossover math</H3>

      <Prose>
        Standard softmax attention has training cost <Code>{"O(L^2 \\cdot d)"}</Code>; linear attention has <Code>{"O(L \\cdot d^2)"}</Code>. The crossover at which linear is cheaper is <Code>{"L = d"}</Code>. For typical model dimensions (<Code>{"d = 1024"}</Code> for a small model, <Code>{"d = 4096"}</Code> for a 7B model, <Code>{"d = 12288"}</Code> for GPT-3-scale), the crossover is at <Code>{"L = 1024, 4096, 12288"}</Code> respectively. In practice constant factors matter: FlashAttention's softmax implementation is 2-4× faster than a naive linear-attention kernel at equal FLOPs, so the wall-clock crossover sits a few factors above the theoretical <Code>L = d</Code>. At <Code>{"L = 100{,}000"}</Code> the advantage is unambiguous regardless of implementation quality.
      </Prose>

      <H3>8.2 Quality gap has narrowed from 5-10% to 1-3%</H3>

      <Prose>
        In 2020, Katharopoulos's linear attention was 5-10 percent worse in perplexity than softmax at matched parameters. By 2024, GLA and RWKV-6 sit at 1-3 percent perplexity gap, depending on the benchmark. RWKV-7 Goose at 14B parameters claims parity with or slight improvement over LLaMA-2-13B on several downstream benchmarks. The progression is driven by architectural enrichment: <em>fixed</em> feature maps (2020) gave way to <em>random</em> feature maps (Performers, 2021) gave way to <em>data-dependent</em> decay (GLA, RWKV-6 dynamic, 2023-2024) gave way to <em>matrix-valued state</em> + <em>delta rule</em> updates (RWKV-7, 2025). Each step added expressivity at the cost of training kernel complexity. The remaining gap is attributed to softmax's ability to do exact indexed retrieval — linear attention's compressive <Code>{"d \\times d"}</Code> state can only store a finite amount of information, so long contexts eventually saturate it.
      </Prose>

      <H3>8.3 Chunked parallel training is the scaling enabler</H3>

      <Prose>
        Before chunked parallel kernels, linear attention was faster than softmax in theory but slower in practice — the recurrent form is linear-cost but sequential, which wastes GPU parallelism. Yang et al. 2024 (GLA) introduced the chunkwise parallel form: within a chunk of size <Code>C</Code>, compute attention as a <Code>{"C \\times C"}</Code> BMM (GPU-friendly); across chunks, carry a <Code>{"d \\times d"}</Code> state (a small sequential loop). This gave linear attention the same GPU utilization as softmax, eliminating the constant-factor penalty that had kept it research-only. The same trick applies to RWKV (from RWKV-5 onwards) and Mamba-2 (which has its own selective-scan kernel). By 2026, any production-scale linear-attention training uses chunkwise parallelism at <Code>C</Code> between 64 and 256.
      </Prose>

      <H3>8.4 State size determines information capacity</H3>

      <Prose>
        The recurrent state in a linear-attention layer is <Code>{"d \\times d"}</Code> (the numerator) plus <Code>{"d"}</Code> (the denominator). This state is a <em>fixed-size compression</em> of the entire past. Information theory gives a bound: the entropy of the past cannot exceed <Code>{"d^2 \\cdot 16"}</Code> bits for fp16. For <Code>{"d = 4096"}</Code> this is ~256Mbit, or roughly 32MB — plenty for most tasks but a hard wall for tasks requiring exact retrieval over very long contexts. This capacity limit explains why linear-attention models systematically underperform softmax on passkey/needle benchmarks past 32K tokens: once the state is saturated, new information overwrites old. RWKV-7's "delta rule" update and matrix-valued states (Eagle) both try to mitigate this by using the state more efficiently, but the fundamental capacity remains <Code>{"O(d^2)"}</Code>.
      </Prose>

      <H3>8.5 Hardware trajectory: kernels are catching up</H3>

      <Prose>
        In 2021 there was no Triton kernel for linear attention. By 2024 the <Code>fla-org</Code> library and Mamba's selective-scan had production-grade kernels. By 2026 NVIDIA and AMD both provide first-party kernels for gated linear attention in cuBLAS-LT and ROCm-BLAS. Inference vendors (Together, Fireworks, Lambda) list RWKV and Mamba endpoints alongside transformers, priced similarly on a tokens-per-second basis. The infrastructure story has converged enough that "use linear attention" no longer means "commit to a research-grade toolchain." This catch-up is what has made RWKV-7 and Mamba-2 practical options in 2026, where three years ago they would have been niche.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES
          ====================================================================== */}
      <H2>9. Failure modes</H2>

      <H3>9.1 Wrong feature map dramatically hurts quality</H3>

      <Prose>
        The feature map <Code>{"\\phi"}</Code> in linear attention determines what kernel you are computing. Poor choices give poor results. For example, using <Code>{"\\phi(x) = x"}</Code> (identity) without non-negativity produces a kernel that can be negative, which breaks the softmax-like normalization: the denominator can be zero or negative, producing NaN outputs. Using <Code>{"\\phi(x) = \\text{ReLU}(x)"}</Code> enforces non-negativity but collapses information from the negative half-space. The <Code>{"\\text{elu}(x) + 1"}</Code> map used in Katharopoulos 2020 is the minimal choice that is non-negative and differentiable; Performers's random features are asymptotically optimal for approximating softmax. Fix: always use a validated feature map (elu+1 for simplicity, random features for higher fidelity) and verify output non-negativity after the feature map.
      </Prose>

      <H3>9.2 Missing or incorrect causal mask</H3>

      <Prose>
        Linear attention gets its causality from the running sum <Code>{"S_t = \\sum_{i \\le t} \\phi(K_i) V_i^T"}</Code>. If you accidentally compute <Code>{"S = \\sum_i \\phi(K_i) V_i^T"}</Code> (no upper bound on <Code>i</Code>), you have leaked future information and the model will memorize the training data but fail on generation. The bug is silent during training because the loss still decreases; it only shows up at inference when the model produces gibberish. Symptom: perfect training loss, catastrophic generation. Fix: always use <Code>cumsum</Code> or a chunked/recurrent implementation; never materialize a <em>non-masked</em> sum over all positions.
      </Prose>

      <CodeBlock language="python">
{`# BUG: no cumsum — sums over the full sequence, leaking future into past
S = (phiK.unsqueeze(-1) * V.unsqueeze(-2)).sum(dim=0)   # [d, dv], shared!
out = torch.einsum("ld,dv->lv", phiQ, S)                # every token sees future

# FIX: cumsum — running sum, respects causality
running = (phiK.unsqueeze(-1) * V.unsqueeze(-2)).cumsum(dim=0)   # [L, d, dv]
out = torch.einsum("ld,ldv->lv", phiQ, running)`}
      </CodeBlock>

      <H3>9.3 RWKV pure-PyTorch is slow — needs custom CUDA for training speed</H3>

      <Prose>
        The RWKV time-mix kernel, naively implemented in PyTorch, runs as a Python-level for-loop over the sequence dimension. At inference with batch size 1 this is fine; at training on an A100 at batch size 64, sequence 2048, it is 10-50× slower than the custom CUDA kernel. The reason: PyTorch's operator-level dispatch overhead dominates when each op is small. The fix is one of (a) the <Code>rwkv-kernel</Code> CUDA package, which implements the whole time-mix recurrence as one fused kernel; (b) the <Code>fla-org/flash-linear-attention</Code> library, which provides a Triton kernel for RWKV-6; or (c) <Code>torch.compile</Code> with the <Code>mode="max-autotune"</Code> option, which can fuse the Python loop into a single kernel in favorable cases. Symptom: "my RWKV training is 50× slower than my transformer baseline." Fix: use the proper kernel, or prototype at small scale on CPU and scale up only after you have the kernel set up.
      </Prose>

      <H3>9.4 Quality ceiling differs sharply by task</H3>

      <Prose>
        Linear attention models have a distinctive quality profile across tasks. On language modeling perplexity: close to softmax (1-3% gap). On multi-task reasoning (MMLU, GSM8K): close to softmax at 7B+ scale. On exact recall tasks (passkey retrieval at 32K+, needle-in-a-haystack): systematically worse, often dramatically — the compressive <Code>{"d \\times d"}</Code> state cannot store arbitrary long-range indexed information. On in-context learning (few-shot accuracy): worse than softmax, particularly for tasks where the model needs to attend to specific few-shot examples. On structured output (JSON, code): comparable. Symptom: "my RWKV agent fails on tasks where it needs to copy something from 20K tokens ago." Fix: use a hybrid architecture with some full-attention layers, or accept that long-range exact recall is not linear attention's strength and route those queries to a softmax model.
      </Prose>

      <H3>9.5 Hallucination rate higher than matched-scale transformer</H3>

      <Prose>
        As of 2024 evaluations, linear-attention models at matched parameter count show higher hallucination rates on factuality benchmarks (TruthfulQA, FactScore) than equivalently-sized transformers. Specifically, RWKV-4-14B's TruthfulQA score was ~5 points below LLaMA-2-13B. The hypothesis: the compressive state averages rather than indexes past information, making it easier for the model to confabulate plausible-but-false details rather than retrieve accurate ones. Fix: apply retrieval augmentation (RAG) to ground generations, use instruction-tuning on factual datasets, or use hybrid architectures. The gap has narrowed with RWKV-6 and RWKV-7 but remains observable.
      </Prose>

      <H3>9.6 Fine-tuning methodology differs from transformers</H3>

      <Prose>
        Linear-attention models respond differently to fine-tuning than transformers. The learned time-decay parameter <Code>w</Code> in RWKV can be destabilized by aggressive learning rates — fine-tuning typically uses 5-10× lower LR than would be appropriate for a transformer of the same size. LoRA adapters on RWKV should target the R, K, V, and time-decay parameters (not just Q/K/V as in transformers); naive LoRA configs miss the time-decay and get inferior adaptation. Symptom: "my RWKV fine-tune is unstable or produces worse results than the base model on held-out data." Fix: use RWKV-specific fine-tuning recipes from the RWKV community (e.g., the <Code>rwkv-lm-tune</Code> or <Code>RWKV-LM-LoRA</Code> repos), which include the extra modules and lower LR defaults.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        Read in roughly chronological order to follow the development from Katharopoulos's original linear-attention observation through the RWKV architecture line and the SSM-attention duality.
      </Prose>

      <StepTrace
        label="primary literature"
        steps={[
          {
            label: "Katharopoulos et al. 2020 — Linear attention / Transformers are RNNs (arXiv:2006.16236)",
            render: () => (
              <Prose>
                Katharopoulos, A., Vyas, A., Pappas, N., and Fleuret, F. (2020). "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention." ICML 2020. arXiv:2006.16236. Available at arxiv.org/abs/2006.16236. The foundational paper for the linear-attention family. Section 3 gives the derivation: softmax attention with a factorizable kernel becomes a running sum, which is an RNN. Section 4 proposes the elu+1 feature map and compares against standard softmax on image generation and speech recognition. The experimental gains are modest but the framing — "autoregressive transformers are a specific kind of RNN" — reshaped subsequent architecture research. Cited by every subsequent linear-attention paper.
              </Prose>
            ),
          },
          {
            label: "Wang et al. 2020 — Linformer (arXiv:2006.04768)",
            render: () => (
              <Prose>
                Wang, S., Li, B.Z., Khabsa, M., Fang, H., and Ma, H. (2020). "Linformer: Self-Attention with Linear Complexity." arXiv:2006.04768. Available at arxiv.org/abs/2006.04768. An alternative route to linear attention: low-rank projection of K and V along the sequence axis, producing a <Code>{"L \\times k"}</Code> attention matrix where <Code>k</Code> is fixed. Cost is <Code>{"O(L \\cdot k \\cdot d)"}</Code>. Limitation: the learned projection is per-length, so the model only works at the sequence length it was trained on. Useful for fixed-length encoder tasks (sentence-pair classification, document embedding) but not autoregressive generation. Less influential than Katharopoulos but an important contemporaneous data point.
              </Prose>
            ),
          },
          {
            label: "Choromanski et al. 2021 — Performers (arXiv:2009.14794)",
            render: () => (
              <Prose>
                Choromanski, K., Likhosherstov, V., Dohan, D., Song, X., Gane, A., Sarlos, T., Hawkins, P., Davis, J., Mohiuddin, A., Kaiser, L., Belanger, D., Colwell, L., and Weller, A. (2021). "Rethinking Attention with Performers." ICLR 2021. arXiv:2009.14794. Available at arxiv.org/abs/2009.14794. Proves that the softmax kernel <Code>{"\\exp(q \\cdot k)"}</Code> can be approximated by random features (FAVOR+) with variance <Code>{"O(1/m)"}</Code> in the number of random features <Code>m</Code>. First linear-attention method to come within 1-2% of softmax at matched scale on Wikitext-103. Section 2 gives the theoretical derivation; section 3 presents the FAVOR+ algorithm; section 4 empirically validates. The paper also introduced the "positive random features" trick that ensures non-negativity without biasing the estimator.
              </Prose>
            ),
          },
          {
            label: "Peng et al. 2023 — RWKV-4 (arXiv:2305.13048)",
            render: () => (
              <Prose>
                Peng, B., Alcaide, E., Anthony, Q., Albalak, A., Arcadinho, S., Cao, H., Cheng, X., Chung, M., Grella, M., GV, K.K., He, X., Hou, H., Kazienko, P., Kocoń, J., Kong, J., Koptyra, B., Lau, H., Mantri, K.S.I., Mom, F., Saito, A., Tang, X., Wang, B., Wind, J.S., Woźniak, S., Zhang, R., Zhang, Z., Zhao, Q., Zhou, P., Zhu, J., and Zhu, R.-J. (2023). "RWKV: Reinventing RNNs for the Transformer Era." EMNLP 2023 Findings. arXiv:2305.13048. Available at arxiv.org/abs/2305.13048. The RWKV-4 paper. Section 3 describes the time-mix (WKV) and channel-mix blocks; section 4 gives training details including the custom CUDA kernel; section 5 reports benchmarks on Pile perplexity and downstream tasks. RWKV-4-14B was the first linear-attention model to be trained at 10B+ scale. The paper is notable for being led by an independent community effort (BlinkDL) rather than a major lab, and for open-sourcing all checkpoints.
              </Prose>
            ),
          },
          {
            label: "Peng et al. 2024 — Eagle and Finch / RWKV-5/6 (arXiv:2404.05892)",
            render: () => (
              <Prose>
                Peng, B., Goldstein, D., Anthony, Q., Albalak, A., Alcaide, E., Biderman, S., Cheah, E., Du, X., Ferdinan, T., Hou, H., Kazienko, P., GV, K.K., Kocoń, J., Koptyra, B., Krishna, S., McClelland, R., Muennighoff, N., Obeid, F., Saito, A., Song, G., Tu, H., Woźniak, S., Zhang, R., Zhao, B., Zhao, Q., Zhou, P., Zhu, J., and Zhu, R.-J. (2024). "Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence." arXiv:2404.05892. Available at arxiv.org/abs/2404.05892. Two architectural upgrades in one paper. Eagle (RWKV-5) replaces RWKV-4's scalar time-decay with a matrix-valued state, increasing capacity without changing training cost. Finch (RWKV-6) adds data-dependent (dynamic) time decay, closing part of the expressivity gap with softmax attention. The paper shows Eagle-7B and Finch-7B matching or exceeding LLaMA-2-7B on several benchmarks.
              </Prose>
            ),
          },
          {
            label: "Sun et al. 2023 — RetNet / Retentive Network (arXiv:2307.08621)",
            render: () => (
              <Prose>
                Sun, Y., Dong, L., Huang, S., Ma, S., Xia, Y., Xue, J., Wang, J., and Wei, F. (2023). "Retentive Network: A Successor to Transformer for Large Language Models." arXiv:2307.08621. Available at arxiv.org/abs/2307.08621. Introduces the retention mechanism: linear attention with a fixed per-head exponential decay <Code>{"\\gamma"}</Code>. The key contribution is showing three equivalent computational forms — parallel (for training), recurrent (for inference), and chunkwise parallel (for long-context training) — all producing identical outputs. Section 4 benchmarks against transformers at 2.7B parameters and reports comparable perplexity on The Pile with 7× higher inference throughput. RetNet has remained influential as a research baseline but the production ecosystem (open checkpoints, community fine-tunes) is thinner than RWKV's.
              </Prose>
            ),
          },
          {
            label: "Yang et al. 2024 — Gated Linear Attention (arXiv:2312.06635)",
            render: () => (
              <Prose>
                Yang, S., Wang, B., Shen, Y., Panda, R., and Kim, Y. (2024). "Gated Linear Attention Transformers with Hardware-Efficient Training." ICML 2024. arXiv:2312.06635. Available at arxiv.org/abs/2312.06635. Generalizes RetNet's fixed retention to a data-dependent gate — each key-value pair's contribution to the state is modulated by a learned, input-conditional gating vector. Equally important, the paper provides a Triton kernel for chunkwise parallel training that matches FlashAttention's wall-clock efficiency at short sequences and dominates at long sequences. GLA is the leading pure-linear-attention architecture as of 2026 and is the basis for the <Code>fla-org/flash-linear-attention</Code> library.
              </Prose>
            ),
          },
          {
            label: "Dao & Gu 2024 — Mamba-2 / Transformers are SSMs (arXiv:2405.21060)",
            render: () => (
              <Prose>
                Dao, T. and Gu, A. (2024). "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality." ICML 2024. arXiv:2405.21060. Available at arxiv.org/abs/2405.21060. Unifies state-space models (Mamba line) and linear attention (Katharopoulos/RetNet/GLA line) through the observation that selective SSMs with scalar-times-identity state transitions are formally equivalent to masked linear attention. This duality explains why the two research threads had been converging independently and provides a common framework for analysis. The paper also introduces Mamba-2, which uses the duality to provide a faster selective-scan kernel than Mamba-1. The SSM-attention duality has become a standard conceptual tool; any paper on linear-attention or SSMs in 2025+ will cite it.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          11. SELF-CHECK
          ====================================================================== */}
      <H2>11. Self-check</H2>

      <Prose>
        Attempt all five before reading the answers. Exercises 1-2 test the math of linear attention; 3 tests the RWKV time-mix kernel; 4 tests architecture judgment; 5 tests debugging.
      </Prose>

      <H3>Exercise 1 (linear attention cost derivation)</H3>
      <Prose>
        A transformer has <Code>d = 2048</Code> model dimension and attention is applied over sequences of length <Code>L</Code>. Compute (a) the FLOP cost of softmax attention as a function of <Code>L</Code>, (b) the FLOP cost of linear attention, (c) the sequence length at which linear becomes cheaper. What happens to the crossover if we use multi-query attention (shared K/V across heads) with <Code>h = 16</Code> query heads and 1 KV head?
      </Prose>
      <Callout accent="green">
        <strong>Answer 1.</strong> (a) Softmax cost per layer: <Code>{"O(L^2 \\cdot d)"}</Code>. Specifically, <Code>{"Q K^T"}</Code> is <Code>{"L^2 \\cdot d"}</Code> FLOPs, softmax is <Code>{"L^2"}</Code> (negligible), and <Code>{"A V"}</Code> is <Code>{"L^2 \\cdot d"}</Code> FLOPs. Total: <Code>{"\\sim 2 L^2 d"}</Code> FLOPs. (b) Linear cost: the running <Code>{"S_t = S_{t-1} + \\phi(K_t) V_t^T"}</Code> is <Code>{"d^2"}</Code> per step; <Code>{"O_t = \\phi(Q_t)^T S_t"}</Code> is <Code>{"d^2"}</Code> per step. Total: <Code>{"\\sim 2 L d^2"}</Code> FLOPs. (c) Crossover when <Code>{"2 L^2 d = 2 L d^2"}</Code>, i.e., <Code>{"L = d = 2048"}</Code>. (d) Multi-query attention with <Code>{"h = 16"}</Code> query heads and 1 KV head reduces softmax's <Code>{"Q K^T"}</Code> cost by a factor of <Code>h</Code> on the KV side, but the dominant cost <Code>{"L^2"}</Code> is unchanged. Softmax still scales as <Code>{"O(L^2 \\cdot d)"}</Code>; linear still scales as <Code>{"O(L \\cdot d^2)"}</Code>. The crossover moves slightly — softmax's constant improves, so linear now wins only at larger <Code>L</Code>, maybe <Code>{"L = 2 \\cdot d = 4096"}</Code> in practice. But asymptotically, linear still wins.
      </Callout>

      <H3>Exercise 2 (feature map and causality)</H3>
      <Prose>
        You have access to two feature maps: (i) <Code>{"\\phi(x) = \\text{elu}(x) + 1"}</Code>, (ii) <Code>{"\\phi(x) = \\text{ReLU}(x)"}</Code>. Which one can be used as-is for causal linear attention? What goes wrong with the other? Write the running-sum update for causal linear attention with feature map <Code>{"\\phi"}</Code> applied to <Code>Q, K</Code>.
      </Prose>
      <Callout accent="green">
        <strong>Answer 2.</strong> Both can be used in principle because both produce non-negative outputs. ReLU has a subtle problem: if many <Code>{"K_i"}</Code> values are negative in all coordinates after projection, <Code>{"\\phi(K_i) = 0"}</Code> and those positions contribute nothing to the state, which equates to the model "not storing" those tokens at all — a non-smooth information bottleneck that hurts training stability. elu+1 is strictly positive everywhere, which is why it's the default. The running-sum update is: <Code>{"S_t = S_{t-1} + \\phi(K_t) V_t^T"}</Code> (numerator, shape <Code>{"d \\times d_v"}</Code>) and <Code>{"z_t = z_{t-1} + \\phi(K_t)"}</Code> (denominator, shape <Code>d</Code>); output is <Code>{"O_t = \\phi(Q_t)^T S_t / (\\phi(Q_t)^T z_t + \\epsilon)"}</Code>. Causality is automatic because the sums only include positions <Code>{"i \\le t"}</Code>.
      </Callout>

      <H3>Exercise 3 (RWKV time-decay interpretation)</H3>
      <Prose>
        In RWKV-4's time-mix, the weight applied to the contribution from position <Code>i</Code> at time <Code>t</Code> (for <Code>{"i < t"}</Code>) is <Code>{"\\exp(w \\cdot (t - i - 1) + k_i)"}</Code>. (a) If <Code>{"w = -0.05"}</Code> per channel (a "slow" channel), what fraction of the original contribution remains after 100 steps? After 1000 steps? (b) If <Code>{"w = -1.0"}</Code> (a "fast" channel), what fraction remains after 5 steps? (c) What is the role of the per-channel <Code>{"u"}</Code> bonus? Why not simply treat the current token the same as past tokens?
      </Prose>
      <Callout accent="green">
        <strong>Answer 3.</strong> (a) <Code>{"\\exp(-0.05 \\cdot 100) = \\exp(-5) = 0.0067"}</Code>, so 0.67% after 100 steps; <Code>{"\\exp(-0.05 \\cdot 1000) = \\exp(-50) \\approx 2 \\cdot 10^{-22}"}</Code>, negligible after 1000 steps. (b) <Code>{"\\exp(-1 \\cdot 5) = \\exp(-5) = 0.0067"}</Code>, so the fast channel also hits 0.67% but after only 5 steps. Different channels have different memory horizons, all governed by <Code>w</Code>. (c) The bonus <Code>u</Code> gives the current token a <em>different</em> multiplicative weight than past tokens. Without it, the current token's weight would be <Code>{"\\exp(w \\cdot (t - t - 1) + k_t) = \\exp(-w + k_t)"}</Code> — which for <Code>{"w < 0"}</Code> makes the current token's weight <em>larger</em> than a 1-step-past token, but not under the same functional form as past tokens. RWKV-4 instead uses <Code>{"\\exp(u + k_t)"}</Code> for the current token and a separate formula for past tokens, treating them asymmetrically. The role of <Code>u</Code>: compensate for the fact that the current token has <em>zero</em> time-decay (no <Code>{"\\exp(w \\cdot \\Delta t)"}</Code> factor to apply), so it needs an explicit per-channel learnable weight to be comparable with the past. In RWKV-5 onwards this is simplified by treating the current token with <Code>{"\\Delta t = 0"}</Code> uniformly.
      </Callout>

      <H3>Exercise 4 (architecture selection)</H3>
      <Prose>
        You are asked to build a language model for a production chat assistant with the following requirements: (a) context up to 200K tokens (for long-document QA), (b) interactive latency ({"<"}200ms first-token latency), (c) 14B parameter budget, (d) deployable on a single A100-80GB. Which architecture do you choose and why? What are the known trade-offs of your choice?
      </Prose>
      <Callout accent="green">
        <strong>Answer 4.</strong> Choose a linear-attention architecture — RWKV-7 Goose (14B), Mamba-2, or a hybrid like Jamba. Rationale: 200K tokens on a 14B transformer would need a KV cache of roughly <Code>{"2 \\cdot 200{,}000 \\cdot 5120 \\cdot 40 \\cdot 2"}</Code> bytes ~= 160GB, impossible on a single A100-80GB. A linear-attention model has constant state size (<Code>{"\\sim 100"}</Code> MB) regardless of context, comfortably fitting. Interactive latency at 200ms first-token: transformers suffer here because the initial forward pass is <Code>{"O(L^2)"}</Code>; at 200K tokens this is slow. Linear attention has <Code>{"O(L)"}</Code> first-token cost, so you can prefill a 200K context in seconds. Trade-offs: (i) 1-3% quality gap vs. softmax 14B — acceptable for chat, measurable on benchmarks. (ii) Poorer exact-recall over very long contexts (passkey at 200K fails more often). (iii) Less mature ecosystem — fewer fine-tuning recipes, fewer instruction-tuned checkpoints. (iv) If you need perfect factuality with precise citations from anywhere in the 200K context, full attention might still be preferable and you'd instead use retrieval to pre-filter the context. The hybrid answer (e.g., Jamba) often gives the best quality-per-cost: ~10% of layers are full attention (for exact recall), rest are linear. That's the current 2026 production sweet spot for this kind of spec.
      </Callout>

      <H3>Exercise 5 (debugging RWKV training instability)</H3>
      <Prose>
        You are training a 1.5B RWKV-6 model from scratch. Around step 5000, training loss becomes NaN. Re-starting from the last checkpoint with a smaller learning rate fixes it temporarily, but NaN returns after another 2000 steps. What are three likely causes, and how would you diagnose each?
      </Prose>
      <Callout accent="green">
        <strong>Answer 5.</strong> Three likely causes:
        <br />
        (1) <strong>Time-decay parameter <Code>w</Code> escaped its constraint.</strong> RWKV requires <Code>{"w \\le 0"}</Code> (non-positive) for the running sum to be bounded. If <Code>w</Code> drifts positive during training, the accumulator <Code>a</Code> grows unboundedly and overflows. Typical parameterization: <Code>{"w = -\\exp(\\text{learnable})"}</Code>, so the "learnable" parameter is unconstrained but <Code>w</Code> is automatically non-positive. Diagnose: print <Code>max(w)</Code> after every step; it should stay below zero. Fix: if the parameterization is <Code>{"w = -\\text{softplus}(\\text{learnable})"}</Code> or similar, your training just drifted beyond the intended range; use a tighter reparameterization.
        <br />
        (2) <strong>Gradient explosion in the WKV kernel.</strong> The log-space running max trick keeps the <em>forward</em> pass stable, but the backward pass still has a product of <Code>{"\\exp(w \\cdot \\Delta t)"}</Code> terms that can underflow to zero for very long sequences. Vanishing gradients show up as tiny updates, not NaN — but if you have <em>explosion</em> you likely have an unstable softmax-like computation somewhere in your implementation. Diagnose: check per-parameter gradient norms; look for individual parameters with norm &gt; 100x the running average. Fix: gradient clipping at norm 1.0 (standard for RWKV training), or reduce max sequence length until you can debug the kernel.
        <br />
        (3) <strong>Bad data sample.</strong> NaN losses often trace to malformed training samples: tokens out of vocabulary, extremely long sequences that trigger numerical issues, or content that confuses the tokenizer. Diagnose: save a copy of the training batch just before the NaN step; replay it in isolation to reproduce. Fix: filter the training data, add NaN-check-and-skip logic, or use bfloat16 instead of float16 (bfloat16 has the same exponent range as float32 and is much more forgiving of poor scaling).
      </Callout>

    </div>
  ),
};

export default rwkvLinearAttentionContent;
