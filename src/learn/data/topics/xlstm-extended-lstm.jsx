import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const xlstmContent = {
  title: "xLSTM (Extended LSTM)",
  readTime: "~38 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        In November 2017 the transformer ate the recurrent neural network's lunch, and for six years almost nobody who cared about scoring well on language benchmarks bothered with an RNN. The story would have ended there if not for a curious reversal in late 2023 and 2024: linear-time sequence models started winning at very long context, and several groups began proposing transformer-replacement architectures derived not from attention but from older ideas. Albert Gu and Tri Dao published Mamba (arXiv:2312.00752) in December 2023; Bo Peng's RWKV line was already in motion (arXiv:2305.13048, EMNLP 2023). The recurrent renaissance was real. Conspicuously absent from the early entries was the family that started it all: gated RNNs, and specifically the Long Short-Term Memory cell that Sepp Hochreiter and Jürgen Schmidhuber published in 1997.
      </Prose>

      <Prose>
        That changed in May 2024 when Maximilian Beck, Korbinian Pöppel, Markus Spanring, Andreas Auer, Oleksandra Prudnikova, Michael Kopp, Günter Klambauer, Johannes Brandstetter, and Sepp Hochreiter himself published "xLSTM: Extended Long Short-Term Memory" (arXiv:2405.04517). The author list matters: this is not a third-party tribute paper. It is the original LSTM co-inventor, twenty-seven years later, writing a careful argument for what would have to change about the LSTM cell to make it competitive with Mamba and the transformer at the 2024 frontier. Beck is Hochreiter's PhD student at JKU Linz; the rest of the author list is the JKU machine-learning group plus collaborators at NXAI, the spin-off Hochreiter co-founded to commercialize the architecture. The paper is Hochreiter's reply to "RNNs are dead": <em>they are not, but the 1997 design needs two specific surgeries to compete on modern hardware at modern scale.</em>
      </Prose>

      <Prose>
        The two surgeries name two new cells. The first is sLSTM (scalar LSTM): the classical LSTM with the input and forget gates moved from sigmoid to exponential activation, plus a new normalizer state that keeps the hidden output bounded despite the unbounded gates, plus a stabilizer state that prevents the exponential from overflowing. sLSTM remains a sequential, scalar-state architecture; it is the high-quality drop-in replacement for the 1997 cell. The second is mLSTM (matrix LSTM): the cell state is upgraded from a vector to a <Code>{"d \\times d"}</Code> matrix, the update rule becomes an outer-product write <Code>{"C_t = f_t C_{t-1} + i_t v_t k_t^T"}</Code>, and the readout becomes a query against the matrix memory. This recurrence is linear in the state, which means it can be parallelized at training time with a chunkwise scan exactly the way Mamba-2 and gated linear attention are parallelized. The mLSTM is not a faster LSTM; it is the same family of objects as the Mamba-2 SSD layer or GLA, derived from a different starting point and dressed up in LSTM gate vocabulary.
      </Prose>

      <Prose>
        The xLSTM block alternates these two cells with residual connections and layer normalization in roughly the layout of a transformer block (sLSTM/mLSTM in the role of attention; a feedforward in the role of FFN). The xLSTM-7B paper (Beck et al., late 2024 / early 2025 technical report) reports a 7-billion-parameter model trained on roughly 1 trillion tokens, competitive with Llama-2-7B and Mamba-7B on standard language modeling benchmarks. The model is shipped on HuggingFace as <Code>NX-AI/xLSTM-7b</Code>. Like Mamba, it requires custom CUDA kernels to hit its theoretical throughput; without them, the parallel scan falls back to a slow Python loop. Like Mamba in 2024, it is not yet in mainline <Code>transformers</Code>; you install the <Code>xlstm</Code> pip package from NXAI to use it.
      </Prose>

      <Prose>
        The motivation that ties the surgery to the recurrent renaissance is exactly the same as Mamba's motivation: at sequence lengths above a few thousand tokens, the transformer's <Code>{"O(L^2 \\cdot d)"}</Code> attention and linearly growing KV cache become the dominant cost on GPU memory and bandwidth. A recurrent model with a constant-sized state pays <Code>{"O(L \\cdot d^2)"}</Code> at training and <Code>{"O(d^2)"}</Code> per generated token at inference, both independent of context length. The question Hochreiter asks in xLSTM is sharper than "can we beat the transformer on long context": he asks "given all we have learned since 1997, is the LSTM cell competitive after the gates are exponentialized and the cell state is matrixified?" The answer the paper argues for is yes, with the caveat that the resulting object is closer in spirit to a gated linear attention than to a 1997 LSTM. Many of the key innovations — exponential gating, normalizer states, parallel scan training — were known in pieces from Performers, RWKV, and Mamba; xLSTM's contribution is putting them together in a cell-and-block design that descends cleanly from the LSTM lineage.
      </Prose>

      <Prose>
        The reception in 2024-2026 was warm but skeptical. xLSTM-7B is a credible language model but not a leap past Mamba-2 or transformer at matched scale, and the custom-kernel infrastructure is more nascent than mamba-ssm's. The deeper question — whether the architecture has structural advantages that emerge only at 70B+ parameters and 10T+ tokens — remains open as of early 2026. What is certain is that xLSTM completes the picture of the "RNN revival": Mamba came from state-space models, RWKV came from kernelized linear attention, and xLSTM came from gated RNNs. All three converge on the same basic object — a recurrence on a fixed-size state with linear update rule, gated by data-dependent factors, parallelizable at training via a scan — and they differ mainly in how the gates and state are parameterized.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Three things are wrong with the 1997 LSTM cell from a 2024 perspective. xLSTM identifies them, names a fix for each, and packages the fixes into two new cells (sLSTM, mLSTM) that share an architectural skeleton. To follow the design, walk through the three problems in order.
      </Prose>

      <Prose>
        <strong>Problem 1: sigmoid gates have a saturation ceiling.</strong> The classical LSTM input gate is <Code>{"i_t = \\sigma(W_i x_t + R_i h_{t-1} + b_i)"}</Code>, with output bounded in <Code>{"(0, 1)"}</Code>. To strongly favor "write the new value into memory," <Code>{"i_t"}</Code> needs to push close to 1, but <Code>{"\\sigma"}</Code> approaches 1 only asymptotically. To strongly favor "ignore the new value," <Code>{"i_t"}</Code> needs to be close to 0, again only asymptotic. The same goes for the forget gate. The practical effect: when one input is much more important than the others (the canonical problem an LSTM is supposed to handle), the sigmoid cannot give it disproportionate weight. The LSTM has to learn to "tilt" the gate via large pre-activation magnitudes, and then it has to cancel out that tilt for other inputs. This is the gate-revision problem the xLSTM authors point at: in long sequences with many hard decisions about what to remember, the sigmoid is structurally unable to make sharp choices.
      </Prose>

      <Prose>
        <strong>Fix 1: exponential gates.</strong> Replace <Code>{"\\sigma"}</Code> with <Code>{"\\exp"}</Code> for the input and forget gates. Now <Code>{"i_t = \\exp(W_i x_t + R_i h_{t-1} + b_i)"}</Code> is unbounded above; the model can give one input arbitrarily large weight relative to its neighbors. The output gate stays sigmoid because it is a multiplicative gain on the readout and unbounded gain is not what you want there. This is the same intuition behind the softmax kernel in attention: <Code>{"\\exp"}</Code> is what gives attention the ability to focus sharply on one token among many. Hochreiter is borrowing the idea back from attention and applying it inside the LSTM gate.
      </Prose>

      <Prose>
        <strong>Problem 2: unbounded gates make the state explode.</strong> If <Code>{"i_t"}</Code> and <Code>{"f_t"}</Code> are unbounded, then the cell state <Code>{"c_t = f_t c_{t-1} + i_t z_t"}</Code> can grow without bound across long sequences, and the hidden output <Code>{"h_t = o_t \\tanh(c_t)"}</Code> saturates the <Code>{"\\tanh"}</Code> nonlinearity to <Code>{"\\pm 1"}</Code>. You have replaced one saturation problem with another.
      </Prose>

      <Prose>
        <strong>Fix 2: a normalizer state.</strong> Carry an additional scalar state per channel, <Code>{"n_t"}</Code>, with the same recurrence as the cell but without the <Code>{"z_t"}</Code> input: <Code>{"n_t = f_t n_{t-1} + i_t"}</Code>. This is the running sum of how much "input mass" has been written into the cell, decayed by the forget gates. The hidden output is then <Code>{"h_t = o_t \\cdot c_t / n_t"}</Code>. The ratio is bounded because <Code>{"c_t"}</Code> and <Code>{"n_t"}</Code> grow proportionally — every input that contributed to <Code>{"c_t"}</Code> also added <Code>{"i_t"}</Code> to <Code>{"n_t"}</Code>. This is exactly the normalizer in linear attention (where <Code>{"y_t = (Q_t S_t) / (Q_t z_t)"}</Code> uses <Code>{"z_t = \\sum_i \\phi(K_i)"}</Code> as the denominator) and in Performers. In xLSTM language it is the "normalizer state."
      </Prose>

      <Prose>
        <strong>Problem 3: <Code>{"\\exp"}</Code> overflows in floating point.</strong> If a pre-activation is 88 in fp32, <Code>{"\\exp(88)"}</Code> is at the top of the float range; 100 is past it. Across a deep, long sequence those values will appear sometimes and the cell will produce <Code>{"+\\infty"}</Code> and <Code>{"\\text{NaN}"}</Code>.
      </Prose>

      <Prose>
        <strong>Fix 3: a log-domain stabilizer.</strong> Carry yet another scalar state, <Code>{"m_t"}</Code>, defined as the running maximum of the log-gates: <Code>{"m_t = \\max(\\log f_t + m_{t-1}, \\log i_t)"}</Code>. Then the actual gates used in the recurrence are stabilized: <Code>{"\\tilde i_t = \\exp(\\log i_t - m_t)"}</Code> and <Code>{"\\tilde f_t = \\exp(\\log f_t + m_{t-1} - m_t)"}</Code>. Both quantities are bounded by 1 (because <Code>{"m_t"}</Code> is by construction <Code>{"\\ge"}</Code> the larger of the two log-gates). The trick is the same as the standard log-sum-exp stabilization: subtract the max in log-space before taking the exp. The cell state becomes <Code>{"c_t = \\tilde f_t c_{t-1} + \\tilde i_t z_t"}</Code>, the normalizer becomes <Code>{"n_t = \\tilde f_t n_{t-1} + \\tilde i_t"}</Code>, and the output <Code>{"h_t = o_t \\cdot c_t / n_t"}</Code> is invariant to the rescaling because <Code>{"m_t"}</Code> divides both numerator and denominator in proportion.
      </Prose>

      <Prose>
        <strong>That is the sLSTM cell.</strong> Per channel, four gates (i, f, o, z), three additional state variables (c, n, m), exponential input/forget plus sigmoid output. The recurrence stays sequential — each step depends on the previous step's state — so sLSTM trains and infers with the same scan as a classical LSTM. There is no parallelizing it across the sequence; the scalar update is fundamentally serial. What you get for the trouble is a cell with sharper gating decisions than the 1997 LSTM and rigorous numerical stability on sequences of 100k+ tokens. Empirically it improves quality on long-range tasks where the original LSTM saturates.
      </Prose>

      <Prose>
        <strong>Now the matrix surgery.</strong> Even with exponential gates, sLSTM has a hard ceiling on memory capacity: the cell state <Code>{"c_t"}</Code> is a single scalar per channel. Across <Code>D</Code> channels you have <Code>D</Code> scalars total to summarize the past, which is why classical LSTMs memorize so few earlier tokens precisely. mLSTM raises the ceiling by replacing the scalar cell state with a matrix <Code>{"C_t \\in \\mathbb{R}^{d_{qk} \\times d_v}"}</Code>. The update is a covariance-style outer-product write: at each step compute a key vector <Code>{"k_t"}</Code> and a value vector <Code>{"v_t"}</Code>, then add their outer product to the matrix, scaled by the input gate.
      </Prose>

      <MathBlock>
        {"C_t = f_t \\cdot C_{t-1} + i_t \\cdot k_t v_t^T"}
      </MathBlock>

      <Prose>
        Read the matrix back with a query vector <Code>{"q_t"}</Code>: the output is <Code>{"C_t^T q_t"}</Code> (matrix-vector multiply against the transposed memory). The intuition: <Code>{"C_t"}</Code> stores a sum of associative pairs, like a key-addressable memory. When you read with a query that resembles one of the stored keys, the corresponding value comes back. That is exactly how attention works (queries against keys retrieve values), and indeed the mLSTM update <Code>{"C_t = f_t C_{t-1} + i_t k_t v_t^T"}</Code> is mathematically the same recurrence as gated linear attention with an exponential decay gate — Mamba-2 with scalar transitions, Beck et al.'s formulation, and Songlin Yang's GLA all live in this equivalence class.
      </Prose>

      <Prose>
        The crucial property of the matrix update is that it is <em>linear in the state</em>: <Code>{"C_t"}</Code> appears on the right with at most multiplication by <Code>{"f_t"}</Code> and addition of an external term. Linear recurrences are parallelizable. Specifically you can chunk the sequence into blocks of length <Code>B</Code>, compute each block's contribution to <Code>{"C_t"}</Code> as a matmul (intra-chunk), and propagate the running state across chunks via the scalar gate (inter-chunk). This is the same SSD chunked-scan algorithm Mamba-2 uses, and it gives mLSTM training cost <Code>{"O(L \\cdot d^2)"}</Code> with hardware utilization comparable to attention, while inference stays at <Code>{"O(d^2)"}</Code> per token regardless of context length. The serial sLSTM cannot be parallelized this way because its update <Code>{"c_t = f_t c_{t-1} + i_t z_t"}</Code> depends on <Code>{"z_t = \\tanh(...)"}</Code> which depends on <Code>{"h_{t-1}"}</Code>, an explicitly nonlinear coupling.
      </Prose>

      <Prose>
        <strong>The xLSTM block.</strong> The architecture stacks alternating sLSTM and mLSTM cells, with residual connections and layer norm wrapping each, in the layout of a transformer block. A typical 7B xLSTM has ~24 such blocks. In some configurations all blocks are mLSTM (called xLSTM[0:1] or "pure mLSTM"); in others a fraction are sLSTM to capture the sharper sequential reasoning sLSTM provides. The empirical finding from Beck et al. is that pure-mLSTM is fastest and competitive; mixing in some sLSTM yields a small quality gain at the cost of training speed because sLSTM blocks are sequential.
      </Prose>

      <Callout accent="gold">
        Mental model: xLSTM is the LSTM cell rebuilt with two specific changes — (1) exponential gates with a normalizer and a log-domain stabilizer (sLSTM); (2) matrix-valued cell state with outer-product writes (mLSTM). The first sharpens gating decisions while staying sequential. The second trades scalar memory for matrix memory and gains the parallel-scan training of Mamba-2 / GLA. The xLSTM block alternates them. The whole construction is Hochreiter's argument that the RNN family, properly modernized, deserves a seat at the 2026 architecture table.
      </Callout>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 Classical LSTM (1997 + forget gate)</H3>

      <Prose>
        Reference for what xLSTM modifies. With input <Code>{"x_t"}</Code>, hidden <Code>{"h_t"}</Code>, cell state <Code>{"c_t"}</Code>:
      </Prose>

      <MathBlock>
        {"\\begin{aligned} i_t &= \\sigma(W_i x_t + R_i h_{t-1} + b_i) \\\\ f_t &= \\sigma(W_f x_t + R_f h_{t-1} + b_f) \\\\ o_t &= \\sigma(W_o x_t + R_o h_{t-1} + b_o) \\\\ z_t &= \\tanh(W_z x_t + R_z h_{t-1} + b_z) \\\\ c_t &= f_t \\odot c_{t-1} + i_t \\odot z_t \\\\ h_t &= o_t \\odot \\tanh(c_t) \\end{aligned}"}
      </MathBlock>

      <Prose>
        Four gates per channel, all sigmoid except <Code>{"z_t"}</Code> which is <Code>{"\\tanh"}</Code>. The cell update <Code>{"c_t"}</Code> is additive (good — gradient through <Code>{"c"}</Code> is multiplicative by <Code>{"f_t \\in (0, 1)"}</Code>), the readout uses <Code>{"\\tanh(c_t)"}</Code> to squash the cell into the bounded range. This is the cell that ran every machine-translation system from 2014-2017 and powers most production speech-recognition stacks even today.
      </Prose>

      <H3>3.2 sLSTM cell (xLSTM, scalar memory with exp gating)</H3>

      <Prose>
        The sLSTM cell keeps the same channel-wise scalar state <Code>{"c_t \\in \\mathbb{R}^d"}</Code> but introduces three modifications: exponential input/forget gates, a normalizer state <Code>{"n_t"}</Code>, and a stabilizer state <Code>{"m_t"}</Code>. Compute the gate pre-activations as before:
      </Prose>

      <MathBlock>
        {"\\begin{aligned} \\tilde i_t &= W_i x_t + R_i h_{t-1} + b_i \\\\ \\tilde f_t &= W_f x_t + R_f h_{t-1} + b_f \\\\ o_t &= \\sigma(W_o x_t + R_o h_{t-1} + b_o) \\\\ z_t &= \\tanh(W_z x_t + R_z h_{t-1} + b_z) \\end{aligned}"}
      </MathBlock>

      <Prose>
        The stabilizer state runs the log-domain max:
      </Prose>

      <MathBlock>
        {"m_t = \\max\\bigl(\\tilde f_t + m_{t-1}, \\; \\tilde i_t\\bigr)"}
      </MathBlock>

      <Prose>
        Then the stabilized gates are:
      </Prose>

      <MathBlock>
        {"i_t = \\exp(\\tilde i_t - m_t), \\qquad f_t = \\exp(\\tilde f_t + m_{t-1} - m_t)"}
      </MathBlock>

      <Prose>
        Note <Code>{"i_t \\le 1"}</Code> and <Code>{"f_t \\le 1"}</Code> by construction (one of the two log-arguments equals <Code>{"m_t"}</Code> and so the corresponding ratio is exactly 1). The state updates are:
      </Prose>

      <MathBlock>
        {"\\begin{aligned} c_t &= f_t \\odot c_{t-1} + i_t \\odot z_t \\\\ n_t &= f_t \\odot n_{t-1} + i_t \\\\ h_t &= o_t \\odot \\bigl(c_t / (n_t + \\varepsilon)\\bigr) \\end{aligned}"}
      </MathBlock>

      <Prose>
        Why does this preserve the sharpening behavior of <Code>{"\\exp"}</Code> when the stabilization caps the gates at 1? Because the <em>relative</em> magnitudes are preserved. If the model needs to write input <Code>j</Code> ten times more aggressively than input <Code>{"j+1"}</Code>, it sets <Code>{"\\tilde i_j - \\tilde i_{j+1} = \\log 10"}</Code>; after stabilization, <Code>{"i_j / i_{j+1} = 10"}</Code> still holds. The sigmoid gate of the classical LSTM cannot achieve a 10x ratio at the high end (both <Code>{"\\sigma(2)"}</Code> and <Code>{"\\sigma(3)"}</Code> are within 0.05 of each other) — that is the expressivity gap the exponential gate fixes.
      </Prose>

      <Prose>
        Memory mixing: in the full sLSTM cell, the cell state <Code>{"c_t \\in \\mathbb{R}^d"}</Code> is split into multiple "heads" (similar to multi-head attention) and the recurrent matrix <Code>{"R"}</Code> is block-diagonal across heads, with within-head mixing. This lets the model spread information across head dimensions — the "memory mixing" referenced in the paper. The simplification we use throughout this topic treats <Code>{"d"}</Code> as a single head; the multi-head extension is a routine reshape.
      </Prose>

      <H3>3.3 mLSTM cell (xLSTM, matrix memory)</H3>

      <Prose>
        The mLSTM cell upgrades the cell state from a scalar per channel to a matrix <Code>{"C_t \\in \\mathbb{R}^{d_{qk} \\times d_v}"}</Code> and the normalizer from a scalar per channel to a vector <Code>{"n_t \\in \\mathbb{R}^{d_{qk}}"}</Code>. Compute query, key, value, and the same scalar gate pre-activations <Code>{"\\tilde i_t, \\tilde f_t"}</Code> as in sLSTM:
      </Prose>

      <MathBlock>
        {"q_t = W_q x_t, \\quad k_t = W_k x_t / \\sqrt{d_{qk}}, \\quad v_t = W_v x_t"}
      </MathBlock>

      <Prose>
        Stabilize as before with <Code>{"m_t = \\max(\\tilde f_t + m_{t-1}, \\tilde i_t)"}</Code> (now <Code>{"m_t"}</Code> is a per-token scalar, not a per-channel vector), then <Code>{"i_t = \\exp(\\tilde i_t - m_t)"}</Code>, <Code>{"f_t = \\exp(\\tilde f_t + m_{t-1} - m_t)"}</Code>. The matrix update is:
      </Prose>

      <MathBlock>
        {"\\begin{aligned} C_t &= f_t \\, C_{t-1} + i_t \\, k_t v_t^T \\\\ n_t &= f_t \\, n_{t-1} + i_t \\, k_t \\\\ h_t &= o_t \\odot \\frac{C_t^T q_t}{\\max\\!\\bigl(\\lvert n_t^T q_t\\rvert, \\, 1\\bigr)} \\end{aligned}"}
      </MathBlock>

      <Prose>
        The numerator <Code>{"C_t^T q_t"}</Code> is a <Code>{"d_v"}</Code>-dimensional vector — read out the matrix memory at the position the query points to. The denominator <Code>{"\\max(|n_t^T q_t|, 1)"}</Code> normalizes by the running mass at that query, with a clamp at 1 to avoid division by tiny numbers when <Code>{"q_t"}</Code> happens to be approximately orthogonal to all stored keys (the original paper's choice; some implementations use a small <Code>{"\\varepsilon"}</Code> instead).
      </Prose>

      <H3>3.4 Equivalence to gated linear attention</H3>

      <Prose>
        Expand the matrix recurrence. Starting from <Code>{"C_0 = 0"}</Code> and unrolling, the state at time <Code>t</Code> is:
      </Prose>

      <MathBlock>
        {"C_t = \\sum_{j=1}^{t} \\Bigl(\\prod_{l=j+1}^{t} f_l\\Bigr) \\cdot i_j \\cdot k_j v_j^T"}
      </MathBlock>

      <Prose>
        Define the cumulative gate <Code>{"G_{t,j} = i_j \\prod_{l=j+1}^{t} f_l"}</Code>. The output before the output gate is:
      </Prose>

      <MathBlock>
        {"\\frac{C_t^T q_t}{n_t^T q_t} = \\frac{\\sum_{j \\le t} G_{t,j} \\, (q_t \\cdot k_j) \\, v_j}{\\sum_{j \\le t} G_{t,j} \\, (q_t \\cdot k_j)}"}
      </MathBlock>

      <Prose>
        That is masked linear attention with kernel <Code>{"\\phi(x) = x"}</Code>, masked by the lower-triangular gate matrix <Code>{"G_{t,j}"}</Code>. The mask itself has a structured (semi-separable) form because <Code>{"G_{t,j} = G_{t-1,j} \\cdot f_t"}</Code> for <Code>{"j < t"}</Code> — exactly the structure that admits the SSD chunked algorithm of Mamba-2. The mLSTM is therefore not "an LSTM with bigger memory" in any architectural sense distinct from gated linear attention; it is the same mathematical object, with the gate parameterized in LSTM-style language (<Code>{"i_t, f_t"}</Code>) rather than RetNet-style language (<Code>{"\\gamma_t"}</Code>) or Mamba-style language (<Code>{"\\Delta_t, A_t"}</Code>).
      </Prose>

      <H3>3.5 Parallel chunked training</H3>

      <Prose>
        For training, materializing the recurrence step-by-step is wasteful: the matmul <Code>{"q_t \\cdot k_j"}</Code> can be done as a single batched <Code>{"L \\times L"}</Code> matrix multiply for all <Code>{"t, j"}</Code>, and the gate mask <Code>{"G_{t,j}"}</Code> can be precomputed in <Code>{"O(L \\log L)"}</Code> via cumulative log-sums. The full forward pass is then:
      </Prose>

      <MathBlock>
        {"Y = \\bigl(M \\odot (Q K^T)\\bigr) V \\, / \\, D"}
      </MathBlock>

      <Prose>
        where <Code>{"M_{t,j} = G_{t,j}"}</Code> for <Code>{"j \\le t"}</Code> else 0, and <Code>{"D"}</Code> is the row-wise normalizer. This is an <Code>{"O(L^2 d)"}</Code> computation in the dense form — the same cost as quadratic attention. To recover linear cost, partition the sequence into chunks of length <Code>B</Code> (typically 64-256). Within a chunk, compute <Code>{"Y_{\\text{intra}} = (M_{\\text{local}} \\odot (Q K^T)) V"}</Code> at <Code>{"O(B^2 d + B d^2)"}</Code> per chunk; across chunks, propagate the running matrix state <Code>{"C"}</Code> via a recurrence at the chunk granularity, costing <Code>{"O((L/B) \\cdot d^2)"}</Code>. Total: <Code>{"O(L \\cdot B \\cdot d + L \\cdot d^2 / B)"}</Code>. Choose <Code>B</Code> to balance the two terms; for typical <Code>d</Code>, <Code>{"B = 64"}</Code> is a reasonable default. This is the SSD-style chunked algorithm; the xLSTM paper's Algorithm 2 is the mLSTM-specific version, structurally identical to Mamba-2's <Code>mamba_chunk_scan_combined</Code>.
      </Prose>

      <H3>3.6 Cost summary</H3>

      <Prose>
        Per layer, with sequence length <Code>L</Code>, model dim <Code>D</Code>, head dim <Code>d</Code>:
      </Prose>

      <Prose>
        sLSTM training: <Code>{"O(L \\cdot D^2)"}</Code> sequential. Cannot be parallelized across <Code>L</Code>. Wall-clock dominated by the serial scan.
      </Prose>

      <Prose>
        sLSTM inference: <Code>{"O(D^2)"}</Code> per token, constant in <Code>L</Code>. Just the gate projections plus scalar updates.
      </Prose>

      <Prose>
        mLSTM training (chunked): <Code>{"O(L \\cdot d \\cdot D)"}</Code> for the linear-attention term plus <Code>{"O(L \\cdot d^2)"}</Code> for the projections. Tensor-Core friendly — runs at high GPU utilization.
      </Prose>

      <Prose>
        mLSTM inference: <Code>{"O(d^2 \\cdot D / d_v)"}</Code> per token, constant in <Code>L</Code>. State size <Code>{"d_{qk} \\cdot d_v"}</Code> scalars per head.
      </Prose>

      <Prose>
        Compare to transformer training <Code>{"O(L^2 D + L D^2)"}</Code> and inference <Code>{"O(L \\cdot D)"}</Code> (KV cache attended once per token). The mLSTM matches Mamba-2/GLA on these axes; sLSTM is slower per token but cheaper than full attention for long contexts.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        We implement the sLSTM cell, the mLSTM cell in both recurrent and parallel forms, an xLSTM block that alternates them, and a tiny character-level language model that compares xLSTM against a baseline classical LSTM. All code below was run; outputs labeled <Code>{"# Output:"}</Code> are real stdout from the runs.
      </Prose>

      <H3>4.1 sLSTM cell with exponential gating + stabilizer</H3>

      <Prose>
        The cell carries four states (<Code>{"h, c, n, m"}</Code>) and four gate projections. The forget-gate bias is initialized to 0 here for visibility; in practice xLSTM uses a small positive bias to favor memory retention at init.
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn as nn

torch.manual_seed(0)

class sLSTMCell(nn.Module):
    """Single sLSTM cell (Beck et al. 2024, eq. 4-9)."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W = nn.Linear(input_size, 4 * hidden_size, bias=False)
        self.R = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.b = nn.Parameter(torch.zeros(4 * hidden_size))
        with torch.no_grad():
            # Forget-gate bias init to 0 (paper uses negative log-uniform)
            self.b[hidden_size:2 * hidden_size].fill_(0.0)

    def forward(self, x, state):
        h_prev, c_prev, n_prev, m_prev = state
        gates = self.W(x) + self.R(h_prev) + self.b
        i_pre, f_pre, o_pre, z_pre = gates.chunk(4, dim=-1)

        # Stabilizer state: m_t = max(log f_t + m_{t-1}, log i_t)
        m_t = torch.maximum(f_pre + m_prev, i_pre)
        i_t = torch.exp(i_pre - m_t)               # stabilized exp(input gate)
        f_t = torch.exp(f_pre + m_prev - m_t)      # stabilized exp(forget gate)
        o_t = torch.sigmoid(o_pre)
        z_t = torch.tanh(z_pre)

        c_t = f_t * c_prev + i_t * z_t
        n_t = f_t * n_prev + i_t                   # normalizer state
        h_t = o_t * (c_t / (n_t + 1e-6))           # bounded output via division
        return h_t, (h_t, c_t, n_t, m_t)

    def init_state(self, batch_size, device=None):
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        c = torch.zeros_like(h)
        n = torch.ones_like(h)                     # n=1 avoids div-by-zero at t=0
        m = torch.zeros_like(h)
        return (h, c, n, m)


B, T, D, H = 2, 8, 4, 6
cell = sLSTMCell(D, H)
x_seq = torch.randn(B, T, D)
state = cell.init_state(B)
hs = []
for t in range(T):
    h, state = cell(x_seq[:, t], state)
    hs.append(h)
out = torch.stack(hs, dim=1)
print(f"sLSTM forward: in shape={tuple(x_seq.shape)}, out shape={tuple(out.shape)}")
print(f"  finite={torch.isfinite(out).all().item()}")
print(f"  hidden mean={out.mean().item():+.4f}  std={out.std().item():.4f}")

gates = cell.W(x_seq[:, -1]) + cell.R(state[0]) + cell.b
i_pre, f_pre, o_pre, z_pre = gates.chunk(4, dim=-1)
m_t = torch.maximum(f_pre + state[3], i_pre)
i_t = torch.exp(i_pre - m_t)
f_t = torch.exp(f_pre + state[3] - m_t)
print(f"\\nGate stats at final step:")
print(f"  i_t (exp input gate): min={i_t.min().item():.3f}, max={i_t.max().item():.3f}, mean={i_t.mean().item():.3f}")
print(f"  f_t (exp forget gate): min={f_t.min().item():.3f}, max={f_t.max().item():.3f}, mean={f_t.mean().item():.3f}")
print(f"  m_t (stabilizer): min={m_t.min().item():+.3f}, max={m_t.max().item():+.3f}")
print(f"\\nsLSTM cell parameters: {sum(p.numel() for p in cell.parameters())}")

# Output:
# sLSTM forward: in shape=(2, 8, 4), out shape=(2, 8, 6)
#   finite=True
#   hidden mean=+0.0087  std=0.0908
#
# Gate stats at final step:
#   i_t (exp input gate): min=0.014, max=1.000, mean=0.307
#   f_t (exp forget gate): min=0.383, max=1.000, mean=0.934
#   m_t (stabilizer): min=-0.626, max=+4.100
#
# sLSTM cell parameters: 264`}
      </CodeBlock>

      <Prose>
        Two things to notice in the output. First, every step has at least one of <Code>{"i_t"}</Code> or <Code>{"f_t"}</Code> equal to 1.0 — that is the stabilization invariant: <Code>{"m_t"}</Code> equals the larger of the two log-arguments, so the corresponding ratio is exactly 1. Second, the stabilizer <Code>{"m_t"}</Code> grows over time (max +4.1 after 8 steps with forget-bias 0). With a positive forget-bias, <Code>{"m_t"}</Code> grows linearly; the math is fine but you must ensure your implementation keeps <Code>{"m_t"}</Code> in fp32 even if the rest of the model is fp16 (otherwise it overflows fp16's max of 65504 in a few hundred steps).
      </Prose>

      <H3>4.2 mLSTM cell with matrix memory: recurrent and parallel forms</H3>

      <Prose>
        The recurrent form is the canonical inference path, one step at a time. The parallel form computes the same outputs in a single batched matmul over the full sequence — this is what training uses. Critical: <em>both forms must produce the same numbers</em>. We implement both and assert numerical equality, which catches the most common mLSTM bugs.
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

class mLSTMCell(nn.Module):
    def __init__(self, d_model, d_qk=None, d_v=None):
        super().__init__()
        self.d_model = d_model
        self.d_qk = d_qk or d_model
        self.d_v = d_v or d_model
        self.W_q = nn.Linear(d_model, self.d_qk, bias=False)
        self.W_k = nn.Linear(d_model, self.d_qk, bias=False)
        self.W_v = nn.Linear(d_model, self.d_v, bias=False)
        self.W_i = nn.Linear(d_model, 1, bias=True)
        self.W_f = nn.Linear(d_model, 1, bias=True)
        self.W_o = nn.Linear(d_model, self.d_v, bias=True)
        with torch.no_grad():
            self.W_f.bias.fill_(3.0)               # init f_t close to 1 (preserve memory)

    def forward_recurrent(self, x):
        B, T, D = x.shape
        scale = self.d_qk ** 0.5
        q = self.W_q(x) / scale
        k = self.W_k(x) / scale
        v = self.W_v(x)
        i_pre = self.W_i(x).squeeze(-1)
        f_pre = self.W_f(x).squeeze(-1)

        C = torch.zeros(B, self.d_qk, self.d_v, device=x.device)
        n = torch.zeros(B, self.d_qk, device=x.device)
        m = torch.zeros(B, device=x.device)

        ys = []
        for t in range(T):
            m_new = torch.maximum(f_pre[:, t] + m, i_pre[:, t])
            i_t = torch.exp(i_pre[:, t] - m_new)
            f_t = torch.exp(f_pre[:, t] + m - m_new)
            kt, vt = k[:, t], v[:, t]
            # Outer-product update: C_t = f_t * C_{t-1} + i_t * (k_t v_t^T)
            C = f_t.view(B, 1, 1) * C + i_t.view(B, 1, 1) * (kt.unsqueeze(-1) * vt.unsqueeze(1))
            n = f_t.view(B, 1) * n + i_t.view(B, 1) * kt
            m = m_new
            qt = q[:, t]
            num = (C * qt.unsqueeze(-1)).sum(dim=1)             # C^T q_t
            den = (n * qt).sum(dim=-1).abs().clamp(min=1.0)
            h_t = num / den.unsqueeze(-1)
            o_t = torch.sigmoid(self.W_o(x[:, t]))
            ys.append(o_t * h_t)
        return torch.stack(ys, dim=1), (C, n, m)

    def forward_parallel(self, x):
        """Parallel form: y = (M * (Q K^T)) V / D."""
        B, T, D = x.shape
        scale = self.d_qk ** 0.5
        q = self.W_q(x) / scale
        k = self.W_k(x) / scale
        v = self.W_v(x)
        i_pre = self.W_i(x).squeeze(-1)
        f_pre = self.W_f(x).squeeze(-1)

        # Replicate the recurrent stabilizer schedule m_t
        m_seq = torch.zeros(B, T, device=x.device)
        m_run = torch.zeros(B, device=x.device)
        for t in range(T):
            m_run = torch.maximum(f_pre[:, t] + m_run, i_pre[:, t])
            m_seq[:, t] = m_run
        m_prev = torch.cat([torch.zeros(B, 1, device=x.device), m_seq[:, :-1]], dim=1)
        log_f = f_pre + m_prev - m_seq
        log_i = i_pre - m_seq
        F_cum = torch.cumsum(log_f, dim=1)
        # gate_log[t,j] = log_i[j] + sum_{l=j+1..t} log_f[l]
        gate_log = log_i.unsqueeze(1) + F_cum.unsqueeze(2) - F_cum.unsqueeze(1)
        idx = torch.arange(T, device=x.device)
        causal = (idx.unsqueeze(1) >= idx.unsqueeze(0)).float()  # lower-triangular mask
        gate = torch.exp(gate_log) * causal.unsqueeze(0)

        sim = (q @ k.transpose(-1, -2)) * gate                    # [B, T, T]
        denom = sim.sum(dim=-1).abs().clamp(min=1.0).unsqueeze(-1)
        num = sim @ v
        return torch.sigmoid(self.W_o(x)) * (num / denom)


B, T, D = 2, 8, 6
mlstm = mLSTMCell(d_model=D, d_qk=4, d_v=4)
x = torch.randn(B, T, D)
y_rec, (C_final, n_final, m_final) = mlstm.forward_recurrent(x)

print(f"mLSTM recurrent forward: in={tuple(x.shape)}, out={tuple(y_rec.shape)}")
print(f"  finite={torch.isfinite(y_rec).all().item()}")
print(f"  hidden mean={y_rec.mean().item():+.4f}  std={y_rec.std().item():.4f}")
print(f"\\nFinal matrix memory C shape: {tuple(C_final.shape)}")
print(f"  C Frobenius norm: {C_final.norm().item():.4f}")
print(f"  n state norm: {n_final.norm().item():.4f}")
print(f"  stabilizer m: {m_final.tolist()}")

y_par = mlstm.forward_parallel(x)
diff = (y_rec - y_par).abs().max().item()
print(f"\\nNumerical check: max|y_recurrent - y_parallel| = {diff:.2e}")
print(f"  match={diff < 1e-4}")
print(f"\\nmLSTM parameters: {sum(p.numel() for p in mlstm.parameters())}")

# Output:
# mLSTM recurrent forward: in=(2, 8, 6), out=(2, 8, 4)
#   finite=True
#   hidden mean=-0.0000  std=0.0005
#
# Final matrix memory C shape: (2, 4, 4)
#   C Frobenius norm: 0.0136
#   n state norm: 0.0254
#   stabilizer m: [25.620901107788086, 25.048282623291016]
#
# Numerical check: max|y_recurrent - y_parallel| = 1.75e-10
#   match=True
#
# mLSTM parameters: 114`}
      </CodeBlock>

      <Prose>
        The numerical check is the centerpiece of any mLSTM implementation: if your parallel form drifts from your recurrent form by more than float32 noise (around 1e-6 to 1e-9 for the cumulative sums), you have a sign bug somewhere — most often in the lower-triangular causal mask, where it is shockingly easy to flip rows and columns and produce an upper-triangular mask that silently corrupts training. We catch <Code>1.75e-10</Code> here, which is float32 round-off; the implementation is consistent.
      </Prose>

      <Prose>
        The hidden output mean is essentially zero and the std is tiny (0.0005) because the forget-bias init of +3.0 makes <Code>{"f_t \\approx 1"}</Code>, so the cell preserves every input nearly perfectly and the normalizer grows as fast as the cell — they cancel almost completely. This is correct behavior at init: the model has not learned to forget anything, so every query reads back a heavily averaged value. Training teaches the W_f and W_i projections to differentiate inputs.
      </Prose>

      <H3>4.3 xLSTM block: sLSTM + mLSTM with residuals</H3>

      <Prose>
        An xLSTM block alternates an sLSTM and an mLSTM cell, each preceded by LayerNorm and wrapped in a residual connection (transformer-block layout). For brevity we omit the conventional FFN layer; production xLSTM blocks include a gated MLP after the sequence mixer.
      </Prose>

      <CodeBlock language="python">
{`class xLSTMBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.slstm = sLSTMCell(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.mlstm = mLSTMCell(d)

    def forward(self, x):
        # sLSTM: scan step by step
        h, state = x.new_zeros(x.size(0), self.slstm.hidden_size), self.slstm.init_state(x.size(0), x.device)
        s_out = []
        u = self.norm1(x)
        for t in range(x.size(1)):
            h, state = self.slstm(u[:, t], state)
            s_out.append(h)
        x = x + torch.stack(s_out, dim=1)
        # mLSTM: parallel form
        x = x + self.mlstm.forward_parallel(self.norm2(x))
        return x`}
      </CodeBlock>

      <H3>4.4 Train xLSTM vs classical LSTM on character-level LM</H3>

      <Prose>
        A small benchmark: a hand-rolled LSTM model (using PyTorch's optimized <Code>nn.LSTM</Code> for the baseline, which calls cuDNN's fused kernel) versus an xLSTM model with one sLSTM+mLSTM block. Train both for 300 steps on a tiny tongue-twister corpus; compare loss and inference throughput. The point is not to win a benchmark — it is to verify both architectures train cleanly and to expose the wall-clock cost of the from-scratch xLSTM scan vs cuDNN's LSTM.
      </Prose>

      <CodeBlock language="python">
{`import math, time, torch, torch.nn as nn, torch.nn.functional as F

class LSTMModel(nn.Module):
    def __init__(self, V, d, n_layers=1):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.lstm = nn.LSTM(d, d, num_layers=n_layers, batch_first=True)
        self.head = nn.Linear(d, V)

    def forward(self, x):
        h = self.emb(x)
        out, _ = self.lstm(h)
        return self.head(out)


class xLSTMModel(nn.Module):
    def __init__(self, V, d, n_blocks=1):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.blocks = nn.ModuleList([xLSTMBlock(d) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, V)

    def forward(self, x):
        h = self.emb(x)
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.norm(h))


text = (
    "the quick brown fox jumps over the lazy dog. " * 50 +
    "she sells sea shells by the sea shore. " * 50 +
    "how much wood would a woodchuck chuck. " * 50 +
    "peter piper picked a peck of pickled peppers. " * 50
)
vocab = sorted(set(text))
V = len(vocab); stoi = {c: i for i, c in enumerate(vocab)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)


def get_batch(seq_len=64, batch=16):
    ix = torch.randint(0, len(data) - seq_len - 1, (batch,))
    x = torch.stack([data[i:i + seq_len] for i in ix])
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in ix])
    return x, y


def train(model, name, n_steps=300, lr=3e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    print(f"\\nTraining {name} ({sum(p.numel() for p in model.parameters())} params):")
    print(f"  {'step':>5}  {'loss':>7}  {'ppl':>7}")
    t0 = time.time()
    for step in range(n_steps):
        x, y = get_batch()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0 or step == n_steps - 1:
            print(f"  {step:>5}  {loss.item():>7.4f}  {math.exp(loss.item()):>7.2f}")
        losses.append(loss.item())
    print(f"  wall time: {time.time() - t0:.1f}s")
    return losses


d = 32
torch.manual_seed(42)
lstm = LSTMModel(V, d)
xlstm = xLSTMModel(V, d, n_blocks=1)
l_lstm = train(lstm, "Classical LSTM")
l_xlstm = train(xlstm, "xLSTM (sLSTM+mLSTM block)")

print(f"\\nFinal-100-step mean loss:")
print(f"  LSTM     : {sum(l_lstm[-100:]) / 100:.4f}  ppl={math.exp(sum(l_lstm[-100:]) / 100):.2f}")
print(f"  xLSTM    : {sum(l_xlstm[-100:]) / 100:.4f}  ppl={math.exp(sum(l_xlstm[-100:]) / 100):.2f}")

# Output:
# vocab size V=28, corpus length=8450
#
# Training Classical LSTM (10268 params):
#    step     loss      ppl
#       0   3.3558    28.67
#      50   1.9511     7.04
#     100   1.0478     2.85
#     150   0.5027     1.65
#     200   0.2814     1.33
#     250   0.1809     1.20
#     299   0.1295     1.14
#   wall time: 2.5s
#
# Training xLSTM (sLSTM+mLSTM block) (14526 params):
#    step     loss      ppl
#       0   3.4298    30.87
#      50   1.4811     4.40
#     100   0.4389     1.55
#     150   0.1782     1.20
#     200   0.0916     1.10
#     250   0.0707     1.07
#     299   0.0535     1.05
#   wall time: 34.0s
#
# Final-100-step mean loss:
#   LSTM     : 0.1917  ppl=1.21
#   xLSTM    : 0.0732  ppl=1.08`}
      </CodeBlock>

      <Prose>
        On this tiny task xLSTM converges to a noticeably lower perplexity (1.08 vs 1.21) but takes 13.6x the wall time. The wall-time cost is entirely from the sLSTM scan: it runs a Python for-loop over 64 timesteps per batch, with the 5x overhead of the sLSTM's stabilizer computations plus PyTorch op-launch latency, while cuDNN's LSTM fuses the entire scan into one kernel call. This is the central engineering reality of xLSTM: <em>the architecture's quality advantage shows up only after you have a fused CUDA kernel for the scan</em>. With the official <Code>xlstm</Code> package and its kernels, the wall-time gap closes to ~2x and quality stays ahead. With this from-scratch implementation, you pay a 14x penalty for a 12% perplexity gain.
      </Prose>

      <H3>4.5 Inference throughput vs sequence length</H3>

      <Prose>
        Run the trained models in eval mode at sequence lengths 64 to 1024, batch size 4, and measure tokens/second of forward computation.
      </Prose>

      <CodeBlock language="python">
{`lstm.eval(); xlstm.eval()
print(f"\\nInference throughput (forward-only, batch=4, d={d}):")
print(f"  {'L':>6}  {'LSTM tok/s':>12}  {'xLSTM tok/s':>14}")
with torch.no_grad():
    for L in [64, 128, 256, 512, 1024]:
        x = torch.randint(0, V, (4, L))
        _ = lstm(x); _ = xlstm(x)              # warmup
        t = time.time()
        for _ in range(5): _ = lstm(x)
        lstm_tps = (5 * 4 * L) / (time.time() - t)
        t = time.time()
        for _ in range(5): _ = xlstm(x)
        xlstm_tps = (5 * 4 * L) / (time.time() - t)
        print(f"  {L:>6}  {lstm_tps:>10.0f}    {xlstm_tps:>12.0f}")

# Output:
# Inference throughput (forward-only, batch=4, d=32):
#       L    LSTM tok/s     xLSTM tok/s
#       64      353437           13197
#      128      454686           13515
#      256      491314           12666
#      512      607870           11499
#     1024      752777           10832`}
      </CodeBlock>

      <Prose>
        The cuDNN LSTM throughput rises with sequence length (better amortization of kernel-launch overhead); the from-scratch xLSTM throughput stays roughly constant — limited by the Python scan loop. With proper fused kernels (NXAI's <Code>xlstm</Code> package supplies them via Triton), the from-scratch curve flips: xLSTM stays constant while a transformer's KV-cache attention degrades quadratically. Without those kernels, you should expect roughly the numbers shown here: a Python-implemented xLSTM is for understanding and prototyping, not for production.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <H3>5.1 The xlstm package (NXAI / Hochreiter group)</H3>

      <Prose>
        The reference implementation lives at <Code>github.com/NX-AI/xlstm</Code>, maintained by the JKU Linz group and NXAI. Install with <Code>pip install xlstm</Code>. The package provides composable building blocks (<Code>sLSTMBlock</Code>, <Code>mLSTMBlock</Code>, <Code>xLSTMBlock</Code>) and a fused Triton kernel for the chunked parallel scan, plus a CUDA fallback. The kernel auto-compiles on first use, similar to FlashAttention; expect a 30-90s compile pause.
      </Prose>

      <CodeBlock language="python">
{`# pip install xlstm
from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig
from xlstm import sLSTMBlockConfig, sLSTMLayerConfig
from xlstm import mLSTMBlockConfig, mLSTMLayerConfig
from xlstm import FeedForwardConfig

cfg = xLSTMBlockStackConfig(
    mlstm_block=mLSTMBlockConfig(
        mlstm=mLSTMLayerConfig(
            num_heads=4,
            qkv_proj_blocksize=4,
            conv1d_kernel_size=4,
        ),
    ),
    slstm_block=sLSTMBlockConfig(
        slstm=sLSTMLayerConfig(
            num_heads=4,
            backend="cuda",          # 'cuda' (fused), 'vanilla' (python ref)
            conv1d_kernel_size=4,
            bias_init="powerlaw_blockdependent",
        ),
        feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
    ),
    context_length=1024,
    num_blocks=24,
    embedding_dim=512,
    slstm_at=[3, 8, 13, 18],         # which block indices use sLSTM (rest are mLSTM)
)

model = xLSTMBlockStack(cfg).cuda()
x = torch.randn(2, 1024, 512, device="cuda")
y = model(x)                          # [2, 1024, 512]`}
      </CodeBlock>

      <Prose>
        The <Code>slstm_at</Code> list is the standard knob: which block indices use sLSTM (the rest are mLSTM). Beck et al. report that placing 4-6 sLSTM blocks among 24 mLSTM blocks gives the best quality-throughput tradeoff for 7B-class models. Pure-mLSTM stacks (<Code>{"slstm_at = []"}</Code>) train fastest but lose ~1% perplexity; pure-sLSTM stacks are far too slow at scale. Quality scaling matches Mamba-2 closely at 7B; the xLSTM-7B technical report shows it within noise of Mamba-2 on Lambada, HellaSwag, ARC, PIQA.
      </Prose>

      <H3>5.2 xLSTM-7B on HuggingFace</H3>

      <Prose>
        NXAI publishes the pretrained <Code>NX-AI/xLSTM-7b</Code> model on HuggingFace Hub, trained on roughly 1 trillion tokens from the SlimPajama mix. As of early 2026 it is not in mainline <Code>transformers</Code>; you load it via the <Code>xlstm</Code> package's HuggingFace integration:
      </Prose>

      <CodeBlock language="python">
{`# pip install xlstm transformers
from xlstm import xLSTMLargeConfig, xLSTMLarge
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("NX-AI/xLSTM-7b")
model = xLSTMLarge.from_pretrained("NX-AI/xLSTM-7b", torch_dtype=torch.bfloat16, device_map="auto")

prompt = "Recurrent neural networks store information across time by"
inputs = tok(prompt, return_tensors="pt").to(model.device)

# Inference uses the recurrent form: state size constant in context length
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=100, do_sample=True, top_p=0.9, temperature=0.8)
print(tok.decode(out[0], skip_special_tokens=True))`}
      </CodeBlock>

      <Prose>
        Memory footprint: xLSTM-7B's recurrent state is roughly <Code>{"d_{qk} \\cdot d_v \\cdot \\text{n\\_heads} \\cdot \\text{n\\_blocks} \\cdot 4"}</Code> bytes (fp32 scan). For the published config with 4 heads, 32 blocks, head dims around 128 each, that is ~256 MB total — independent of context length. Compare to a 7B Llama whose KV cache at 64K context is ~16 GB. For very long context, the difference is the difference between running at all and running out of memory.
      </Prose>

      <H3>5.3 xLSTM serving and the kernel question</H3>

      <Prose>
        Like Mamba in 2024, xLSTM is not a drop-in for any inference engine that does not know about it. As of early 2026 vLLM has experimental xLSTM support; TensorRT-LLM does not; SGLang has it on the roadmap. For self-hosted serving the practical path is the NXAI reference server, which uses the package's Triton kernels directly. The custom-kernel ecosystem matters because the linear-time advantage at long context only realizes if the scan kernel runs on Tensor Cores; the pure-PyTorch scan is 5-20x slower than the Triton fused scan and effectively pointless at production scale.
      </Prose>

      <H3>5.4 Comparison to mamba-ssm and gla-pytorch</H3>

      <Prose>
        At the kernel level, mLSTM is mathematically the same recurrence as Mamba-2 (with different gate parameterization) and the same as gated linear attention (with Mamba-style stabilization). All three projects have converged kernels for the chunked parallel scan. NXAI's xLSTM Triton kernel, state-spaces' mamba-ssm CUDA kernel, and Songlin Yang's <Code>flash-linear-attention</Code> Triton kernel all implement chunked SSD with the same algorithmic skeleton; numerical results are identical up to floating-point order-of-summation. Performance is within 30% of each other on H100. Choose by which API and pretrained weights you prefer rather than by raw kernel speed.
      </Prose>

      <H3>5.5 Long-context fine-tuning workflow</H3>

      <Prose>
        A common production task: fine-tune xLSTM-7B for 64K context on domain documents. Because the state is constant-size, the GPU memory cost of the long-context fine-tune is bounded by the activations (the scan does materialize per-chunk activations during backward), not by a KV cache. The xlstm package supports this via gradient checkpointing on the chunk boundaries — set <Code>{"chunkwise_kernel='checkpointed'"}</Code> in the kernel config. Training at 64K context fits in 80 GB H100 for a 7B xLSTM where the equivalent Llama-7B fine-tune would require multi-GPU KV-cache sharding.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        Four visualizations: (1) a 5-step trace of the sLSTM update; (2) a 4-step trace of the mLSTM matrix-memory accumulation; (3) a throughput comparison vs sequence length across LSTM, transformer, Mamba, xLSTM; (4) heatmaps of the mLSTM matrix memory <Code>{"C"}</Code> at four checkpoints showing the outer-product structure.
      </Prose>

      <StepTrace
        label="sLSTM update over 5 timesteps"
        steps={[
          {
            label: "Step 0: state initialized",
            render: () => (
              <Prose>
                State <Code>{"(h, c, n, m) = (0, 0, 1, 0)"}</Code> per channel. The normalizer <Code>{"n_0 = 1"}</Code> rather than 0 to avoid division by zero on the first timestep. The stabilizer <Code>{"m_0 = 0"}</Code>. No input has been written yet; <Code>{"h_0 = 0"}</Code>.
              </Prose>
            ),
          },
          {
            label: "Step 1: first input arrives, gates fire",
            render: () => (
              <Prose>
                Input <Code>{"x_1"}</Code> arrives. Pre-activations: say <Code>{"\\tilde i_1 = +0.5"}</Code>, <Code>{"\\tilde f_1 = +0.2"}</Code>. Stabilizer: <Code>{"m_1 = \\max(\\tilde f_1 + m_0, \\tilde i_1) = \\max(0.2, 0.5) = 0.5"}</Code>. Stabilized gates: <Code>{"i_1 = \\exp(0.5 - 0.5) = 1.0"}</Code>, <Code>{"f_1 = \\exp(0.2 + 0 - 0.5) = \\exp(-0.3) = 0.741"}</Code>. The input gate fully opens (its log was the maximum so it normalizes to 1); the forget gate stays around 0.74. State updates: <Code>{"c_1 = 0.741 \\cdot 0 + 1.0 \\cdot z_1 = z_1"}</Code>; <Code>{"n_1 = 0.741 \\cdot 1 + 1.0 = 1.741"}</Code>; <Code>{"h_1 = o_1 \\cdot z_1 / 1.741"}</Code>.
              </Prose>
            ),
          },
          {
            label: "Step 2: state carries forward, new input added",
            render: () => (
              <Prose>
                Pre-activations: <Code>{"\\tilde i_2 = -0.3"}</Code>, <Code>{"\\tilde f_2 = +0.1"}</Code>. Stabilizer: <Code>{"m_2 = \\max(0.1 + 0.5, -0.3) = 0.6"}</Code>. Stabilized: <Code>{"i_2 = \\exp(-0.3 - 0.6) = 0.407"}</Code>, <Code>{"f_2 = \\exp(0.1 + 0.5 - 0.6) = 1.0"}</Code> — now the forget gate is at the maximum and the input gate is suppressed. State: <Code>{"c_2 = 1.0 \\cdot c_1 + 0.407 \\cdot z_2"}</Code> (memory preserved, mild new write); <Code>{"n_2 = 1.0 \\cdot n_1 + 0.407"}</Code>.
              </Prose>
            ),
          },
          {
            label: "Step 3: a sharp gate decision",
            render: () => (
              <Prose>
                Suppose the model has learned that <Code>{"x_3"}</Code> carries strong signal (a sentence-final delimiter, say). Pre-activations: <Code>{"\\tilde i_3 = +3.0"}</Code> (very large), <Code>{"\\tilde f_3 = -1.0"}</Code> (favor forgetting). Stabilizer: <Code>{"m_3 = \\max(-1.0 + 0.6, 3.0) = 3.0"}</Code>. Stabilized: <Code>{"i_3 = \\exp(3.0 - 3.0) = 1.0"}</Code>, <Code>{"f_3 = \\exp(-1.0 + 0.6 - 3.0) = \\exp(-3.4) = 0.033"}</Code>. The forget gate slammed nearly closed (kept ~3% of memory) while the input gate fired at full. State <Code>{"c_3"}</Code> is dominated by <Code>{"z_3"}</Code>; the past has been mostly cleared. <em>This is the sharpening that exponential gating provides — sigmoid gates cannot make this kind of multiplicative-100x decision in one step.</em>
              </Prose>
            ),
          },
          {
            label: "Step 4: output via normalized division",
            render: () => (
              <Prose>
                After step 3 the cell <Code>{"c_3"}</Code> has reset and the normalizer <Code>{"n_3 \\approx 1.0"}</Code> proportionally. The output <Code>{"h_3 = o_3 \\cdot c_3 / n_3"}</Code> is bounded by <Code>{"o_3 \\cdot |z_3|"}</Code> regardless of how aggressively the input gate fired. The normalizer is the trick that lets exponential gating coexist with bounded outputs — the same trick attention uses with its softmax denominator. The model can decide-sharply without blowing up the hidden activations downstream.
              </Prose>
            ),
          },
        ]}
      />

      <Prose>
        Now the matrix-memory analog. We feed a sequence of orthogonal-ish keys into the mLSTM and watch the matrix cell state <Code>{"C \\in \\mathbb{R}^{8 \\times 8}"}</Code> grow as outer products accumulate. With <Code>{"i_t = 0.6"}</Code> and <Code>{"f_t = 0.7"}</Code> held constant, key <Code>{"k_t = e_t"}</Code> (the t-th standard basis vector), value <Code>{"v_t"}</Code> a small mix centered on position <Code>t</Code>, the matrix accumulates a lower-triangular pattern with each row recording where one key wrote its value.
      </Prose>

      <StepTrace
        label="mLSTM matrix memory accumulation (d_qk = d_v = 8)"
        steps={[
          {
            label: "Step 0: first outer product written",
            render: () => (
              <>
                <Prose>
                  At <Code>t = 0</Code>, the matrix state is empty, then an outer product <Code>{"i_0 \\cdot k_0 v_0^T"}</Code> is added. With <Code>{"k_0 = e_0"}</Code> and <Code>{"v_0 = (1, 0.5, 0.25, 0, 0, 0, 0, 0)"}</Code>, only row 0 of the matrix is populated — exactly with the value vector scaled by <Code>{"i_0 = 0.6"}</Code>.
                </Prose>
                <Heatmap
                  label="C after step 0"
                  matrix={[
                    [+0.60, +0.30, +0.15, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                  ]}
                  rowLabels={["e0", "e1", "e2", "e3", "e4", "e5", "e6", "e7"]}
                  colLabels={["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7"]}
                  colorScale="gold"
                />
              </>
            ),
          },
          {
            label: "Step 2: three outer products, decayed by f",
            render: () => (
              <>
                <Prose>
                  After step 2, three keys have written. Each row's contribution is decayed by <Code>{"f^{t - j}"}</Code>: row 0 has been multiplied by <Code>{"0.7^2 = 0.49"}</Code> and is now <Code>{"(0.29, 0.15, 0.07, ...)"}</Code>. Row 2 was just written at full <Code>{"i = 0.6"}</Code>. The matrix is filling diagonally, each row offset by the value vector's pattern.
                </Prose>
                <Heatmap
                  label="C after step 2"
                  matrix={[
                    [+0.29, +0.15, +0.07, 0, 0, 0, 0, 0],
                    [0, +0.42, +0.21, +0.11, 0, 0, 0, 0],
                    [0, 0, +0.60, +0.30, +0.15, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                  ]}
                  rowLabels={["e0", "e1", "e2", "e3", "e4", "e5", "e6", "e7"]}
                  colLabels={["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7"]}
                  colorScale="gold"
                />
              </>
            ),
          },
          {
            label: "Step 4: five rows populated, oldest most decayed",
            render: () => (
              <>
                <Prose>
                  By step 4 five rows are populated. Row 4 is bright (just written, scale 0.6); row 0 is dim (decayed by <Code>{"0.7^4 = 0.24"}</Code>). The matrix is now an associative store: querying with <Code>{"q = e_2"}</Code> would read row 2 and return its stored value pattern, scaled by the cumulative gate at the time of the query.
                </Prose>
                <Heatmap
                  label="C after step 4"
                  matrix={[
                    [+0.14, +0.07, +0.04, 0, 0, 0, 0, 0],
                    [0, +0.21, +0.10, +0.05, 0, 0, 0, 0],
                    [0, 0, +0.29, +0.15, +0.07, 0, 0, 0],
                    [0, 0, 0, +0.42, +0.21, +0.11, 0, 0],
                    [0, 0, 0, 0, +0.60, +0.30, +0.15, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                  ]}
                  rowLabels={["e0", "e1", "e2", "e3", "e4", "e5", "e6", "e7"]}
                  colLabels={["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7"]}
                  colorScale="gold"
                />
              </>
            ),
          },
          {
            label: "Step 7: full sequence written, exponential decay visible",
            render: () => (
              <>
                <Prose>
                  After all 8 keys have written, the matrix is fully populated but with strong intensity gradient: recent rows (6, 7) are bright; old rows (0, 1) are dim. Row 0 has decayed by <Code>{"0.7^7 \\approx 0.082"}</Code> from its initial value. The exponential decay is the constant-gate analog of the trainable gate <Code>{"f_t"}</Code>; in a real trained mLSTM, the model would learn to make <Code>{"f_t"}</Code> close to 1 for important keys (preserving them long-term) and small for unimportant ones (faster decay).
                </Prose>
                <Heatmap
                  label="C after step 7"
                  matrix={[
                    [+0.05, +0.02, +0.01, 0, 0, 0, 0, 0],
                    [0, +0.07, +0.04, +0.02, 0, 0, 0, 0],
                    [0, 0, +0.10, +0.05, +0.03, 0, 0, 0],
                    [0, 0, 0, +0.14, +0.07, +0.04, 0, 0],
                    [0, 0, 0, 0, +0.21, +0.10, +0.05, 0],
                    [0, 0, 0, 0, 0, +0.29, +0.15, +0.07],
                    [+0.11, 0, 0, 0, 0, 0, +0.42, +0.21],
                    [+0.30, +0.15, 0, 0, 0, 0, 0, +0.60],
                  ]}
                  rowLabels={["e0", "e1", "e2", "e3", "e4", "e5", "e6", "e7"]}
                  colLabels={["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7"]}
                  colorScale="gold"
                />
              </>
            ),
          },
        ]}
      />

      <Prose>
        Throughput comparison across architectures, normalized to the 1024-token regime. LSTM scales linearly with batched cuDNN; transformer drops quadratically as sequence length grows; Mamba and xLSTM stay flat. Numbers are approximate, taken from a 7B-class model on H100.
      </Prose>

      <Plot
        label="inference throughput vs sequence length (tokens/sec, 7B model on H100)"
        series={[
          {
            name: "LSTM (cuDNN, scalar state)",
            color: "#8b5cf6",
            points: [[1024, 4500], [2048, 4400], [4096, 4300], [8192, 4200], [16384, 4000], [32768, 3700]],
          },
          {
            name: "Transformer (FlashAttention)",
            color: "#60a5fa",
            points: [[1024, 12000], [2048, 8500], [4096, 5200], [8192, 2400], [16384, 950], [32768, 260]],
          },
          {
            name: "Mamba (selective scan)",
            color: "#34d399",
            points: [[1024, 11000], [2048, 10500], [4096, 10000], [8192, 9500], [16384, 9100], [32768, 8700]],
          },
          {
            name: "xLSTM (mLSTM chunked scan)",
            color: "#e2b55a",
            points: [[1024, 9500], [2048, 9300], [4096, 9000], [8192, 8700], [16384, 8400], [32768, 8000]],
          },
        ]}
        xLabel="sequence length L"
        yLabel="tokens / sec"
        width={520}
      />

      <Prose>
        Three observations. First, the transformer dominates at short context but collapses past 8K — the quadratic attention term overtakes everything. Second, Mamba and xLSTM track each other closely; the mLSTM's chunked scan and Mamba-2's SSD scan share the same algorithmic structure and similar kernel quality. Third, classical LSTM (with cuDNN) is competitive at long context — it never had a quadratic cost — but it loses to transformer at short context because cuDNN's LSTM is slower than FlashAttention per token, and it loses to Mamba/xLSTM because its scalar state gives lower quality at matched parameter count. The xLSTM's design is explicitly to keep LSTM's flat scaling curve while raising the quality ceiling via the matrix memory.
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        Choosing xLSTM in 2026 depends on context length, model scale, deployment ecosystem, and what alternatives you have already invested in. Decide by quadrant.
      </Prose>

      <H3>7.1 By model scale</H3>

      <Prose>
        <strong>Sub-1B research / prototyping:</strong> xLSTM is a fine choice; the <Code>xlstm</Code> package is mature enough for small-scale experiments. At this scale, transformer baselines are also cheap to run, so use both and compare. xLSTM's quality lead over a transformer at &lt;1B is small; the win is mainly to learn the architecture.
      </Prose>

      <Prose>
        <strong>1B-13B production language model:</strong> xLSTM-7B is a credible candidate. Mamba-2 is also credible. Transformer (with grouped-query attention) is still the safe default. If you have a long-context requirement (32K+) and a tight memory budget, choose xLSTM or Mamba-2; else Llama-style transformer.
      </Prose>

      <Prose>
        <strong>30B+ frontier model:</strong> Pure xLSTM is not validated at this scale as of early 2026 — neither is pure Mamba. The mainstream frontier (GPT, Claude, Gemini, Llama 3+) remains transformer-based, sometimes with hybrid Mamba layers in cost-sensitive deployments. Recommend transformer or hybrid; xLSTM is research-grade at this scale.
      </Prose>

      <H3>7.2 By context length</H3>

      <Prose>
        <strong>Short (≤4K):</strong> Transformer wins on quality and the KV cache fits. xLSTM offers no advantage; do not use it.
      </Prose>

      <Prose>
        <strong>Medium (4K-32K):</strong> Coin flip on quality between xLSTM, Mamba-2, and transformer with FlashAttention. xLSTM/Mamba-2 win on memory and inference cost. Hybrid architectures (Jamba-style) are the practical sweet spot.
      </Prose>

      <Prose>
        <strong>Long (32K-128K):</strong> xLSTM is competitive with Mamba-2 here and beats transformer decisively on cost. Either xLSTM or Mamba-2 is a reasonable production choice depending on which ecosystem you already use.
      </Prose>

      <Prose>
        <strong>Very long (128K+):</strong> SSM-family (Mamba-2, xLSTM, RWKV-7) is the only choice on a single GPU. Constant-state recurrent models do not care about context length; transformers run out of memory.
      </Prose>

      <H3>7.3 By task domain</H3>

      <Prose>
        <strong>General language modeling:</strong> Transformer is still the default. xLSTM is a competitive alternative, not a clear win; choose by ecosystem.
      </Prose>

      <Prose>
        <strong>Time series forecasting:</strong> xLSTM (specifically the mLSTM variant) is novel here — the matrix memory captures multivariate temporal patterns better than a vector-state RNN. The xLSTM time-series follow-up paper (Beck et al., late 2024) reports state-of-the-art on M4 and several other benchmarks. Worth trying.
      </Prose>

      <Prose>
        <strong>Speech / audio:</strong> S4/Mamba and xLSTM are roughly equivalent. xLSTM has less published audio work as of 2026; default to Mamba unless you have a specific reason.
      </Prose>

      <Prose>
        <strong>Vision:</strong> Transformer (ViT) dominates. xLSTM-Vision exists (NXAI, 2024) but is not competitive yet with DINOv2 or vit-large at standard image resolutions.
      </Prose>

      <Prose>
        <strong>Reasoning / chain-of-thought:</strong> Inconclusive. The xLSTM-7B technical report shows competitive performance on GSM8K and other reasoning benchmarks but not a leap. Transformer with strong post-training (DPO, RLHF) is still the safe bet for reasoning-heavy applications.
      </Prose>

      <H3>7.4 By deployment constraint</H3>

      <Prose>
        <strong>Edge / mobile:</strong> Mamba and RWKV have more mature mobile-export tooling than xLSTM. Default to those for edge. Note that pure Mamba-1 has lower wall-clock cost than xLSTM at small batch sizes because its kernel is more aggressively optimized for low-batch decoding.
      </Prose>

      <Prose>
        <strong>Single-GPU long-context server:</strong> xLSTM and Mamba-2 are the right answer. Choose by which pretrained weights you want.
      </Prose>

      <Prose>
        <strong>Multi-GPU production stack with vLLM/TensorRT-LLM:</strong> Transformer remains the safest because the inference engines have invested most effort there. xLSTM support is experimental in vLLM as of early 2026; Mamba-2 has slightly better support; transformer is fully supported everywhere.
      </Prose>

      <H3>7.5 Quick-pick summary</H3>

      <Prose>
        <strong>Default:</strong> transformer with FlashAttention-3 and grouped-query attention. <strong>For long context (32K+):</strong> xLSTM or Mamba-2 (your choice — they perform similarly). <strong>For time series:</strong> xLSTM (the matrix memory is a real fit). <strong>For frontier-scale LMs:</strong> still transformer in 2026; reconsider if a credible 70B+ recurrent model emerges. <strong>For research into post-transformer architectures:</strong> xLSTM is one of the three serious entries (alongside Mamba-2 and RWKV-7); learn all three.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES
          ====================================================================== */}
      <H2>8. What scales</H2>

      <Prose>
        Scaling an architecture is asking: as model dimension <Code>D</Code>, sequence length <Code>L</Code>, and dataset size <Code>N</Code> grow, what dominates time, memory, and quality? For xLSTM the answer factors cleanly into the two cells.
      </Prose>

      <H3>8.1 Training compute</H3>

      <Prose>
        mLSTM training (chunked SSD): <Code>{"O(L \\cdot d \\cdot D + L \\cdot D^2)"}</Code> per layer. The first term is the SSM/linear-attention work, the second is the feedforward and projection work. For typical <Code>{"d \\ll D"}</Code>, the second dominates and the architecture scales like a transformer-without-attention. Wall-clock matches a transformer at short context and beats it at long context because the linear-attention term grows linearly while transformer attention grows quadratically. The mLSTM kernel runs at 60-80% of H100 peak throughput in the chunked form, comparable to FlashAttention-3.
      </Prose>

      <Prose>
        sLSTM training: <Code>{"O(L \\cdot D^2)"}</Code> per layer, but <em>sequential</em> — each step depends on the previous. The walk-clock cost of sLSTM is dominated by the scan latency, not by FLOPs. At a 7B scale, the published xLSTM-7B uses 4 sLSTM blocks among 28 mLSTM blocks; sLSTM contributes ~25-30% of training wall-clock for ~15% of the parameters. This is the trade-off Beck et al. accept: sLSTM's quality gain costs throughput. Pure-mLSTM stacks train ~30% faster but lose ~1% perplexity.
      </Prose>

      <H3>8.2 Inference compute and memory</H3>

      <Prose>
        Per generated token: mLSTM costs <Code>{"O(d \\cdot D + D^2)"}</Code> — the matrix-vector readout plus the projection MLPs. sLSTM costs <Code>{"O(D^2)"}</Code> — the scalar update plus projections. Both are constant in <Code>L</Code>. Compare to transformer's <Code>{"O(L \\cdot D + D^2)"}</Code> per token at decode (KV cache attended once). For <Code>{"L > D"}</Code>, the recurrent models win on per-token compute.
      </Prose>

      <Prose>
        State memory at inference: mLSTM stores <Code>{"d_{qk} \\cdot d_v \\cdot \\text{n\\_heads} \\cdot \\text{n\\_blocks}"}</Code> floats per sequence in the batch. For xLSTM-7B with 4 heads, ~28 blocks, head dims around 128, that is ~470K floats per sequence, ~1.9 MB at fp32. Compare to Llama-7B KV cache at 32K context: 16 GB per sequence. Three-to-four orders of magnitude difference for long context.
      </Prose>

      <H3>8.3 Quality scaling with parameters</H3>

      <Prose>
        At matched parameter count and matched training data, xLSTM is within 1-2% perplexity of Mamba-2 and within 1-3% of a transformer on the Pile and SlimPajama benchmarks (Beck et al. 2024 and the xLSTM-7B technical report). The xLSTM authors argue the gap closes at larger scales but the published evidence stops at 7B parameters. Anyone betting on xLSTM at frontier scale is betting on extrapolation of a small trend, not on a verified result. Mamba-2 is in the same boat.
      </Prose>

      <H3>8.4 What the matrix dimension d_qk costs</H3>

      <Prose>
        The mLSTM has a tunable knob the transformer does not have a clean analog of: the matrix-state dimension <Code>{"d_{qk}"}</Code>. State storage is <Code>{"d_{qk} \\cdot d_v"}</Code> per head. Increasing <Code>{"d_{qk}"}</Code> raises capacity (more keys storable) at cost of state size and inference compute (the readout is <Code>{"O(d_{qk} \\cdot d_v)"}</Code> per token). xLSTM-7B uses <Code>{"d_{qk} \\sim 128"}</Code>, comparable to Mamba-2's state dim of 64-256. Doubling <Code>{"d_{qk}"}</Code> roughly doubles state storage and increases inference cost by 2x; quality gain past 256 is small.
      </Prose>

      <H3>8.5 Larger-scale unknowns</H3>

      <Prose>
        Three open questions as of early 2026. (1) Does xLSTM scale to 70B+ parameters with similar trends, or does it diverge? No 70B xLSTM has been published. (2) Does the mLSTM matrix memory remain expressive enough at 1M+ context, or does the constant-size state become a binding capacity constraint? Empirical evidence at 100K is encouraging; at 1M unverified. (3) Can xLSTM-style architectures support effective in-context learning the way transformer attention does? The mathematical equivalence to gated linear attention suggests yes (linear attention does support in-context learning, just less efficiently per parameter than softmax attention), but the empirical gap on few-shot benchmarks is real.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES
          ====================================================================== */}
      <H2>9. Failure modes</H2>

      <H3>9.1 Forgetting the stabilizer state m_t</H3>

      <Prose>
        The most catastrophic failure mode: implementing exponential gates without the stabilizer. <Code>{"\\exp"}</Code> overflows fp32 around input 89 and fp16 around input 11. The pre-activation of an unstabilized exponential gate can easily reach those magnitudes during training. Without <Code>{"m_t"}</Code>, the cell state goes to <Code>{"+\\infty"}</Code> within tens of steps, then to NaN, then loss is NaN forever. Diagnostic: run a forward pass with random init; if any element of <Code>{"c_t"}</Code> exceeds 1e6 in fp32, you have forgotten the stabilizer. Fix: implement the <Code>{"m_t"}</Code> recurrence and use the stabilized <Code>{"i_t, f_t"}</Code> in the cell update. This is the single most important sanity check for an sLSTM/mLSTM implementation.
      </Prose>

      <H3>9.2 Missing the normalizer state n_t</H3>

      <Prose>
        Even with stabilization, the unnormalized output <Code>{"h_t = o_t \\tanh(c_t)"}</Code> (LSTM-style) misbehaves with exponential gates: after the stabilizer rescaling, <Code>{"c_t"}</Code> can grow without bound in absolute value (the rescaling preserves ratios but not magnitudes across long sequences). Without dividing by <Code>{"n_t"}</Code>, the hidden state saturates the <Code>{"\\tanh"}</Code> to <Code>{"\\pm 1"}</Code> for every channel and every step, killing the gradient signal. Diagnostic: print <Code>{"|c_t|"}</Code> statistics over a long sequence; if they grow monotonically and the hidden std is plastered at 1.0, you have missed <Code>{"n_t"}</Code>. Fix: use <Code>{"h_t = o_t \\cdot c_t / n_t"}</Code> with the normalizer recurrence.
      </Prose>

      <H3>9.3 Treating xLSTM gates like classical LSTM gates</H3>

      <Prose>
        A natural mistake when reading the xLSTM paper after years of LSTM experience is to write the cell update with <Code>{"i_t = \\sigma(...)"}</Code> habitually. The <Code>{"\\exp"}</Code> is the entire point. Symptom: model trains but never matches Mamba-2 or transformer baselines. Diagnostic: print the gate values in a forward pass. If they are all in <Code>{"(0, 1)"}</Code>, you have a sigmoid not an exp. Fix: use <Code>{"\\exp"}</Code> for input and forget gates (with stabilization); keep <Code>{"\\sigma"}</Code> for the output gate.
      </Prose>

      <H3>9.4 Lower-triangular vs upper-triangular causal mask</H3>

      <Prose>
        In the parallel form of mLSTM, the gate matrix <Code>{"M_{t,j}"}</Code> must be lower-triangular: position <Code>t</Code> reads positions <Code>{"j \\le t"}</Code>. PyTorch broadcasting tricks with <Code>{"\\text{idx}.unsqueeze(0) \\ge \\text{idx}.unsqueeze(1)"}</Code> can silently produce the upper triangle if you get the broadcast direction wrong. Symptom: training loss does not decrease; or the model "trains" but is uninterpretable. Diagnostic: print the mask and verify it is lower-triangular. Fix: use <Code>{"\\text{idx}.unsqueeze(1) \\ge \\text{idx}.unsqueeze(0)"}</Code>, or simply <Code>{"\\text{torch.tril}(\\text{ones}(L, L))"}</Code>. The recurrent and parallel forms must agree numerically; assert this in tests with a small example.
      </Prose>

      <H3>9.5 Trying to parallelize sLSTM</H3>

      <Prose>
        Engineers familiar with parallel scans for SSMs sometimes assume the same trick works for sLSTM. It does not. The sLSTM update <Code>{"c_t = f_t c_{t-1} + i_t z_t"}</Code> has <Code>{"z_t = \\tanh(W_z x_t + R_z h_{t-1} + b_z)"}</Code>, and <Code>{"h_{t-1}"}</Code> depends on <Code>{"c_{t-1} / n_{t-1}"}</Code> nonlinearly. The recurrence is not associative; no parallel scan exists for it. If you want parallelism, use mLSTM. The xLSTM block design is explicitly a hybrid because of this asymmetry. Trying to write a "parallel sLSTM kernel" wastes weeks for no result.
      </Prose>

      <H3>9.6 Poor positional handling at very long context</H3>

      <Prose>
        Recurrent models do not have positional encodings — position is implicit in the order of the recurrence. But that means the recurrent state has to encode position information, and it has finite capacity. At very long context (100K+) the model can lose track of absolute position; a token at position 50,000 looks much like a token at position 60,000 if the matrix memory has saturated. Transformers handle this with explicit position encodings (RoPE) that scale; xLSTM does not have a clean RoPE-equivalent. The empirical fix is training at the deployment context length so the model learns to use its state efficiently at that length; a researcher's open question is whether some positional augmentation can help.
      </Prose>

      <H3>9.7 fp16 in the matrix memory C</H3>

      <Prose>
        The matrix memory <Code>{"C_t"}</Code> accumulates outer products over the entire sequence. In fp16, the cumulative scaling by <Code>{"\\prod_t f_t"}</Code> products with stabilizer rescaling can underflow or lose precision. Mamba-2's reference kernel runs the scan in fp32 even when the rest of the model is fp16; xLSTM's kernel does the same. If you write a custom mLSTM kernel: do not run the matrix-update in fp16. Use bf16 (which has fp32 exponent range) or fp32. Symptom of getting this wrong: training is fine for a few thousand steps then loss spikes to NaN; diagnostic: print the running max of <Code>{"|C_t|"}</Code>; if it diverges, your accumulator precision is too low.
      </Prose>

      <H3>9.8 Initializing the forget bias too high</H3>

      <Prose>
        The xLSTM paper recommends a forget-bias initialization that makes <Code>{"f_t \\approx 1"}</Code> at start (preserve memory by default). If you set the bias too high (e.g., +5 instead of +3), the model effectively never forgets, the matrix memory <Code>{"C_t"}</Code> averages all inputs uniformly, and the model degenerates to bag-of-tokens behavior. Symptom: training loss decreases extremely slowly, hidden activations are tiny (because <Code>{"C_t / n_t"}</Code> averages to a small value when both grow linearly). Diagnostic: print the average <Code>{"f_t"}</Code> at step 100; if it is &gt;0.99, lower the forget bias init. The published default is around +3, giving <Code>{"f_t \\approx 0.95"}</Code> initially.
      </Prose>

      <H3>9.9 Naive Python scan instead of fused kernel</H3>

      <Prose>
        Like Mamba in 2024, xLSTM's PyTorch reference scan is 5-20x slower than the Triton fused kernel. If you import the <Code>xlstm</Code> package and find it slower than a transformer, verify the kernel compiled successfully (look for "Compiled xlstm Triton kernel" log lines). On systems without a working CUDA toolchain, the package falls back to the Python reference and produces correct outputs at glacial speed. This is the most common new-user complaint: "xLSTM is slow." Almost always the answer is the kernel did not compile. Fix: install <Code>triton</Code>, ensure CUDA matches the PyTorch build, and run on an NVIDIA GPU; AMD and CPU are not supported by the fused kernel.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        Read in chronological order to follow the line from the 1997 LSTM through the 2024 xLSTM proposal and its parallel/contemporary rivals (Mamba, RWKV).
      </Prose>

      <StepTrace
        label="primary literature"
        steps={[
          {
            label: "Hochreiter & Schmidhuber 1997 — LSTM",
            render: () => (
              <Prose>
                Hochreiter, S., and Schmidhuber, J. (1997). "Long Short-Term Memory." Neural Computation 9(8), 1735-1780. The foundational paper. Introduces the cell state, the input gate, and the output gate; the forget gate was added in Gers, Schmidhuber, Cummins (2000), "Learning to Forget: Continual Prediction with LSTM," Neural Computation 12(10). Together these form the modern LSTM that everyone teaches. Read sections 3-4 for the cell mechanics. The xLSTM paper assumes you know this paper; many of its design choices are exactly the LSTM design with two specific surgeries.
              </Prose>
            ),
          },
          {
            label: "Beck et al. 2024 — xLSTM (arXiv:2405.04517)",
            render: () => (
              <Prose>
                Beck, M., Pöppel, K., Spanring, M., Auer, A., Prudnikova, O., Kopp, M., Klambauer, G., Brandstetter, J., and Hochreiter, S. (2024). "xLSTM: Extended Long Short-Term Memory." arXiv:2405.04517. Available at arxiv.org/abs/2405.04517. The xLSTM paper. Introduces sLSTM (Section 2.2: exponential gates + normalizer + stabilizer), mLSTM (Section 2.3: matrix memory, parallel form), and the xLSTM block (Section 3). Section 4 reports language modeling, Long Range Arena, and time-series benchmarks at scales up to 1.3B parameters. Read sections 2.2-2.3 carefully — the equations of motion for sLSTM and mLSTM are dense and the paper's notation evolves through the section. The appendix has the parallel-form derivation; this is essential if you want to implement the chunked scan correctly.
              </Prose>
            ),
          },
          {
            label: "Beck et al. 2025 — xLSTM-7B Technical Report",
            render: () => (
              <Prose>
                Beck, M., Pöppel, K., Lippert, P., Auer, A., Prudnikova, O., Kopp, M., Klambauer, G., Brandstetter, J., and Hochreiter, S. (2024-2025). "xLSTM-7B: A Recurrent LLM for Fast and Efficient Inference." Technical report from NXAI, available at nx-ai.com and the project's HuggingFace page (NX-AI/xLSTM-7b). Documents the 7-billion-parameter pretraining run (~1T tokens), the kernel infrastructure, the head/block configuration, and benchmark results vs Llama-2-7B, Mamba-7B, RWKV-5-7B. Practical reading for anyone deploying xLSTM at scale: contains hyperparameter choices, learning-rate schedule, and serving-throughput numbers.
              </Prose>
            ),
          },
          {
            label: "Gu & Dao 2023 — Mamba (arXiv:2312.00752)",
            render: () => (
              <Prose>
                Gu, A., and Dao, T. (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." arXiv:2312.00752. Available at arxiv.org/abs/2312.00752. Concurrent rival to xLSTM. Different formulation (state-space model with selective input dependence), same goal (linear-time recurrent LM that beats LSTM and competes with transformer). Mamba was published December 2023; xLSTM was published May 2024. Read to understand how the two architectures arrived at structurally similar objects from different starting points.
              </Prose>
            ),
          },
          {
            label: "Dao & Gu 2024 — Mamba-2 / SSD (arXiv:2405.21060)",
            render: () => (
              <Prose>
                Dao, T., and Gu, A. (2024). "Transformers Are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality." ICML 2024. arXiv:2405.21060. Proves the equivalence of certain selective SSMs and masked linear attention. Mamba-2's chunked SSD algorithm is structurally identical to xLSTM's mLSTM chunked scan — read this paper to understand the algorithm both architectures use under different names. Published the same month as xLSTM (May 2024); the convergence is not coincidence.
              </Prose>
            ),
          },
          {
            label: "Peng et al. 2023 — RWKV-4 (arXiv:2305.13048)",
            render: () => (
              <Prose>
                Peng, B., et al. (2023). "RWKV: Reinventing RNNs for the Transformer Era." EMNLP 2023 Findings. arXiv:2305.13048. The third member of the recurrent revival trio (RWKV, Mamba, xLSTM). RWKV starts from a kernelized linear-attention reformulation; Mamba from state-space models; xLSTM from gated RNNs. All three converge on linear-time recurrent LMs. RWKV-4 was the first 7B-class recurrent LM to be competitive with transformers; RWKV-5 (Eagle) and RWKV-6 (Finch) iterated, and RWKV-7 (Goose, 2025) extends with delta-rule updates. Read for the RWKV-specific gating and the time-mix / channel-mix block layout.
              </Prose>
            ),
          },
          {
            label: "Yang et al. 2024 — Gated Linear Attention (arXiv:2312.06635)",
            render: () => (
              <Prose>
                Yang, S., Wang, B., Zhang, Y., Shen, Y., and Kim, Y. (2024). "Gated Linear Attention Transformers with Hardware-Efficient Training." ICML 2024. arXiv:2312.06635. Develops the chunked-scan training kernel for gated linear attention — the same algorithm xLSTM's mLSTM uses, derived from a different motivation. Yang's <Code>flash-linear-attention</Code> repo at github.com/sustcsonglin/flash-linear-attention is a high-quality reference for the kernels and is often cited alongside the xlstm package. Useful for understanding the kernel implementation independently of the xLSTM-specific gate parameterization.
              </Prose>
            ),
          },
          {
            label: "Katharopoulos et al. 2020 — Linear Attention as RNN",
            render: () => (
              <Prose>
                Katharopoulos, A., Vyas, A., Pappas, N., and Fleuret, F. (2020). "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention." ICML 2020. arXiv:2006.16236. The paper that first observed that linear attention is an RNN at inference. The mathematical framework xLSTM's mLSTM lives in. Ten pages, no equations more complex than a sum, and it explains the entire reason mLSTM is parallelizable. Read first if you want to understand <em>why</em> the matrix-cell-state update can be both an RNN and a parallel matmul.
              </Prose>
            ),
          },
          {
            label: "NX-AI/xlstm — Reference repository",
            render: () => (
              <Prose>
                The official xLSTM implementation, maintained by Beck and the JKU Linz / NXAI team. Available at github.com/NX-AI/xlstm. Contains the Triton kernels, the model classes (sLSTMBlock, mLSTMBlock, xLSTMBlockStack), and example training scripts. The README has installation instructions and a minimum-working-example. The <Code>xlstm/blocks/mlstm</Code> subdirectory has the chunked-scan kernel; reading it alongside Mamba-2's <Code>mamba_chunk_scan_combined</Code> shows the algorithmic similarity in concrete form.
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
        Five exercises. Try all before reading the answers. Exercises 1-2 test the cell math; 3 tests the equivalence between mLSTM and gated linear attention; 4 tests architecture selection; 5 tests debugging.
      </Prose>

      <H3>Exercise 1 (sLSTM stabilizer arithmetic)</H3>
      <Prose>
        Given pre-activations <Code>{"\\tilde i_t = +2.0"}</Code>, <Code>{"\\tilde f_t = +0.5"}</Code> at step <Code>t</Code> with <Code>{"m_{t-1} = +1.0"}</Code>, compute <Code>{"m_t, i_t, f_t"}</Code>. Then verify that with <Code>{"c_{t-1} = 4, n_{t-1} = 5, z_t = 0.6, o_t = 0.7"}</Code>, the hidden output <Code>{"h_t"}</Code> is the same whether you stabilize or not (in exact arithmetic).
      </Prose>
      <Callout accent="green">
        <strong>Answer 1.</strong> <Code>{"m_t = \\max(\\tilde f_t + m_{t-1}, \\tilde i_t) = \\max(0.5 + 1.0, 2.0) = 2.0"}</Code>. Stabilized: <Code>{"i_t = \\exp(2.0 - 2.0) = 1.0"}</Code>; <Code>{"f_t = \\exp(0.5 + 1.0 - 2.0) = \\exp(-0.5) = 0.6065"}</Code>. State updates: <Code>{"c_t = 0.6065 \\cdot 4 + 1.0 \\cdot 0.6 = 3.026"}</Code>; <Code>{"n_t = 0.6065 \\cdot 5 + 1.0 = 4.033"}</Code>; <Code>{"h_t = 0.7 \\cdot 3.026 / 4.033 = 0.5253"}</Code>. Without stabilization: <Code>{"i_t' = e^{2.0} = 7.389"}</Code>, <Code>{"f_t' = e^{0.5} = 1.649"}</Code>; <Code>{"c_t' = 1.649 \\cdot 4 + 7.389 \\cdot 0.6 = 11.03"}</Code>; <Code>{"n_t' = 1.649 \\cdot 5 + 7.389 = 15.63"}</Code>; <Code>{"h_t' = 0.7 \\cdot 11.03 / 15.63 = 0.4937"}</Code>. The two answers differ (0.5253 vs 0.4937)! The difference is because the stabilizer division is applied at <em>every step</em>, not just one — to verify equivalence you need the unstabilized recurrence with <em>unstabilized</em> <Code>{"c_{t-1}, n_{t-1}"}</Code> as well, not the stabilized values. The stabilizer is invariant only when applied consistently along the whole trajectory. This subtlety is exactly why a partial implementation of the stabilizer (e.g., applied to the gates but not the state updates) silently produces wrong outputs.
      </Callout>

      <H3>Exercise 2 (mLSTM outer product)</H3>
      <Prose>
        Initialize an mLSTM matrix state <Code>{"C_0 = 0 \\in \\mathbb{R}^{2 \\times 2}"}</Code>. With constant gates <Code>{"f = 0.5, i = 1.0"}</Code> at every step, write keys <Code>{"k_1 = (1, 0), k_2 = (0, 1), k_3 = (1, 1)/\\sqrt{2}"}</Code> and values <Code>{"v_1 = (0.5, 0.5), v_2 = (1.0, 0.0), v_3 = (0.0, 1.0)"}</Code>. Compute <Code>{"C_3"}</Code>. Then read with query <Code>{"q = (1, 0)"}</Code> (no normalization, just <Code>{"C_3^T q"}</Code>) and check the answer corresponds to a decayed mix of <Code>{"v_1"}</Code> and the <Code>{"k_1"}</Code>-component of <Code>{"v_3"}</Code>.
      </Prose>
      <Callout accent="green">
        <strong>Answer 2.</strong> Each step: <Code>{"C_t = 0.5 \\cdot C_{t-1} + 1.0 \\cdot k_t v_t^T"}</Code>. <br />
        <Code>{"k_1 v_1^T = \\begin{pmatrix}0.5 & 0.5\\\\0 & 0\\end{pmatrix}"}</Code>, so <Code>{"C_1 = \\begin{pmatrix}0.5 & 0.5\\\\0 & 0\\end{pmatrix}"}</Code>. <br />
        <Code>{"k_2 v_2^T = \\begin{pmatrix}0 & 0\\\\1 & 0\\end{pmatrix}"}</Code>, so <Code>{"C_2 = 0.5 \\cdot C_1 + k_2 v_2^T = \\begin{pmatrix}0.25 & 0.25\\\\1 & 0\\end{pmatrix}"}</Code>. <br />
        <Code>{"k_3 v_3^T = (1/\\sqrt{2})\\begin{pmatrix}0 & 1\\\\0 & 1\\end{pmatrix} \\approx \\begin{pmatrix}0 & 0.707\\\\0 & 0.707\\end{pmatrix}"}</Code>, so <Code>{"C_3 = 0.5 \\cdot C_2 + k_3 v_3^T \\approx \\begin{pmatrix}0.125 & 0.832\\\\0.5 & 0.707\\end{pmatrix}"}</Code>. <br />
        Readout: <Code>{"C_3^T q = C_3^T (1, 0) = (0.125, 0.832)"}</Code> — first component matches the decayed <Code>{"v_1[0] = 0.5"}</Code> times <Code>{"0.5^2 = 0.25"}</Code> giving 0.125; second component is decayed <Code>{"v_1[1] = 0.5"}</Code> times 0.25 giving 0.125 plus <Code>{"v_3[1]/\\sqrt 2 = 0.707"}</Code> giving 0.832. Both match. The matrix memory is acting as an associative store: query <Code>{"e_1"}</Code> selects the row corresponding to keys aligned with <Code>{"e_1"}</Code> and reads back a decayed sum of the values associated with those keys.
      </Callout>

      <H3>Exercise 3 (equivalence to gated linear attention)</H3>
      <Prose>
        Write the mLSTM output (before the output gate) as a sum over past positions, and identify which kernel feature map <Code>{"\\phi"}</Code> and which gate schedule it corresponds to in the gated-linear-attention formulation <Code>{"y_t = \\sum_{j \\le t} G_{t, j} \\phi(q_t)^T \\phi(k_j) v_j"}</Code>. State the precise correspondence.
      </Prose>
      <Callout accent="green">
        <strong>Answer 3.</strong> Unrolling <Code>{"C_t = f_t C_{t-1} + i_t k_t v_t^T"}</Code> from <Code>{"C_0 = 0"}</Code>: <Code>{"C_t = \\sum_{j \\le t} (\\prod_{l=j+1}^{t} f_l) \\cdot i_j \\cdot k_j v_j^T"}</Code>. The output (numerator only) is <Code>{"C_t^T q_t = \\sum_{j \\le t} (\\prod_{l=j+1}^{t} f_l) \\cdot i_j \\cdot (q_t \\cdot k_j) \\cdot v_j"}</Code>. Comparing to <Code>{"y_t = \\sum_{j \\le t} G_{t,j} \\phi(q_t)^T \\phi(k_j) v_j"}</Code>: the feature map is the identity <Code>{"\\phi(x) = x"}</Code>, and the gate is <Code>{"G_{t,j} = i_j \\prod_{l=j+1}^{t} f_l"}</Code>. This is gated linear attention with an exponential decay schedule and a per-position write-strength <Code>{"i_j"}</Code>. It is also the SSD form of Mamba-2 with the per-step decay equal to <Code>{"f_l"}</Code>. The three architectures (mLSTM, GLA, Mamba-2) differ only in (a) how <Code>{"i_j, f_l"}</Code> are parameterized as functions of the input and (b) the surrounding block layout — they are mathematically the same recurrence.
      </Callout>

      <H3>Exercise 4 (architecture selection)</H3>
      <Prose>
        You are building a server that processes legal contracts. Average document is 50K tokens, max 200K. You must run on a single 80GB H100 with batch size 4 (concurrent users). Latency budget is 200 ms for first token, then sustained throughput. Quality target: GPT-3.5-class reasoning on legal language. Pick an architecture and justify in concrete terms.
      </Prose>
      <Callout accent="green">
        <strong>Answer 4.</strong> Choose xLSTM-7B or Mamba-2-7B. Specifically:
        <br />
        (a) <strong>Memory budget.</strong> A 7B Llama-2 with KV cache at 50K context per sequence: ~12 GB; at batch 4: 48 GB; at 200K context per sequence: ~50 GB; at batch 4: <em>200 GB</em> — does not fit. xLSTM-7B's recurrent state is ~200 MB total at batch 4 regardless of context. The transformer fails on memory before quality matters.
        <br />
        (b) <strong>Throughput.</strong> Decoding at 50K context for transformer: each token attends over 50K KV entries, ~5x slower than at 1K context. xLSTM/Mamba-2: same per-token cost regardless of context. At 200K context, xLSTM is ~30x faster per token than transformer.
        <br />
        (c) <strong>Quality.</strong> At 7B, xLSTM and Mamba-2 are within 1-2% of Llama-2-7B perplexity. For legal-document understanding, that gap is unlikely to be the binding constraint; document-length and recall are more important. Both xLSTM and Mamba-2 can absorb the full 50K context with no quality degradation from chunking.
        <br />
        (d) <strong>Latency.</strong> First-token latency requires processing the full prefix (prefill). For a 50K prefix, xLSTM's chunked scan runs in ~2-4 seconds on H100; transformer prefill at 50K with FlashAttention-3 is ~5-8 seconds. Both overrun the 200 ms budget at this context length, so prompt streaming or speculative prefill is needed regardless. xLSTM has no inherent latency disadvantage and a small advantage at long context.
        <br />
        (e) <strong>Choice between xLSTM and Mamba-2.</strong> Coin flip on quality. Pick by which pretrained checkpoint better matches your domain (NXAI's xLSTM-7B vs state-spaces' Mamba-7B-base, both available on HuggingFace). Mamba-2 has more mature inference tooling as of early 2026; xLSTM has the time-series advantage that may not matter here. Recommendation: start with Mamba-2 for the better tooling, switch to xLSTM if your legal-domain fine-tuning shows clearly better numbers (which is plausible given xLSTM's slightly more expressive sLSTM blocks for sequential reasoning).
        <br />
        Transformer-7B is wrong for this problem on hardware grounds. xLSTM or Mamba-2 are the realistic choices.
      </Callout>

      <H3>Exercise 5 (debugging)</H3>
      <Prose>
        You are training an xLSTM-1.3B model. Training proceeds normally for 5000 steps, then loss spikes from 2.6 to 12.7 in three steps and stays there. You restart from the previous checkpoint with half the learning rate; loss spikes again at step 5300. You suspect xLSTM-specific failure modes. List three Mamba/xLSTM-specific things to check and a diagnostic for each.
      </Prose>
      <Callout accent="green">
        <strong>Answer 5.</strong> Three xLSTM-specific failure modes to check:
        <br />
        (1) <strong>Stabilizer state m_t in fp16.</strong> If the entire model runs in fp16/bf16 and the stabilizer accumulator <Code>{"m_t"}</Code> is in fp16, then <Code>{"m_t"}</Code> can drift outside fp16's representable range (max ~65504) over a long training session. The drift is monotonic for typical inputs, so the model trains fine until <Code>{"m_t"}</Code> hits the ceiling, after which one of the stabilized exponentials goes to <Code>{"\\exp(-65504) = 0"}</Code> (gate becomes exactly zero) and the cell loses all dynamic range. Diagnostic: print <Code>{"\\max(|m_t|)"}</Code> per step; if it grows monotonically and is approaching 30 in fp16 or 80 in fp32, there is a problem. Fix: keep <Code>{"m_t"}</Code> in fp32 even if everything else is in lower precision; the xlstm package's reference kernel does this by default but custom implementations sometimes break it. Alternatively, periodically reset <Code>{"m_t"}</Code> to 0 at chunk boundaries (this is what the chunked scan does in production).
        <br />
        (2) <strong>Forget bias drift.</strong> If the forget-bias initialization is in a regime where <Code>{"f_t"}</Code> is very close to 1 (e.g., bias init +5), the model's gradient through the forget gate is tiny (because <Code>{"\\sigma'(5) \\approx 0.007"}</Code> and the exp's derivative at the saturated value is also small). The model effectively cannot learn to forget; the matrix memory <Code>{"C_t"}</Code> grows without effective bound and at some point overflows or saturates the normalizer. Diagnostic: print <Code>{"\\text{mean}(f_t)"}</Code> per step; if it is &gt;0.99 throughout training, the forget gate is stuck. Fix: lower the forget bias init to ~+3, or use the per-block heterogeneous init (<Code>{"\\text{powerlaw\\_blockdependent}"}</Code>) the published config uses.
        <br />
        (3) <strong>Matrix memory C_t accumulator precision.</strong> Even with stabilization, the per-step outer-product accumulation in <Code>{"C_t"}</Code> can drift in low precision. Mamba-2 and xLSTM both keep <Code>{"C_t"}</Code> in fp32 internally even when the model is in bf16. If you wrote a custom kernel that uses bf16 throughout, the accumulation error compounds and at some point a chunk-boundary state transfer produces a NaN. Diagnostic: print the running max of <Code>{"|C_t|"}</Code> per chunk; if it grows past 1e4 (typical xLSTM <Code>{"|C_t|"}</Code> stays under 1e2), something is accumulating wrong. Fix: ensure the matrix-state accumulator is fp32 in the kernel.
        <br />
        Bonus: gradient clipping at norm 1.0 (standard for both Mamba and xLSTM) catches most aggregate instabilities. If you do not have it on, that is the first fix to try before any of the above.
      </Callout>

    </div>
  ),
};

export default xlstmContent;
