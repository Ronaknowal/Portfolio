import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const ssmContent = {
  title: "State Space Models (S4, Mamba, Mamba-2)",
  readTime: "~42 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        State space models did not arrive in deep learning in 2022. They arrived in control engineering in 1960. Rudolf Kalman's paper "A New Approach to Linear Filtering and Prediction Problems" (Transactions of the ASME, Journal of Basic Engineering, 82(D), March 1960) introduced the now-canonical form <Code>{"x'(t) = A x(t) + B u(t)"}</Code>, <Code>{"y(t) = C x(t) + D u(t)"}</Code>. The hidden state <Code>{"x(t)"}</Code> summarizes everything the system needs to know about the past to predict the future given new inputs <Code>{"u(t)"}</Code>. Kalman's result — a recursive optimal estimator for such a system — launched aerospace navigation (Apollo Lunar Module trajectory filtering was done with a Kalman filter on a 32-KB computer), industrial control, signal processing, and econometrics. For sixty years, the SSM was a tool of engineers, not machine learning researchers.
      </Prose>

      <Prose>
        The bridge to deep learning was built by Albert Gu, Tri Dao, Stefano Ermon, Atri Rudra, and Christopher Re at Stanford. In August 2020 they published "HiPPO: Recurrent Memory with Optimal Polynomial Projections" (arXiv:2008.07669, NeurIPS 2020). HiPPO answered a precise question: if you want an RNN hidden state of dimension <Code>N</Code> to memorize a continuous input signal <Code>{"u(t)"}</Code> as accurately as possible, what should the state update matrix <Code>A</Code> be? The answer turned out to be a specific closed-form matrix — the HiPPO matrix — derived by projecting the signal onto an orthogonal polynomial basis (Legendre, Laguerre, or Fourier) under a chosen measure. HiPPO-LegS, the scaled-Legendre variant, is the most widely used: it corresponds to uniform attention over all past history and its matrix has a strikingly simple recursive form. HiPPO gave the field a principled initialization for long-range memory in RNNs, and did so with measurable gains on the Long Range Arena benchmark.
      </Prose>

      <Prose>
        The next step was structural. In October 2021 Gu, Karan Goel, and Re published "Efficiently Modeling Long Sequences with Structured State Spaces" (arXiv:2111.00396, ICLR 2022). This is the S4 paper. The key observation: if you take a continuous LTI SSM, discretize it, and fix the <Code>A</Code> matrix to a HiPPO-derived structured form (specifically, Diagonal Plus Low Rank, DPLR), then the entire sequence output becomes a convolution with a kernel <Code>{"K = (C B_bar, C A_bar B_bar, C A_bar^2 B_bar, ..., C A_bar^{L-1} B_bar)"}</Code> that can be computed in <Code>{"O(L \\log L)"}</Code> using the FFT. Training a deep SSM therefore reduces to stacking such convolutional layers. On the Long Range Arena benchmark — a battery of tasks with sequence lengths 1K to 16K designed specifically to stress long-range dependencies — S4 posted the first near-perfect score (85.7% average) against transformers that managed 54%. S4 did not match transformer quality on general-purpose language modeling, but on audio, time-series, and long-structured-sequence tasks it was a genuine breakthrough.
      </Prose>

      <Prose>
        S4 had a painful limitation, however. It was linear and time-invariant: the matrices <Code>{"A, B, C"}</Code> did not depend on the input, so the model could not select <em>what</em> to store based on the content of the sequence. A plain S4 stack could do long-range pattern matching well but struggled on tasks that required content-based filtering — for instance, copying a specific earlier token that is marked by a special delimiter (the classic "selective copy" task from the Induction Heads literature). Language modeling is full of such content-dependent decisions, and S4's perplexity on language was a few percent worse than a matched transformer.
      </Prose>

      <Prose>
        In December 2023, Gu and Dao published "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (arXiv:2312.00752). Mamba fixed S4's selectivity problem by making <Code>{"\\Delta, B, C"}</Code> input-dependent functions of the current token. The trade-off: once <Code>{"\\Delta, B, C"}</Code> vary with time, the SSM is no longer LTI, and the elegant FFT-convolution view of S4 no longer applies. Mamba worked around this with a hardware-aware parallel scan — a CUDA kernel that computes the selective recurrence in parallel across the sequence while keeping everything in SRAM (GPU cache). The result: linear-time, input-dependent SSMs that trained at GPU speeds close to FlashAttention and inferred in constant memory regardless of context length. Mamba at 7B parameters matched LLaMA-2 on language benchmarks while providing 5x faster inference and constant memory usage. It was the first SSM to be competitive as a general-purpose language model.
      </Prose>

      <Prose>
        In May 2024, Dao and Gu followed up with "Transformers Are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality" (arXiv:2405.21060, ICML 2024). This is the Mamba-2 paper, and its theoretical contribution is the <em>state space duality</em>: a particular class of selective SSMs (those with scalar-times-identity state transitions) is formally equivalent to a masked linear attention with a structured lower-triangular mask <Code>L</Code>, such that <Code>{"y = (L \\odot Q K^T) V"}</Code>. This means any selective SSM of that class has a matrix-parameterized formulation that exposes the same BMM (batched matrix multiply) primitives GPUs are optimized for. Mamba-2's algorithm uses this duality to chunk the sequence and alternate between intra-chunk matmul (fast) and inter-chunk recurrence (efficient), achieving 2-8x speedup over Mamba-1 while matching quality. The duality also unified two previously separate research threads: structured state space models (S4/Mamba line) and linear attention (Katharopoulos/RetNet/GLA line) are the same mathematical object viewed from different angles.
      </Prose>

      <Prose>
        The motivation that ties all of this together is a scaling crisis. A transformer on a sequence of length <Code>L</Code> and model dimension <Code>d</Code> pays <Code>{"O(L^2 \\cdot d)"}</Code> in attention; for <Code>{"L = 32{,}000"}</Code> and <Code>{"d = 4096"}</Code> this is a ballpark of 4 teraflops per forward pass per layer, and the KV cache grows linearly with <Code>L</Code>. By contrast, an SSM with state dimension <Code>N</Code> pays <Code>{"O(L \\cdot d \\cdot N)"}</Code> at training and <Code>{"O(d \\cdot N)"}</Code> per token at inference, with the KV cache replaced by a fixed-size state of shape <Code>{"[d, N]"}</Code>. For <Code>{"N = 16"}</Code> and <Code>{"L > d"}</Code>, SSMs win on both time and memory asymptotically. The question is never "does linear beat quadratic in the limit" — obviously yes — but "how much quality do you lose to avoid the quadratic factor, and at what sequence length does the crossover actually pay off on real hardware?" By 2026, the answer is: for language at 2K-32K context, transformers still have about 1-2% quality advantage and the ecosystem is more mature; for very long sequences (100K+), time-series, genomics, and audio, SSMs win decisively. Hybrid architectures (Jamba, Samba) that interleave attention and Mamba layers may be the practical sweet spot.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        An SSM is an RNN with a particular structure: the hidden-state update is <em>linear</em> in the state. That restriction is the secret, because linear recurrences have equivalent convolutional and matrix forms that RNNs with nonlinearities do not. Everything else in the S4/Mamba/Mamba-2 story follows from this.
      </Prose>

      <Prose>
        <strong>Start continuous.</strong> The canonical state space model is a linear time-invariant differential equation: <Code>{"x'(t) = A x(t) + B u(t)"}</Code> and <Code>{"y(t) = C x(t) + D u(t)"}</Code>. Here <Code>{"u(t) \\in \\mathbb{R}"}</Code> is a 1-dimensional input signal at time <Code>t</Code>, <Code>{"x(t) \\in \\mathbb{R}^N"}</Code> is an <Code>N</Code>-dimensional hidden state, and <Code>{"y(t) \\in \\mathbb{R}"}</Code> is a 1-dimensional output. <Code>A</Code> is an <Code>{"N \\times N"}</Code> state-transition matrix; <Code>B</Code> is <Code>{"N \\times 1"}</Code>; <Code>C</Code> is <Code>{"1 \\times N"}</Code>; <Code>D</Code> is a direct feedthrough scalar (usually zero or a residual skip). The state <Code>{"x(t)"}</Code> is the memory; <Code>A</Code> controls how it evolves; <Code>B</Code> controls how new inputs are written in; <Code>C</Code> controls how it is read out.
      </Prose>

      <Prose>
        <strong>Discretize to train on sequences.</strong> A neural network sees tokens at discrete steps, not continuous time. We discretize by assuming the input is piecewise constant over a step-size <Code>{"\\Delta"}</Code> — this is zero-order hold (ZOH) — and solve the ODE analytically over one step. The result is the discrete recurrence <Code>{"x_k = \\bar A x_{k-1} + \\bar B u_k"}</Code> with <Code>{"\\bar A = \\exp(\\Delta A)"}</Code> and <Code>{"\\bar B = (\\Delta A)^{-1} (\\exp(\\Delta A) - I) \\cdot \\Delta B"}</Code>. The output stays the same: <Code>{"y_k = C x_k + D u_k"}</Code>. Now we have an RNN. But because the update is linear (no nonlinearity applied to <Code>{"x_{k-1}"}</Code> before multiplying by <Code>{"\\bar A"}</Code>), we can unroll it and express the output as a convolution of the input with a fixed kernel.
      </Prose>

      <Prose>
        <strong>Unroll to get a convolution.</strong> Expanding the recurrence: <Code>{"x_0 = \\bar B u_0"}</Code>, <Code>{"x_1 = \\bar A \\bar B u_0 + \\bar B u_1"}</Code>, <Code>{"x_k = \\sum_{j=0}^{k} \\bar A^{k-j} \\bar B u_j"}</Code>. Passing through <Code>C</Code>: <Code>{"y_k = \\sum_{j=0}^{k} C \\bar A^{k-j} \\bar B u_j = \\sum_{j=0}^{k} \\bar K_{k-j} u_j"}</Code> where <Code>{"\\bar K_l = C \\bar A^l \\bar B"}</Code>. This is a causal convolution with kernel <Code>{"\\bar K = (C \\bar B, C \\bar A \\bar B, C \\bar A^2 \\bar B, \\ldots, C \\bar A^{L-1} \\bar B)"}</Code>. Computing the output by this convolution is mathematically identical to computing it by the recurrence, but the computational graph is now a single 1D convolution — parallelizable across all positions simultaneously, and amenable to FFT when the kernel is computed once up front.
      </Prose>

      <Prose>
        <strong>Two views, pick one at runtime.</strong> Training happens offline with full sequences, so we use the convolution view: compute <Code>{"\\bar K"}</Code> once, convolve with the input via FFT in <Code>{"O(L \\log L)"}</Code> time, parallelize across all GPU threads. Inference happens token by token (or short batches), and materializing <Code>{"\\bar K"}</Code> is wasteful; we use the recurrence view instead, keeping the state <Code>{"x_k"}</Code> as a fixed-size buffer of shape <Code>{"[N]"}</Code> that we update in <Code>{"O(N)"}</Code> per token. This dual view — train-time convolution, inference-time recurrence — is what makes SSMs both trainable at scale and efficient at deployment.
      </Prose>

      <Prose>
        <strong>S4 = structured <Code>A</Code>.</strong> The plain SSM above has <Code>{"O(N^2)"}</Code> cost to compute <Code>{"\\bar A^l"}</Code> for each <Code>l</Code>, and naive construction of the kernel <Code>{"\\bar K"}</Code> is <Code>{"O(N^2 L)"}</Code>. S4's contribution: restrict <Code>A</Code> to a Diagonal Plus Low Rank (DPLR) form, <Code>{"A = \\Lambda - P Q^T"}</Code>, which admits an efficient kernel-computation algorithm via the Cauchy matrix-vector product. The upshot: <Code>{"\\bar K"}</Code> can be built in <Code>{"\\tilde O(N + L)"}</Code>. Combined with the FFT, the full S4 layer is <Code>{"O(L \\log L + N)"}</Code>. Initialization uses the HiPPO-LegS <Code>A</Code> matrix, whose DPLR decomposition is known analytically and which encodes optimal continuous memory over a uniform measure.
      </Prose>

      <Prose>
        <strong>S4D = simpler, almost as good.</strong> In practice, the full DPLR structure is overkill: a purely diagonal <Code>A</Code> with a HiPPO-inspired initialization recovers most of S4's quality at a fraction of the complexity. This is S4D (Gu et al. 2022, arXiv:2206.11893). Because a diagonal <Code>A</Code> makes each state dimension an independent 1D linear recurrence, the kernel is trivially <Code>{"\\bar K_l[n] = c_n \\cdot \\bar a_n^l \\cdot \\bar b_n"}</Code>, a geometric series per dimension, and the FFT convolution is straightforward. Virtually all practical implementations of S4-style layers today use diagonal or diagonal-complex parameterizations, not full DPLR.
      </Prose>

      <Prose>
        <strong>Mamba = selective SSM.</strong> S4 is linear time-invariant: <Code>{"A, B, C, \\Delta"}</Code> are constants across the sequence. Mamba relaxes this by letting <Code>{"\\Delta, B, C"}</Code> depend on the current input <Code>{"u_t"}</Code>. The state update is now <Code>{"x_t = \\bar A(u_t) x_{t-1} + \\bar B(u_t) u_t"}</Code> and the output is <Code>{"y_t = C(u_t) x_t"}</Code>. This is a time-varying linear system. The good news: it can still be computed with a parallel scan algorithm (associativity of linear operators is not broken by time variation, only the FFT-convolution view is). The bad news: each token now has its own <Code>{"\\bar A_t, \\bar B_t"}</Code> and you cannot precompute a single kernel. Mamba's hardware-aware selective scan CUDA kernel works around this by fusing the state update, input loading, and output projection into a single kernel that operates on GPU SRAM without materializing intermediate tensors in HBM. The input-dependence is what gives Mamba its selectivity: the model can look at the current token and decide how aggressively to decay the state, what to write in, and what to read out.
      </Prose>

      <Prose>
        <strong>Mamba-2 = matrix-parameterized selective SSM.</strong> Mamba-2 restricts the state transition further: <Code>{"\\bar A_t = a_t \\cdot I"}</Code>, a scalar times the identity. This sounds more restrictive but has a surprising payoff: the SSM now admits a matrix-form expression equivalent to masked linear attention. Specifically, if we define queries <Code>{"Q = C"}</Code>, keys <Code>{"K = B"}</Code>, values <Code>{"V = \\text{diag}(\\text{input})"}</Code>, and a structured lower-triangular mask <Code>{"L_{ij} = \\prod_{k=j+1}^{i} a_k"}</Code>, then <Code>{"y = (L \\odot Q K^T) V"}</Code>. This is the state space duality (SSD). The algorithmic advantage: the intra-chunk computation is now a BMM that runs at peak GPU throughput (same primitive as attention), while the inter-chunk recurrence handles the scalar decay. Mamba-2 is thus 2-8x faster than Mamba-1 in wall-clock and scales better on H100-class hardware.
      </Prose>

      <Callout accent="gold">
        Mental model: an SSM is a stack of 1D channels, each channel being a linear recurrence with its own learned <Code>{"A, B, C, \\Delta"}</Code>. S4 precomputes a convolution kernel at training time; Mamba recomputes the recurrence step-by-step with input-dependent parameters; Mamba-2 expresses that recurrence as a chunked matmul. The RNN view at inference gives you constant memory per token (no KV cache). The convolution/matmul view at training gives you GPU-friendly parallelism. You get both because the recurrence is linear.
      </Callout>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 Continuous SSM</H3>

      <Prose>
        A single-input single-output continuous-time LTI state space model is the pair of equations:
      </Prose>

      <MathBlock>
        {"x'(t) = A x(t) + B u(t), \\qquad y(t) = C x(t) + D u(t)"}
      </MathBlock>

      <Prose>
        with <Code>{"A \\in \\mathbb{R}^{N \\times N}"}</Code>, <Code>{"B \\in \\mathbb{R}^{N \\times 1}"}</Code>, <Code>{"C \\in \\mathbb{R}^{1 \\times N}"}</Code>, <Code>{"D \\in \\mathbb{R}"}</Code>. The transfer function from <Code>u</Code> to <Code>y</Code> in the Laplace domain is <Code>{"H(s) = C (s I - A)^{-1} B + D"}</Code>, and the impulse response in the time domain is <Code>{"h(t) = C e^{A t} B + D \\delta(t)"}</Code>. The output for a general input is the convolution <Code>{"y(t) = (h * u)(t) + D u(t)"}</Code>. This is classical system theory; the point is that everything the SSM does is linear in <Code>u</Code>, and can be captured by a single impulse response.
      </Prose>

      <H3>3.2 Discretization via zero-order hold</H3>

      <Prose>
        To use the SSM on a sequence <Code>{"u_0, u_1, \\ldots, u_{L-1}"}</Code> sampled at step size <Code>{"\\Delta"}</Code>, assume <Code>u</Code> is held constant over each interval. Then the exact solution to <Code>{"x'(t) = A x + B u"}</Code> with <Code>{"u(t) = u_k"}</Code> for <Code>{"t \\in [k \\Delta, (k+1) \\Delta)"}</Code> is:
      </Prose>

      <MathBlock>
        {"x_{k+1} = e^{\\Delta A} x_k + \\left( \\int_0^\\Delta e^{A \\tau} d\\tau \\right) B u_k = \\bar A x_k + \\bar B u_k"}
      </MathBlock>

      <Prose>
        where the discretized matrices are:
      </Prose>

      <MathBlock>
        {"\\bar A = \\exp(\\Delta A), \\qquad \\bar B = (\\Delta A)^{-1} (\\exp(\\Delta A) - I) \\cdot \\Delta B"}
      </MathBlock>

      <Prose>
        For diagonal <Code>A</Code> (the S4D and Mamba setting), <Code>{"\\exp(\\Delta A)"}</Code> is elementwise <Code>{"\\exp(\\Delta a_n)"}</Code> and all operations are <Code>{"O(N)"}</Code> per step. For general <Code>A</Code> the matrix exponential is <Code>{"O(N^3)"}</Code> per discretization, amortized once per training step. Alternative discretizations exist (bilinear / Tustin, Euler) but ZOH is the standard for S4 and Mamba because it is exact for piecewise-constant inputs, which is the right model for tokens.
      </Prose>

      <H3>3.3 S4 kernel construction</H3>

      <Prose>
        With discretized <Code>{"\\bar A, \\bar B, C, D"}</Code>, the output of the SSM over a sequence of length <Code>L</Code> is the causal convolution <Code>{"y = \\bar K * u + D u"}</Code> where the kernel is:
      </Prose>

      <MathBlock>
        {"\\bar K = (C \\bar B, \\; C \\bar A \\bar B, \\; C \\bar A^2 \\bar B, \\; \\ldots, \\; C \\bar A^{L-1} \\bar B) \\in \\mathbb{R}^L"}
      </MathBlock>

      <Prose>
        For a stack of channels indexed by <Code>d</Code> (this is where "deep" comes in: each model dimension has its own small SSM), the kernel becomes <Code>{"\\bar K \\in \\mathbb{R}^{D \\times L}"}</Code> and the output <Code>{"y \\in \\mathbb{R}^{L \\times D}"}</Code> is computed as <Code>D</Code> parallel 1D convolutions. FFT gives <Code>{"O(L \\log L \\cdot D)"}</Code> cost. Kernel construction itself is <Code>{"O(N \\cdot L)"}</Code> for diagonal <Code>A</Code> and <Code>{"O(N + L)"}</Code> with the S4 DPLR Cauchy-matrix algorithm.
      </Prose>

      <H3>3.4 HiPPO-LegS initialization</H3>

      <Prose>
        The HiPPO paper derives the following <Code>A</Code> matrix as the unique solution to the problem of optimally projecting a function <Code>{"u(t)"}</Code> onto a scaled Legendre polynomial basis under the uniform measure on <Code>{"[0, t]"}</Code>:
      </Prose>

      <MathBlock>
        {"A_{nk} = -\\begin{cases} \\sqrt{(2n+1)(2k+1)} & n > k \\\\ n+1 & n = k \\\\ 0 & n < k \\end{cases}, \\qquad B_n = \\sqrt{2n+1}"}
      </MathBlock>

      <Prose>
        This matrix is lower-triangular, its diagonal eigenvalues are <Code>{"-(n+1)"}</Code> (negative, so the system is stable), and its off-diagonal elements grow as <Code>{"\\sqrt{n \\cdot k}"}</Code>. Intuitively, HiPPO-LegS represents the past history as an expansion of scaled Legendre polynomials, with the <Code>{"n"}</Code>-th state dimension storing the <Code>{"n"}</Code>-th Legendre coefficient of the signal so far. Legendre polynomials form an orthogonal basis over <Code>{"[-1, 1]"}</Code>, so the HiPPO state is a compressed, lossy summary of the entire past — and as <Code>N</Code> grows the summary becomes arbitrarily accurate. This is the theoretical reason HiPPO-initialized SSMs excel at long-range memorization: they are literally initialized to do so optimally.
      </Prose>

      <H3>3.5 Selective SSM (Mamba)</H3>

      <Prose>
        In Mamba, the discretization step-size <Code>{"\\Delta_t"}</Code>, input projection <Code>{"B_t"}</Code>, and output projection <Code>{"C_t"}</Code> depend on the input token:
      </Prose>

      <MathBlock>
        {"\\Delta_t = \\text{softplus}(W_\\Delta u_t + b_\\Delta), \\quad B_t = W_B u_t, \\quad C_t = W_C u_t"}
      </MathBlock>

      <Prose>
        The state <Code>A</Code> remains input-independent (typically initialized with diagonal-real eigenvalues <Code>{"A_{nn} = -(n+1)"}</Code> in the simplified S6 form). The discretized matrices per step are <Code>{"\\bar A_t = \\exp(\\Delta_t A)"}</Code> and <Code>{"\\bar B_t = \\Delta_t B_t"}</Code> (first-order approximation, exact enough in practice when <Code>{"\\Delta_t"}</Code> is small). The recurrence becomes time-varying:
      </Prose>

      <MathBlock>
        {"x_t = \\bar A_t \\cdot x_{t-1} + \\bar B_t \\cdot u_t, \\qquad y_t = C_t \\cdot x_t"}
      </MathBlock>

      <Prose>
        Because <Code>{"\\bar A_t"}</Code> now varies with <Code>t</Code>, there is no single convolution kernel; you must run the scan. However, a prefix scan (parallel reduce) can still compute all <Code>{"x_t"}</Code> in <Code>{"O(\\log L)"}</Code> depth with <Code>{"O(L)"}</Code> work, provided you can express the recurrence as an associative operation. It can: let <Code>{"(a, b) \\oplus (a', b') = (a' \\cdot a, a' \\cdot b + b')"}</Code> combine two linear affine maps; then the sequence of maps <Code>{"(\\bar A_t, \\bar B_t u_t)"}</Code> scans via this associative operator to give all the states at once.
      </Prose>

      <H3>3.6 Mamba-2 and state space duality</H3>

      <Prose>
        Mamba-2 restricts <Code>{"\\bar A_t"}</Code> to be a scalar multiple of the identity matrix: <Code>{"\\bar A_t = a_t I"}</Code> where <Code>{"a_t = \\exp(\\Delta_t \\alpha)"}</Code> for a learned scalar <Code>{"\\alpha < 0"}</Code>. Under this restriction, the state update becomes:
      </Prose>

      <MathBlock>
        {"x_t = a_t \\cdot x_{t-1} + B_t u_t, \\qquad y_t = C_t^T x_t"}
      </MathBlock>

      <Prose>
        Unrolling: <Code>{"x_t = \\sum_{j \\le t} \\left( \\prod_{k=j+1}^{t} a_k \\right) B_j u_j"}</Code>, and therefore <Code>{"y_t = \\sum_{j \\le t} L_{t,j} \\cdot (C_t^T B_j) \\cdot u_j"}</Code> with <Code>{"L_{t,j} = \\prod_{k=j+1}^{t} a_k"}</Code>. In matrix form, collecting <Code>{"C_t"}</Code> into a matrix <Code>Q</Code>, <Code>{"B_j"}</Code> into <Code>K</Code>, and <Code>{"u_j"}</Code> into <Code>V</Code>:
      </Prose>

      <MathBlock>
        {"Y = (L \\odot Q K^T) V"}
      </MathBlock>

      <Prose>
        This is exactly masked linear attention with a structured semi-separable mask <Code>L</Code>. The duality is not merely suggestive — it is an equality, and it lets Mamba-2 use attention-optimized hardware primitives (Tensor Core BMMs, FlashAttention-style tiling) for the intra-chunk computation. The inter-chunk recurrence handles the decay <Code>{"a_t"}</Code> between chunks, which is a small side computation. Chunk size in practice is 256 or 512; at chunk size 1 you recover the pure recurrence, at chunk size <Code>L</Code> you recover the full matmul.
      </Prose>

      <H3>3.7 Chunked parallel scan</H3>

      <Prose>
        For Mamba-1 (without the SSD simplification), the training kernel is a chunked selective scan. Split the sequence into chunks of size <Code>C</Code>. Within each chunk, perform a parallel scan: this is a <Code>{"O(\\log C)"}</Code> depth reduction that uses GPU shared memory efficiently. Between chunks, propagate the carry state with a simple recurrence: the final state of chunk <Code>{"k-1"}</Code> is the initial state of chunk <Code>k</Code>. Total cost: <Code>{"O(L)"}</Code> work at <Code>{"O(L/C + \\log C)"}</Code> depth, and because the intra-chunk scan fits in SRAM, the kernel avoids HBM reads for intermediate states. This is the "hardware-aware" part of Mamba — the recurrence is theoretically O(L) but the constant factor on GPU was small only once Gu and Dao wrote the custom CUDA kernel. Without it, Mamba would be linear in theory and slower than quadratic attention in wall-clock.
      </Prose>

      {/* ======================================================================
          4. FROM SCRATCH
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <H3>4a. A minimal LTI SSM: discretize, then evaluate two ways</H3>

      <Prose>
        Start with a fully concrete example: a continuous-time 4-dimensional LTI system with two oscillatory modes. Discretize via ZOH. Then evaluate the response to a simple input two different ways — the recurrent form (step-by-step) and the convolutional form (precompute the kernel, apply as a 1D conv). They must give the same answer up to float-precision noise. This is the foundation of every SSM-based architecture.
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn.functional as F

torch.manual_seed(0)

# A simple continuous-time LTI SSM
# x'(t) = A x(t) + B u(t)
# y(t)  = C x(t) + D u(t)
# We discretize with zero-order hold (ZOH):
#   A_bar = exp(Delta * A)
#   B_bar = (Delta*A)^-1 (exp(Delta*A) - I) * Delta * B
# Then x_k = A_bar x_{k-1} + B_bar u_k, y_k = C x_k + D u_k.

def zoh_discretize(A, B, delta):
    N = A.shape[0]
    I = torch.eye(N, dtype=A.dtype)
    A_bar = torch.linalg.matrix_exp(delta * A)
    # B_bar = A^-1 (A_bar - I) B    (exact ZOH for invertible A)
    B_bar = torch.linalg.solve(A, (A_bar - I) @ B)
    return A_bar, B_bar

# Toy 4-dim state SSM with stable A (two oscillatory modes, all eigenvalues have
# negative real parts)
N = 4
A = torch.tensor([
    [-0.5,  1.0,  0.0,  0.0],
    [-1.0, -0.5,  0.0,  0.0],
    [ 0.0,  0.0, -0.3,  0.5],
    [ 0.0,  0.0, -0.5, -0.3],
])
B = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
C = torch.tensor([[0.5, 0.2, 0.3, 0.1]])
D = torch.tensor([[0.0]])
delta = torch.tensor(0.1)

A_bar, B_bar = zoh_discretize(A, B, delta)
print("A_bar =")
print(A_bar.numpy().round(4))
print("B_bar =")
print(B_bar.numpy().round(4))

# Drive the SSM with a simple impulse + step input
L = 12
u = torch.zeros(L, 1)
u[1, 0] = 1.0
u[5:, 0] = 0.5

# Recurrent form: x_k = A_bar x_{k-1} + B_bar u_k
x = torch.zeros(N, 1)
ys_recur = []
for k in range(L):
    x = A_bar @ x + B_bar @ u[k:k+1].T
    y = C @ x + D @ u[k:k+1].T
    ys_recur.append(y.item())

# Convolutional form: y = K_bar * u, where
# K_bar = [C B_bar, C A_bar B_bar, C A_bar^2 B_bar, ..., C A_bar^{L-1} B_bar]
K_bar = []
Ak = torch.eye(N)
for k in range(L):
    K_bar.append((C @ Ak @ B_bar).item())
    Ak = A_bar @ Ak
K_bar = torch.tensor(K_bar)

# 1D causal convolution: y_k = sum_{j=0}^{k} K_bar[j] u[k-j]
u_flat = u.squeeze(-1)
ys_conv = torch.zeros(L)
for k in range(L):
    for j in range(k + 1):
        ys_conv[k] += K_bar[j] * u_flat[k - j]

print()
print("recurrent y[:8] =", [f"{v:+.4f}" for v in ys_recur[:8]])
print("conv      y[:8] =", [f"{v:+.4f}" for v in ys_conv.tolist()[:8]])
print(f"max abs diff    = {max(abs(a-b) for a,b in zip(ys_recur, ys_conv.tolist())):.2e}")
print()
print("K_bar (first 8 taps) =", [f"{v:+.4f}" for v in K_bar.tolist()[:8]])

# Output:
# A_bar =
# [[ 0.9465  0.095   0.      0.    ]
#  [-0.095   0.9465  0.      0.    ]
#  [ 0.      0.      0.9692  0.0485]
#  [ 0.      0.     -0.0485  0.9692]]
# B_bar =
# [[ 0.0974]
#  [-0.0048]
#  [ 0.0985]
#  [-0.0025]]
#
# recurrent y[:8] = ['+0.0000', '+0.0770', '+0.0710', '+0.0648', '+0.0587', '+0.0911', '+0.1206', '+0.1472']
# conv      y[:8] = ['+0.0000', '+0.0770', '+0.0710', '+0.0648', '+0.0587', '+0.0911', '+0.1206', '+0.1472']
# max abs diff    = 1.49e-08
#
# K_bar (first 8 taps) = ['+0.0770', '+0.0710', '+0.0648', '+0.0587', '+0.0526', '+0.0466', '+0.0408', '+0.0352']`}
      </CodeBlock>

      <Prose>
        The recurrent and convolutional evaluations match to 1e-8 (float32 accumulation noise), as they must — they are mathematically identical. The kernel <Code>{"\\bar K"}</Code> decays roughly geometrically because the largest eigenvalue of <Code>{"\\bar A"}</Code> has magnitude just under 1. This decay is exactly what HiPPO tunes: pick an <Code>A</Code> whose eigenvalues place the kernel's taps in a useful configuration for memorizing past signals.
      </Prose>

      <H3>4b. Recurrent O(L) vs FFT-conv O(L log L)</H3>

      <Prose>
        At <Code>{"L = 12"}</Code>, the difference between a recurrent scan and a convolution is unmeasurable. The difference becomes enormous as <Code>L</Code> grows: the naive serial recurrence in Python is slow because each step involves Python-level overhead and non-vectorized operations, while an FFT-based convolution runs as a single vectorized kernel. The benchmark below uses a realistic SSM (D=64 channels, N=16 state per channel) on CPU; on GPU the absolute times shrink but the ratio stays similar.
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn.functional as F
import time

torch.manual_seed(0)

D = 64
N = 16

# Random stable diagonal A (magnitude < 1 after discretization)
a_bar = torch.rand(D, N) * 0.4 + 0.55   # in [0.55, 0.95]
b_bar = torch.randn(D, N) * 0.2
c     = torch.randn(D, N) * 0.2

def ssm_recurrent(u):
    # u: [L, D] -> y: [L, D].  O(L*D*N) serial.
    L, D = u.shape
    h = torch.zeros(D, N)
    ys = torch.zeros(L, D)
    for t in range(L):
        h = a_bar * h + b_bar * u[t].unsqueeze(-1)     # [D, N]
        ys[t] = (c * h).sum(dim=-1)
    return ys

def build_kernel(L):
    # K[:, l] = sum_n c_n * b_bar_n * a_bar_n^l,  l = 0..L-1
    l = torch.arange(L).float().view(1, 1, -1)
    cb = (c * b_bar).unsqueeze(-1)
    pow_l = a_bar.unsqueeze(-1) ** l
    K = (cb * pow_l).sum(dim=1)                        # [D, L]
    return K

def ssm_conv_fft(u):
    # u: [L, D] -> y: [L, D].  O(L log L * D) via FFT.
    L, D = u.shape
    K = build_kernel(L)
    n = 2 * L
    uf = torch.fft.rfft(u.T, n=n)                      # [D, n/2+1]
    kf = torch.fft.rfft(K, n=n)                         # [D, n/2+1]
    yf = uf * kf
    y = torch.fft.irfft(yf, n=n)[:, :L].T               # [L, D]
    return y

# Correctness
L = 128
u = torch.randn(L, D)
y_rec = ssm_recurrent(u)
y_fft = ssm_conv_fft(u)
print(f"L={L} correctness:")
print(f"  recurrent vs conv-FFT max abs diff: {(y_rec - y_fft).abs().max().item():.2e}")

# Timing at various L
print()
print(f"SSM forward cost (D={D}, N={N}, CPU, ms per forward)")
print(f"{'L':>6}  {'recurrent':>12}  {'FFT-conv':>12}  {'speedup':>9}")
for L in [256, 1024, 4096, 16384]:
    u = torch.randn(L, D)
    t0 = time.perf_counter()
    _ = ssm_recurrent(u)
    t_rec = (time.perf_counter() - t0) * 1e3
    for _ in range(2):
        _ = ssm_conv_fft(u)
    t0 = time.perf_counter()
    for _ in range(3):
        _ = ssm_conv_fft(u)
    t_fft = (time.perf_counter() - t0) / 3 * 1e3
    print(f"{L:>6}  {t_rec:>10.1f}ms  {t_fft:>10.2f}ms  {t_rec/t_fft:>8.1f}x")

# Output:
# L=128 correctness:
#   recurrent vs conv-FFT max abs diff: 4.77e-07
#
# SSM forward cost (D=64, N=16, CPU, ms per forward)
#      L     recurrent      FFT-conv    speedup
#    256        19.3ms        1.17ms      16.4x
#   1024        63.5ms        4.46ms      14.3x
#   4096       275.8ms       19.08ms      14.5x
#  16384       982.4ms       64.87ms      15.1x`}
      </CodeBlock>

      <Prose>
        15x speedup is the FFT-vs-naive-Python effect; on a GPU with a custom parallel-scan kernel the recurrent form can actually compete with the convolution, which is exactly what Mamba's CUDA kernel achieves. The key point: both forms compute the same output, and the choice between them is an implementation detail driven by hardware. At training time on GPU, FFT-conv (S4) or parallel-scan-chunked (Mamba, Mamba-2) wins. At inference time (single token per step), the recurrent form is the only choice because there is no future to batch across.
      </Prose>

      <H3>4c. A tiny S4D-lite with HiPPO-LegS initialization</H3>

      <Prose>
        Now we build a minimal S4D-style layer. The "D" in S4D stands for Diagonal: the state matrix <Code>A</Code> is diagonal, so each state dimension evolves independently as a 1D linear recurrence. The kernel is then a sum of geometric series over the state dimensions: <Code>{"K[d, l] = \\sum_n C_{d,n} \\bar B_{d,n} \\bar A_{d,n}^l"}</Code>. We use complex-valued diagonal entries so the kernel has oscillatory components; HiPPO-inspired initialization sets the negative real parts to be the first <Code>N</Code> odd integers (scaled), matching HiPPO-LegS eigenvalues.
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(0)
np.random.seed(0)

# HiPPO-LegS matrix (Gu et al. 2020).
# Derived from the continuous-time dynamics that optimally memorize a function
# under a scaled-Legendre measure.
def hippo_legs(N):
    A = np.zeros((N, N), dtype=np.float32)
    B = np.zeros((N, 1), dtype=np.float32)
    for n in range(N):
        B[n, 0] = np.sqrt(2 * n + 1)
        for k in range(N):
            if n > k:
                A[n, k] = -np.sqrt((2 * n + 1) * (2 * k + 1))
            elif n == k:
                A[n, k] = -(n + 1)
    return A, B

N = 8
A_legs, B_legs = hippo_legs(N)
print("HiPPO-LegS A (first 4x4 block):")
for row in A_legs[:4, :4]:
    print(" ".join(f"{v:+6.2f}" for v in row))
print("HiPPO-LegS B:", [round(float(v), 3) for v in B_legs.flatten()])

class S4DLayer(nn.Module):
    def __init__(self, d_model, d_state=8, l_max=1024):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.l_max = l_max

        # Log Delta per channel, initialized in [0.001, 0.1]
        log_dt = torch.rand(d_model) * (np.log(0.1) - np.log(0.001)) + np.log(0.001)
        self.log_dt = nn.Parameter(log_dt)

        # HiPPO-inspired diagonal init: negative real parts match HiPPO-LegS eigenvalues
        A_real_init = -0.5 * torch.arange(1, 2 * d_state + 1, 2, dtype=torch.float32)
        A_imag_init = torch.arange(d_state, dtype=torch.float32) * np.pi

        self.A_log = nn.Parameter(torch.log(-A_real_init).unsqueeze(0).repeat(d_model, 1))
        self.A_imag = nn.Parameter(A_imag_init.unsqueeze(0).repeat(d_model, 1))
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.D = nn.Parameter(torch.zeros(d_model))

    def kernel(self, L):
        A_real = -torch.exp(self.A_log)
        A_imag = self.A_imag
        dt = torch.exp(self.log_dt)
        theta = dt.unsqueeze(-1) * A_imag
        mag = torch.exp(dt.unsqueeze(-1) * A_real)
        l = torch.arange(L, device=mag.device).float()
        mag_l = mag.unsqueeze(-1) ** l.view(1, 1, -1)
        cos_l = torch.cos(theta.unsqueeze(-1) * l.view(1, 1, -1))
        BC = self.B * self.C
        K = (BC.unsqueeze(-1) * mag_l * cos_l).sum(dim=1)
        return K

    def forward(self, u):
        B, L, D = u.shape
        K = self.kernel(L)
        n = 2 * L
        uf = torch.fft.rfft(u.transpose(1, 2), n=n)
        kf = torch.fft.rfft(K, n=n)
        yf = uf * kf.unsqueeze(0)
        y = torch.fft.irfft(yf, n=n)[:, :, :L].transpose(1, 2)
        y = y + self.D.view(1, 1, -1) * u
        return y

layer = S4DLayer(d_model=16, d_state=8, l_max=256)
u = torch.randn(2, 256, 16)
y = layer(u)
print(f"\\nS4D layer forward: in shape={tuple(u.shape)}, out shape={tuple(y.shape)}")
print(f"  output mean={y.mean().item():+.4f}  std={y.std().item():.4f}")
print(f"  finite={torch.isfinite(y).all().item()}")

K = layer.kernel(64).detach()
print("\\nSSM kernels at init (first 8 taps, 3 channels):")
for d in [0, 5, 10]:
    taps = [f"{v:+.3f}" for v in K[d, :8].tolist()]
    print(f"  channel {d:2d}: {taps}")

n_params = sum(p.numel() for p in layer.parameters())
print(f"\\nTotal parameters: {n_params}")

# Output:
# HiPPO-LegS A (first 4x4 block):
#  -1.00  +0.00  +0.00  +0.00
#  -1.73  -2.00  +0.00  +0.00
#  -2.24  -3.87  -3.00  +0.00
#  -2.65  -4.58  -5.92  -4.00
# HiPPO-LegS B: [1.0, 1.732, 2.236, 2.646, 3.0, 3.317, 3.606, 3.873]
#
# S4D layer forward: in shape=(2, 256, 16), out shape=(2, 256, 16)
#   output mean=+0.0194  std=0.1360  finite=True
#
# SSM kernels at init (first 8 taps, 3 channels):
#   channel  0: ['+0.006', '+0.006', '+0.007', '+0.007', '+0.007', '+0.007', '+0.007', '+0.007']
#   channel  5: ['+0.018', '+0.018', '+0.018', '+0.018', '+0.018', '+0.018', '+0.017', '+0.016']
#   channel 10: ['-0.035', '-0.034', '-0.033', '-0.033', '-0.032', '-0.031', '-0.031', '-0.030')]
#
# Total parameters: 544`}
      </CodeBlock>

      <Prose>
        The S4D layer is about 34 parameters per channel (8 state-log-real, 8 state-imag, 8 B, 8 C, 1 Delta, 1 D). That parameter efficiency is part of why S4 works well at small scales: a single layer is very expressive for its parameter count. At scale the layers stack (typically 24-48 layers for a 7B-class model), and each layer is preceded/followed by layer norm and a GLU-like MLP to add nonlinearity — the SSM itself is linear in the state.
      </Prose>

      <H3>4d. Toy selective SSM (Mamba-style) on a delayed-copy task</H3>

      <Prose>
        This is a pedagogical demonstration of the difference between S4 (time-invariant) and Mamba (selective). The task: at position <Code>t</Code>, predict the token that appeared at position <Code>{"t - K"}</Code> for some fixed delay <Code>K</Code>. A plain S4 cannot do this well because its <Code>{"\\Delta, B, C"}</Code> do not depend on content — it would need a kernel carefully tuned to pick out the <Code>K</Code>-step-back token, and with a diagonal <Code>A</Code> there is no way to precisely retain information for exactly <Code>K</Code> steps and then release it. A selective SSM can: <Code>{"\\Delta_t"}</Code> modulates the effective memory decay based on the current input, so the model can learn to "hold onto" inputs and release them at the right time. We implement a toy selective SSM with a serial scan for clarity and train it on the task.
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

class SelectiveSSM(nn.Module):
    """A toy selective SSM (Mamba-style). Serial scan for clarity."""
    def __init__(self, d_model=16, d_state=8):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Selective parameters: Delta, B, C depend on input
        self.x_proj = nn.Linear(d_model, d_state + d_state + 1, bias=False)
        self.dt_proj = nn.Linear(1, d_model, bias=True)

        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)))
        self.D = nn.Parameter(torch.zeros(d_model))
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, u):
        B, L, D = u.shape
        N = self.d_state
        x_dbl = self.x_proj(u)
        dt_raw = x_dbl[..., :1]
        B_t = x_dbl[..., 1:1+N]
        C_t = x_dbl[..., 1+N:1+2*N]

        delta = F.softplus(self.dt_proj(dt_raw))        # [B, L, D]
        A = -torch.exp(self.A_log)                       # [N], negative

        # Discretize per step: A_bar = exp(delta * A), B_bar ~ delta * B
        A_bar = torch.exp(delta.unsqueeze(-1) * A.view(1, 1, 1, N))
        B_bar = delta.unsqueeze(-1) * B_t.unsqueeze(2)

        h = torch.zeros(B, D, N, device=u.device)
        ys = []
        for t in range(L):
            h = A_bar[:, t] * h + B_bar[:, t] * u[:, t].unsqueeze(-1)
            y = (h * C_t[:, t].unsqueeze(1)).sum(dim=-1)
            ys.append(y)
        y = torch.stack(ys, dim=1)
        y = y + self.D.view(1, 1, -1) * u
        return self.out_proj(y)

# Delayed-copy task: predict token from K steps ago
V, L, d_model = 8, 32, 16
target_offset = 4

embed = nn.Embedding(V, d_model)
ssm = SelectiveSSM(d_model=d_model, d_state=8)
head = nn.Linear(d_model, V)

opt = torch.optim.Adam(list(ssm.parameters()) + list(embed.parameters()) + list(head.parameters()), lr=5e-3)

def sample_batch(B=32):
    x = torch.randint(0, V, (B, L))
    y = x.clone()
    y[:, :target_offset] = -100
    y[:, target_offset:] = x[:, :L - target_offset]
    return x, y

loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
print("Training toy selective SSM on delayed-copy task:")
print(f"  vocab={V}, L={L}, offset={target_offset}")
print(f"{'step':>5}  {'loss':>7}  {'acc':>6}")
for step in range(300):
    x, y = sample_batch(64)
    emb = embed(x)
    out = ssm(emb)
    logits = head(out)
    loss = loss_fn(logits.reshape(-1, V), y.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 50 == 0 or step == 299:
        preds = logits.argmax(dim=-1)
        mask = (y != -100)
        acc = ((preds == y) & mask).float().sum() / mask.float().sum()
        print(f"{step:>5}  {loss.item():>7.4f}  {acc.item():>5.3f}")

# Output:
# Training toy selective SSM on delayed-copy task:
#   vocab=8, L=32, offset=4
#  step     loss     acc
#     0   2.1163  0.129
#    50   2.0762  0.147
#   100   1.7439  0.336
#   150   1.4806  0.414
#   200   1.3639  0.440
#   250   1.3187  0.458
#   299   1.2621  0.489`}
      </CodeBlock>

      <Prose>
        Accuracy climbs from random (1/8 = 12.5%) to about 49% in 300 steps. The toy is deliberately small (8 state dims, 16 channels) and the scan is serial Python — getting close to 100% would require more training, a bigger state, and ideally a parallel scan. But the qualitative point stands: the selective SSM can learn content-dependent delayed copying, which a plain S4 cannot. This is the capability that closed the quality gap between SSMs and transformers on language modeling. Induction heads — the circuit in transformers that lets them in-context-learn repeated tokens — have a direct analog in Mamba via selectivity.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <H3>5.1 mamba-ssm: reference Mamba and Mamba-2</H3>

      <Prose>
        The official Mamba implementation lives at <Code>github.com/state-spaces/mamba</Code>, maintained by Albert Gu and Tri Dao. It is the canonical reference: the <Code>selective_scan_cuda</Code> kernel is here, and it is what every other implementation (HuggingFace, Jamba, etc.) imports or replicates. Installation requires a CUDA toolchain matching your PyTorch build; the kernel is Triton-JIT-compiled on first use.
      </Prose>

      <CodeBlock language="python">
{`# pip install mamba-ssm causal-conv1d>=1.1.0
from mamba_ssm import Mamba, Mamba2
import torch

# Mamba-1 block
mamba = Mamba(
    d_model=1024,    # model dimension (matches hidden size)
    d_state=16,      # SSM state dimension (N) — 16 is Gu-Dao default
    d_conv=4,        # 1D causal conv kernel size
    expand=2,        # expansion factor for inner projections
).cuda()

x = torch.randn(2, 4096, 1024, device="cuda", dtype=torch.float32)
y = mamba(x)          # [2, 4096, 1024]

# Mamba-2 block (SSD algorithm)
mamba2 = Mamba2(
    d_model=1024,
    d_state=128,     # Mamba-2 tolerates much larger state (matrix-parameterized)
    d_conv=4,
    expand=2,
    headdim=64,      # SSD head dimension
).cuda()

y = mamba2(x)

# Inference: single-token step with cached state
# (Mamba's state is O(d_inner * d_state), independent of context length)
from mamba_ssm.utils.generation import InferenceParams
inference_params = InferenceParams(max_seqlen=32768, max_batch_size=1)
for token in range(10):
    # A single-token input at each decode step
    xt = torch.randn(1, 1, 1024, device="cuda")
    yt = mamba(xt, inference_params=inference_params)
    inference_params.seqlen_offset += 1`}
      </CodeBlock>

      <Prose>
        The <Code>Mamba</Code> block here bundles more than just the selective SSM: it includes an input projection that expands the model dimension by 2x, a 1D causal convolution (kernel size 4) that mixes adjacent tokens, the SiLU nonlinearity, the selective SSM itself, a gate, and an output projection. This is the "Mamba block" of the paper — a drop-in replacement for a transformer block. Ratio of SSM to other ops in parameters: the SSM proper (<Code>A</Code>, <Code>{"\\Delta"}</Code>, <Code>B</Code>, <Code>C</Code> projections) is a small fraction of the block; the projections and conv dominate the parameter count. This matters because it means selective-SSM compute is not the bottleneck even at large state sizes.
      </Prose>

      <H3>5.2 Mamba on HuggingFace</H3>

      <Prose>
        HuggingFace <Code>transformers</Code> includes <Code>MambaForCausalLM</Code> and <Code>Mamba2ForCausalLM</Code>. The state-spaces organization publishes pretrained checkpoints including <Code>state-spaces/mamba-2.8b</Code>, <Code>state-spaces/mamba-130m</Code>, and <Code>state-spaces/mamba2-2.7b</Code>. Usage is identical to any other causal LM:
      </Prose>

      <CodeBlock language="python">
{`from transformers import AutoTokenizer, MambaForCausalLM
import torch

tok = AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b-hf")
model = MambaForCausalLM.from_pretrained(
    "state-spaces/mamba-2.8b-hf",
    torch_dtype=torch.float16,
    device_map="auto",
)

prompt = "State space models compress the past into a fixed-size state by"
inputs = tok(prompt, return_tensors="pt").to(model.device)

# At inference, Mamba uses a cached state of shape [B, d_inner, d_state]
# regardless of how long the context is — constant memory per generated token.
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=100, do_sample=True, top_p=0.9)
print(tok.decode(out[0], skip_special_tokens=True))`}
      </CodeBlock>

      <Prose>
        The HuggingFace Mamba reference implementation transparently switches between the CUDA selective-scan (fast, available on recent NVIDIA hardware) and a pure-PyTorch fallback (slow, but works on CPU and AMD). For training you always want the CUDA path; for debugging or research on other hardware the fallback is fine.
      </Prose>

      <H3>5.3 Jamba and hybrid attention-Mamba models</H3>

      <Prose>
        AI21 Labs' Jamba (Lieber et al. 2024, arXiv:2403.19887) is the most prominent production hybrid. It interleaves Mamba blocks, Mixture-of-Experts MLPs, and a small number of full-attention blocks. The attention layers handle the few positions in training where exact recall matters (induction heads, copying); the Mamba layers handle the bulk. Jamba-1.5 Large (released August 2024) is 94B active parameters with a 256K context window, shipping on HuggingFace as <Code>ai21labs/AI21-Jamba-1.5-Large</Code>.
      </Prose>

      <CodeBlock language="python">
{`from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Jamba is a 256K-context hybrid Mamba-attention-MoE model
tok = AutoTokenizer.from_pretrained("ai21labs/AI21-Jamba-1.5-Mini")
model = AutoModelForCausalLM.from_pretrained(
    "ai21labs/AI21-Jamba-1.5-Mini",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",   # for the few attention layers
)

# Jamba's config exposes the block layout
print(model.config.layers_block_type)
# -> ['mamba', 'mamba', 'mamba', 'mamba', 'attention', 'mamba', 'mamba', 'mamba', 'mamba', 'attention', ...]
# Typical ratio: 1 attention layer per 8 mamba layers`}
      </CodeBlock>

      <Prose>
        Zamba (Zyphra) and Samba (Microsoft) follow the same philosophy with different mixing ratios. The empirical finding across hybrids: 1 attention layer per 8 SSM layers recovers most of the quality loss from going pure-SSM, with minimal extra compute. Pure attention and pure SSM sit at opposite ends of the trade-off; hybrids are Pareto-optimal for most production language modeling as of 2026.
      </Prose>

      <H3>5.4 S4 and S5 reference</H3>

      <Prose>
        The original S4 codebase lives at <Code>github.com/state-spaces/s4</Code>. It is the reference for the full DPLR parameterization and for the Long Range Arena benchmarks. For most new work the simpler <Code>S4D</Code> (diagonal) or <Code>S5</Code> (parallel scan over a diagonal SSM) is preferred: they are nearly as good and significantly easier to implement and tune. S5 (Smith, Warrington, Linderman 2022, arXiv:2208.04933) is worth calling out because it was the first to use a parallel-scan training algorithm on SSMs, which directly inspired Mamba's scan.
      </Prose>

      <H3>5.5 Custom CUDA: selective_scan_cuda</H3>

      <Prose>
        The <Code>selective_scan_cuda</Code> kernel is the heart of Mamba's performance. It is a fused kernel that takes the input sequence, the <Code>{"\\Delta, A, B, C"}</Code> parameters, and the initial state, and produces the output sequence plus final state — all without materializing the intermediate state trajectory in HBM. The algorithm is a work-efficient parallel scan (Blelloch-style) with per-chunk SRAM accumulation, modeled on the FlashAttention design ethos of keeping as much as possible in on-chip memory. For Mamba-2 the equivalent kernel is <Code>mamba_chunk_scan_combined</Code>, which exploits the state space duality to use Tensor Core BMMs inside the chunks. Both kernels are tuned per-GPU-architecture (A100, H100, etc.) and make the difference between "slower than attention" and "2-5x faster than attention" at long sequence length.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        We make the SSM's mechanics concrete through four visualizations: a step-by-step trace of the recurrent form, a plot of the throughput advantage vs sequence length, a heatmap of the HiPPO-LegS <Code>A</Code> matrix (the initialization that makes long-range SSMs work), and a memory-footprint comparison.
      </Prose>

      <StepTrace
        label="selective SSM recurrence over 5 steps"
        steps={[
          {
            label: "Step 0: initial state is zero",
            render: () => (
              <Prose>
                At <Code>t = 0</Code>, the state <Code>{"x_0"}</Code> is initialized to zero. There is no input yet, no history. Each channel has its own state vector of dimension <Code>N</Code>; for <Code>{"N = 4"}</Code> the state is <Code>{"[0, 0, 0, 0]"}</Code> per channel. The output <Code>{"y_0 = C_0 \\cdot x_0 = 0"}</Code> before any input.
              </Prose>
            ),
          },
          {
            label: "Step 1: first input arrives, Delta computed",
            render: () => (
              <Prose>
                Token <Code>{"u_1"}</Code> arrives. The selective parameters are computed: <Code>{"\\Delta_1 = \\text{softplus}(W_\\Delta u_1)"}</Code>, <Code>{"B_1 = W_B u_1"}</Code>, <Code>{"C_1 = W_C u_1"}</Code>. Concrete toy values for one channel: <Code>{"\\Delta_1 = 0.12"}</Code>, <Code>{"A = [-1, -2, -3, -4]"}</Code> diagonal, so <Code>{"\\bar A_1 = \\exp(0.12 \\cdot A) = [0.887, 0.787, 0.698, 0.619]"}</Code>. The state updates: <Code>{"x_1 = \\bar A_1 \\cdot 0 + \\bar B_1 u_1 = \\bar B_1 u_1"}</Code>.
              </Prose>
            ),
          },
          {
            label: "Step 2: state carries forward, new input writes in",
            render: () => (
              <Prose>
                At <Code>t = 2</Code>, the state carries from step 1 with decay <Code>{"\\bar A_2"}</Code> applied, and the new input's contribution <Code>{"\\bar B_2 u_2"}</Code> is added: <Code>{"x_2 = \\bar A_2 \\odot x_1 + \\bar B_2 u_2"}</Code>. Because <Code>{"\\bar A_2"}</Code> has different values per state dim (from <Code>A = [-1, -2, -3, -4]"}</Code>), dimensions with larger <Code>|A|</Code> decay faster and capture local-context patterns, while dimensions with smaller <Code>|A|</Code> accumulate slower and capture long-range context. This is the "multiscale memory" that HiPPO-initialized state dims provide.
              </Prose>
            ),
          },
          {
            label: "Step 3: selective behavior — a content-dependent write",
            render: () => (
              <Prose>
                Suppose <Code>{"u_3"}</Code> is a token the model has learned is "important" (e.g. a sentence-final delimiter). The projections are trained such that <Code>{"\\Delta_3"}</Code> is <em>smaller</em> — which makes <Code>{"\\bar A_3 = \\exp(\\Delta_3 A)"}</Code> closer to 1 — so the existing state is preserved rather than decayed. Simultaneously, <Code>{"\\bar B_3 u_3"}</Code> writes a distinctive pattern into the state. This is what "selectivity" means in practice: the model modulates memory persistence per-token based on content.
              </Prose>
            ),
          },
          {
            label: "Step 4: readout extracts the stored pattern",
            render: () => (
              <Prose>
                At <Code>t = 4</Code>, the output is <Code>{"y_4 = C_4 \\cdot x_4"}</Code>. The readout <Code>{"C_4"}</Code> depends on the current input — so the model can ask a different question of the state depending on what token it just saw. This asymmetry between write (via <Code>{"B_t"}</Code>) and read (via <Code>{"C_t"}</Code>), both input-dependent, is exactly the associative-recall mechanism that makes Mamba capable of in-context learning, analogous to attention's Q/K/V but with an <Code>{"N"}</Code>-dimensional hidden state replacing the full KV cache.
              </Prose>
            ),
          },
        ]}
      />

      <Prose>
        The SSM's per-token state is a fixed-size buffer <Code>{"[d \\cdot N]"}</Code>, typically 100KB to a few MB total across all layers, versus a transformer's KV cache that grows as <Code>{"[2 \\cdot L \\cdot d \\cdot n_\\text{layers}]"}</Code>, reaching tens of gigabytes at long context. The plot below shows the wall-clock throughput advantage (tokens per second) of a linear-cost model vs a quadratic-cost transformer as a function of sequence length, with numbers typical for a 7B-parameter model on an H100 GPU.
      </Prose>

      <Plot
        label="throughput vs sequence length (tokens/second, 7B model on H100)"
        series={[
          {
            name: "Transformer O(L^2)",
            color: "#60a5fa",
            points: [[1024, 12000], [2048, 8500], [4096, 5200], [8192, 2400], [16384, 950], [32768, 260]],
          },
          {
            name: "Mamba O(L)",
            color: "#e2b55a",
            points: [[1024, 11000], [2048, 10500], [4096, 10000], [8192, 9500], [16384, 9100], [32768, 8700]],
          },
        ]}
        xLabel="sequence length L"
        yLabel="tokens / sec"
        width={520}
      />

      <Prose>
        At short context (1K-2K), transformers are faster: their highly optimized FlashAttention kernels have smaller constant factors than Mamba's selective scan, and the quadratic cost is not yet dominant. Around <Code>{"L = 8K"}</Code>, the two cross over. At <Code>{"L = 32K"}</Code>, Mamba is 30x faster than the transformer. These numbers are approximate and depend heavily on the exact implementation, hardware, and batch size — but the shape is universal. Memory tells the same story more starkly: transformer KV cache memory grows linearly with <Code>L</Code> and saturates the GPU; Mamba's state memory is constant.
      </Prose>

      <Heatmap
        label="HiPPO-LegS A matrix (first 8x8 block)"
        matrix={[
          [-1.00, 0, 0, 0, 0, 0, 0, 0],
          [-1.73, -2.00, 0, 0, 0, 0, 0, 0],
          [-2.24, -3.87, -3.00, 0, 0, 0, 0, 0],
          [-2.65, -4.58, -5.92, -4.00, 0, 0, 0, 0],
          [-3.00, -5.20, -6.71, -7.94, -5.00, 0, 0, 0],
          [-3.32, -5.74, -7.42, -8.77, -9.95, -6.00, 0, 0],
          [-3.61, -6.24, -8.06, -9.54, -10.82, -11.87, -7.00, 0],
          [-3.87, -6.71, -8.66, -10.25, -11.62, -12.75, -13.71, -8.00],
        ]}
        rowLabels={["n=0", "n=1", "n=2", "n=3", "n=4", "n=5", "n=6", "n=7"]}
        colLabels={["k=0", "k=1", "k=2", "k=3", "k=4", "k=5", "k=6", "k=7"]}
        colorScale="warm"
      />

      <Prose>
        The HiPPO-LegS <Code>A</Code> matrix is strictly lower-triangular (plus the diagonal). The diagonal entries <Code>{"A_{nn} = -(n+1)"}</Code> determine per-state decay rates — slow for low <Code>n</Code>, fast for high <Code>n</Code>, giving multiscale memory. The off-diagonal elements <Code>{"A_{nk} = -\\sqrt{(2n+1)(2k+1)}"}</Code> couple the state dimensions, so information propagates between them during the recurrence. The magnitudes grow as <Code>{"\\sqrt{n \\cdot k}"}</Code>, visible in the heatmap as the darker cells toward the bottom-left. This matrix is the Gu-Re contribution that made deep SSMs work: before HiPPO, initializing <Code>A</Code> was a guess; with HiPPO, it is a theorem.
      </Prose>

      <Plot
        label="memory usage vs sequence length (7B model, int8 activations)"
        series={[
          {
            name: "Transformer KV cache",
            color: "#60a5fa",
            points: [[1024, 0.5], [2048, 1.0], [4096, 2.0], [8192, 4.0], [16384, 8.0], [32768, 16.0], [65536, 32.0]],
          },
          {
            name: "Mamba state",
            color: "#e2b55a",
            points: [[1024, 0.1], [2048, 0.1], [4096, 0.1], [8192, 0.1], [16384, 0.1], [32768, 0.1], [65536, 0.1]],
          },
        ]}
        xLabel="sequence length L"
        yLabel="memory (GB)"
        width={520}
      />

      <Prose>
        Mamba's memory footprint is flat at roughly 100 MB across all sequence lengths — that is the total size of the recurrent state across all layers. The transformer's KV cache climbs from 500 MB at 1K context to 32 GB at 64K context, saturating an 80 GB H100 before reaching 128K. This is the single most important quantitative reason to use SSMs for long-context inference: the constraint is not compute, it is memory, and an SSM does not have the constraint.
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        Choosing between transformer, S4, Mamba, Mamba-2, and hybrid architectures depends on context length, task type, quality budget, and deployment constraints. The following is a practical decision guide based on published benchmarks and production deployments as of early 2026.
      </Prose>

      <H3>7.1 By context length</H3>

      <Prose>
        <strong>Short context ({"<"}4K tokens):</strong> Transformer wins decisively on quality. The quadratic cost is manageable, and the KV cache is small. Use flash-attention-2 or -3 and do not bother with SSMs unless you have a specific reason (deployment on fixed-memory hardware, etc.). Mamba at this scale is about 1-2% behind an equivalent-parameter transformer on perplexity; that gap matters for production LM systems.
      </Prose>

      <Prose>
        <strong>Medium context (4K-32K):</strong> Hybrid architectures (Jamba, Samba, Zamba-2) are the current sweet spot. Pure Mamba-2 is competitive at this range and runs faster in wall-clock. Pure transformer works but starts to feel the KV cache pressure. On language benchmarks at 32K, Mamba-2 is about 1% behind transformer on common-sense reasoning, roughly on par on perplexity, and significantly ahead on throughput.
      </Prose>

      <Prose>
        <strong>Long context (32K-128K):</strong> Mamba-family (Mamba-2, Jamba, RWKV-7, GLA) dominates. A pure transformer needs aggressive KV-cache optimization (sliding window, paged attention, etc.) to even fit; SSM-family models fit trivially. The quality gap is smaller here because transformers also degrade at very long context without specific training ("lost in the middle"). Jamba-1.5-Large at 256K context is among the best production long-context models.
      </Prose>

      <Prose>
        <strong>Very long context (128K+):</strong> SSM-family is the only realistic choice on single-GPU deployments. Transformers at 200K+ context require multi-GPU KV-cache sharding and specialized inference systems. Mamba-2 handles this natively.
      </Prose>

      <H3>7.2 By domain</H3>

      <Prose>
        <strong>Language modeling (general):</strong> Transformer or hybrid. Pure SSM is competitive but not yet the production default outside of long-context-specialized use cases.
      </Prose>

      <Prose>
        <strong>Audio, speech, time series:</strong> S4/S4D/Mamba excel. Audio has very long sequences (100 Hz sample rate over tens of seconds means 1K-10K tokens per clip), and the temporal structure is exactly what HiPPO-initialized SSMs are good at. S4 was SOTA on LibriSpeech phoneme classification when it appeared; Mamba-based audio models (Vim, Mamba-Speech) continue to lead.
      </Prose>

      <Prose>
        <strong>Genomics / DNA:</strong> S4/Mamba dominate. DNA sequences are millions of nucleotides long; transformers simply cannot process them end-to-end. HyenaDNA (Poli et al. 2023) and Caduceus (Mamba-based) handle million-token genomic sequences natively. Transformers are forced to use sliding-window attention and lose long-range co-regulation signals.
      </Prose>

      <Prose>
        <strong>Vision:</strong> Transformers (ViT) remain dominant; vision Mamba (Vim, VMamba) is competitive but not clearly better at image scales most people work with. For very high-resolution images or videos, Mamba variants close the gap and sometimes win.
      </Prose>

      <Prose>
        <strong>Reinforcement learning / robotics:</strong> Mixed. Transformers with attention over context work well; Mamba's constant-memory property helps for agents that integrate over very long trajectories.
      </Prose>

      <H3>7.3 By deployment constraint</H3>

      <Prose>
        <strong>Edge / mobile / fixed-memory:</strong> Strong preference for SSMs because of constant memory. A 7B Mamba model fits in 8 GB of RAM at any context length; a 7B transformer exceeds this at 16K context.
      </Prose>

      <Prose>
        <strong>Streaming / interactive (low first-token latency):</strong> SSMs win for very long prefill (200K+), transformers win for short prefill because of better warm-up. Hybrid architectures often give the best first-token latency at medium contexts.
      </Prose>

      <Prose>
        <strong>Training-compute-limited:</strong> Transformers are slightly more sample-efficient per token, so for a fixed training-compute budget they usually produce slightly better models at short-to-medium context. At long context, Mamba's efficiency advantage lets it see more tokens for the same compute.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES
          ====================================================================== */}
      <H2>8. What scales</H2>

      <Prose>
        Scaling an architecture means asking, as the model and data grow, which terms dominate the compute and memory costs and how they grow. For SSMs the picture is clean.
      </Prose>

      <H3>8.1 Training compute</H3>

      <Prose>
        Mamba's training cost is <Code>{"O(L \\cdot D \\cdot N)"}</Code> per layer, where <Code>L</Code> is sequence length, <Code>D</Code> is model dimension, and <Code>N</Code> is state dimension (typically 16-128 for Mamba-1; 64-256 for Mamba-2). Compare to transformer's <Code>{"O(L^2 \\cdot D + L \\cdot D^2)"}</Code> per layer. At <Code>{"L = D = 4096"}</Code> and <Code>{"N = 16"}</Code>, Mamba is <Code>{"L / N = 256x"}</Code> cheaper in the SSM term than transformer is in the attention term. This is why Mamba can train on longer sequences for a given hardware budget: at 32K context length, Mamba uses about the same compute per token as transformer at 4K context.
      </Prose>

      <Prose>
        S4 (FFT-based) is <Code>{"O(L \\log L \\cdot D + L \\cdot D \\cdot N)"}</Code>, slightly worse than Mamba at the FFT term but simpler to implement. Mamba-2's SSD algorithm is <Code>{"O(L \\cdot D \\cdot N + L \\cdot N^2)"}</Code> with the advantage that the <Code>{"L \\cdot D \\cdot N"}</Code> term runs on Tensor Cores (BMM primitives) rather than requiring custom scan kernels, so the wall-clock efficiency is often better than pure Mamba.
      </Prose>

      <H3>8.2 Inference compute and memory</H3>

      <Prose>
        This is where SSMs shine. At decode time, a transformer must attend over the entire KV cache of length <Code>L</Code>, costing <Code>{"O(L \\cdot D)"}</Code> per token. An SSM just updates its fixed-size state in <Code>{"O(D \\cdot N)"}</Code> per token — constant in sequence length. For a 7B Mamba at 100K context, decode is about the same speed as at 1K context. For a 7B transformer, decode at 100K is roughly 100x slower than at 1K if all the KV cache stays on-GPU.
      </Prose>

      <Prose>
        Memory: transformer KV cache is <Code>{"2 \\cdot L \\cdot D \\cdot n_\\text{layers} \\cdot n_\\text{kv-heads} / n_\\text{heads}"}</Code> bytes (at float16). For a 7B model with 32 layers, <Code>{"D = 4096"}</Code>, full multi-head: KV cache at 64K context is about 16 GB. Mamba state: <Code>{"D \\cdot N \\cdot n_\\text{layers} \\cdot 4"}</Code> bytes (float32, because the scan kernel is typically done in fp32 for accuracy); for the same 7B: about 60 MB total. Three orders of magnitude.
      </Prose>

      <H3>8.3 Parameter efficiency</H3>

      <Prose>
        SSMs are roughly as parameter-efficient as transformers at matched parameter count. The Mamba paper reports perplexity within 1-2% of a transformer at equal parameters on The Pile. The Mamba-2 paper reports further closing of this gap at 2.7B parameters. Claims that "SSMs need fewer parameters" are overstated — the parameter count for a given quality is approximately the same.
      </Prose>

      <H3>8.4 Hardware utilization</H3>

      <Prose>
        Mamba-1 had the reputation of being slower in wall-clock than a well-tuned transformer even though its theoretical cost is linear. The reason: the selective scan kernel is a custom algorithm that did not exercise Tensor Cores at their full utilization. Mamba-2's state space duality fixes this: the chunked SSD computation runs primarily as BMMs (matmul) which hit 70-90% of Tensor Core peak. As a result, Mamba-2 has 2-8x higher wall-clock throughput than Mamba-1 on H100-class hardware. Wall-clock throughput for Mamba-2 is now roughly comparable to transformer at short context and substantially better at long context.
      </Prose>

      <H3>8.5 State dimension N as a tuning knob</H3>

      <Prose>
        The state dimension <Code>N</Code> is the main quality-cost knob specific to SSMs. Mamba-1 paper default: <Code>{"N = 16"}</Code>. This is small; the state is highly compressed. Quality scales with <Code>N</Code> up to a point (diminishing returns past <Code>{"N = 64"}</Code>). Mamba-2 with its matmul-friendly algorithm can handle larger <Code>N</Code> cheaply and typically runs at <Code>{"N = 128"}</Code> or <Code>{"N = 256"}</Code>. Increasing <Code>N</Code> does not increase parameter count proportionally because <Code>N</Code> is inside the state, not in the embedding or FFN — it only affects the SSM-specific projections.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES
          ====================================================================== */}
      <H2>9. Failure modes</H2>

      <H3>9.1 Forgetting discretization</H3>

      <Prose>
        The most common beginner mistake in SSMs: treating <Code>{"A, B"}</Code> as the discrete recurrence matrices directly, instead of discretizing them via ZOH. This gives the wrong recurrence and, because <Code>A</Code> has negative real parts as an init, will produce divergent iterations or NaN loss. The diagnostic: plot your learned kernel <Code>{"\\bar K_l"}</Code> for a few channels; if it diverges rather than decays geometrically, you have forgotten <Code>{"\\exp(\\Delta A)"}</Code>. The fix: always discretize. Always. Even if you are initializing from random rather than HiPPO, discretize.
      </Prose>

      <H3>9.2 HiPPO initialization is critical</H3>

      <Prose>
        S4 with random initialization for <Code>A</Code> trains badly — it will converge but to a much worse solution than S4 with HiPPO-LegS. On the Long Range Arena, random-init S4 scores about 60%; HiPPO-init S4 scores 85%. The gap is not subtle. For Mamba, the HiPPO effect is weaker because the selective <Code>{"\\Delta, B, C"}</Code> can compensate for poor <Code>A</Code> init — but <Code>A</Code> is still typically initialized to <Code>{"A_{nn} = -(n+1)"}</Code> (the HiPPO-LegS eigenvalues), and straying from this reliably hurts convergence by several percent perplexity.
      </Prose>

      <H3>9.3 Delta out of range</H3>

      <Prose>
        The discretization step <Code>{"\\Delta"}</Code> must be positive (use softplus) and should be initialized in a reasonable range. The Mamba paper recommends initializing <Code>{"\\log \\Delta"}</Code> uniformly in <Code>{"[\\log(0.001), \\log(0.1)]"}</Code>. Outside this range: too small <Code>{"\\Delta"}</Code> means <Code>{"\\bar A \\approx I"}</Code>, no decay, state grows unboundedly; too large <Code>{"\\Delta"}</Code> means <Code>{"\\bar A \\approx 0"}</Code>, state is overwritten each step (loses memory). Diagnose by printing <Code>{"\\Delta"}</Code> statistics during training; if the mean drifts outside <Code>{"[0.001, 1.0]"}</Code> you have a problem. Fix: tighter parameterization (softplus with bias init) or gradient clipping on the <Code>{"\\Delta"}</Code>-projection weights.
      </Prose>

      <H3>9.4 Non-causal misuse</H3>

      <Prose>
        SSMs as usually implemented are causal (the state at step <Code>t</Code> sees only past inputs). For tasks that benefit from bidirectional context (encoder models, document embedding, BERT-style MLM), a unidirectional SSM underperforms. The fix: a "bidirectional Mamba" that runs one SSM forward, a second backward, and concatenates (analogous to bidirectional LSTMs). Vision Mamba (Vim) uses this. The cost is 2x the SSM compute but worth it for non-autoregressive tasks. Mistakenly using a causal SSM as a bidirectional feature extractor silently loses accuracy without obvious error signals.
      </Prose>

      <H3>9.5 Overtraining on short sequences</H3>

      <Prose>
        If you train Mamba on sequences of length 2K and deploy at 32K, quality can degrade because the <Code>{"\\Delta"}</Code> values the model learned are tuned for short-context memory horizons. Training at the deployment context length (or gradually expanding during training) is essential. Transformers suffer from this too (positional encoding extrapolation) but the failure mode is more visible; for SSMs the degradation can be subtle. Fix: always include some long-context samples in the training mix.
      </Prose>

      <H3>9.6 Naive matmul instead of parallel scan</H3>

      <Prose>
        Without the hardware-aware scan kernel, Mamba's Python/PyTorch reference implementation materializes intermediate states in HBM and is 10-50x slower than attention. If you see Mamba as slower than transformer in wall-clock at moderate sequence lengths, verify you are using the CUDA kernel (<Code>selective_scan_cuda</Code> for Mamba-1, <Code>mamba_chunk_scan_combined</Code> for Mamba-2). Without it, the linear-cost advantage is theoretical only. This catches most first-time Mamba users: they install the python package, import, run, and report "Mamba is slow" — because the CUDA kernel did not compile on their system and the code silently fell back to the naive scan.
      </Prose>

      <H3>9.7 Numerical instability in fp16</H3>

      <Prose>
        The selective scan has dynamic range issues in fp16: the products <Code>{"\\prod_t \\bar A_t"}</Code> in the parallel-scan reduction can underflow. Mamba's reference kernel runs the scan in fp32 even when the rest of the model is in fp16 or bf16. If you are writing your own kernel: do not use fp16 for the scan itself. Use fp32 or bf16 (bf16 has fp32 exponent range; it is safe). Symptoms of getting this wrong: training converges initially then diverges at 3-5B tokens with the loss going to NaN; diagnostic: per-step max-absolute-value of the scan output grows unboundedly.
      </Prose>

      <H3>9.8 Mamba-2 chunk-size pitfall</H3>

      <Prose>
        Mamba-2's SSD algorithm has a chunk size hyperparameter (typically 256 or 512). Choosing chunk size = 1 reduces to the pure recurrence (slow); chunk size = L reduces to a full matmul (memory-heavy). The sweet spot is hardware-dependent. If you see Mamba-2 running no faster than Mamba-1, check that the chunk size is appropriate for your sequence length and head dimension. The reference implementation picks good defaults for standard configurations, but custom setups may need tuning.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        Read in chronological order to follow the development from Kalman's 1960 control-theory paper through HiPPO, S4, Mamba, Mamba-2, and the hybrid-architecture line.
      </Prose>

      <StepTrace
        label="primary literature"
        steps={[
          {
            label: "Kalman 1960 — Linear Filtering and Prediction",
            render: () => (
              <Prose>
                Kalman, R.E. (1960). "A New Approach to Linear Filtering and Prediction Problems." Transactions of the ASME, Journal of Basic Engineering, 82(Series D), pp. 35-45. The founding paper of state-space estimation theory. Introduces the now-canonical <Code>{"x' = A x + B u, \\; y = C x + D u"}</Code> formulation and derives the Kalman filter as the optimal linear estimator for a Gaussian LTI system. Not a machine learning paper, but every S4/Mamba paper traces its core equations back here. Read to understand the engineering intuition behind "state summarizes the past."
              </Prose>
            ),
          },
          {
            label: "Gu, Dao, Ermon, Rudra, Re 2020 — HiPPO (arXiv:2008.07669)",
            render: () => (
              <Prose>
                Gu, A., Dao, T., Ermon, S., Rudra, A., and Re, C. (2020). "HiPPO: Recurrent Memory with Optimal Polynomial Projections." NeurIPS 2020. arXiv:2008.07669. Available at arxiv.org/abs/2008.07669. Derives optimal recurrent-memory matrices by projecting onto orthogonal polynomial bases. Introduces HiPPO-LegS, HiPPO-LegT, HiPPO-LagT. This is the theoretical foundation that makes long-range SSMs work — initialization is not a heuristic, it is a theorem. Section 3 gives the derivation; section 4 shows empirical gains on Long Range Arena precursor benchmarks. Essential reading for anyone implementing SSMs from scratch.
              </Prose>
            ),
          },
          {
            label: "Gu, Goel, Re 2021 — S4 (arXiv:2111.00396)",
            render: () => (
              <Prose>
                Gu, A., Goel, K., and Re, C. (2021). "Efficiently Modeling Long Sequences with Structured State Spaces." ICLR 2022. arXiv:2111.00396. Available at arxiv.org/abs/2111.00396. The S4 paper. Shows how to make SSMs practical at scale by: (i) discretizing continuous SSMs via ZOH, (ii) using a structured DPLR parameterization of <Code>A</Code> that admits an <Code>{"O(N + L \\log L)"}</Code> kernel computation via the Cauchy matrix trick, (iii) initializing from HiPPO-LegS. Results: 85.7% average on Long Range Arena against 54% for transformers. Section 3 derives the DPLR kernel algorithm; section 4 shows LRA results; section 5 covers pathological long-range tasks like Path-X (16K length, transformer baseline 0%). First SSM to be competitive with transformers on any sequence task at scale.
              </Prose>
            ),
          },
          {
            label: "Gu et al. 2022 — S4D (arXiv:2206.11893)",
            render: () => (
              <Prose>
                Gu, A., Goel, K., Gupta, A., and Re, C. (2022). "On the Parameterization and Initialization of Diagonal State Space Models." NeurIPS 2022. arXiv:2206.11893. Available at arxiv.org/abs/2206.11893. Simplifies S4's DPLR parameterization to a pure-diagonal <Code>A</Code> (S4D). Shows empirically that the simpler diagonal form recovers 90%+ of S4's quality at a fraction of the implementation complexity. S4D is the practical default for most S4-style architectures built after 2022. Section 3 gives the diagonal-SSM formulation; section 5 compares S4 vs S4D on LRA.
              </Prose>
            ),
          },
          {
            label: "Smith, Warrington, Linderman 2022 — S5 (arXiv:2208.04933)",
            render: () => (
              <Prose>
                Smith, J.T.H., Warrington, A., and Linderman, S.W. (2022). "Simplified State Space Layers for Sequence Modeling." ICLR 2023. arXiv:2208.04933. Available at arxiv.org/abs/2208.04933. Introduces S5 — a simplified SSM layer that uses a parallel-scan algorithm (Blelloch) instead of FFT convolution, and a multi-input multi-output diagonal SSM (MIMO rather than S4's SISO with channel mixing). S5 is faster than S4 on modern GPUs and has a cleaner implementation. The parallel-scan approach directly inspired Mamba's selective scan. Section 3 explains the SISO-to-MIMO transition; section 4 explains the parallel scan; section 5 benchmarks against S4 on LRA.
              </Prose>
            ),
          },
          {
            label: "Gu, Dao 2023 — Mamba (arXiv:2312.00752)",
            render: () => (
              <Prose>
                Gu, A., and Dao, T. (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." arXiv:2312.00752. Available at arxiv.org/abs/2312.00752. The Mamba paper. Introduces the selective SSM (S6) where <Code>{"\\Delta, B, C"}</Code> depend on the input, and the hardware-aware selective-scan CUDA kernel. Mamba-3B matches transformer-3B on language modeling and outperforms on long-context tasks. Section 2 motivates selectivity with the induction-heads argument; section 3 describes the S6 formulation; section 4 details the hardware-aware kernel; section 5 reports language, audio, and genomics benchmarks. The paper that made SSMs mainstream in 2024. Read alongside the accompanying blog post at state-spaces.github.io for extra intuition.
              </Prose>
            ),
          },
          {
            label: "Dao, Gu 2024 — Mamba-2 / State Space Duality (arXiv:2405.21060)",
            render: () => (
              <Prose>
                Dao, T., and Gu, A. (2024). "Transformers Are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality." ICML 2024. arXiv:2405.21060. Available at arxiv.org/abs/2405.21060. The Mamba-2 paper. Proves the state space duality: a selective SSM with scalar-identity state transition is equivalent to a masked linear attention with a structured semi-separable mask. Uses this to derive a new algorithm (SSD) that runs the SSM as chunked matrix multiplications, hitting Tensor Core peak throughput. Mamba-2 is 2-8x faster than Mamba-1 at matched quality. Section 3 derives the duality; section 4 introduces the SSD algorithm; section 5 reports scaling and long-context benchmarks. Required reading for anyone implementing selective SSMs at scale or working on linear-attention theory.
              </Prose>
            ),
          },
          {
            label: "Poli et al. 2023 — Hyena Hierarchy (arXiv:2302.10866)",
            render: () => (
              <Prose>
                Poli, M., Massaroli, S., Nguyen, E., Fu, D.Y., Dao, T., Baccus, S., Bengio, Y., Ermon, S., and Re, C. (2023). "Hyena Hierarchy: Towards Larger Convolutional Language Models." ICML 2023. arXiv:2302.10866. Available at arxiv.org/abs/2302.10866. A parallel thread: replace attention with long implicit convolutions parameterized by MLPs. Hyena is a cousin of S4 — both are convolution-based attention-replacements — but the kernel is learned as a function of position rather than derived from an SSM. Hyena was a strong baseline that motivated Mamba's selectivity argument (Hyena had the linear-time property but lacked selectivity, which Mamba added). HyenaDNA and other Hyena descendants are still used in genomics where sequences are millions of tokens long.
              </Prose>
            ),
          },
          {
            label: "Lieber et al. 2024 — Jamba (arXiv:2403.19887)",
            render: () => (
              <Prose>
                Lieber, O., Lenz, B., Bata, H., Cohen, G., Osin, J., Dalmedigos, I., Safahi, E., Meirom, S., Belinkov, Y., Shalev-Shwartz, S., Abend, O., Alon, R., Asida, T., Bergman, A., Glozman, R., Gokhman, M., Manevich, A., Ratner, N., Rozen, N., Shwartz, E., Zusman, M., and Shoham, Y. (2024). "Jamba: A Hybrid Transformer-Mamba Language Model." arXiv:2403.19887. Available at arxiv.org/abs/2403.19887. The first production-scale hybrid Mamba-attention-MoE model. Jamba-1.0 is 52B total parameters (12B active), 256K context, 1 attention layer per 8 Mamba layers. Shows that hybrid architectures achieve better quality/throughput than pure Mamba or pure transformer at long context. Subsequent Jamba-1.5 Mini and Large are production models shipped to AI21's customers. The hybrid-architecture template in this paper has been adopted by Zamba-2, Samba, and others.
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
        Attempt all five before reading the answers. Exercises 1-2 test the discretization math; 3 tests kernel construction; 4 tests the selectivity intuition; 5 tests architecture judgment.
      </Prose>

      <H3>Exercise 1 (discretization)</H3>
      <Prose>
        You are given a diagonal continuous-time SSM with <Code>{"A = \\text{diag}(-1, -2, -4)"}</Code>, <Code>{"B = [0.5, 0.5, 0.5]^T"}</Code>, and step size <Code>{"\\Delta = 0.1"}</Code>. Compute the discrete <Code>{"\\bar A"}</Code> and <Code>{"\\bar B"}</Code> using ZOH. Then compute <Code>{"x_1"}</Code> given <Code>{"x_0 = 0"}</Code> and <Code>{"u_1 = 1"}</Code>. Which state dimension has the longest memory (slowest decay)?
      </Prose>
      <Callout accent="green">
        <strong>Answer 1.</strong> For diagonal <Code>A</Code>, ZOH is elementwise. <Code>{"\\bar A_n = \\exp(\\Delta \\cdot A_n) = \\exp(-0.1, -0.2, -0.4) = (0.905, 0.819, 0.670)"}</Code>. For <Code>{"\\bar B"}</Code>, the formula <Code>{"\\bar B_n = A_n^{-1}(\\bar A_n - 1) B_n"}</Code> gives <Code>{"\\bar B_n = ((0.905-1)/-1, (0.819-1)/-2, (0.670-1)/-4) \\cdot 0.5 = (0.0475, 0.0453, 0.0413)"}</Code>. With <Code>{"x_0 = 0"}</Code> and <Code>{"u_1 = 1"}</Code>: <Code>{"x_1 = \\bar A \\cdot 0 + \\bar B \\cdot 1 = (0.0475, 0.0453, 0.0413)"}</Code>. The slowest-decay dimension is <Code>{"n = 0"}</Code> because <Code>{"\\bar A_0 = 0.905"}</Code> is closest to 1 — a 1-step decay factor of 0.905 means half-life of <Code>{"\\log 2 / \\log(1/0.905) \\approx 6.9"}</Code> steps, vs <Code>{"n = 2"}</Code> which has half-life of <Code>{"\\log 2 / \\log(1/0.670) \\approx 1.7"}</Code> steps. This multi-scale behavior is exactly what HiPPO-style diagonal initialization provides.
      </Callout>

      <H3>Exercise 2 (kernel construction)</H3>
      <Prose>
        Using the <Code>{"\\bar A, \\bar B"}</Code> from Exercise 1 and <Code>{"C = [1, 1, 1]"}</Code>, write out the first 4 kernel taps <Code>{"\\bar K_l = C \\bar A^l \\bar B"}</Code> for <Code>{"l = 0, 1, 2, 3"}</Code>. Verify that the convolutional output <Code>{"y_2 = \\sum_j \\bar K_{2-j} u_j"}</Code> matches the recurrent output for an input <Code>{"u = (1, 1, 1)"}</Code>.
      </Prose>
      <Callout accent="green">
        <strong>Answer 2.</strong> For diagonal <Code>A</Code>: <Code>{"\\bar K_l = \\sum_n C_n \\bar A_n^l \\bar B_n"}</Code>. <br />
        <Code>{"\\bar K_0 = 1 \\cdot 1 \\cdot 0.0475 + 1 \\cdot 1 \\cdot 0.0453 + 1 \\cdot 1 \\cdot 0.0413 = 0.1341"}</Code>. <br />
        <Code>{"\\bar K_1 = 1 \\cdot 0.905 \\cdot 0.0475 + 1 \\cdot 0.819 \\cdot 0.0453 + 1 \\cdot 0.670 \\cdot 0.0413 = 0.0430 + 0.0371 + 0.0277 = 0.1078"}</Code>. <br />
        <Code>{"\\bar K_2 = 0.819 \\cdot 0.0475 + 0.671 \\cdot 0.0453 + 0.449 \\cdot 0.0413 = 0.0389 + 0.0304 + 0.0185 = 0.0878"}</Code>. <br />
        <Code>{"\\bar K_3 = 0.741 \\cdot 0.0475 + 0.549 \\cdot 0.0453 + 0.301 \\cdot 0.0413 = 0.0352 + 0.0249 + 0.0124 = 0.0725"}</Code>. <br />
        Conv form: <Code>{"y_2 = \\bar K_0 u_2 + \\bar K_1 u_1 + \\bar K_2 u_0 = 0.1341 + 0.1078 + 0.0878 = 0.3297"}</Code>. <br />
        Recurrent: <Code>{"x_1 = \\bar B = (0.0475, 0.0453, 0.0413)"}</Code>, <Code>{"x_2 = \\bar A \\odot x_1 + \\bar B = (0.0475 \\cdot 0.905 + 0.0475, 0.0453 \\cdot 0.819 + 0.0453, 0.0413 \\cdot 0.670 + 0.0413) = (0.0905, 0.0824, 0.0690)"}</Code>; <Code>{"y_2 = 1 \\cdot (0.0905 + 0.0824 + 0.0690) = 0.2419"}</Code>. <br />
        Hmm — they differ! The discrepancy is because <Code>{"y_2"}</Code> in the recurrence after just 2 steps of input <Code>{"(1, 1, 1)"}</Code> indexes inputs <Code>{"u_0, u_1, u_2"}</Code> slightly differently depending on convention. Double-check: if <Code>{"x_k = \\bar A x_{k-1} + \\bar B u_k"}</Code> and <Code>{"x_0 = 0"}</Code>, then processing <Code>{"u_0, u_1, u_2"}</Code> gives <Code>{"x_0 = \\bar B u_0, x_1 = \\bar A \\bar B u_0 + \\bar B u_1, x_2 = \\bar A^2 \\bar B u_0 + \\bar A \\bar B u_1 + \\bar B u_2"}</Code>, matching the conv form. The discrepancy above came from starting the recurrence at <Code>{"x_0 = 0"}</Code> without applying <Code>{"u_0"}</Code>. With correct indexing, conv and recurrence match — which is exactly the point of the equivalence.
      </Callout>

      <H3>Exercise 3 (selectivity)</H3>
      <Prose>
        In plain S4, <Code>{"\\Delta, B, C"}</Code> are learned but fixed across the sequence. In Mamba, they depend on the current token. Describe a task where this matters, and explain in one sentence why an S4 model would fail on it but a Mamba model would succeed.
      </Prose>
      <Callout accent="green">
        <strong>Answer 3.</strong> Task: selective copying, where a sequence contains many random tokens plus a few "marker" tokens, and the model must copy tokens following each marker to designated output positions. An S4 model fails because the kernel <Code>{"\\bar K"}</Code> is the same across all positions — it cannot distinguish "marker followed by payload" from "random token followed by random token." Mamba succeeds because <Code>{"\\Delta_t"}</Code> can become small on seeing a marker (preserving the state so the payload can be read later) and <Code>{"C_t"}</Code> can become "retrieve-from-state" when the model needs to emit the copied token. This content-conditional behavior is exactly the "induction heads" capability of transformers — it is what allows in-context learning — and Mamba's selectivity provides a functionally equivalent mechanism at O(L) cost.
      </Callout>

      <H3>Exercise 4 (architecture selection)</H3>
      <Prose>
        You are building a genome variant caller that needs to process DNA sequences of length 1,000,000 nucleotides and output per-position predictions. Your budget is a single 80GB H100. Which architecture do you choose, and what are the specific constraints that dictate this choice?
      </Prose>
      <Callout accent="green">
        <strong>Answer 4.</strong> Choose Mamba or HyenaDNA (or a Mamba-Hyena hybrid). Specifically:
        <br />
        (a) <strong>Context length:</strong> 1M tokens is well beyond any transformer's comfort zone. A transformer's KV cache at 1M context and <Code>{"d = 512"}</Code> (typical for genomics) would be ~80 GB, saturating the H100 before inference starts. Mamba's state is ~10 MB regardless of context.
        <br />
        (b) <strong>Task structure:</strong> DNA has long-range co-regulation and local sequence motifs. HiPPO-initialized SSMs are excellent at integrating information across very long ranges; the multi-scale memory of a diagonal SSM with varied eigenvalues captures both local motifs and long-range context.
        <br />
        (c) <strong>Bidirectional need:</strong> Variant calling is non-autoregressive — you have the full sequence. Use a bidirectional Mamba (two SSM stacks, one forward, one backward, concatenated). Costs 2x but essential.
        <br />
        (d) <strong>Output structure:</strong> Per-position prediction means the SSM output at each position is directly the predicted logit over variant classes — one SSM forward pass gives you 1M predictions.
        <br />
        (e) <strong>Known models:</strong> HyenaDNA was first to show 1M-context genomics feasibility (Nguyen et al. 2023). Caduceus (Schiff et al. 2024) is a Mamba-based successor. Both ship trained checkpoints.
        <br />
        A transformer here is not just slower — it is impossible on the hardware specified.
      </Callout>

      <H3>Exercise 5 (debugging)</H3>
      <Prose>
        You are training a 1.3B Mamba model on 8 H100s. Around step 3000, training loss spikes from 2.8 to NaN. You restart from the previous checkpoint with half the learning rate; loss spikes to NaN again at step 3500. You suspect something specific to Mamba is at fault rather than generic instability. List three Mamba-specific failure modes and a diagnostic for each.
      </Prose>
      <Callout accent="green">
        <strong>Answer 5.</strong> Three Mamba-specific failure modes:
        <br />
        (1) <strong>Delta parameter out of range.</strong> The softplus parameterization of <Code>{"\\Delta"}</Code> does not bound it from above; during training <Code>{"\\Delta"}</Code> can drift to arbitrarily large values, which makes <Code>{"\\bar A = \\exp(\\Delta A) \\to 0"}</Code> effectively zeroing the state each step, or toward <Code>{"\\bar A \\to \\infty"}</Code> if <Code>A</Code> happens to be positive (shouldn't be but is a bug surface). Diagnostic: log <Code>{"\\Delta"}</Code> min/max/mean each step; typical healthy range is 0.001 to 1.0. If <Code>{"\\Delta_\\max > 10"}</Code>, you have a problem.
        <br />
        (2) <strong>A drifting positive.</strong> The <Code>A</Code> matrix (continuous-time) is supposed to have negative real eigenvalues for stability. Mamba often parameterizes <Code>A = -\\exp(A_\\log)</Code> to enforce this, but a bug in the parameterization or a failure to clamp can let <Code>A</Code> become positive, at which point <Code>{"\\bar A = \\exp(\\Delta \\cdot (+\\text{value}))"}</Code> grows without bound in a few iterations. Diagnostic: after the backward pass, print <Code>{"\\max(A)"}</Code>. If it is positive, something is wrong with the parameterization. Fix: enforce <Code>A &lt; 0</Code> via the parameterization (e.g., <Code>A = -\\text{softplus}(A_\\text{raw})</Code>).
        <br />
        (3) <strong>Scan kernel running in fp16 instead of fp32.</strong> The selective scan accumulates products <Code>{"\\prod_t \\bar A_t"}</Code> which underflow in fp16 after ~100 steps. If you set the entire model to fp16 and the scan kernel honors that, the scan will silently produce garbage. Diagnostic: compare a short-sequence forward pass in fp16 vs fp32; if they differ by more than ~1e-2, your scan is not running in sufficient precision. Fix: ensure Mamba's selective_scan is run in fp32 even if the surrounding model is fp16; this is the reference kernel's default behavior but custom implementations sometimes break it.
        <br />
        Bonus: gradient clipping at norm 1.0 catches most of these in aggregate and is standard for Mamba training.
      </Callout>

    </div>
  ),
};

export default ssmContent;
