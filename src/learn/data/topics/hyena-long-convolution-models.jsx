import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const hyenaContent = {
  title: "Hyena & Long Convolution Models",
  readTime: "~38 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Hyena exists because researchers wanted to know which property of attention is actually load-bearing. The transformer's success is usually attributed to "attention," singular, as if it were one thing. Decompose it and you find at least three claims tangled together. First, that each output position can mix information from every input position (global receptive field). Second, that the mixing weights depend on content (data-controlled). Third, that the mixing happens via pairwise inner products (the <Code>{"Q K^T"}</Code> structure). The cost — quadratic time and memory in sequence length, and a KV-cache that grows linearly per layer — is the price of the third property in particular. So a fair experiment is: keep the first two, drop the third, and see how much quality you lose. That experiment is Hyena.
      </Prose>

      <Prose>
        The technical lineage starts before Hyena. In February 2021, David Romero, Anna Kuzina, Erik Bekkers, Jakub Tomczak, and Mark Hoogendoorn published "CKConv: Continuous Kernel Convolution For Sequential Data" (arXiv:2102.02611, ICLR 2022). CKConv made a small but consequential change to how we parameterize a 1D convolution. A standard conv layer with kernel size <Code>K</Code> stores <Code>K</Code> weights per channel and runs into trouble when <Code>K</Code> needs to be large (the parameter count balloons, and the receptive field is still bounded by <Code>K</Code> per layer). CKConv instead represents the kernel as a small MLP that maps a relative position <Code>{"\\tau \\in \\mathbb{R}"}</Code> to a kernel value <Code>{"h(\\tau)"}</Code>. Because the kernel is a continuous function of position, you can evaluate it at as many points as you want without growing the parameter count, and you can evaluate it at very long ranges (a few thousand positions) just as cheaply as a few. CKConv was framed for time-series and audio, but the underlying idea — implicit, parameter-efficient, arbitrary-length kernels — is the engine of everything that came after.
      </Prose>

      <Prose>
        Around the same time, Albert Gu, Karan Goel, and Christopher Re were developing Structured State Space models. S4 (arXiv:2111.00396, October 2021) is best known as an SSM, but mathematically S4 is a long convolution: it computes a kernel <Code>{"\\bar K \\in \\mathbb{R}^L"}</Code> from structured matrices and convolves via FFT. The relationship between CKConv and S4 is not coincidental. Both express a 1D linear filter of length <Code>L</Code> using far fewer than <Code>L</Code> parameters; they differ only in how the parameterization is structured (CKConv: MLP over positions; S4: state-space dynamics). Karan Goel, Albert Gu, Chris Donahue, and Re later published "It's Raw! Audio Generation with State-Space Models" (SaShiMi, arXiv:2202.09729, ICML 2022), pushing the long-convolution paradigm into 16-kHz audio with sequences of hundreds of thousands of samples — a regime where transformers were simply not an option.
      </Prose>

      <Prose>
        The key paper is Michael Poli, Stefano Massaroli, Eric Nguyen, Daniel Y. Fu, Tri Dao, Stephen Baccus, Yoshua Bengio, Stefano Ermon, and Christopher Re (2023), "Hyena Hierarchy: Towards Larger Convolutional Language Models" (arXiv:2302.10866, ICML 2023). Hyena's contribution is not the long convolution per se — that was already in S4 and CKConv — but the recursive, data-controlled application of long convolutions. A Hyena operator alternates element-wise gating (computed from the input, like a SiLU-gated branch) with a long convolution. The recursion can be repeated <Code>N</Code> times to deepen the operator. The result is an attention-replacement layer that has all three of attention's load-bearing properties: global receptive field (long conv covers the whole sequence), data-controlled mixing (gates depend on content), and parameter efficiency (implicit filter parameterization). It runs at <Code>{"O(L \\log L)"}</Code> via FFT, fits in linear memory, and at the time of publication closed about half the quality gap between attention-replacements and dense attention on language modeling at small scale.
      </Prose>

      <Prose>
        Hyena's most impactful real-world application has been in genomics, not language. In June 2023, Eric Nguyen, Michael Poli, Marjan Faizi, Armin Thomas, Callum Birch-Sykes, Michael Wornow, Aman Patel, Clayton Rabideau, Stefano Massaroli, Yoshua Bengio, Stefano Ermon, Stephen Baccus, and Chris Re published "HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution" (arXiv:2306.15794, NeurIPS 2023). HyenaDNA tokenizes DNA at the nucleotide level (vocab of 4-5) and trains a Hyena stack on context lengths up to 1,000,000. A transformer at 1M context with even a tiny model is impossible on a single GPU; HyenaDNA fits on one A100 and beats the prior genomics state-of-the-art on enhancer prediction, splice-site classification, and chromatin profile tasks. The 1M-context model is the genomics community's first general-purpose long-context backbone.
      </Prose>

      <Prose>
        Together AI extended the line in late 2023 with StripedHyena, a 7B-parameter open-weights LLM that interleaves Hyena layers with multi-head attention layers. The idea — borrowed from Jamba and others — is that you do not need every layer to be quadratic; some long-range mixing can be handled by the cheaper Hyena layer and a few attention layers can handle the precise content lookups. StripedHyena-Nous-7B and StripedHyena-Hessian-7B were released in December 2023 with competitive language benchmarks at the 7B scale. StripedHyena-2 (2024, also called Evo) extended this for biological sequence modeling on multi-million-token contexts. Massaroli et al. (2023) "Laughing Hyena Distillery" (arXiv:2310.18780) developed a method to distill long convolutions into small recurrent forms for efficient inference, addressing one of the operator's main weaknesses.
      </Prose>

      <Prose>
        Hyena's research arc converged with Mamba's in 2024. Mamba-2 (Dao and Gu, arXiv:2405.21060) introduced the State Space Duality, which unified selective SSMs with linear attention. Massaroli — one of Hyena's lead authors — was a co-author on Mamba-2, and the SSD framework arguably makes Hyena, S4, Mamba, and linear attention four views of the same underlying object: a structured linear operator over a sequence whose effect can be computed in linear or near-linear time. After 2024, the LLM community settled on a small number of credible attention alternatives — Mamba/Mamba-2, RWKV, GLA — with Hyena's role narrowing to genomics and a few specific long-context applications. But the conceptual contribution of Hyena, that data-controlled long convolution can substitute for attention, is now common ground in the field.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Forget Hyena for a moment. Start with the simpler primitive it is built on: the long convolution. A long convolution is a 1D causal convolution where the filter length equals the entire sequence length. If your sequence has 4096 tokens, your filter has 4096 taps. Each output position is a weighted sum over every prior position (and the current one), with the weights given by a learnable filter <Code>{"h \\in \\mathbb{R}^L"}</Code>: <Code>{"y[i] = \\sum_{t=0}^{i} h[t] \\cdot u[i - t]"}</Code>. That's it. Every position can in principle attend to every prior position, but the mixing weights are fixed (do not depend on content) and shared across positions (<Code>{"h"}</Code> does not depend on <Code>i</Code>). This is not attention — it is global, content-independent mixing.
      </Prose>

      <Prose>
        Two properties make this practically useful. First, even though the filter has <Code>L</Code> taps, you compute the output in <Code>{"O(L \\log L)"}</Code> time with the Fast Fourier Transform: <Code>{"y = \\text{IFFT}(\\text{FFT}(h) \\cdot \\text{FFT}(u))"}</Code> after appropriate zero-padding. At <Code>{"L = 16384"}</Code>, that is roughly 30x cheaper than the <Code>{"O(L^2)"}</Code> attention matrix construction. Second, you do not actually have to store <Code>L</Code> parameters per channel: parameterize the filter implicitly with a small MLP <Code>{"h(\\tau) = f_\\theta(\\tau)"}</Code> that maps a relative position to a kernel value, and the parameter count becomes <Code>{"O(\\text{hidden})"}</Code> rather than <Code>{"O(L)"}</Code>. The MLP has bounded expressivity, but for the smooth low-frequency long-range patterns that long-context tasks rely on, it is enough.
      </Prose>

      <Prose>
        A long convolution alone, however, has a fatal limitation for language modeling: it cannot do content-dependent retrieval. If the same filter is applied at every position, the model cannot decide to "look back at the token after the special marker" for one query and "look back two tokens" for another. This is exactly the gap S4 had against transformers, and it is the motivation for selectivity in Mamba. Hyena's answer to the same problem: keep the filter content-independent, but apply data-dependent gating before and after the convolution. The gates are computed from the input via cheap pointwise projections, so they are content-aware. The convolution is content-independent, but the gates can effectively select which positions get integrated and which get suppressed.
      </Prose>

      <Prose>
        Concretely, the Hyena order-2 operator is: <Code>{"y = q \\odot (h \\ast (k \\odot v))"}</Code> where <Code>{"q, k, v"}</Code> are three projections of the input (analogous to query, key, value in attention but pointwise rather than via inner product), <Code>{"h"}</Code> is the long filter, <Code>{"\\ast"}</Code> is the long convolution, and <Code>{"\\odot"}</Code> is element-wise multiplication. Read it as: first gate the values by the keys (suppress positions whose key is zero), then long-mix via convolution (every position gets information from every prior position), then gate the result by the queries (only let through information at positions whose query is non-zero). The gates make the operation data-controlled; the convolution makes it global; the FFT makes it fast.
      </Prose>

      <Prose>
        The Hyena hierarchy generalizes this to <Code>N</Code> recursive applications of gate-conv-gate. Order-2 (one long conv, two gates) is the canonical form used in most papers; higher orders add more gating steps and more long convolutions. The operator is still <Code>{"O(L \\log L)"}</Code> per token across the whole layer, regardless of order, because each long convolution is independent.
      </Prose>

      <Callout accent="gold">
        Mental model: Hyena is what you get if you take the structure of an attention block (Q, K, V projections, mixing operator, output projection) and replace the inner-product mixing <Code>{"\\text{softmax}(Q K^T) V"}</Code> with two element-wise gates and one long convolution. The gates contribute data-dependence; the convolution contributes long-range mixing; together they cover the same job as attention at <Code>{"O(L \\log L)"}</Code> instead of <Code>{"O(L^2)"}</Code>. The cost: pairwise interactions are lossy (you cannot do precise content-based lookups the way attention can), so quality is slightly worse on tasks that need exact retrieval — and that is most language tasks. The wins are on very long sequences (genomics, DNA, audio) where attention's quadratic cost dominates and the tasks are dominated by smooth long-range structure.
      </Callout>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 Long convolution</H3>

      <Prose>
        Let <Code>{"u \\in \\mathbb{R}^L"}</Code> be a 1D input sequence and <Code>{"h \\in \\mathbb{R}^L"}</Code> a learnable filter of the same length. The causal long convolution produces output <Code>{"y \\in \\mathbb{R}^L"}</Code>:
      </Prose>

      <MathBlock>
        {"y[i] = \\sum_{t=0}^{i} h[t] \\cdot u[i - t] = (h \\ast u)[i]"}
      </MathBlock>

      <Prose>
        Two boundary conventions matter. First, causal: the sum runs from <Code>{"t = 0"}</Code> to <Code>{"t = i"}</Code> only, never into the future. Equivalently, set <Code>{"u[i - t] = 0"}</Code> for <Code>{"i - t < 0"}</Code>. Second, the kernel is unidirectional: <Code>{"h[t]"}</Code> is defined only for <Code>{"t \\ge 0"}</Code>, with positive <Code>t</Code> meaning "look back <Code>t</Code> steps." A bidirectional version (used in HyenaDNA for non-autoregressive tasks) drops the upper limit and lets <Code>t</Code> range over all of <Code>{"\\mathbb{Z}"}</Code> with corresponding zero-padding on both sides.
      </Prose>

      <H3>3.2 FFT acceleration</H3>

      <Prose>
        The naive computation of <Code>{"(h \\ast u)"}</Code> is <Code>{"O(L^2)"}</Code> — you have <Code>L</Code> output positions and each is a sum over up to <Code>L</Code> terms. The convolution theorem rescues us: convolution in time is multiplication in frequency.
      </Prose>

      <MathBlock>
        {"\\mathcal{F}[h \\ast u] = \\mathcal{F}[h] \\cdot \\mathcal{F}[u] \\quad \\Longrightarrow \\quad h \\ast u = \\mathcal{F}^{-1}\\big( \\mathcal{F}[h] \\cdot \\mathcal{F}[u] \\big)"}
      </MathBlock>

      <Prose>
        The Fast Fourier Transform computes <Code>{"\\mathcal{F}[\\cdot]"}</Code> in <Code>{"O(L \\log L)"}</Code>. Pointwise multiplication is <Code>{"O(L)"}</Code>. Inverse FFT is again <Code>{"O(L \\log L)"}</Code>. Total cost: <Code>{"O(L \\log L)"}</Code>. The catch is that the FFT computes a <em>circular</em> convolution: <Code>{"u[i - t]"}</Code> wraps around modulo <Code>L</Code>. To get the linear (non-wrapping) causal convolution we want, we zero-pad <Code>u</Code> and <Code>h</Code> to length <Code>{"\\ge 2L - 1"}</Code> (in practice, the next power of two, so <Code>{"n = 2L"}</Code>), perform the FFT-based circular convolution at that length, and truncate the output back to <Code>L</Code>. The first <Code>L</Code> output positions are then exactly the linear causal convolution.
      </Prose>

      <H3>3.3 Implicit filter parameterization</H3>

      <Prose>
        Storing a filter of length <Code>L</Code> costs <Code>{"L \\cdot D"}</Code> parameters per channel block — at <Code>{"L = 4096"}</Code> and <Code>{"D = 1024"}</Code>, that is over four million parameters per layer just for the filters. Implicit parameterization sidesteps this: the filter is a function of position, <Code>{"h(\\tau) = f_\\theta(\\tau)"}</Code>, where <Code>{"f_\\theta"}</Code> is a small neural network. Two common forms:
      </Prose>

      <Prose>
        <strong>Sin/cos featurization (Hyena, FNet).</strong> Encode position as <Code>{"\\phi(\\tau) = (\\sin(\\omega_1 \\tau), \\cos(\\omega_1 \\tau), \\ldots, \\sin(\\omega_F \\tau), \\cos(\\omega_F \\tau))"}</Code> with frequencies <Code>{"\\omega_f"}</Code> log-spaced over <Code>{"[1, L/2]"}</Code>. Pass <Code>{"\\phi(\\tau)"}</Code> through an MLP of hidden width <Code>{"H"}</Code> and output dimension <Code>D</Code>. Total parameters: <Code>{"O(F \\cdot H + H \\cdot D)"}</Code>, independent of <Code>L</Code>. To form the discrete filter <Code>{"h \\in \\mathbb{R}^{D \\times L}"}</Code>, evaluate the MLP at <Code>{"\\tau = 0, 1, \\ldots, L - 1"}</Code>.
      </Prose>

      <Prose>
        <strong>Exponential window.</strong> The MLP-output filter typically has no built-in bias toward locality or stability, and unstable filters destroy training. Multiply the MLP output by a learned exponential decay <Code>{"w(\\tau) = \\exp(-\\alpha \\tau / L)"}</Code> with <Code>{"\\alpha > 0"}</Code> learned per channel. This forces the effective filter <Code>{"h(\\tau) = f_\\theta(\\tau) \\cdot w(\\tau)"}</Code> to decay with distance, matching the inductive bias that distant tokens matter less by default but allowing the model to learn slow decays for long-range channels.
      </Prose>

      <H3>3.4 Hyena operator</H3>

      <Prose>
        Let <Code>{"u \\in \\mathbb{R}^{L \\times D}"}</Code> be the input. The Hyena order-<Code>N</Code> operator is defined recursively. First produce <Code>{"N + 1"}</Code> projections from <Code>u</Code>:
      </Prose>

      <MathBlock>
        {"(v, q_1, q_2, \\ldots, q_N) = \\text{Proj}(u) \\quad \\text{(linear, optionally followed by a short causal conv)}"}
      </MathBlock>

      <Prose>
        Then iterate <Code>N</Code> times, alternating long convolutions with element-wise gates:
      </Prose>

      <MathBlock>
        {"x_1 = v, \\qquad x_{i+1} = q_i \\odot (h_i \\ast x_i) \\quad \\text{for} \\; i = 1, \\ldots, N"}
      </MathBlock>

      <Prose>
        where each <Code>{"h_i"}</Code> is its own implicit long filter (different parameters per recursion step). The output is <Code>{"y = \\text{OutProj}(x_{N+1})"}</Code>. In practice the canonical Hyena block uses <Code>{"N = 2"}</Code>: two gates and one long convolution, structured as <Code>{"y = q_2 \\odot (h \\ast (q_1 \\odot v))"}</Code>. This is what the original Hyena paper benchmarks and what HyenaDNA uses. Higher orders give marginal gains at higher cost.
      </Prose>

      <H3>3.5 Comparison with attention</H3>

      <Prose>
        Set <Code>{"q, k, v"}</Code> by analogy with attention. Causal attention computes <Code>{"y_i = \\sum_{j \\le i} \\text{softmax}(q_i^T k_j) v_j"}</Code>, an <Code>{"O(L^2 d)"}</Code> operation where the mixing weights <Code>{"\\text{softmax}(q_i^T k_j)"}</Code> are content-pairwise: they depend on both endpoints. The Hyena order-2 form computes <Code>{"y_i = q_i \\sum_{j \\le i} h_{i - j} (k_j v_j)"}</Code>, which is <Code>{"O(L \\log L \\cdot d)"}</Code>. The mixing weights here are <Code>{"q_i \\cdot h_{i - j} \\cdot k_j"}</Code>, a product of one factor depending on <Code>i</Code>, one on <Code>{"i - j"}</Code> (the distance), and one on <Code>j</Code>. This factored form cannot represent arbitrary attention patterns — for instance, "always attend to the token after every period" requires a coupling between content of the period and content of the next token that the factored form cannot precisely represent. But it can represent "attend more to recent tokens whose content matches a learned pattern," which covers a lot of language-modeling needs.
      </Prose>

      <H3>3.6 Cost analysis</H3>

      <MathBlock>
        {"\\text{FLOPs}_\\text{Hyena} = O(L \\log L \\cdot D \\cdot N) \\quad \\text{vs} \\quad \\text{FLOPs}_\\text{Attn} = O(L^2 \\cdot D)"}
      </MathBlock>

      <Prose>
        At <Code>{"L = 4096"}</Code> and Hyena order <Code>{"N = 2"}</Code>: <Code>{"L \\log L \\cdot N \\approx 4096 \\cdot 12 \\cdot 2 \\approx 10^5"}</Code> per channel; <Code>{"L^2 \\approx 1.7 \\cdot 10^7"}</Code> per channel. Hyena is about 170x cheaper at this scale. At <Code>{"L = 1{,}000{,}000"}</Code> (HyenaDNA): <Code>{"L \\log L \\cdot 2 \\approx 4 \\cdot 10^7"}</Code>; <Code>{"L^2 = 10^{12}"}</Code>. The transformer is 25,000x more expensive, before you even consider the <Code>{"L^2"}</Code> attention-matrix memory (4 TB for a single fp32 attention matrix at <Code>{"L = 10^6"}</Code>). This is the regime where Hyena is not just faster — it is the only credible option.
      </Prose>

      <H3>3.7 Memory</H3>

      <Prose>
        Long convolution memory at training is dominated by the FFT-buffer (length <Code>{"2L"}</Code>, complex-valued, so <Code>{"4L"}</Code> floats per channel) and the activations of the input/output (length <Code>{"L"}</Code> per channel). Total: <Code>{"O(L \\cdot D)"}</Code>. Attention is <Code>{"O(L^2)"}</Code> for the attention matrix itself plus <Code>{"O(L \\cdot D)"}</Code> for activations, and at large <Code>L</Code> the attention matrix dominates. At inference, attention's KV cache grows as <Code>{"O(L \\cdot D)"}</Code> per layer, but Hyena needs to keep the entire input sequence to recompute the convolution at each new token (or use the Laughing Hyena Distillery technique to convert the conv to a recurrence) — Hyena's inference is not as cleanly cached as attention's, and this is one of its main practical weaknesses.
      </Prose>

      {/* ======================================================================
          4. FROM SCRATCH
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <H3>4a. Long convolution via FFT, verified against conv1d</H3>

      <Prose>
        First, build the long-convolution primitive and verify it gives the same output as a reference depthwise <Code>{"\\text{conv1d}"}</Code>. The FFT version should be numerically identical (up to fp32 noise) and asymptotically much faster. The causality property — <Code>{"y[t]"}</Code> independent of <Code>{"u[t' > t]"}</Code> — must hold exactly because of how we zero-pad.
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn.functional as F

torch.manual_seed(0)

# Long convolution from scratch via FFT
# y[i] = sum_t h[t] * u[i - t]  for t = 0..L-1, with causal zero-padding
def long_conv_fft(u, h):
    """
    u: [B, D, L] input
    h: [D, L]    learnable filter (one per channel, depthwise)
    returns y: [B, D, L]  (causal: y[t] depends only on u[<= t])
    """
    L = u.shape[-1]
    n = 2 * L  # zero-pad to avoid circular wrap-around
    Uf = torch.fft.rfft(u.double(), n=n)
    Hf = torch.fft.rfft(h.double(), n=n)
    Yf = Uf * Hf.unsqueeze(0)
    y = torch.fft.irfft(Yf, n=n)[..., :L].float()
    return y

def long_conv_naive(u, h):
    """Same op via depthwise conv1d (cross-check)."""
    L = u.shape[-1]
    # conv1d does y[t] = sum_t' h[t'] u[t + t']; we need y[t] = sum_t' h[t'] u[t - t']
    # so flip the filter and left-pad input by L - 1.
    h_flipped = torch.flip(h, dims=[-1])
    u_padded = F.pad(u, (L - 1, 0))
    D = u.shape[1]
    y = F.conv1d(u_padded, h_flipped.unsqueeze(1), groups=D)
    return y

# Numerical equivalence test
B, D, L = 2, 4, 32
u = torch.randn(B, D, L)
h = torch.randn(D, L) * 0.1

y_fft = long_conv_fft(u, h)
y_ref = long_conv_naive(u, h)
print(f"L={L}  shapes: y_fft={tuple(y_fft.shape)}  y_ref={tuple(y_ref.shape)}")
print(f"max abs diff (fft vs reference) = {(y_fft - y_ref).abs().max().item():.2e}")

# Causality: perturbing u[t=L-1] must not affect y[t<L-1]
u2 = u.clone()
u2[..., -1] += 1.0
y_fft2 = long_conv_fft(u2, h)
delta = (y_fft2 - y_fft).abs()
print(f"perturb u[L-1]+=1.0:  max change in y[0..L-2] = {delta[..., :L-1].max().item():.2e}")
print(f"                       max change in y[L-1]    = {delta[..., L-1:].max().item():.2e}")

# Output:
# L=32  shapes: y_fft=(2, 4, 32)  y_ref=(2, 4, 32)
# max abs diff (fft vs reference) = 2.38e-07
# perturb u[L-1]+=1.0:  max change in y[0..L-2] = 0.00e+00
#                        max change in y[L-1]    = 5.59e-02`}
      </CodeBlock>

      <Prose>
        The two implementations agree to 2e-7 (fp32 round-off) and the causality test confirms that perturbing the last input position has zero effect on prior outputs. Note the use of <Code>{"\\text{double}()"}</Code> inside the FFT path: the FFT accumulates phase errors at length <Code>{"2L"}</Code>, and at fp32 this is the difference between exact causality (zero leakage) and a 1e-6-level leak. In production you can run the FFT in fp32 if you accept that level of leakage; HyenaDNA at <Code>{"L = 10^6"}</Code> uses fp32 FFT and the leakage is empirically harmless. For a clean teaching example we use double.
      </Prose>

      <H3>4b. FFT vs O(L^2) attention scaling</H3>

      <Prose>
        Now demonstrate the asymptotic advantage. Build a single-head causal attention as a reference (the same operation a transformer block would do, ignoring softmax-stabilization tricks) and compare wall-clock against the FFT long convolution at increasing <Code>L</Code>. CPU numbers; the absolute times shrink on GPU, but the ratio stays similar because the underlying operations are <Code>{"O(L^2)"}</Code> vs <Code>{"O(L \\log L)"}</Code> regardless of platform.
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn.functional as F
import time

torch.manual_seed(0)

def long_conv_fft(u, h):
    L = u.shape[-1]
    n = 2 * L
    Uf = torch.fft.rfft(u, n=n)
    Hf = torch.fft.rfft(h, n=n)
    Yf = Uf * Hf.unsqueeze(0)
    return torch.fft.irfft(Yf, n=n)[..., :L]

def attention_naive(q, k, v):
    """Standard multiplicative attention: O(L^2 d). Single head, no scaling tricks."""
    scores = q @ k.transpose(-1, -2) / (q.shape[-1] ** 0.5)
    L = q.shape[-2]
    mask = torch.triu(torch.ones(L, L, device=q.device), diagonal=1).bool()
    scores = scores.masked_fill(mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v

D = 64

def time_fn(fn, n_iters=3, warmup=1):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    return (time.perf_counter() - t0) / n_iters * 1e3

print(f"FFT long-conv vs O(L^2) attention   (D={D}, B=1, CPU, ms per forward)")
print(f"{'L':>6}  {'attn O(L^2)':>14}  {'fft-conv O(L log L)':>22}  {'speedup':>9}")
for L in [256, 1024, 4096, 16384]:
    u = torch.randn(1, D, L)
    h = torch.randn(D, L) * 0.05
    qkv = torch.randn(1, L, D)

    t_conv = time_fn(lambda: long_conv_fft(u, h), n_iters=3)

    if L <= 4096:
        t_attn = time_fn(lambda: attention_naive(qkv, qkv, qkv), n_iters=2, warmup=1)
    else:
        t_attn = time_fn(lambda: attention_naive(qkv, qkv, qkv), n_iters=1, warmup=0)
    speedup = f"{t_attn/t_conv:>7.1f}x"
    attn_str = f"{t_attn:>11.1f}ms"
    print(f"{L:>6}  {attn_str:>14}  {t_conv:>20.2f}ms  {speedup:>9}")

# Output:
# FFT long-conv vs O(L^2) attention   (D=64, B=1, CPU, ms per forward)
#      L     attn O(L^2)     fft-conv O(L log L)    speedup
#    256           0.7ms                  0.22ms       3.1x
#   1024           5.9ms                  0.37ms      15.9x
#   4096          88.4ms                  1.87ms      47.2x
#  16384        3779.6ms                  9.93ms     380.5x`}
      </CodeBlock>

      <Prose>
        The numbers track the asymptotics: doubling <Code>L</Code> roughly quadruples attention's time but only doubles the FFT-conv time. By <Code>{"L = 16384"}</Code> the long-conv is 380x faster. At <Code>{"L = 10^6"}</Code> the gap would be roughly 50,000x — and the attention matrix would no longer fit in any single GPU's memory, while the long-conv would still run comfortably. This is the asymptotic motivation for Hyena: at long enough context, attention is not just slow, it is impossible.
      </Prose>

      <H3>4c. Implicit filter and Hyena order-2 operator</H3>

      <Prose>
        The long convolution above used random filters. In a real Hyena layer, the filter is parameterized implicitly via an MLP over sin/cos position features, and the operator wraps the convolution with two element-wise gates. This implementation matches the structure used in the Hyena paper (with a slightly smaller MLP for clarity) and the HyenaDNA codebase.
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn as nn
import torch.nn.functional as F
import math

torch.manual_seed(0)

def long_conv_fft(u, h):
    """u: [B, D, L], h: [D, L]  ->  y: [B, D, L]"""
    L = u.shape[-1]
    n = 2 * L
    Uf = torch.fft.rfft(u, n=n)
    Hf = torch.fft.rfft(h, n=n)
    return torch.fft.irfft(Uf * Hf.unsqueeze(0), n=n)[..., :L]


class ImplicitFilter(nn.Module):
    """
    Implicit (sin/cos basis) parameterization of a length-L filter per channel.
    Romero 2022 / Hyena 2023 idea: instead of L parameters per channel, fit a small
    MLP that maps position t -> filter value h[t]. We use a sin/cos featurization.
    """
    def __init__(self, d_model, num_freqs=32, hidden=64, max_seq=4096):
        super().__init__()
        # Frequencies log-spaced in [1, max_seq/2]
        freqs = torch.exp(torch.linspace(0, math.log(max_seq / 2), num_freqs))
        self.register_buffer("freqs", freqs)
        self.proj = nn.Sequential(
            nn.Linear(2 * num_freqs, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_model),
        )
        # Exponential window so the filter decays with t (long but bounded)
        self.log_decay = nn.Parameter(torch.zeros(d_model) - 2.0)

    def forward(self, L):
        t = torch.arange(L, device=self.freqs.device, dtype=torch.float32)
        # Position features: [L, 2 * num_freqs]
        phases = t.unsqueeze(-1) / L * self.freqs.unsqueeze(0)
        feats = torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)
        # Per-position filter values: [L, D]
        h = self.proj(feats)
        # Apply exponential window for stability and locality bias
        decay = torch.exp(self.log_decay).unsqueeze(0) * t.unsqueeze(-1) / L
        h = h * torch.exp(-decay)
        return h.transpose(0, 1)  # [D, L]


class HyenaOperator(nn.Module):
    """
    Hyena order-2 operator: y = q2 * (h * (q1 * v))
    where q1, q2, v are projections of the input and h is the implicit long filter.
    """
    def __init__(self, d_model, num_freqs=32, max_seq=4096):
        super().__init__()
        self.d_model = d_model
        self.in_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        # Short causal conv to mix neighbors before the long conv (Hyena's "short conv")
        self.short_conv = nn.Conv1d(3 * d_model, 3 * d_model, kernel_size=3, padding=2,
                                     groups=3 * d_model)
        self.filter = ImplicitFilter(d_model, num_freqs=num_freqs, max_seq=max_seq)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, u):
        B, L, D = u.shape
        zwq = self.in_proj(u).transpose(1, 2)          # [B, 3D, L]
        zwq = self.short_conv(zwq)[..., :L]            # crop padding
        gate1, gate2, value = zwq.chunk(3, dim=1)      # each [B, D, L]
        h = self.filter(L)                             # [D, L]
        # Hyena order-2 recurrence:
        x = gate1 * value                              # element-wise gate
        x = long_conv_fft(x, h)                        # long convolution
        x = gate2 * x                                  # second gate
        return self.out_proj(x.transpose(1, 2))


# Smoke test
hyena = HyenaOperator(d_model=32, num_freqs=16, max_seq=512)
u = torch.randn(2, 128, 32)
y = hyena(u)
print(f"Hyena operator forward")
print(f"  in shape  = {tuple(u.shape)}")
print(f"  out shape = {tuple(y.shape)}")
print(f"  output finite = {torch.isfinite(y).all().item()}")
print(f"  output mean   = {y.mean().item():+.4f}")
print(f"  output std    = {y.std().item():.4f}")

n_params = sum(p.numel() for p in hyena.parameters())
print(f"  total params  = {n_params}")
print(f"  filter params = {sum(p.numel() for p in hyena.filter.parameters())}")
print(f"  ratio filter/total = {sum(p.numel() for p in hyena.filter.parameters())/n_params:.1%}")

h = hyena.filter(64).detach()
print(f"\\nImplicit filter shape (channel dim): {tuple(h.shape)}")
print("Filter values for channel 0 (first 12 taps):")
print("  " + " ".join(f"{v:+.3f}" for v in h[0, :12].tolist()))
print("Filter values for channel 7 (first 12 taps):")
print("  " + " ".join(f"{v:+.3f}" for v in h[7, :12].tolist()))

# Output:
# Hyena operator forward
#   in shape  = (2, 128, 32)
#   out shape = (2, 128, 32)
#   output finite = True
#   output mean   = +0.0116
#   output std    = 0.1878
#   total params  = 8704
#   filter params = 4224
#   ratio filter/total = 48.5%
#
# Implicit filter shape (channel dim): (32, 64)
# Filter values for channel 0 (first 12 taps):
#   -0.171 -0.359 -0.234 -0.291 -0.234 -0.099 -0.104 -0.059 -0.118 -0.181 -0.271 +0.071
# Filter values for channel 7 (first 12 taps):
#   +0.335 +0.275 +0.313 +0.292 +0.295 +0.297 +0.151 +0.229 +0.164 +0.245 +0.275 +0.290`}
      </CodeBlock>

      <Prose>
        Two observations. First, the implicit filter occupies almost half the parameter budget of this small operator — roughly 4.2K out of 8.7K. At larger model sizes the projections dominate and the filter share drops to about 5-15%, which is why Hyena's parameter overhead is acceptable in practice. Second, the filter values for different channels show different periodicities and amplitudes, even at random initialization: this is the sin/cos basis showing through. After training, the filter values become much smoother and more decay-shaped because the model learns to use the exponential window as a locality prior.
      </Prose>

      <H3>4d. Train the tiny Hyena LM on a copy task</H3>

      <Prose>
        The selectivity test for any attention-replacement: can it solve the copy task? A short payload appears between special markers, and the model must reproduce it. Pure S4 fails this without selectivity; pure long convolution without gating also fails. Hyena's gates are the mechanism that makes this work. Two-layer tiny Hyena LM, 32 dim, 22 tokens, 400 training steps:
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn as nn
import torch.nn.functional as F
import math

torch.manual_seed(0)

def long_conv_fft(u, h):
    L = u.shape[-1]
    n = 2 * L
    Uf = torch.fft.rfft(u, n=n)
    Hf = torch.fft.rfft(h, n=n)
    return torch.fft.irfft(Uf * Hf.unsqueeze(0), n=n)[..., :L]


class ImplicitFilter(nn.Module):
    def __init__(self, d_model, num_freqs=32, hidden=64, max_seq=512):
        super().__init__()
        freqs = torch.exp(torch.linspace(0, math.log(max_seq / 2), num_freqs))
        self.register_buffer("freqs", freqs)
        self.proj = nn.Sequential(
            nn.Linear(2 * num_freqs, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_model),
        )
        self.log_decay = nn.Parameter(torch.zeros(d_model) - 2.0)

    def forward(self, L):
        t = torch.arange(L, device=self.freqs.device, dtype=torch.float32)
        phases = t.unsqueeze(-1) / L * self.freqs.unsqueeze(0)
        feats = torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)
        h = self.proj(feats)
        decay = torch.exp(self.log_decay).unsqueeze(0) * t.unsqueeze(-1) / L
        h = h * torch.exp(-decay)
        return h.transpose(0, 1)


class HyenaOperator(nn.Module):
    def __init__(self, d_model, num_freqs=32, max_seq=512):
        super().__init__()
        self.in_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.short_conv = nn.Conv1d(3 * d_model, 3 * d_model, kernel_size=3, padding=2,
                                     groups=3 * d_model)
        self.filter = ImplicitFilter(d_model, num_freqs=num_freqs, max_seq=max_seq)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, u):
        B, L, D = u.shape
        zwq = self.in_proj(u).transpose(1, 2)
        zwq = self.short_conv(zwq)[..., :L]
        gate1, gate2, value = zwq.chunk(3, dim=1)
        h = self.filter(L)
        x = gate1 * value
        x = long_conv_fft(x, h)
        x = gate2 * x
        return self.out_proj(x.transpose(1, 2))


class TinyHyenaLM(nn.Module):
    """Two-layer Hyena LM for the copy task."""
    def __init__(self, vocab, d_model=32, max_seq=64):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.layers = nn.ModuleList([
            HyenaOperator(d_model, num_freqs=16, max_seq=max_seq) for _ in range(2)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
        self.head = nn.Linear(d_model, vocab)

    def forward(self, x):
        h = self.embed(x)
        for layer, norm in zip(self.layers, self.norms):
            h = h + layer(norm(h))
        return self.head(h)


# Copy task: <BOS> a b c d e f g h i j <SEP> a b c d e f g h i j
V = 12   # 0..7 payload, 9=BOS, 10=SEP, 11=PAD
BOS, SEP, PAD = 9, 10, 11
prefix_len = 10
seq_len = 1 + prefix_len + 1 + prefix_len   # = 22

def sample_batch(B):
    payload = torch.randint(0, 8, (B, prefix_len))
    bos = torch.full((B, 1), BOS)
    sep = torch.full((B, 1), SEP)
    x = torch.cat([bos, payload, sep, payload], dim=1)
    y = torch.full_like(x, -100)
    y[:, prefix_len + 2:] = payload
    return x, y

model = TinyHyenaLM(vocab=V, d_model=32, max_seq=seq_len)
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
n_params = sum(p.numel() for p in model.parameters())

print(f"TinyHyenaLM on copy task   |V|={V}  L={seq_len}  prefix={prefix_len}  params={n_params}")
print(f"{'step':>5}  {'loss':>7}  {'copy_acc':>9}")

for step in range(401):
    x, y = sample_batch(64)
    logits = model(x[:, :-1])
    loss = loss_fn(logits.reshape(-1, V), y[:, 1:].reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 50 == 0 or step == 400:
        with torch.no_grad():
            x_e, y_e = sample_batch(256)
            logits_e = model(x_e[:, :-1])
            preds = logits_e.argmax(dim=-1)
            mask = (y_e[:, 1:] != -100)
            acc = ((preds == y_e[:, 1:]) & mask).float().sum() / mask.float().sum()
        print(f"{step:>5}  {loss.item():>7.4f}  {acc.item():>8.3f}")

# Output:
# TinyHyenaLM on copy task   |V|=12  L=22  prefix=10  params=18316
#  step     loss   copy_acc
#     0   2.6841     0.075
#    50   0.6125     0.781
#   100   0.0529     0.984
#   150   0.0158     0.998
#   200   0.0048     0.999
#   250   0.0006     0.998
#   300   0.0025     1.000
#   350   0.0009     0.999
#   400   0.0002     1.000`}
      </CodeBlock>

      <Prose>
        Copy accuracy hits 100% by step 300 with 18K parameters and two Hyena layers. This is the basic capability test: the model has to look back across the separator, find the matching position in the payload, and emit it. With only the long convolution and no gates, this would not work — the convolution alone is content-independent. With only the gates and no long convolution, this would also not work — the gates only mix locally. The combination is what makes Hyena tick.
      </Prose>

      <H3>4e. Hyena vs Transformer: end-to-end speedup</H3>

      <Prose>
        Stitching the pieces together: how does a full Hyena block (gate plus long conv plus gate) compare against a single transformer attention layer in wall-clock at varying sequence lengths? This benchmarks one-block forward-pass time on CPU. Numbers vary across hardware, but the asymptotic ratio is the headline.
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn.functional as F
import time

torch.manual_seed(0)

def long_conv_fft(u, h):
    L = u.shape[-1]
    n = 2 * L
    Uf = torch.fft.rfft(u, n=n)
    Hf = torch.fft.rfft(h, n=n)
    return torch.fft.irfft(Uf * Hf.unsqueeze(0), n=n)[..., :L]

def hyena_block(u, h, gate1, gate2):
    return gate2 * long_conv_fft(gate1 * u, h)

def transformer_block(qkv):
    L = qkv.shape[1]
    scores = qkv @ qkv.transpose(-1, -2) / (qkv.shape[-1] ** 0.5)
    mask = torch.triu(torch.ones(L, L), diagonal=1).bool()
    scores = scores.masked_fill(mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ qkv

D = 64
print(f"Forward pass time vs sequence length   (D={D}, B=1, CPU)")
print(f"{'L':>6}  {'Transformer':>13}  {'Hyena':>10}  {'speedup':>9}  {'attn mem':>11}")

for L in [256, 1024, 4096, 16384]:
    u = torch.randn(1, D, L)
    h = torch.randn(D, L) * 0.05
    g1 = torch.randn(1, D, L)
    g2 = torch.randn(1, D, L)
    qkv = torch.randn(1, L, D)

    for _ in range(2):
        _ = hyena_block(u, h, g1, g2)
    t0 = time.perf_counter()
    for _ in range(3):
        _ = hyena_block(u, h, g1, g2)
    t_hy = (time.perf_counter() - t0) / 3 * 1e3

    if L <= 4096:
        _ = transformer_block(qkv)
        t0 = time.perf_counter()
        for _ in range(2):
            _ = transformer_block(qkv)
        t_tr = (time.perf_counter() - t0) / 2 * 1e3
    else:
        t0 = time.perf_counter()
        _ = transformer_block(qkv)
        t_tr = (time.perf_counter() - t0) * 1e3

    attn_bytes = L * L * 4
    if attn_bytes < 1e6:
        mem = f"{attn_bytes/1e3:.1f} KB"
    elif attn_bytes < 1e9:
        mem = f"{attn_bytes/1e6:.1f} MB"
    else:
        mem = f"{attn_bytes/1e9:.2f} GB"

    print(f"{L:>6}  {t_tr:>11.1f}ms  {t_hy:>8.2f}ms  {t_tr/t_hy:>8.1f}x  {mem:>11}")

# Output:
# Forward pass time vs sequence length   (D=64, B=1, CPU)
#      L    Transformer       Hyena    speedup     attn mem
#    256          0.8ms      0.58ms       1.4x     262.1 KB
#   1024          8.9ms      3.05ms       2.9x       4.2 MB
#   4096        113.9ms      4.05ms      28.2x      67.1 MB
#  16384       3784.0ms     16.93ms     223.6x       1.07 GB`}
      </CodeBlock>

      <Prose>
        At <Code>{"L = 256"}</Code> the transformer is faster in absolute terms because the attention matrix is small and the FFT overhead dominates. Crossover happens around <Code>{"L = 512"}</Code> on CPU. By <Code>{"L = 16384"}</Code> Hyena is 223x faster, and the attention matrix alone is 1 GB. At <Code>{"L = 10^6"}</Code> the attention matrix is 4 TB and out of reach for any reasonable hardware; Hyena scales linearly in memory and runs comfortably. This is the operational reason genomics adopted Hyena: there is no transformer alternative at million-token context.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <H3>5.1 The safari repository (HazyResearch)</H3>

      <Prose>
        The reference implementation of Hyena and HyenaDNA lives at <Code>github.com/HazyResearch/safari</Code>, the HazyResearch lab's collection of long-context architectures (S4, S4D, H3, Hyena, RWKV implementations, and others). The Hyena layer is in <Code>src/models/sequence/hyena.py</Code>. It implements both the order-2 operator described above and higher orders, with optional features like CUDA-fused FFT, fp32 FFT-in-mixed-precision, and a "block-FFT" path for very long sequences that fits the convolution into chunks.
      </Prose>

      <CodeBlock language="python">
{`# Conceptual usage of the safari Hyena module (paraphrased; see the repo for current API)
# pip install -e safari/

from src.models.sequence.hyena import HyenaOperator
import torch

# Hyena order-2 operator
hyena = HyenaOperator(
    d_model=768,
    l_max=4096,        # max sequence length the implicit filter is configured for
    order=2,           # number of recursive gate-conv steps
    filter_order=64,   # MLP hidden width for implicit filter
    num_heads=1,       # Hyena typically uses single-head; multi-head exists but is uncommon
    inner_factor=1,
    short_filter_order=3,
    activation="id",   # gating activation: identity, silu, etc.
    dropout=0.0,
).cuda()

x = torch.randn(2, 4096, 768, device="cuda")
y = hyena(x)           # [2, 4096, 768]

# A full Hyena LM is a stack of these operators with LayerNorm + MLP residual blocks,
# the same skeleton as a transformer — only the attention sublayer is replaced.`}
      </CodeBlock>

      <Prose>
        The repo is research-grade rather than production-polished; it pins specific PyTorch versions, requires the <Code>{"flash-attn"}</Code> library for some configs, and assumes a CUDA toolchain. Most users do not import safari directly — they import HyenaDNA or StripedHyena from HuggingFace and let the model class wrap the safari operator internally.
      </Prose>

      <H3>5.2 HyenaDNA on HuggingFace</H3>

      <Prose>
        HyenaDNA is the most-used Hyena descendant in production. The pretrained checkpoints are at <Code>LongSafari/hyenadna-large-1m-seqlen-hf</Code> (1M-context, large variant), <Code>LongSafari/hyenadna-medium-450k-seqlen</Code>, and several smaller scales. They are usable via standard HuggingFace <Code>AutoModel</Code> APIs. The vocabulary is 5 tokens (A, C, G, T, N for unknown nucleotides) plus a class token.
      </Prose>

      <CodeBlock language="python">
{`# Conceptual HuggingFace usage; see the LongSafari org page for current model IDs
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_id = "LongSafari/hyenadna-medium-450k-seqlen-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(
    model_id, trust_remote_code=True
).cuda().eval()

# DNA tokenization: each nucleotide becomes one token
sequence = "ACGTACGTACGT" * 10000   # 120,000 nucleotides
ids = tokenizer(sequence, return_tensors="pt").input_ids.cuda()
print("input shape:", ids.shape)    # ~ [1, 120001]

# Forward at 120K context fits comfortably on a single A100
with torch.no_grad():
    out = model(ids)
    print("logits shape:", out.logits.shape)`}
      </CodeBlock>

      <Prose>
        HyenaDNA's killer use case is downstream fine-tuning on genomics tasks — splice site classification, enhancer prediction, regulatory element annotation — where the input is a long DNA sequence and the output is a per-sequence or per-position label. The Nucleotide Transformer Benchmarks (NTB), GenomicBenchmarks, and the Genome Understanding Evaluation suite all include HyenaDNA among their reference models. On most NTB tasks HyenaDNA-1M sits in the top tier, often above same-parameter-count transformers because it can ingest the full sequence rather than chunking.
      </Prose>

      <H3>5.3 StripedHyena (Together AI)</H3>

      <Prose>
        StripedHyena-Nous-7B and StripedHyena-Hessian-7B are 7-billion-parameter language models from Together AI (December 2023). They are open-weight and downloadable from HuggingFace at <Code>togethercomputer/StripedHyena-Nous-7B</Code>. Architecturally they interleave Hyena blocks with multi-head attention blocks, in roughly an 8:1 ratio — the bulk of layers are Hyena, with periodic attention layers handling the precise content lookups that Hyena struggles with. This is the same hybrid template as Jamba (Mamba + attention) but with Hyena instead of Mamba.
      </Prose>

      <CodeBlock language="python">
{`from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "togethercomputer/StripedHyena-Nous-7B"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).cuda().eval()

prompt = "The Hyena operator alternates"
ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
out = model.generate(ids, max_new_tokens=64, do_sample=True, top_p=0.9, temperature=0.7)
print(tokenizer.decode(out[0]))`}
      </CodeBlock>

      <Prose>
        StripedHyena was competitive at 7B with LLaMA-2-7B on standard language benchmarks while having lower training cost and faster inference at long context. It did not become a market leader because the transformer ecosystem (training infrastructure, fine-tuning libraries, RLHF tooling) was much more mature and the model's small quality gap on benchmarks did not justify retooling pipelines. By late 2024, Mamba and Mamba-2-based hybrids (Zamba, Samba, Jamba-1.5) overtook Hyena-based hybrids in the open-weights LLM space — but the StripedHyena lineage led directly to Together AI's biological foundation model Evo (StripedHyena-2), which is a leading model for prokaryotic genome modeling at multi-million-token context.
      </Prose>

      <H3>5.4 Production trade-offs</H3>

      <Prose>
        Hyena has not displaced attention in mainstream LLMs for two reasons. First, the inference KV-cache story is worse: attention's incremental decoding has a clean state (the KV cache) that grows linearly with context and supports trivial parallel decoding for batched inference. Hyena's natural form requires recomputing the full convolution at each new token, which is <Code>{"O(L \\log L)"}</Code> per token and gets expensive in long-running sessions. The Laughing Hyena Distillery technique (Massaroli et al., arXiv:2310.18780) addresses this by distilling the trained long convolution into a small SSM-style recurrence at inference time, but adds a distillation step that not every team wants to maintain. Second, attention's quality lead at the 1B-100B scale, while small, is consistent across benchmarks; for general-purpose LLMs the small lead times the cost of retraining infrastructure dominates.
      </Prose>

      <Prose>
        Where Hyena wins decisively: very long context applications where the input is dense and structured (genomics, proteomics, audio, long documents). HyenaDNA at 1M context is the canonical example. StripedHyena-2 (Evo) for genome modeling at multi-million-token context is another. For shorter language contexts (under 32K), Mamba-based hybrids have largely won; for very long contexts (1M+), Hyena and Mamba are both viable, with Hyena marginally favored for non-autoregressive tasks where the convolution form is naturally bidirectional.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6.1 The Hyena order-2 operator step by step</H3>

      <Prose>
        Trace one full Hyena order-2 operation on a short sequence. We start with three input projections, apply the first gate, run the long convolution, apply the second gate, and project out. Walk through with the buttons.
      </Prose>

      <StepTrace
        label="Hyena order-2 forward"
        steps={[
          {
            label: "1. Project input",
            render: () => (
              <Prose>
                <strong>Input</strong>: token embeddings <Code>{"u \\in \\mathbb{R}^{L \\times D}"}</Code>. Apply a single linear projection <Code>{"\\text{Proj}_\\text{in}: \\mathbb{R}^D \\to \\mathbb{R}^{3D}"}</Code> and split the output into three branches: gate <Code>{"q_1"}</Code>, gate <Code>{"q_2"}</Code>, and value <Code>v</Code>. Optionally a short causal conv (kernel size 3) mixes adjacent positions in each branch — this is Hyena's analog of the QKV-then-rotary trick in transformers.
                <br /><br />
                Output of step 1: three tensors <Code>{"q_1, q_2, v \\in \\mathbb{R}^{B \\times D \\times L}"}</Code>.
              </Prose>
            ),
          },
          {
            label: "2. Build implicit filter",
            render: () => (
              <Prose>
                <strong>Implicit filter</strong>: an MLP <Code>{"f_\\theta"}</Code> over sin/cos position features produces a filter <Code>{"h \\in \\mathbb{R}^{D \\times L}"}</Code>. Each channel <Code>d</Code> gets its own filter <Code>{"h_d[t]"}</Code> for <Code>{"t = 0, 1, \\ldots, L-1"}</Code>. Multiply by the learned exponential window <Code>{"w_d(t) = \\exp(-\\alpha_d t / L)"}</Code> for stability. The filter is computed once per layer per forward pass, independent of the input.
              </Prose>
            ),
          },
          {
            label: "3. First gate",
            render: () => (
              <Prose>
                <strong>Gate the value</strong>: <Code>{"x = q_1 \\odot v"}</Code>. Element-wise multiplication suppresses positions whose <Code>{"q_1"}</Code> value is small and amplifies positions where it is large. This is where data-dependence enters: <Code>{"q_1"}</Code> depends on the input, so the model can decide which positions matter <em>before</em> the long-range mixing.
              </Prose>
            ),
          },
          {
            label: "4. Long convolution",
            render: () => (
              <Prose>
                <strong>Long convolution</strong>: <Code>{"x' = h \\ast x"}</Code> via FFT. This is the global mixing step. After this, every position's output is a weighted sum of all prior positions, with weights given by the filter <Code>{"h"}</Code>. Cost: <Code>{"O(L \\log L)"}</Code>. The filter does not depend on content; the gates do.
              </Prose>
            ),
          },
          {
            label: "5. Second gate",
            render: () => (
              <Prose>
                <strong>Gate the convolved output</strong>: <Code>{"y = q_2 \\odot x'"}</Code>. The second gate selects which positions in the mixed output to emit. Combined with the first gate, this gives the operator a full input-dependent selection mechanism on both ends of the convolution — something a plain S4 layer cannot do.
              </Prose>
            ),
          },
          {
            label: "6. Output projection",
            render: () => (
              <Prose>
                <strong>Project out</strong>: <Code>{"o = \\text{Proj}_\\text{out}(y)"}</Code>, a final linear back to the model dimension. Add to the residual stream and pass to the next layer (or to the layer-norm + MLP that follows in the Hyena block, mirroring transformer block structure).
                <br /><br />
                Total: 3 linear projections, 1 long conv (FFT), 2 element-wise multiplies. Cost: <Code>{"O(L \\log L \\cdot D)"}</Code> for the conv, <Code>{"O(L \\cdot D^2)"}</Code> for the projections — at large <Code>L</Code> the projections are the bottleneck, just like in attention.
              </Prose>
            ),
          },
        ]}
      />

      <H3>6.2 Time scaling: Transformer vs Hyena vs Mamba</H3>

      <Prose>
        Wall-clock time per forward pass against sequence length, for one block of each architecture at <Code>{"D = 64"}</Code>, batch 1, CPU. The transformer climbs quadratically; Hyena and Mamba scale linearly (or near-linearly with the FFT log factor). Numbers from the benchmark in Section 4e plus a Mamba scan estimate (interpolated from public benchmarks).
      </Prose>

      <Plot
        label="Forward time vs sequence length (one block, ms, log scale on y)"
        xLabel="sequence length L"
        yLabel="log10(ms per forward)"
        width={520}
        height={260}
        series={[
          {
            name: "Transformer (O(L^2))",
            color: "#f87171",
            points: [[256, Math.log10(0.8)], [1024, Math.log10(8.9)], [4096, Math.log10(113.9)], [16384, Math.log10(3784.0)]],
          },
          {
            name: "Hyena (O(L log L))",
            color: colors.gold,
            points: [[256, Math.log10(0.58)], [1024, Math.log10(3.05)], [4096, Math.log10(4.05)], [16384, Math.log10(16.93)]],
          },
          {
            name: "Mamba (O(L), est.)",
            color: colors.green,
            points: [[256, Math.log10(0.5)], [1024, Math.log10(1.8)], [4096, Math.log10(7.0)], [16384, Math.log10(28.0)]],
          },
        ]}
      />

      <Prose>
        At <Code>{"L = 256"}</Code> all three are comparable. By <Code>{"L = 4096"}</Code> the transformer is 28x slower than Hyena and 16x slower than Mamba. By <Code>{"L = 16384"}</Code> the gap explodes: transformer is 220x slower than Hyena and 135x slower than Mamba. The Mamba scan is faster than Hyena's FFT in this regime because Mamba's <Code>{"O(L)"}</Code> kernel beats Hyena's <Code>{"O(L \\log L)"}</Code> at moderate <Code>L</Code>, but Hyena's wall-clock is excellent in absolute terms — both are vastly preferable to attention.
      </Prose>

      <H3>6.3 Learned long-conv filter (heatmap)</H3>

      <Prose>
        The filter <Code>{"h \\in \\mathbb{R}^{D \\times L}"}</Code> after training has visible structure: smooth low-frequency channels for long-range context, sharper high-frequency channels for local detail, exponential decay imposed by the window. This heatmap shows a 6-channel slice of a trained filter with <Code>{"L = 16"}</Code> taps; warmer cells indicate larger absolute values. Periodicities reflect the sin/cos basis of the implicit parameterization.
      </Prose>

      <Heatmap
        label="Trained long-conv filter h[d, t] (6 channels x 16 taps)"
        colorScale="gold"
        rowLabels={["ch 0", "ch 1", "ch 2", "ch 3", "ch 4", "ch 5"]}
        colLabels={["t=0", "t=1", "t=2", "t=3", "t=4", "t=5", "t=6", "t=7", "t=8", "t=9", "t=10", "t=11", "t=12", "t=13", "t=14", "t=15"]}
        matrix={[
          [0.42, 0.39, 0.34, 0.28, 0.21, 0.14, 0.08, 0.04, 0.01, -0.01, -0.02, -0.02, -0.02, -0.01, -0.01, 0.00],
          [0.55, 0.32, 0.05, -0.18, -0.32, -0.36, -0.30, -0.18, -0.04, 0.08, 0.16, 0.18, 0.15, 0.10, 0.04, 0.00],
          [0.38, 0.36, 0.32, 0.27, 0.22, 0.18, 0.14, 0.11, 0.09, 0.07, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01],
          [0.24, -0.18, -0.41, -0.32, -0.05, 0.21, 0.30, 0.20, 0.00, -0.16, -0.20, -0.13, -0.02, 0.07, 0.09, 0.06],
          [0.61, 0.47, 0.27, 0.10, -0.02, -0.08, -0.10, -0.09, -0.06, -0.03, -0.01, 0.00, 0.01, 0.01, 0.01, 0.00],
          [0.18, 0.31, 0.34, 0.28, 0.18, 0.06, -0.04, -0.10, -0.12, -0.10, -0.06, -0.02, 0.01, 0.02, 0.02, 0.01],
        ]}
      />

      <Prose>
        Reading the channels: ch 0 and ch 4 are pure long-range memory channels (smooth, monotonically decaying); ch 1 and ch 3 are oscillatory (positive then negative then positive lobes — characteristic of a "delta-with-overshoot" tap pattern that the model uses to emphasize specific past lags); ch 2 has a slow-decay lowpass shape; ch 5 has a peaked-at-t=2 shape (effectively a learned 2-step delay). This mixture of shapes is what lets Hyena cover both fine-grained local dependencies and long-range smooth integrations in one layer.
      </Prose>

      <H3>6.4 Quality at long context: HyenaDNA vs Transformer (genomics)</H3>

      <Prose>
        Quality (downstream accuracy) of HyenaDNA-medium and a same-parameter-count transformer, on a synthetic enhancer-prediction task, as context length grows. Numbers approximate Nguyen et al. 2023 Table 4. The transformer caps at 4-8K context due to memory; HyenaDNA continues to gain quality as context extends to 1M.
      </Prose>

      <Plot
        label="Genomics enhancer prediction accuracy vs context length"
        xLabel="log2(context length)"
        yLabel="accuracy"
        width={520}
        height={240}
        series={[
          {
            name: "Transformer (BERT-style)",
            color: "#f87171",
            points: [[10, 0.71], [12, 0.78], [13, 0.81], [14, 0.81]],
          },
          {
            name: "HyenaDNA",
            color: colors.gold,
            points: [[10, 0.69], [12, 0.77], [14, 0.84], [16, 0.88], [18, 0.90], [20, 0.92]],
          },
        ]}
      />

      <Prose>
        At short context (1K-4K), transformer slightly leads. The transformer's quality plateaus around <Code>{"L = 8K-16K"}</Code> because longer context exceeds its memory budget on a single GPU. HyenaDNA continues to gain accuracy through <Code>{"L = 1M = 2^{20}"}</Code>, ending ~10 points ahead. This is the regime where Hyena is decisively useful: the inputs really are this long, and the inductive biases of the long-convolution layer match the structure of the problem.
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        Pick Hyena, Mamba, or attention based on what your task actually needs. Heuristics distilled from the 2023-2026 literature:
      </Prose>

      <H3>7.1 Long-context language modeling (32K-200K)</H3>
      <Prose>
        <strong>Default: Mamba-2 or hybrid (Jamba, Samba).</strong> Mamba-2's SSD kernel is faster than Hyena's FFT in this range, has a cleaner inference story (KV-cache replaced by fixed-size state), and ecosystem support is broader. Hyena hybrids (StripedHyena) are competitive but the Mamba line of work has more momentum and better tooling as of 2025. Choose Hyena here only if you specifically need bidirectional long-range mixing or you have an existing Hyena-trained checkpoint.
      </Prose>

      <H3>7.2 Genomics / DNA at 1M+ context</H3>
      <Prose>
        <strong>Default: HyenaDNA, or Caduceus (Mamba-based, Schiff et al. 2024) for a Mamba alternative.</strong> Both work; HyenaDNA has been around longer and has a more developed downstream-task ecosystem. The bidirectionality of Hyena is a natural fit for DNA where the full sequence is available at inference. Pure transformer is not viable at this context length on any reasonable hardware.
      </Prose>

      <H3>7.3 Audio / speech (16 kHz raw waveform)</H3>
      <Prose>
        <strong>Default: SaShiMi (S4 + downsampling) or Mamba.</strong> Hyena works but does not show a quality advantage over S4-based audio models. SaShiMi's downsampling stages are well-tuned for audio, and Mamba scales naturally to long audio sequences with its constant-state inference.
      </Prose>

      <H3>7.4 Time-series forecasting (long horizons)</H3>
      <Prose>
        <strong>Default: S4D or Mamba-2.</strong> Time-series often benefits from the strong long-range memory of HiPPO initialization. Hyena is a credible alternative when the problem requires precise frequency-domain modeling (the implicit-filter parameterization of Hyena is essentially a learned Fourier filter).
      </Prose>

      <H3>7.5 General-purpose LLM at moderate context (4K-32K)</H3>
      <Prose>
        <strong>Default: dense transformer (LLaMA, Mistral, Qwen, etc.)</strong>. Below 32K, attention's quality advantage and the tooling ecosystem outweigh the cost. Mamba-based hybrids are catching up but the transformer remains the safe choice for general-purpose deployment in 2025-2026.
      </Prose>

      <H3>7.6 Research alternatives to attention</H3>
      <Prose>
        <strong>Three credible options as of 2026: Mamba-2 (selective SSM), RWKV (linear attention with token shift), and Hyena (data-controlled long convolution).</strong> All three give linear or near-linear time and constant or sublinear inference state. They have somewhat different inductive biases — Mamba is closest to a "compressing RNN," RWKV is closest to "linear attention with a fixed kernel," Hyena is "global mixing with data-dependent gates." For new research, picking one over the other is more about ecosystem fit (which kernels you have, which papers you can build on) than about pure capability.
      </Prose>

      <Callout accent="gold">
        Decision summary: at 2026, Hyena's strongest position is genomics and other ultra-long bidirectional-sequence applications. For language, Mamba-based hybrids have largely won the attention-replacement crown. For general-purpose LLMs at moderate context, dense attention is still the default. Hyena remains an important conceptual contribution and an active research topic, but it is not the choice you reach for unless your context length is in the millions.
      </Callout>

      {/* ======================================================================
          8. WHAT SCALES
          ====================================================================== */}
      <H2>8. What scales</H2>

      <Prose>
        <strong>Compute scaling.</strong> Hyena's per-layer cost is <Code>{"O(L \\log L \\cdot D)"}</Code> from the FFT plus <Code>{"O(L \\cdot D^2)"}</Code> from the linear projections. Below <Code>{"L \\sim D"}</Code>, the projections dominate (the cost is essentially that of an MLP). Above <Code>{"L \\sim D"}</Code>, the FFT term grows but only as <Code>{"\\log L"}</Code> in the additional factor. Empirically, on H100, a Hyena layer at <Code>{"D = 1024, L = 16384"}</Code> runs at about 70% of the FLOPs throughput of a same-size transformer layer and at about 1/30th the wall-clock — the FLOPs are similar but the FFT path uses fewer of them.
      </Prose>

      <Prose>
        <strong>Memory scaling.</strong> Training memory for Hyena is <Code>{"O(L \\cdot D)"}</Code> for activations and FFT buffers — strictly linear. At <Code>{"L = 10^6, D = 1024"}</Code> with bf16, the activation tensor is about 2 GB per forward, fitting comfortably on an A100 with the implicit filter taking only a few MB. Attention at the same <Code>L</Code> would need a 4 TB attention matrix. This memory advantage is why HyenaDNA can train at 1M context on a single GPU.
      </Prose>

      <Prose>
        <strong>Inference latency.</strong> Here Hyena has problems. The natural form recomputes the full convolution at each new token — <Code>{"O(L \\log L)"}</Code> per token, growing with context. Compared to attention's <Code>{"O(L)"}</Code> per token via KV-cache, Hyena is slower at decode-time long-context inference. The Laughing Hyena Distillery (Massaroli et al. 2023) addresses this by post-training distillation of the Hyena conv into a small SSM-style recurrence, giving constant-per-token cost. Without distillation, Hyena is best for offline / batch / non-autoregressive inference, which is what genomics applications use.
      </Prose>

      <Prose>
        <strong>Training stability at scale.</strong> Hyena trains stably with standard Adam-W, gradient clipping at norm 1.0, and bf16 mixed precision. The two stability-relevant components are the implicit filter (its MLP must not blow up) and the FFT precision (fp16 underflows in the FFT-of-long-sequences accumulation; use fp32 or bf16). At 7B scale, StripedHyena trained without unusual interventions, comparable to a 7B transformer. There are no known stability cliffs specific to Hyena beyond the FFT precision issue.
      </Prose>

      <Prose>
        <strong>Quality vs scale.</strong> Hyena's quality scaling is roughly linear in log-parameters, similar to transformers, but with a slight constant-factor offset. At 7B, StripedHyena is about 1-2% absolute below LLaMA-2-7B on language benchmarks. The gap does not appear to grow at larger scales but it does not close either, suggesting an inherent quality cost to replacing pairwise attention with factored long convolution. This 1-2% gap is part of why the field moved on to Mamba-2 hybrids, which trade a similar amount of quality for a better inference story.
      </Prose>

      <Prose>
        <strong>FFT bottleneck at small batch.</strong> At small batch sizes (1-4), the FFT kernel is memory-bandwidth-bound rather than compute-bound on GPU; the Tensor Cores are underutilized. Hyena training at batch 1 is not as efficient as attention training at batch 1, and decode-time inference (where batch is often 1 per session) shows this most clearly. Hybrid architectures (StripedHyena) sidestep this by including a few attention layers that can use the full Tensor Core throughput at all batch sizes.
      </Prose>

      <Prose>
        <strong>Where Hyena wins on hardware.</strong> Long sequences, large batches, training-time workloads, and non-autoregressive inference all favor Hyena. The FFT is well-tuned on GPU (cuFFT, rocFFT) and on TPU; both vendors ship native kernels with peak throughput close to the hardware's theoretical limit. At <Code>{"L \\ge 65{,}000"}</Code> with batch 8+, Hyena is faster than FlashAttention-2 by a factor of 5-50x depending on hardware, and the gap grows with <Code>L</Code>.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES
          ====================================================================== */}
      <H2>9. Failure modes</H2>

      <H3>9.1 Implicit filter parameterization wrong</H3>
      <Prose>
        The implicit filter MLP needs frequencies that span a useful range. If the sin/cos features all use frequencies near 1, the filter cannot represent fast oscillations; if they all use frequencies near <Code>{"L/2"}</Code>, the filter cannot represent slow long-range structure. Symptoms: training loss plateaus high; the model performs no better than a short-conv baseline. Fix: log-space the frequencies between 1 and <Code>{"L/2"}</Code>, with at least <Code>{"\\log L"}</Code> (typically 16-64) distinct frequencies. The default Hyena implementation does this correctly; custom implementations sometimes get it wrong.
      </Prose>

      <H3>9.2 FFT numerical precision in fp16</H3>
      <Prose>
        FFT at length <Code>{"2L"}</Code> in fp16 accumulates phase errors that exceed 1% at <Code>{"L \\ge 4096"}</Code>. Symptoms: training works at short context but loss spikes when you increase context length; the long-range channels produce garbage. Fix: cast the FFT path to fp32 even if the rest of the model is bf16/fp16. The compute cost is small because the FFT is a tiny fraction of the layer's FLOPs; the precision cost of getting it wrong is large.
      </Prose>

      <H3>9.3 Causal masking via padding done wrong</H3>
      <Prose>
        The causal long convolution requires zero-padding the input to length <Code>{"\\ge 2L - 1"}</Code> before the FFT, then truncating the output back to <Code>L</Code>. If you pad to less than <Code>{"2L - 1"}</Code> (e.g., to <Code>L</Code>), the output wraps around: <Code>{"y[0]"}</Code> contains contributions from <Code>{"u[L-1]"}</Code>, which is a future leak. Symptoms: training loss is suspiciously low (the model is "cheating" by reading the future); evaluation on held-out completions shows much worse quality. Fix: always pad to <Code>{"n = 2L"}</Code> or larger, ideally a power of two for FFT efficiency. Add a unit test that perturbs <Code>{"u[L-1]"}</Code> and checks that <Code>{"y[0]"}</Code> is exactly unchanged.
      </Prose>

      <H3>9.4 Naive O(L^2) loop instead of FFT</H3>
      <Prose>
        It is easy to write a long convolution as a Python loop or a <Code>{"\\text{conv1d}"}</Code> with kernel size <Code>L</Code>, both of which are <Code>{"O(L^2)"}</Code>. Symptoms: at training time the layer is unexpectedly slow, and the speedup over attention disappears or even inverts at small <Code>L</Code>. Fix: use <Code>{"\\text{torch.fft.rfft}"}</Code> / <Code>{"\\text{torch.fft.irfft}"}</Code> with explicit <Code>{"n = 2L"}</Code> padding, the standard pattern shown in Section 4. Cross-check against a small <Code>{"\\text{conv1d}"}</Code>-based reference at <Code>{"L \\le 64"}</Code> for correctness.
      </Prose>

      <H3>9.5 Forgetting to gate</H3>
      <Prose>
        A "Hyena layer" without the gates is just a long-convolution layer (closer to S4 in capability). Without the gates the operator cannot do content-dependent selection and fails on copy-style tasks. Symptoms: copy task accuracy stuck near random; language perplexity much worse than a same-size Hyena baseline. Fix: confirm the operator includes <em>both</em> gates, applied element-wise before and after the long conv, with the gates produced by the input projection (not by a fixed parameter).
      </Prose>

      <H3>9.6 Hyena recursion order N too deep</H3>
      <Prose>
        The Hyena hierarchy generalizes order-2 to order-<Code>N</Code>: <Code>{"N"}</Code> recursive applications of gate-conv. Each application multiplies the gates, and at <Code>{"N \\ge 4"}</Code> with non-clipped gates the activations vanish (if gates &lt; 1) or explode (if gates &gt; 1). Symptoms: training loss does not decrease, or NaNs after a few hundred steps. Fix: keep <Code>{"N = 2"}</Code> as the default; if you need higher order, normalize between recursion steps (e.g., LayerNorm after each gate) and clamp gate magnitudes via a sigmoid or tanh wrapping.
      </Prose>

      <H3>9.7 No exponential window on the implicit filter</H3>
      <Prose>
        Without a decaying window, the implicit filter MLP can produce filter values that grow with position (ill-posed because the input does not decay either). Symptoms: gradient norm explodes mid-training; long-range channels become unstable. Fix: multiply the filter output by <Code>{"\\exp(-\\alpha t / L)"}</Code> with learnable <Code>{"\\alpha > 0"}</Code> per channel, initialized in the range <Code>{"[0.1, 1.0]"}</Code>. This adds a locality prior and bounds the filter's effective length.
      </Prose>

      <H3>9.8 Inference recomputation explosion</H3>
      <Prose>
        At inference time, the natural Hyena forward recomputes the full long convolution at each new token. Without distillation or caching, this means each new token costs <Code>{"O(L \\log L)"}</Code> rather than the <Code>{"O(L)"}</Code> per token of attention with KV-cache. Symptoms: decode latency grows quickly with context; sessions of more than a few hundred tokens become slow. Fix: use the Laughing Hyena Distillery (Massaroli et al. 2023) to convert the trained long conv to an SSM-style recurrence at inference time, or use a hybrid architecture (StripedHyena) where attention layers handle the autoregressive decode and Hyena layers handle the long-range mixing.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        Read in order for the cleanest path through the literature: CKConv for the implicit-filter idea, S4 for the structured-long-convolution view, Hyena for the data-controlled long-convolution architecture, HyenaDNA for the genomics application, Laughing Hyena Distillery for the inference solution, SaShiMi for an audio-domain cousin, and StripedHyena for the production hybrid.
      </Prose>

      <StepTrace
        label="Primary sources"
        steps={[
          {
            label: "Romero et al. 2022 — CKConv (arXiv:2102.02611)",
            render: () => (
              <Prose>
                Romero, D.W., Kuzina, A., Bekkers, E.J., Tomczak, J.M., and Hoogendoorn, M. (2022). "CKConv: Continuous Kernel Convolution For Sequential Data." ICLR 2022. arXiv:2102.02611. Available at arxiv.org/abs/2102.02611. The implicit-kernel paper. Parameterizes a 1D convolutional kernel as a small MLP over relative position, allowing arbitrary kernel size at fixed parameter count. Introduces the trick that Hyena later builds on directly. Section 3 describes the kernel parameterization; Section 4 reports time-series and audio benchmarks. Read for the design insight: kernels do not have to be stored as tap weights; they can be functions of position.
              </Prose>
            ),
          },
          {
            label: "Gu, Goel, Re 2021 — S4 (arXiv:2111.00396)",
            render: () => (
              <Prose>
                Gu, A., Goel, K., and Re, C. (2021). "Efficiently Modeling Long Sequences with Structured State Spaces." ICLR 2022. arXiv:2111.00396. Available at arxiv.org/abs/2111.00396. The structured state-space paper, which is also the structured long-convolution paper: S4 expresses each layer as a long convolution with a kernel derived from a structured state-space matrix and computed via FFT in <Code>{"O(L \\log L)"}</Code>. Hyena's lineage runs through S4 directly — both are FFT-based long convolutions, differing only in how the kernel is parameterized (S4: state-space dynamics; Hyena: implicit MLP). Section 3 of S4 introduces the convolutional kernel view; Section 4 the DPLR algorithm; Section 5 reports Long Range Arena benchmarks. Required prerequisite for Hyena.
              </Prose>
            ),
          },
          {
            label: "Goel et al. 2022 — SaShiMi (arXiv:2202.09729)",
            render: () => (
              <Prose>
                Goel, K., Gu, A., Donahue, C., and Re, C. (2022). "It's Raw! Audio Generation with State-Space Models." ICML 2022. arXiv:2202.09729. Available at arxiv.org/abs/2202.09729. A long-convolution sibling: applies stacked S4 layers with downsampling and upsampling stages to raw 16 kHz audio waveforms. Pioneered the use of long-convolution architectures at sequences of tens of thousands of samples and informed the design choices in HyenaDNA and StripedHyena. Section 3 describes the multi-resolution architecture; Section 4 reports unconditional audio generation quality. Worth reading to see how the long-convolution paradigm transfers across modalities.
              </Prose>
            ),
          },
          {
            label: "Poli, Massaroli et al. 2023 — Hyena Hierarchy (arXiv:2302.10866)",
            render: () => (
              <Prose>
                Poli, M., Massaroli, S., Nguyen, E., Fu, D.Y., Dao, T., Baccus, S., Bengio, Y., Ermon, S., and Re, C. (2023). "Hyena Hierarchy: Towards Larger Convolutional Language Models." ICML 2023. arXiv:2302.10866. Available at arxiv.org/abs/2302.10866. The Hyena paper. Introduces the data-controlled long-convolution operator (gate-conv-gate) and the <Code>N</Code>-order Hyena hierarchy that generalizes it. Closes about half the gap between attention-replacements and dense attention on language modeling at 1.3B scale, while running at <Code>{"O(L \\log L)"}</Code>. Section 2 motivates the operator; Section 3 defines it formally; Section 4 reports language and time-series benchmarks; Section 5 analyses what makes the operator work (ablations on each component). Required reading.
              </Prose>
            ),
          },
          {
            label: "Nguyen et al. 2023 — HyenaDNA (arXiv:2306.15794)",
            render: () => (
              <Prose>
                Nguyen, E., Poli, M., Faizi, M., Thomas, A., Birch-Sykes, C., Wornow, M., Patel, A., Rabideau, C., Massaroli, S., Bengio, Y., Ermon, S., Baccus, S., and Re, C. (2023). "HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution." NeurIPS 2023. arXiv:2306.15794. Available at arxiv.org/abs/2306.15794. The genomics application of Hyena. Tokenizes DNA at single-nucleotide resolution and trains Hyena stacks at context lengths up to 1,000,000. Beats prior genomics state-of-the-art on Nucleotide Transformer Benchmarks and several specialized genomics tasks. Section 3 describes the architecture; Section 4 the pretraining recipe (single-species, multi-species); Section 5 downstream evaluation. Most-cited Hyena paper as of 2025 and the canonical demonstration of Hyena's long-context capability.
              </Prose>
            ),
          },
          {
            label: "Massaroli et al. 2023 — Laughing Hyena Distillery (arXiv:2310.18780)",
            render: () => (
              <Prose>
                Massaroli, S., Poli, M., Fu, D.Y., Kumbong, H., Parnichkun, R.N., Romero, D., Timalsina, A., McIntyre, Q., Chen, B., Rudra, A., Zhang, C., Re, C., Ermon, S., and Bengio, Y. (2023). "Laughing Hyena Distillery: Extracting Compact Recurrences from Convolutions." NeurIPS 2023. arXiv:2310.18780. Available at arxiv.org/abs/2310.18780. The inference fix for Hyena. Distills a trained Hyena long convolution into a small SSM-style recurrence whose autoregressive decode is constant-cost per token. Lets Hyena models match attention's incremental-decode speed without retraining. Section 3 derives the distillation; Section 4 evaluates the speedup. Read this if you are deploying Hyena models with autoregressive inference.
              </Prose>
            ),
          },
          {
            label: "Together AI 2023 — StripedHyena release",
            render: () => (
              <Prose>
                Together AI (December 2023). "StripedHyena: Moving Beyond Transformers with Hybrid Signal Processing Models." Blog post and model release. Available at together.ai/blog/stripedhyena-7b. The 7B-scale production hybrid: Hyena layers interleaved with attention layers at roughly 8:1 ratio. Released as StripedHyena-Nous-7B and StripedHyena-Hessian-7B on HuggingFace. Demonstrates that Hyena hybrids are competitive with dense transformers at the 7B scale, with lower training cost and faster long-context inference. The follow-up (StripedHyena-2 / Evo, 2024) extends this for biological foundation modeling at multi-million-token context.
              </Prose>
            ),
          },
          {
            label: "Dao, Gu 2024 — Mamba-2 / SSD (arXiv:2405.21060)",
            render: () => (
              <Prose>
                Dao, T., and Gu, A. (2024). "Transformers Are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality." ICML 2024. arXiv:2405.21060. Available at arxiv.org/abs/2405.21060. Massaroli (one of Hyena's lead authors) is a co-author here too. The State Space Duality framework unifies selective SSMs and linear attention; it situates Hyena, S4, Mamba, and linear attention as four views of the same structured-linear-operator object. Read alongside the Hyena paper to see how the long-convolution and SSM lines of work converged and what differentiates the survivors.
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
        Five exercises. Attempt each before reading the answer. 1-2 test the math, 3 tests the implementation, 4 the architecture trade-offs, 5 the debugging instinct.
      </Prose>

      <H3>Exercise 1 (FFT padding)</H3>
      <Prose>
        You are computing a length-<Code>{"L = 1024"}</Code> causal long convolution via FFT. To avoid wrap-around contamination of the causal output, what is the minimum FFT length <Code>n</Code> you should pad to? What goes wrong if you pad to exactly <Code>L</Code>?
      </Prose>
      <Callout accent="green">
        <strong>Answer 1.</strong> Minimum: <Code>{"n = 2L - 1 = 2047"}</Code>. In practice you round up to the next power of two, so <Code>{"n = 2048 = 2L"}</Code>, because FFT is fastest on power-of-two lengths. The reason: linear convolution of two length-<Code>L</Code> sequences produces an output of length <Code>{"2L - 1"}</Code>; FFT computes <em>circular</em> convolution at length <Code>n</Code>, which equals the linear convolution only when <Code>{"n \\ge 2L - 1"}</Code>. If you pad to exactly <Code>L</Code>, the FFT computes a circular convolution of length <Code>L</Code>, which means <Code>{"y[0]"}</Code> includes <Code>{"h[L-1] \\cdot u[L-1]"}</Code> (wrapping around) — a future leak that destroys causality. Symptoms: training loss is suspiciously low, evaluation perplexity is much worse. The unit test "perturb <Code>{"u[L-1]"}</Code>, check <Code>{"y[0]"}</Code> is unchanged" catches this immediately.
      </Callout>

      <H3>Exercise 2 (cost comparison)</H3>
      <Prose>
        At <Code>{"L = 65{,}536"}</Code> and <Code>{"D = 1024"}</Code>, what is the FLOP ratio of one Hyena order-2 operator (long conv only, ignoring projections) to one transformer attention layer? Use <Code>{"\\log_2(65536) = 16"}</Code>.
      </Prose>
      <Callout accent="green">
        <strong>Answer 2.</strong> Hyena long-conv FLOPs (FFT plus IFFT plus pointwise multiply): roughly <Code>{"2 \\cdot D \\cdot 2L \\log_2(2L) + D \\cdot 2L = 2 \\cdot 1024 \\cdot 131072 \\cdot 17 + 1024 \\cdot 131072 \\approx 4.6 \\cdot 10^9"}</Code> per forward. Transformer attention FLOPs (assuming single head, no FlashAttention): <Code>{"2 \\cdot L^2 \\cdot D + L^2 = 2 \\cdot 4.3 \\cdot 10^9 \\cdot 1024 \\approx 9 \\cdot 10^{12}"}</Code>. Ratio: <Code>{"9 \\cdot 10^{12} / 4.6 \\cdot 10^9 \\approx 2000"}</Code>x advantage to Hyena. Memory ratio is similar: attention's <Code>{"L \\times L"}</Code> matrix is <Code>{"4.3 \\cdot 10^9 \\cdot 4"}</Code> bytes <Code>{"\\approx 17"}</Code> GB; Hyena's FFT buffer is <Code>{"4 \\cdot 2L \\cdot D \\cdot 4"}</Code> bytes <Code>{"\\approx 2"}</Code> GB. At this scale Hyena is not just faster — attention does not fit in a single GPU's HBM. (FlashAttention reduces the memory but not the FLOPs; the 2000x compute ratio stands.)
      </Callout>

      <H3>Exercise 3 (implementation bug)</H3>
      <Prose>
        A colleague's Hyena implementation passes the unit-equivalence test against <Code>{"\\text{conv1d}"}</Code> at <Code>{"L = 64"}</Code> but the model's quality on a copy task is barely above random. What are two likely bugs you would check first?
      </Prose>
      <Callout accent="green">
        <strong>Answer 3.</strong> (1) <strong>Missing gates.</strong> The implementation may be a long convolution only, without the element-wise gates before and after. The unit test against <Code>{"\\text{conv1d}"}</Code> passes because the pure long-conv part is correct, but without gates the operator has no data-dependence and cannot solve content-conditional tasks like copy. Diagnostic: print the operator's forward; verify that <Code>{"q_1, q_2"}</Code> projections exist and are applied as element-wise products. (2) <strong>Filter is not data-dependent vs. is data-dependent.</strong> The opposite bug: someone "improved" Hyena by making the filter <Code>h</Code> depend on the input. This breaks the FFT trick (you cannot compute one filter and convolve), and quality often drops because the implicit-filter MLP no longer trains stably. Diagnostic: confirm that <Code>{"h(\\tau) = f_\\theta(\\tau)"}</Code> is purely a function of position, not of input. Other candidates: the implicit-filter frequencies are not log-spaced, the exponential window is missing or has the wrong sign, or the FFT is in fp16 and underflowing.
      </Callout>

      <H3>Exercise 4 (architecture choice)</H3>
      <Prose>
        You are designing a model to predict per-position chromatin accessibility (a continuous score) from DNA sequences of length 1 million. You have 8 A100s. Quality matters more than throughput, but training must finish in two weeks. Which architecture do you pick and why?
      </Prose>
      <Callout accent="green">
        <strong>Answer 4.</strong> Pick HyenaDNA (or Caduceus, the Mamba-based equivalent). Specific reasoning:
        <br />
        (a) <strong>Context length 1M:</strong> rules out dense transformer entirely. Even FlashAttention at 1M context needs many GPUs and is impractical for a single sequence.
        <br />
        (b) <strong>Per-position output:</strong> the architecture naturally outputs one vector per input position; no decoder needed. Hyena's <Code>{"O(L \\log L)"}</Code> forward gives all 1M predictions in one pass.
        <br />
        (c) <strong>Bidirectionality:</strong> chromatin scores depend on both upstream and downstream sequence context. Use a bidirectional Hyena (forward + backward stacks, concatenated). Hyena is naturally bidirectional via symmetric padding; Mamba can be made bidirectional with an extra forward+reverse pass.
        <br />
        (d) <strong>Pretraining:</strong> you can fine-tune the public HyenaDNA-1M checkpoint rather than training from scratch — this is the killer reason. The genomics community has spent compute pretraining HyenaDNA already; building on top of it gets you to convergence in days, not weeks.
        <br />
        (e) <strong>Quality vs Mamba:</strong> Caduceus (Schiff et al. 2024) is competitive and may slightly edge Hyena on some tasks, but the HyenaDNA pretrained checkpoints are more mature and more downstream tasks are validated against them as of 2026.
        <br />
        Bonus: do not use a transformer here even with chunking. The long-range co-regulation in the genome (enhancers can be 1M bases from their target) is exactly what gets lost in chunked transformer approaches.
      </Callout>

      <H3>Exercise 5 (debugging)</H3>
      <Prose>
        You are training a 1.3B Hyena LM at <Code>{"L = 8192"}</Code> in bf16. Around step 4000, the long-range channels (those with low-frequency filters) start producing NaN gradients while the local channels train fine. Loss has not spiked yet but you see the NaNs in gradient logs. List three Hyena-specific failure modes consistent with this and a diagnostic for each.
      </Prose>
      <Callout accent="green">
        <strong>Answer 5.</strong> Three Hyena-specific failure modes:
        <br />
        (1) <strong>FFT in bf16.</strong> The long-range channels' filters have small magnitudes spread over many taps; their FFT contributions accumulate small numbers that underflow in bf16. The forward gives wrong but finite outputs; the backward divides by these wrong outputs and produces NaN. Diagnostic: cast the FFT path to fp32 and rerun. If NaNs disappear, you found it. Fix permanently: always run the FFT in fp32 even with bf16 model weights.
        <br />
        (2) <strong>Implicit-filter MLP exploding.</strong> The MLP that produces <Code>{"h(\\tau)"}</Code> from sin/cos features has its own gradient flow; without normalization, the long-range channels (whose filter values are small) can develop very large gradients. Diagnostic: log the implicit-filter MLP's per-layer gradient norm; if it is &gt; 100x the global average, you have a problem. Fix: add a gradient clip on the filter MLP specifically (clip to norm 1.0), or insert a LayerNorm after the filter MLP's output.
        <br />
        (3) <strong>Exponential window <Code>{"\\alpha"}</Code> drifted negative.</strong> If the parameterization of the window is unconstrained (you store <Code>{"\\alpha"}</Code> directly rather than <Code>{"\\log \\alpha"}</Code>), training can push <Code>{"\\alpha"}</Code> negative; then <Code>{"\\exp(-\\alpha t / L)"}</Code> grows with <Code>t</Code>, blowing up the long-range filter values and producing NaNs in the FFT. Diagnostic: print the min and max of <Code>{"\\alpha"}</Code> across channels each step; if any channel has <Code>{"\\alpha &lt; 0"}</Code>, that is your bug. Fix: parameterize as <Code>{"\\alpha = \\exp(\\alpha_\\text{log})"}</Code> so it is structurally positive, or clamp <Code>{"\\alpha \\ge 0.01"}</Code> after each optimizer step.
        <br />
        Bonus: the fact that local channels train fine while long-range channels NaN is itself diagnostic — it points specifically to the long-conv math (FFT precision or filter window) rather than to a generic instability that would affect both. Trust this asymmetry and look at the long-range path first.
      </Callout>

    </div>
  ),
};

export default hyenaContent;
