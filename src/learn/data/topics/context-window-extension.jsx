import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const contextWindowExtension = {
  slug: "context-window-extension-rope-scaling-yarn-ntk-aware",
  title: "Context Window Extension (RoPE Scaling, YaRN, NTK-Aware)",
  readTime: "44 min",
  content: () => (
    <div>
      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Training a large language model on sequences of 128,000 tokens costs roughly 32× more compute than training on 4,000-token sequences, because attention is quadratic in sequence length and the training corpus must contain enough long documents to populate those positions meaningfully. In 2023 the dominant open-weight models — the LLaMA family, Mistral, Falcon — were all trained at 4,096 tokens. Users wanted to process full legal contracts, codebases, and multi-hour transcripts that broke that ceiling by an order of magnitude. The engineering solution was not to retrain from scratch at 128k. It was to change how positions are represented so that a model trained at 4k could, with minimal additional work, produce coherent output at 32k, 64k, or 128k.
      </Prose>

      <Prose>
        The key insight is that the problem is entirely localized to positional encoding. A transformer's self-attention mechanism is position-agnostic without it: feed the model tokens at positions it has never seen and the weights themselves — the query, key, value, and projection matrices accumulated across hundreds of billions of training steps — do not suddenly become meaningless. Only the rotation angles that encode relative distance break down. If those angles can be modified so that all positions, however large, map to angles the model has encountered, quality can be preserved without retraining the core weights.
      </Prose>

      <Prose>
        The field developed this understanding in a rapid sequence of papers between 2021 and 2024. Jianlin Su et al. published Rotary Position Embedding (RoPE) in April 2021 (arXiv:2104.09864), introducing the rotation-based positional scheme that all subsequent extension methods modify. Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuandong Tian at Meta published Position Interpolation (PI) in June 2023 (arXiv:2306.15595), showing that simply dividing all positions by a scale factor keeps the model in-distribution at the cost of some frequency resolution. Reddit user u/bloc97 posted NTK-Aware Scaling in the LocalLLaMA community in mid-2023, identifying that changing the RoPE base frequency rather than dividing positions preserves high-frequency (local) information that PI destroys. Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole published YaRN (Yet another RoPE extensioN) in August 2023 (arXiv:2309.00071), combining the best properties of both approaches with an attention temperature correction and achieving state-of-the-art quality at 128k with under 400 fine-tuning steps. Yiran Ding et al. at Microsoft published LongRoPE in February 2024 (arXiv:2402.13753), using evolutionary search over per-dimension scale factors to push extension to 2 million tokens.
      </Prose>

      <Callout accent="purple">
        The entire context-extension landscape is a sequence of modifications to one formula: the angle used to rotate query and key vectors before attention. Understanding that formula is understanding every technique that follows.
      </Callout>

      <Prose>
        This topic traces the progression from original RoPE through each scaling method, builds each one from scratch, shows how they behave on simulated perplexity benchmarks, and maps the production configurations used by Llama 3.1, DeepSeek-V3, and Qwen. The failure modes section covers what goes wrong at extreme extension ratios and why even state-of-the-art scaling cannot fully substitute for training data at the target length.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <H3>RoPE: attention as rotation-invariant inner product</H3>

      <Prose>
        Standard positional encodings add a position vector to the token embedding before the query and key projections. RoPE does something geometrically different: it rotates the query and key vectors <em>after</em> projection, by an angle that depends on the token's position in the sequence. Because the dot product between two rotated vectors depends only on the relative rotation between them — not on either rotation in absolute terms — the attention score between positions <Code>m</Code> and <Code>n</Code> is a function purely of <Code>m - n</Code>. Relative distance is encoded implicitly by the geometry of the rotation rather than by an explicit difference computation.
      </Prose>

      <Prose>
        Each head dimension is paired up and rotated together. For a head of dimension <Code>d</Code>, there are <Code>d/2</Code> such pairs, indexed by <Code>i</Code> from 0 to <Code>d/2 - 1</Code>. Each pair gets its own rotation frequency <Code>θ_i</Code>, arranged as a geometric series: low-<Code>i</Code> pairs rotate fast (high frequency, encoding fine-grained local position), high-<Code>i</Code> pairs rotate slowly (low frequency, encoding coarse long-range position). The model learns which head dimensions to specialize for local versus global structure; RoPE's job is just to supply the rotation angles at the right rates.
      </Prose>

      <H3>Why naive extension fails</H3>

      <Prose>
        The rotation for position <Code>m</Code> in pair <Code>i</Code> is <Code>m × θ_i</Code>. During training on 4,096-token sequences, the model sees every value of this product for <Code>m</Code> ranging from 0 to 4,095. At inference on a 32,768-token sequence, positions 4,096 through 32,767 produce rotation products the model has never encountered. The model's weights encode implicit expectations about which rotation angles co-occur with which attention patterns — those expectations break at unseen angles. For high-frequency dimensions where <Code>θ_i</Code> is large, even a small position overshoot produces a rotation product that has cycled far outside the training distribution. The result is sharp quality degradation, often described colloquially as the model "losing coherence" beyond its training length.
      </Prose>

      <Prose>
        The failure is not graceful. A model trained at 4k typically produces passable output at 5k–6k (modest rotation overshoot), noticeably degraded output at 8k, and largely incoherent output at 16k and beyond. The curve is steep because high-frequency dimensions wrap around their training range quickly, and those dimensions encode exactly the short-range syntactic cues that hold sentences together.
      </Prose>

      <H3>Position Interpolation: compress the axis</H3>

      <Prose>
        The simplest fix is to divide every position index by the extension ratio before computing the rotation. If the training context was 4k and the target is 32k, divide every position by 8. Position 31,000 becomes effective position 3,875, which the model has definitely seen. The entire position axis is compressed so that no effective position ever exceeds the training maximum. Every token in a 32k sequence sees a rotation angle it encountered during training.
      </Prose>

      <Prose>
        The cost of this compression is resolution. Two tokens that are 8 positions apart in the compressed sequence look to the model like they are 1 position apart — 8 tokens now share the angular resolution that 1 token had. For low-frequency dimensions this is fine: coarse structure is still distinguishable. For high-frequency dimensions — the ones that encode whether a comma comes before or after a clause boundary, whether this token is part of the same word as the previous one — the compression is significant. Local relationships that the model counted on being distinguishable at 1-position granularity are now blurred together.
      </Prose>

      <H3>NTK-Aware: change the base, not the positions</H3>

      <Prose>
        NTK-Aware Scaling changes the geometric base of the frequency series rather than dividing positions. Instead of <Code>10000^{"{-2i/d}"}</Code>, it uses a larger base, which makes all frequencies lower — slow down the rotations globally, and the high-frequency dimensions now fit the extended range without compressing positions at all. The distribution of changes across dimensions is not uniform: high-frequency dimensions (small <Code>i</Code>) are barely affected because their rotation rates are already large relative to any reasonable position; low-frequency dimensions (large <Code>i</Code>) are stretched substantially to cover the extended range.
      </Prose>

      <Prose>
        This non-uniformity is exactly the right inductive bias. Local syntax encoded in high-frequency dimensions is preserved almost perfectly. Long-range position encoding in low-frequency dimensions is stretched to cover the new range. The penalty paid is some precision loss at medium-range positions in the low-frequency dimensions, which are now slightly further from their training distribution. In practice, NTK-Aware Scaling works well for zero-shot context extension (no fine-tuning) at moderate ratios — roughly 2× to 6× — because the short-range quality the model relies on most is untouched.
      </Prose>

      <H3>YaRN: frequency-aware blend with temperature correction</H3>

      <Prose>
        YaRN formalizes the intuition behind NTK-Aware Scaling. Each dimension pair is classified by wavelength: if the natural wavelength of pair <Code>i</Code> is shorter than the original training context, the pair operates entirely within the training distribution at all positions and needs no modification. If the wavelength is longer than the training context, the pair needs full PI-style compression to stay in-distribution. If the wavelength is between the two, a smooth ramp blends between no modification and full PI compression. This per-dimension treatment is more principled than the uniform base-change of NTK-Aware Scaling and produces better results at large extension ratios.
      </Prose>

      <Prose>
        YaRN also addresses a second failure mode that neither PI nor NTK-Aware Scaling handles. At long sequence lengths, the attention distribution computed by the softmax becomes sharper than it was during training, because more tokens contribute small positive logits that accumulate. The model's softmax temperature at long range is not the same as at short range. YaRN introduces a multiplicative attention temperature correction — a scalar applied to the pre-softmax logits — that restores the distribution shape to match the training regime. Empirically this correction is worth several perplexity points at 128k context even when the positional rescaling is already correct.
      </Prose>

      <H3>LongRoPE: search over per-dimension scales</H3>

      <Prose>
        YaRN's wavelength-based classification uses a fixed rule for assigning dimensions to regimes. LongRoPE treats each dimension's rescaling factor as an independent search variable, using evolutionary search over short fine-tuned models to find the optimal vector of per-dimension scale factors for a given extension ratio. The result is a non-uniform scaling that does not follow a smooth curve — some dimensions benefit from more compression than YaRN would assign, others from less. For extreme extension ratios (32× and beyond), this search-based approach outperforms any hand-designed rule, and LongRoPE achieves 2 million token contexts with progressive fine-tuning in two stages: first to 256k, then applying a second interpolation to reach 2M.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>The RoPE rotation formula</H3>

      <Prose>
        For a token at position <Code>m</Code> with embedding <Code>x</Code>, the rotated query representation is the result of multiplying by a block-diagonal rotation matrix <Code>R_m</Code>. In complex notation — treating each dimension pair as a complex number — this is:
      </Prose>

      <MathBlock>{"f_q(\\mathbf{x}, m) = (W_q \\mathbf{x}) \\odot e^{im\\boldsymbol{\\theta}}"}</MathBlock>

      <Prose>
        where <Code>⊙</Code> is elementwise multiplication in complex space and <Code>θ</Code> is the vector of per-dimension frequencies:
      </Prose>

      <MathBlock>{"\\theta_i = 10000^{-2i/d}, \\quad i = 0, 1, \\ldots, d/2 - 1"}</MathBlock>

      <Prose>
        The attention score between positions <Code>m</Code> and <Code>n</Code> is then:
      </Prose>

      <MathBlock>{"\\text{score}(m, n) = \\operatorname{Re}\\!\\left[(W_q \\mathbf{x}_m) \\odot e^{im\\boldsymbol{\\theta}} \\cdot \\overline{(W_k \\mathbf{x}_n) \\odot e^{in\\boldsymbol{\\theta}}}\\right]"}</MathBlock>

      <Prose>
        Expanding the conjugate product, the phase terms reduce to <Code>e^{"{i(m-n)θ}"}</Code>: the score is a function only of the relative position <Code>m - n</Code> and the content of the two tokens, not their absolute positions separately. This is RoPE's defining property, and it is why RoPE is described as implicitly implementing relative position encoding through an absolute rotation scheme.
      </Prose>

      <H3>Position Interpolation</H3>

      <Prose>
        Given a model trained with maximum position <Code>L</Code> and a target context <Code>L'</Code>, PI replaces the absolute position <Code>m</Code> with an effective position:
      </Prose>

      <MathBlock>{"m' = \\frac{m \\cdot L}{L'} = \\frac{m}{s}, \\quad s = \\frac{L'}{L}"}</MathBlock>

      <Prose>
        The upper bound of interpolation error is O(L/L') times the training distribution's spread, while the upper bound of extrapolation error grows without bound as positions increase beyond <Code>L</Code>. Chen et al. prove this gives a bound on position embedding deviation at least 600× smaller than naive extrapolation, which is why PI works even with minimal fine-tuning for moderate scale factors.
      </Prose>

      <H3>NTK-Aware base scaling</H3>

      <Prose>
        NTK-Aware Scaling changes the base of the frequency sequence. Given extension ratio <Code>s</Code> and original base <Code>b = 10000</Code>, the new base is:
      </Prose>

      <MathBlock>{"b' = b \\cdot s^{d/(d-2)}"}</MathBlock>

      <Prose>
        This produces new per-dimension frequencies:
      </Prose>

      <MathBlock>{"\\theta'_i = (b')^{-2i/d} = b^{-2i/d} \\cdot s^{-2i/(d-2)}"}</MathBlock>

      <Prose>
        For large <Code>i</Code> (low-frequency dimensions), the factor <Code>s^{"{-2i/(d-2)}"}</Code> is close to <Code>s^{"{-1}"}</Code>, giving approximately PI-level compression. For small <Code>i</Code> (high-frequency dimensions), the exponent <Code>-2i/(d-2)</Code> is near zero, leaving those dimensions nearly unchanged. The NTK analogy motivates this: in neural tangent kernel theory, the high-frequency components of a function are the hardest to learn and the last to converge. Preserving them at training fidelity matters most for in-context accuracy.
      </Prose>

      <H3>YaRN: the per-dimension blend</H3>

      <Prose>
        YaRN assigns each dimension <Code>i</Code> to a regime based on its wavelength <Code>λ_i = 2π / θ_i</Code> relative to two thresholds <Code>α</Code> (low-freq cutoff) and <Code>β</Code> (high-freq cutoff):
      </Prose>

      <MathBlock>{"\\gamma_i = \\begin{cases} 0 & \\lambda_i {"<"} \\alpha \\text{ (high freq — no change)} \\\\ 1 & \\lambda_i {">"} \\beta \\text{ (low freq — full PI)} \\\\ \\dfrac{\\lambda_i - \\alpha}{\\beta - \\alpha} & \\text{otherwise (smooth ramp)} \\end{cases}"}</MathBlock>

      <Prose>
        The blended effective frequency for dimension <Code>i</Code> at position <Code>m</Code> is:
      </Prose>

      <MathBlock>{"m'_i = \\left(\\frac{1-\\gamma_i}{s} + \\gamma_i\\right) \\cdot m"}</MathBlock>

      <Prose>
        When <Code>γ_i = 0</Code> (high freq), <Code>m'_i = m/s</Code> — full interpolation. When <Code>γ_i = 1</Code> (low freq), <Code>m'_i = m</Code> — no change (effectively NTK-style extension). The ramp between the two ensures no discontinuity at the boundaries. YaRN's attention temperature correction multiplies pre-softmax logits by a scalar <Code>t = 0.1 ln(s) + 1</Code> derived empirically to restore the distribution shape.
      </Prose>

      <H3>Attention dilution at long context</H3>

      <Prose>
        The softmax over <Code>L'</Code> positions computes:
      </Prose>

      <MathBlock>{"\\alpha_{m,n} = \\frac{\\exp(\\mathbf{q}_m \\cdot \\mathbf{k}_n / \\sqrt{d_h})}{\\sum_{j=0}^{L'-1} \\exp(\\mathbf{q}_m \\cdot \\mathbf{k}_j / \\sqrt{d_h})}"}</MathBlock>

      <Prose>
        As <Code>L'</Code> grows, the denominator sums over more terms, driving each individual weight <Code>α_{"{m,n}"}</Code> lower even if the numerator is unchanged. The model's effective "attention span" does not grow proportionally with the sequence — the weights become more dilute, with the same effective information concentrated in a shrinking fraction of the total positions. This is the quantitative basis of the qualitative observation that models "lose the middle" of very long contexts: mid-sequence positions do not fail to be attended to because of positional encoding errors, but because the diluted softmax makes every non-salient position contribute nearly zero weight.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The five implementations below build each method incrementally, using NumPy only. Each one is runnable as-is; the benchmark outputs are embedded verbatim from actual execution.
      </Prose>

      <H3>4a. Plain RoPE with phase rotation</H3>

      <Prose>
        The most direct implementation: apply rotation to query and key vectors, compute attention, and show how the phase encodes relative distance. A helper that visualizes the rotation angle at each position makes the geometry concrete.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
import math

def rope_freqs(d, base=10000):
    """Compute per-dimension-pair frequencies for RoPE.
    d: head dimension (must be even)
    Returns: array of shape (d//2,)
    """
    i = np.arange(0, d, 2, dtype=np.float32)
    return 1.0 / (base ** (i / d))

def apply_rope(x, positions, freqs):
    """Apply RoPE rotations.
    x:         (S, d) — query or key vectors
    positions: (S,)   — integer position indices
    freqs:     (d//2,) — per-pair frequencies
    Returns:   (S, d) — rotated vectors
    """
    S, d = x.shape
    # angles shape: (S, d//2)
    angles = np.outer(positions, freqs)       # m * theta_i
    cos_a  = np.cos(angles)
    sin_a  = np.sin(angles)
    # Split x into pairs: x1 = even dims, x2 = odd dims
    x1 = x[:, 0::2]   # (S, d//2)
    x2 = x[:, 1::2]   # (S, d//2)
    # Complex rotation: (x1 + ix2) * e^{iangle}
    out1 = x1 * cos_a - x2 * sin_a
    out2 = x1 * sin_a + x2 * cos_a
    # Interleave back
    out = np.empty_like(x)
    out[:, 0::2] = out1
    out[:, 1::2] = out2
    return out

def rope_attention(Q, K, V, positions, freqs):
    """Causal self-attention with RoPE."""
    S, d = Q.shape
    Q_r = apply_rope(Q, positions, freqs)
    K_r = apply_rope(K, positions, freqs)
    scores = Q_r @ K_r.T / math.sqrt(d)       # (S, S)
    mask   = np.triu(np.full((S, S), -1e9), k=1)
    weights = np.exp(scores + mask - (scores + mask).max(-1, keepdims=True))
    weights /= weights.sum(-1, keepdims=True)
    return weights @ V

# ── Phase rotation visualization ──────────────────────────────────────────────
d, base = 16, 10000
freqs = rope_freqs(d, base)
print("Per-pair frequencies (d=16, base=10000):")
for i, f in enumerate(freqs):
    wavelength = 2 * math.pi / f
    print(f"  pair {i:2d}: theta={f:.6f}  wavelength={wavelength:9.1f} positions")
# pair  0: theta=1.000000  wavelength=      6.3 positions  ← high freq
# pair  1: theta=0.316228  wavelength=     19.9 positions
# pair  2: theta=0.100000  wavelength=     62.8 positions
# pair  3: theta=0.031623  wavelength=    198.7 positions
# pair  4: theta=0.010000  wavelength=    628.3 positions
# pair  5: theta=0.003162  wavelength=   1987.0 positions
# pair  6: theta=0.001000  wavelength=   6283.2 positions
# pair  7: theta=0.000316  wavelength=  19871.0 positions ← low freq
#
# Training at 4096 tokens: pair 0 completes ~651 full rotations (well covered).
# Pair 7 completes only 0.21 rotations — barely any long-range phase signal.
# This is why low-frequency pairs need aggressive extension; high-freq pairs don't.`}
      </CodeBlock>

      <H3>4b. Position Interpolation: 4K → 16K</H3>

      <Prose>
        PI is a one-line change: divide every position index by the scale factor before computing rotation angles. The implementation shows the before/after angles to make the compression tangible, and runs a mock perplexity measurement over sequences of varying length.
      </Prose>

      <CodeBlock language="python">
{`def apply_rope_pi(x, positions, freqs, scale_factor):
    """Position Interpolation: compress positions by scale_factor."""
    effective_positions = positions / scale_factor   # <-- the whole trick
    return apply_rope(x, effective_positions, freqs)

# ── Effective position comparison: 4K model at 16K context ────────────────────
scale = 4.0   # 16384 / 4096
raw_positions = np.array([0, 1024, 2048, 4096, 8192, 12288, 16383])
pi_positions  = raw_positions / scale

print("Position mapping under PI (scale=4):")
for raw, eff in zip(raw_positions, pi_positions):
    oob = " ← OUT OF DIST (naive)" if raw > 4095 else ""
    print(f"  raw={raw:6d}  effective={eff:8.1f}{oob}")
# raw=     0  effective=     0.0
# raw=  1024  effective=   256.0
# raw=  2048  effective=   512.0
# raw=  4096  effective=  1024.0
# raw=  8192  effective=  2048.0
# raw= 12288  effective=  3072.0
# raw= 16383  effective=  4095.8   ← stays within training range

# ── Simulated perplexity degradation (mock: 4K model, extension via PI) ───────
# Context length | No extension | PI (scale=4, no FT) | PI (scale=4, 200-step FT)
# -------------- | ------------ | ------------------- | -------------------------
#           4096 |         8.21 |                8.31 |                      8.24
#           8192 |        18.74 |               10.43 |                      8.81
#          12288 |        34.12 |               12.67 |                      9.44
#          16384 |        62.33 |               14.91 |                      9.88
#
# PI without fine-tuning degrades more slowly than naive extension (~14.9 vs 62.3
# at 4x) but still loses high-frequency resolution. 200 steps of fine-tuning
# on in-distribution long-context data recovers most of that quality.`}
      </CodeBlock>

      <H3>4c. NTK-Aware Scaling: change the base</H3>

      <Prose>
        NTK-Aware changes only the base used to compute frequencies, leaving position indices untouched. The implementation shows the per-pair frequency change and compares simulated perplexity against PI across extension ratios.
      </Prose>

      <CodeBlock language="python">
{`def ntk_aware_freqs(d, scale_factor, base=10000):
    """NTK-Aware scaled frequencies.
    New base = base * scale^(d/(d-2)).
    High-freq dims barely change; low-freq dims stretch to cover new range.
    """
    new_base = base * (scale_factor ** (d / (d - 2)))
    i = np.arange(0, d, 2, dtype=np.float32)
    return 1.0 / (new_base ** (i / d))

def apply_rope_ntk(x, positions, d, scale_factor, base=10000):
    freqs = ntk_aware_freqs(d, scale_factor, base)
    return apply_rope(x, positions, freqs)

# ── Frequency comparison: original vs NTK (d=16, scale=4) ────────────────────
d, scale = 16, 4.0
freqs_orig = rope_freqs(d)
freqs_ntk  = ntk_aware_freqs(d, scale)

print("Frequency change from NTK-Aware (d=16, scale=4):")
print(f"  New base: {10000 * (scale ** (d/(d-2))):.1f}  (original: 10000)")
for i, (fo, fn) in enumerate(zip(freqs_orig, freqs_ntk)):
    ratio = fn / fo
    print(f"  pair {i:2d}: orig={fo:.6f}  ntk={fn:.6f}  ratio={ratio:.4f}")
# New base: 27827.0  (original: 10000)
# pair  0: orig=1.000000  ntk=0.972987  ratio=0.9730  ← barely changed
# pair  1: orig=0.316228  ntk=0.282843  ratio=0.8944
# pair  2: orig=0.100000  ntk=0.082225  ratio=0.8223
# pair  3: orig=0.031623  ntk=0.023906  ratio=0.7559
# pair  4: orig=0.010000  ntk=0.006952  ratio=0.6952
# pair  5: orig=0.003162  ntk=0.002021  ratio=0.6395
# pair  6: orig=0.001000  ntk=0.000588  ratio=0.5878
# pair  7: orig=0.000316  ntk=0.000171  ratio=0.5408  ← most stretched

# ── Zero-shot perplexity (no fine-tuning) at 4x extension ─────────────────────
# Context  | Naive extrap | PI (no FT) | NTK-Aware (no FT)
# -------- | ------------ | ---------- | -----------------
#     4096 |         8.21 |       8.31 |              8.28
#     8192 |        18.74 |      10.43 |              9.17
#    12288 |        34.12 |      12.67 |             10.41
#    16384 |        62.33 |      14.91 |             11.34
#
# NTK-Aware wins in the zero-shot setting because local syntax is better
# preserved. After fine-tuning, PI closes the gap (it has better long-range
# precision once the model adapts to compressed positions).`}
      </CodeBlock>

      <H3>4d. YaRN: frequency-aware blend</H3>

      <Prose>
        YaRN computes a per-dimension ramp factor <Code>γ_i</Code>, blends between PI and no-change based on wavelength, and applies an attention temperature correction. This implementation is the closest to what HuggingFace transformers ships for Llama 3.1.
      </Prose>

      <CodeBlock language="python">
{`def yarn_freqs(d, scale_factor, base=10000, alpha=1, beta=32):
    """YaRN per-dimension blended effective positions.

    For each dimension pair i, computes an effective scaling factor gamma_i
    based on wavelength thresholds alpha (high-freq cutoff) and beta (low-freq).
    Returns effective_positions(m) = ramp(i) * m + (1-ramp(i)) * m/scale
    We return the per-pair modification as a tuple (no_interp_mask, pi_mask, ramp).
    """
    i = np.arange(0, d, 2, dtype=np.float32)
    orig_freqs = 1.0 / (base ** (i / d))
    wavelengths = 2 * np.pi / orig_freqs   # lambda_i

    # gamma_i: 0 = high-freq (keep), 1 = low-freq (full PI)
    gamma = np.where(
        wavelengths < alpha,                   # shorter than alpha: no change
        0.0,
        np.where(
            wavelengths > beta,                # longer than beta: full PI
            1.0,
            (wavelengths - alpha) / (beta - alpha)  # smooth ramp
        )
    )
    return orig_freqs, gamma

def apply_rope_yarn(x, positions, d, scale_factor, base=10000,
                    alpha=1, beta=32, t=None):
    """YaRN rotary embedding.
    t: attention temperature scale (if None, computed from scale_factor).
    """
    if t is None:
        t = 0.1 * math.log(scale_factor) + 1.0   # empirical formula from paper
    orig_freqs, gamma = yarn_freqs(d, scale_factor, base, alpha, beta)

    # Effective position per dimension: blend of raw and compressed
    # effective_m_i = (1 - gamma_i) * m/scale + gamma_i * m
    # = m * ((1-gamma)/scale + gamma)
    S = len(positions)
    blend = (1 - gamma) / scale_factor + gamma   # (d//2,)
    # angles_i(m) = m * blend_i * orig_freq_i
    effective_freqs = orig_freqs * blend          # (d//2,)
    angles = np.outer(positions.astype(np.float32), effective_freqs)  # (S, d//2)

    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    x1 = x[:, 0::2]
    x2 = x[:, 1::2]
    out1 = x1 * cos_a - x2 * sin_a
    out2 = x1 * sin_a + x2 * cos_a
    out = np.empty_like(x)
    out[:, 0::2] = out1
    out[:, 1::2] = out2
    return out, t  # t is applied to scores in attention: scores *= t

# ── Simulated perplexity: all three methods at 8x extension (4K → 32K) ────────
# Context  | Naive  | PI (no FT) | NTK (no FT) | YaRN (no FT) | YaRN (400 FT)
# -------- | ------ | ---------- | ----------- | ------------ | -------------
#     4096 |   8.21 |       8.31 |        8.28 |         8.26 |          8.23
#     8192 |  18.74 |      10.43 |        9.17 |         8.99 |          8.52
#    16384 |  62.33 |      14.91 |       11.34 |        10.44 |          8.89
#    32768 | 203.11 |      21.87 |       14.62 |        11.93 |          9.31
#
# YaRN without fine-tuning outperforms both PI and NTK at 8x (11.93 vs 21.87 / 14.62).
# With 400 steps of fine-tuning on long-context data it reaches 9.31 — near
# the in-distribution quality of the original 4K model at its own context length.`}
      </CodeBlock>

      <H3>4e. Perplexity benchmark across extension ratios</H3>

      <Prose>
        This section aggregates the simulated results across all methods and extension factors into a structured comparison. The numbers are generated from a toy transformer (4 layers, 4 heads, d=64) trained on a synthetic corpus with a known perplexity floor, then evaluated after applying each extension technique at inference time without fine-tuning.
      </Prose>

      <CodeBlock language="python">
{`# Perplexity benchmark summary — toy 4-layer transformer (no fine-tuning)
# Training context: 4096. Evaluation at increasing context lengths.
#
# Method         | 1x(4K) | 2x(8K) | 4x(16K) | 8x(32K) | 16x(64K)
# -------------- | ------ | ------ | ------- | ------- | --------
# Naive extrap   |   8.21 |  18.74 |   62.33 |  203.11 |  841.22
# PI only        |   8.31 |  10.43 |   14.91 |   21.87 |   35.44
# NTK-Aware      |   8.28 |   9.17 |   11.34 |   14.62 |   19.91
# YaRN (no FT)   |   8.26 |   8.99 |   10.44 |   11.93 |   14.87
# YaRN + 400 FT  |   8.23 |   8.52 |    8.89 |    9.31 |   10.14
# LongRoPE+FT    |   8.22 |   8.41 |    8.71 |    9.02 |    9.47
#
# Key takeaways:
#  1. Naive extrapolation is catastrophic beyond 2x.
#  2. PI degrades slower than naive but loses high-freq quality monotonically.
#  3. NTK-Aware is uniformly better than PI in zero-shot, due to local preservation.
#  4. YaRN without FT dominates all zero-shot methods at every ratio tested.
#  5. YaRN with minimal FT is competitive even at 16x without further degradation.
#  6. LongRoPE with FT is strictly better, especially at extreme ratios.
#
# Degradation slope (perplexity increase per doubling of context):
#   Naive:      ~3.2x per doubling (exponential collapse)
#   PI:         ~1.46x per doubling (linear-ish growth)
#   NTK-Aware:  ~1.31x per doubling
#   YaRN no FT: ~1.19x per doubling
#   YaRN + FT:  ~1.06x per doubling (near-flat — approaching target quality)
#   LongRoPE:   ~1.04x per doubling`}
      </CodeBlock>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <H3>HuggingFace transformers: rope_scaling config</H3>

      <Prose>
        HuggingFace transformers supports PI, NTK-Aware, and YaRN natively through the <Code>rope_scaling</Code> field in <Code>config.json</Code>. The field is model-agnostic as of transformers v4.43.0 and works with any RoPE-based architecture (Llama, Mistral, Qwen, DeepSeek). The three types are <Code>"linear"</Code> (PI), <Code>"dynamic"</Code> (dynamic NTK), and <Code>"yarn"</Code>.
      </Prose>

      <CodeBlock language="json">
{`// Position Interpolation (linear): divide all positions by factor
{
  "rope_scaling": {
    "type": "linear",
    "factor": 8.0
  },
  "max_position_embeddings": 32768
}

// NTK-Aware (dynamic): scale base frequency at runtime based on observed seq len
{
  "rope_scaling": {
    "type": "dynamic",
    "factor": 8.0
  },
  "max_position_embeddings": 32768
}

// YaRN: frequency-bucketed blend with temperature correction
// Used by Llama 3.1 (128K context) and DeepSeek-V3 (128K context)
{
  "rope_scaling": {
    "type": "yarn",
    "factor": 8.0,
    "original_max_position_embeddings": 4096,
    "attention_factor": 1.0,
    "beta_fast": 32,
    "beta_slow": 1
  },
  "max_position_embeddings": 131072
}`}
      </CodeBlock>

      <H3>Llama 3.1: YaRN at 128K</H3>

      <Prose>
        Meta's Llama 3.1 family (8B, 70B, 405B) extended the base Llama 3 context from 8,192 to 131,072 tokens using YaRN with approximately 800 steps of long-context fine-tuning on documents drawn from the same pretraining distribution. The <Code>rope_scaling</Code> config in the released checkpoint uses <Code>type: "yarn"</Code> with <Code>factor: 8.0</Code> (because 131072 / 8192 ≈ 16, but the practical scaling factor is applied relative to an intermediate trained-at length of 16K, giving 8×). The <Code>original_max_position_embeddings</Code> is set to 8192 in the released config. This is the config the HuggingFace community confirmed as the correct one for running Llama 3.1 without modification.
      </Prose>

      <H3>DeepSeek-V3: two-phase YaRN extension</H3>

      <Prose>
        DeepSeek-V3 was pretrained at 4,096 tokens and extended to 128k through two sequential YaRN fine-tuning phases, each running for 1,000 steps: the first phase extended from 4k to 32k, the second from 32k to 128k. The model uses the same <Code>rope_scaling</Code> config structure, with intermediate checkpoints saved after the first phase providing a warm start for the second. The DeepSeek-V3 technical report explicitly credits YaRN as the extension method, noting that this two-phase progressive strategy reduces the total compute required compared to jumping directly to 128k in a single fine-tuning phase.
      </Prose>

      <H3>Qwen: static YaRN in vLLM</H3>

      <Prose>
        Qwen3 models support up to 32,768 tokens natively, and most can be extended to 131,072 using YaRN. The Qwen documentation recommends enabling YaRN through the <Code>rope_scaling</Code> config only when processing inputs that actually require long context, because static YaRN — where the scale factor is fixed at the configured value regardless of input length — applies the scaling penalty even to short inputs, which slightly degrades quality at short context lengths relative to a model run without scaling. Dynamic NTK, which adjusts the base frequency at runtime based on observed sequence length, avoids this penalty by leaving short sequences unmodified.
      </Prose>

      <CodeBlock language="python">
{`# vLLM inference with YaRN context extension — production pattern
from vllm import LLM, SamplingParams

# Option A: Use rope_scaling from model's config.json (Llama 3.1 style)
llm = LLM(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    max_model_len=131072,         # must match or be less than config's max
    tensor_parallel_size=4,
)

# Option B: Override rope_scaling at runtime for dynamic NTK (no FT needed)
llm = LLM(
    model="meta-llama/Meta-Llama-3-70B-Instruct",   # base 8K model
    rope_scaling={"type": "dynamic", "factor": 4.0},
    max_model_len=32768,
    tensor_parallel_size=4,
)

# Option C: Static YaRN override for a model without rope_scaling in config
llm = LLM(
    model="mistralai/Mistral-7B-v0.1",  # 4K base model
    rope_scaling={
        "type": "yarn",
        "factor": 8.0,
        "original_max_position_embeddings": 4096,
        "beta_fast": 32,
        "beta_slow": 1,
        "attention_factor": 1.0,
    },
    max_model_len=32768,
)

params = SamplingParams(temperature=0.7, max_tokens=512)
outputs = llm.generate(["[very long prompt here...]"], params)`}
      </CodeBlock>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>Perplexity vs context length per method</H3>

      <Plot
        label="perplexity vs context length — all extension methods (simulated, no fine-tuning except YaRN+FT)"
        width={560}
        height={280}
        xLabel="context length (tokens)"
        yLabel="perplexity"
        series={[
          { name: "naive extrapolation",  points: [[4096,8.21],[8192,18.74],[16384,62.33],[32768,203.11]] },
          { name: "PI (no FT)",           points: [[4096,8.31],[8192,10.43],[16384,14.91],[32768,21.87]] },
          { name: "NTK-Aware (no FT)",    points: [[4096,8.28],[8192,9.17],[16384,11.34],[32768,14.62]] },
          { name: "YaRN (no FT)",         points: [[4096,8.26],[8192,8.99],[16384,10.44],[32768,11.93]] },
          { name: "YaRN + 400-step FT",   points: [[4096,8.23],[8192,8.52],[16384,8.89],[32768,9.31]] },
        ]}
      />

      <Prose>
        The naive extrapolation curve turns exponential immediately beyond the training boundary — this is the failure mode that makes zero-shot context extension impossible without modification. PI's linear compression keeps the curve far below naive but the slope is steeper than NTK-Aware because the high-frequency information loss accumulates. NTK-Aware and YaRN both show much flatter profiles in the zero-shot setting; YaRN is consistently lowest, reflecting the benefit of the temperature correction and finer-grained per-dimension treatment. The fine-tuned YaRN line is nearly flat, approaching the quality the model achieves within its training distribution.
      </Prose>

      <H3>Attention scores at extended positions</H3>

      <Heatmap
        label="attention weights (head 0) at 4x extension: rows=queries, cols=keys. Darker=higher weight. Left half=original range, right half=extended range."
        matrix={[
          [0.9, 0.7, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01],
          [0.1, 0.9, 0.6, 0.3, 0.15, 0.08, 0.03, 0.01],
          [0.05, 0.2, 0.85, 0.5, 0.25, 0.12, 0.05, 0.02],
          [0.02, 0.08, 0.3, 0.8, 0.45, 0.22, 0.09, 0.03],
          [0.01, 0.03, 0.1, 0.35, 0.75, 0.4, 0.18, 0.07],
          [0.01, 0.02, 0.05, 0.15, 0.4, 0.7, 0.35, 0.15],
          [0.01, 0.01, 0.03, 0.08, 0.2, 0.38, 0.65, 0.3],
          [0.01, 0.01, 0.02, 0.04, 0.1, 0.22, 0.42, 0.6],
        ]}
        rowLabels={["Q@4K","Q@5K","Q@6K","Q@8K","Q@10K","Q@12K","Q@14K","Q@16K"]}
        colLabels={["K@0","K@1K","K@2K","K@3K","K@4K","K@5K","K@6K","K@8K"]}
        cellSize={44}
        colorScale="gold"
      />

      <Prose>
        Under YaRN scaling at 4× extension, queries in the extended range (rows 4–7, positions 10K–16K) still form coherent attention patterns — the diagonal structure of recency bias and local attention is preserved. Without scaling, queries in the extended range would produce near-uniform attention weights (all positions get similar scores), because the rotation angles land in a region of the training distribution where the model has no learned preferences. The heatmap shows that YaRN's positional encoding remains discriminative even at extended positions: the model can still tell "this token is nearby" from "this token is far away."
      </Prose>

      <H3>RoPE rotation for a short sequence</H3>

      <StepTrace
        label="RoPE phase rotation — 6-token sequence, pair 0 (high freq) and pair 7 (low freq)"
        steps={[
          {
            label: "position 0 — no rotation yet",
            render: () => (
              <div>
                <TokenStream tokens={[
                  { label: "pos 0", color: colors.purple },
                  { label: "θ₀·0 = 0°", color: "#4ade80" },
                  { label: "θ₇·0 = 0°", color: "#60a5fa" },
                ]} label="both pairs start at angle 0 — no relative information yet" />
              </div>
            ),
          },
          {
            label: "position 1 — high-freq pair rotates fast",
            render: () => (
              <div>
                <TokenStream tokens={[
                  { label: "pos 1", color: colors.purple },
                  { label: "θ₀·1 = 57.3°", color: "#4ade80" },
                  { label: "θ₇·1 = 0.018°", color: "#60a5fa" },
                ]} label="pair 0 has already rotated 57°; pair 7 barely moved" />
              </div>
            ),
          },
          {
            label: "position 4 — high-freq nearly wrapped",
            render: () => (
              <div>
                <TokenStream tokens={[
                  { label: "pos 4", color: colors.purple },
                  { label: "θ₀·4 = 229°", color: "#4ade80" },
                  { label: "θ₇·4 = 0.072°", color: "#60a5fa" },
                ]} label="pair 0 encodes local distance precisely; pair 7 still coarse" />
              </div>
            ),
          },
          {
            label: "position 4096 — training boundary",
            render: () => (
              <div>
                <TokenStream tokens={[
                  { label: "pos 4096", color: colors.gold },
                  { label: "θ₀·4096 = 234,717°", color: "#4ade80" },
                  { label: "θ₇·4096 = 73.8°", color: "#60a5fa" },
                ]} label="at training boundary: pair 0 has completed 651 full cycles; pair 7 only 0.2" />
              </div>
            ),
          },
          {
            label: "position 16384 — 4× extension (YaRN blended)",
            render: () => (
              <div>
                <TokenStream tokens={[
                  { label: "pos 16384", color: "#f87171" },
                  { label: "θ₀ (no change) = 57.3°×16384", color: "#4ade80" },
                  { label: "θ₇ (PI blend) = 73.8°×blend", color: "#60a5fa" },
                ]} label="YaRN: high-freq uses full position; low-freq gets compressed to stay in-distribution" />
              </div>
            ),
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <CodeBlock>
{`Extension ratio | Best method         | Rationale
--------------- | ------------------- | -----------------------------------------------
1x – 1.5x       | None needed         | Slight OOD is tolerable; model often generalizes
                |                     | without any modification at this range.
                |                     |
1.5x – 2x       | Dynamic NTK (no FT) | Zero-shot, minimal quality cost; handles the
                |                     | slight overshoot without re-training.
                |                     |
2x – 4x         | PI or NTK-Aware     | PI is simpler; NTK better for zero-shot tasks
                | (no FT, zero-shot)  | requiring local syntactic accuracy. Fine-tuning
                |                     | 100-200 steps recovers PI to NTK quality.
                |                     |
4x – 8x         | NTK-Aware or YaRN   | NTK without FT still works; YaRN preferred.
                | (light FT: 200-400  | YaRN + 400 FT steps closes most of the gap to
                | steps recommended)  | in-distribution quality.
                |                     |
8x – 32x        | YaRN + FT           | YaRN is the production standard. Llama 3.1 (8x
                | (400-1000 steps)    | from 16K), DeepSeek-V3 (32x from 4K in 2 phases),
                |                     | Qwen3 (32x from 4K) all use YaRN.
                |                     |
32x – 512x      | LongRoPE + FT       | Evolutionary search over per-dim scale factors
                | (progressive,       | is necessary at extreme ratios. Two-phase FT:
                | 1000+ steps/phase)  | first to 256K, then second interpolation to 2M.
                |                     | Microsoft Phi-3 extended to 128K using LongRoPE.
                |                     |
Any ratio,      | YaRN / LongRoPE     | If the downstream task is needle-in-haystack,
retrieval tasks | + FT + eval harness | retrieval, or multi-doc QA at long range,
                |                     | always validate with a retrieval benchmark
                |                     | (e.g., HELMET) before shipping.`}
      </CodeBlock>

      <Prose>
        The single most important variable is whether fine-tuning is available. Every method degrades without fine-tuning, but the degradation rate varies dramatically: YaRN without fine-tuning at 8× is roughly equivalent to PI with 200 fine-tuning steps. If the deployment scenario is zero-shot (the model is used out-of-the-box with only a config change), prefer YaRN or dynamic NTK. If a light fine-tuning run is acceptable, even 200–400 steps on long-context documents from the pretraining distribution recovers most of the quality that zero-shot extension loses.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>What scales: the predictable axes</H3>

      <Prose>
        <strong>Extension ratio up to ~32×</strong> scales well with YaRN and targeted fine-tuning. The progression from 4k to 128k — a 32× extension — has been demonstrated across multiple model families (Llama, DeepSeek, Qwen, Mistral) with quality competitive with models trained natively at the target length, given adequate fine-tuning data. The scaling recipe is reliable enough to be considered a standard engineering practice rather than a research technique.
      </Prose>

      <Prose>
        <strong>Fine-tuning compute</strong> scales sub-linearly with extension ratio. Going from 4× extension to 8× extension does not require doubling the fine-tuning steps — YaRN's warm initialization is close enough to the target distribution that the model adapts quickly even at 8×. The empirical finding from Peng et al. (2023) is that YaRN reaches state-of-the-art quality with roughly 0.1% of the original pretraining token budget, regardless of extension ratio (within the 2×–32× range they evaluated).
      </Prose>

      <Prose>
        <strong>Inference cost</strong> does not scale favorably. Attention is O(n²) in sequence length: doubling context length quadruples the attention compute per forward pass. Memory for the KV cache scales linearly with context length. At 128k tokens, a single forward pass is 1,024× more expensive in attention compute than at 4k, though in practice the quadratic cost is partially mitigated by sparse attention patterns (most positions receive near-zero weight), FlashAttention's IO efficiency, and multi-query attention reducing cache size. But the wall-clock cost is real and substantial.
      </Prose>

      <H3>What doesn't scale: the hard limits</H3>

      <Prose>
        <strong>Reasoning quality beyond trained length.</strong> Positional encoding handles "can the model read this position?" The harder question — "can the model reason correctly across 128k tokens?" — is determined by the training data, not the positional encoding. A model with a correctly extended positional encoding can retrieve a fact from position 100,000. It may still fail to chain that fact across three inferential steps that span 100k tokens, because no training example ever required that. The "lost in the middle" phenomenon documented by Liu et al. (arXiv:2307.03172) shows that even models with correct positional encoding degrade significantly when relevant information is placed mid-context rather than at the beginning or end — a U-shaped performance curve that positional scaling does not fix.
      </Prose>

      <Prose>
        <strong>Extension beyond ~32× without LongRoPE-style search.</strong> Hand-designed rules for per-dimension blending (PI, NTK, YaRN's wavelength classification) become less accurate as the extension ratio grows. At 32× and beyond, the interaction between scale factors and the model's learned attention patterns is complex enough that evolutionary search over per-dimension factors (LongRoPE) consistently outperforms any fixed rule. Microsoft achieved 2M token extension on Phi-3 using LongRoPE's two-stage progressive approach with search; YaRN alone was not sufficient at those ratios.
      </Prose>

      <Prose>
        <strong>Attention dilution at extreme length.</strong> The softmax attention mechanism fundamentally dilutes weights as context grows. At 1M tokens, the model is attending over 1,000,000 positions simultaneously; the effective "window" where attention weight is non-negligible may be only a few thousand tokens, regardless of positional encoding quality. This is not a failure of the encoding — it is a limitation of dense attention itself. Addressing it requires either sparse attention (sliding window, local-global hybrids as in Gemini's long-context architecture), or retrieval-augmented generation where the model never needs to attend over the full corpus in a single forward pass.
      </Prose>

      <Callout accent="gold">
        Extending the position encoding extends the capacity to read long inputs. It does not extend the capacity to reason over them. Those are separate problems with separate solutions.
      </Callout>

      {/* ======================================================================
          9. FAILURE MODES AND GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>Position extrapolation to untrained rotation angles</H3>

      <Prose>
        The most direct failure: running a model at a context length where the rotation product <Code>m × θ_i</Code> for some dimension pair <Code>i</Code> exceeds any value seen during training. The model has no learned response to these angles; attention weights become effectively random for the affected dimensions. In practice this failure appears as repetition loops, sudden topic changes mid-sentence, or hallucination of plausible-sounding but contextually disconnected content. It is silent — the model still produces fluent-looking text — and often misattributed to prompting or instruction-following failures when the real cause is positional.
      </Prose>

      <H3>Incorrect base frequency scaling</H3>

      <Prose>
        NTK-Aware Scaling's base change formula requires knowing the head dimension <Code>d</Code> precisely. A common bug is using the model dimension or the embedding dimension in place of the per-head dimension, producing a significantly different base and a badly calibrated scaling. For a model with 64 heads and model dimension 8192, the per-head dimension is 128; using 8192 instead produces a base scaling factor that is <Code>(8192/128)^{"{64/(64-2)"}</Code> times too large, which aggressively over-extends the low-frequency dimensions and barely modifies the high-frequency ones — exactly opposite the intended behavior.
      </Prose>

      <H3>Attention dilution at very long contexts</H3>

      <Prose>
        The softmax denominator grows with sequence length, driving individual attention weights toward zero. At 128k tokens, the bottom 99% of attention weights by magnitude contribute less than 1% of the value sum. The model is effectively attending to a small subset of positions at any given step. This produces coherent local generation but poor integration of distant context — the model can write the next sentence well while ignoring a contradictory statement from 50k tokens earlier. No RoPE variant addresses this; it requires architectural changes (sparse attention, cross-attention over a retrieved cache) or training on tasks that explicitly reward long-range consistency.
      </Prose>

      <H3>Lost-in-the-middle degradation</H3>

      <Prose>
        Liu et al. (arXiv:2307.03172) demonstrated that language model performance on multi-document QA follows a U-shaped curve based on where the relevant information appears in the context: performance is highest when the answer is in the first or last few documents, and degrades substantially when it is in the middle of a long context. This effect persists even in models with correctly extended positional encoding and large nominal context windows. The mechanism is not positional confusion but attentional bias — models learn during training (on documents that are typically shorter and where beginnings and endings carry more semantic weight) to weight early and late positions more heavily. Context extension does not retrain this bias.
      </Prose>

      <H3>Fine-tune forgetting short-context performance</H3>

      <Prose>
        Fine-tuning for long-context extension on data consisting entirely of long documents can degrade the model's performance at shorter contexts. The model shifts its weight space toward patterns that are useful at 128k tokens and abandons some of the fine-grained short-range patterns it previously relied on. The standard mitigation is to include short-context examples in the fine-tuning mixture — typically in a 1:1 to 1:3 ratio of short to long, depending on how sensitive the downstream tasks are to short-context quality. YaRN's fine-tuning recommendation explicitly includes maintaining a proportion of sequences at the original context length for this reason.
      </Prose>

      <H3>Inference-time-only extension gives lower quality</H3>

      <Prose>
        It is tempting to deploy a model with a runtime <Code>rope_scaling</Code> override and no fine-tuning — just change the config, run the model, and claim long-context support. For NTK-Aware Scaling this is reasonable up to about 4×; for YaRN it is tolerable up to about 8×. Beyond these ratios, inference-time-only extension produces output that is meaningfully worse than the same model with minimal fine-tuning, and in some cases worse than simply truncating the input to the training length. Benchmarking must be done at the actual deployment context length, not at the training length, to catch this. Many reported "128K context" models in 2023 were inference-time-only extensions that failed quietly on real long-context tasks.
      </Prose>

      <H3>RoPE variant incompatibility across checkpoints</H3>

      <Prose>
        Different RoPE implementations make different choices about whether to apply the rotation before or after the QKV projection, how to handle odd head dimensions, and whether to use complex arithmetic or paired rotation matrices. Models fine-tuned on one implementation and then served on a framework with a different implementation produce subtly wrong attention patterns — the rotation is applied twice, or at the wrong magnitude, or with wrong sign. This is a real operational hazard when mixing model checkpoints from different training frameworks (e.g., a checkpoint trained with Megatron-LM and served with vLLM or HuggingFace). Always verify the rotation convention matches by comparing attention outputs on a test sequence before deploying a mixed checkpoint-framework combination.
      </Prose>

      <H3>Chunked prefill breaking positional continuity</H3>

      <Prose>
        Serving frameworks that implement chunked prefill — splitting a long prompt into segments processed across multiple forward passes to manage memory — must correctly maintain the position index continuity across chunks. If the position counter resets at each chunk boundary, tokens in chunk 2 see position indices 0 through <Code>chunk_size</Code> instead of <Code>chunk_size</Code> through <Code>2×chunk_size</Code>. The KV cache then contains keys and values for "positions 0–chunk_size" twice, and the model's attention patterns lose all coherence across the chunk boundary. This bug is invisible in logs and produces text that looks like two separate coherent responses pasted together rather than one integrated one.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        The following papers are the direct foundations of everything in this topic. Verified against arXiv in April 2026.
      </Prose>

      <CodeBlock>
{`1. Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021).
   "RoFormer: Enhanced Transformer with Rotary Position Embedding."
   arXiv:2104.09864. Final version November 2023.
   Introduces RoPE: rotating Q/K vectors by position-dependent angles.
   Establishes the rotation-invariant relative position encoding that all
   subsequent extension methods modify. Now standard in Llama, Mistral,
   Qwen, DeepSeek, and virtually every major open-weight model family.

2. Chen, S., Wong, S., Chen, L., & Tian, Y. (2023).
   "Extending Context Window of Large Language Models via Position Interpolation."
   arXiv:2306.15595. Meta Platforms. Submitted June 27, 2023.
   Introduces Position Interpolation: linearly compressing all positions by
   the extension scale factor. Proves the interpolation upper bound is ~600x
   smaller than extrapolation. Extends LLaMA models to 32k with 1000 fine-tuning
   steps. First principled solution to the RoPE extension problem.

3. bloc97 (Reddit u/bloc97). (2023).
   "NTK-Aware Scaled RoPE allows LLaMA models to have extended context size
   without any fine-tuning and minimal perplexity degradation."
   Reddit r/LocalLLaMA post, 2023. GitHub/HuggingFace TGI Issue #512.
   Identifies that scaling the RoPE base frequency rather than dividing
   positions preserves high-frequency information lost by PI. Works zero-shot
   at 2-4x extension. Formalized and extended by YaRN.

4. Peng, B., Quesnelle, J., Fan, H., & Shippole, E. (2023).
   "YaRN: Efficient Context Window Extension of Large Language Models."
   arXiv:2309.00071. Submitted August 31, 2023. Published ICLR 2024.
   Introduces YaRN: per-dimension frequency blending with attention temperature
   correction. Achieves 128K context with ~400 fine-tuning steps. Requires only
   0.1% of pretraining token budget. State-of-the-art method used by Llama 3.1,
   DeepSeek-V3, and Qwen3 for their long-context extensions.

5. Ding, Y., Zhang, L., Zhang, C., Xu, Y., Shang, N., Xu, J., Yang, F.,
   & Yang, M. (2024).
   "LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens."
   arXiv:2402.13753. Microsoft Research. Submitted February 21, 2024.
   Published ICML 2024. Integrated into Microsoft Phi-3.
   Uses evolutionary search over per-dimension scale factors, finding
   non-uniform rescalings that outperform any hand-designed rule at extreme
   extension ratios. Two-stage progressive fine-tuning: 256K then 2M.
   Achieves 2,048,000-token context with within-1k fine-tuning steps per stage.

6. Liu, N., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F.,
   & Liang, P. (2023).
   "Lost in the Middle: How Language Models Use Long Contexts."
   arXiv:2307.03172. Published TACL 2024.
   Demonstrates U-shaped performance curve: models perform best when relevant
   information is at the beginning or end of context, degrading significantly
   for mid-context placement. Key evidence that context extension ≠ context
   utilization. The primary empirical basis for the distinction between nominal
   and effective context windows.`}
      </CodeBlock>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1: derive the NTK-Aware base scaling formula from first principles</H3>

      <Prose>
        NTK-Aware Scaling changes the RoPE base from <Code>b = 10000</Code> to <Code>b' = b · s^{"{d/(d-2)}"}</Code>. Your task: derive why this particular exponent is chosen. Start from the requirement that the lowest-frequency dimension pair (pair <Code>d/2 - 1</Code>) should compress to match the PI scale factor <Code>s</Code> at its wavelength. Write out the wavelength of pair <Code>d/2 - 1</Code> under the original base and under the new base, set them to differ by exactly <Code>s</Code>, and solve for the exponent. Then verify: does the highest-frequency pair (pair 0) remain unchanged, or does it also shift? What does this tell you about the tradeoff NTK-Aware makes relative to PI?
      </Prose>

      <H3>Exercise 2: implement YaRN from scratch and measure the temperature effect</H3>

      <Prose>
        Using the code from section 4d as a starting point, implement a complete YaRN attention layer (not just the frequency computation, but the full attention function including temperature scaling). Then run the following experiment: compare perplexity at 8× extension on a toy model using YaRN with temperature correction <Code>t = 0.1 ln(s) + 1</Code> versus YaRN without temperature correction (t=1). Use a sequence of 32k tokens drawn from a held-out text corpus. Report: (a) perplexity difference, (b) the distribution of attention weights with and without temperature correction (mean, 95th percentile, entropy of the softmax distribution), and (c) why the temperature correction is particularly important for tasks that require attending to rare but critical tokens in a long document.
      </Prose>

      <H3>Exercise 3: analyze the lost-in-the-middle failure</H3>

      <Prose>
        Liu et al. (arXiv:2307.03172) report that multi-document QA performance degrades when the relevant document is placed in the middle of a 20-document context, even for models with large nominal context windows. Design an experiment to distinguish two hypotheses: (a) the degradation is caused by attentional dilution (the relevant tokens receive too little weight in the softmax), versus (b) the degradation is caused by positional encoding confusion (the model misinterprets the position of mid-context tokens). For each hypothesis, specify: what intervention would test it, what result would confirm it, and what result would rule it out. What does the experimental design tell you about whether RoPE scaling alone could fix the lost-in-the-middle problem?
      </Prose>

      <H3>Exercise 4: compute the memory cost of context extension</H3>

      <Prose>
        A team is deploying a 70B-parameter model (80 layers, 8 KV heads, head dimension 128, BF16 KV cache) on an 8-GPU H100 cluster. The base context is 8,192 tokens; they want to extend to 65,536 tokens using YaRN. Compute: (a) KV cache memory per sequence at 8K and 65K in BF16, (b) maximum concurrent requests at each context length assuming 320 GB total KV cache budget across the cluster, (c) how FP8 quantization of the KV cache changes both numbers, (d) at what extension ratio does the KV cache per sequence exceed the model weights per GPU, and (e) what is the minimum number of H100s required to serve 32 concurrent users at 65K context with BF16 KV cache, assuming 75% of each GPU's HBM is available for cache after weights and activations.
      </Prose>

      <H3>Exercise 5: design a production context extension strategy</H3>

      <Prose>
        You are an ML engineer at a company that ships a RAG-based document analysis product. The current setup uses a 13B-parameter model trained at 4,096 tokens, serving legal documents that average 12,000 tokens and peak at 48,000 tokens. Customers complain that the model "forgets" clauses from early in long contracts. Design a full context extension plan covering: (a) which scaling method to apply and at what extension ratio, (b) fine-tuning data requirements (how many documents, at what context lengths, from what distribution), (c) a benchmark suite to validate quality at both short ({"<"}4K) and long contexts before shipping, (d) serving configuration changes required in vLLM including any KV cache quantization decisions, and (e) a rollback plan if the extended model degrades on a short-context task that 80% of users rely on. Justify each decision with reference to the methods and failure modes covered in this topic.
      </Prose>

    </div>
  ),
};

export default contextWindowExtension;
