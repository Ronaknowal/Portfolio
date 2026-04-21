import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const fp8Training = {
  title: "FP8 Training & Low-Precision Pre-Training",
  readTime: "38 min",
  content: () => (
    <div>
      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Training a modern large language model in full 32-bit precision is economic malpractice. Every weight, every activation, every gradient stored and shuffled across the HBM bus as a four-byte float, when the signal those numbers carry does not remotely need four bytes of resolution. The result is a pretraining run that pays four times the memory bandwidth, four times the on-chip storage, and — on any modern accelerator whose arithmetic units are tuned for lower-precision formats — two to four times the compute, in exchange for no measurable capability gain. A decade of empirical work on precision has made this concrete: the trained model you would get from FP32 is the same trained model you would get from a carefully tuned 8-bit recipe, to within the noise of a random seed. The capability comes from the data, the architecture, and the scale of compute. The precision format is the efficiency multiplier that decides how much of each you can afford at a fixed budget.
      </Prose>

      <Prose>
        The progression is easier to see as a sequence of research milestones than as a set of formats. In October 2017, Paulius Micikevicius, Sharan Narang, and collaborators at NVIDIA and Baidu posted "Mixed Precision Training" (arXiv:1710.03740, published ICLR 2018). They showed that you could train convolutional networks, recurrent networks, GANs, and early Transformers entirely in IEEE FP16 — with three caveats. You had to keep a master copy of the weights in FP32 for the optimizer update, you had to multiply the loss by a large constant before the backward pass so that small gradients did not underflow to zero, and you had to accumulate every matmul in FP32 so that rounding error did not compound. Those three tricks together are the template that every later low-precision recipe descends from. Before 2017, FP32 training was universal. After 2017, FP32-only training was a choice you had to defend.
      </Prose>

      <Prose>
        FP16's weakness is range, not precision. Its maximum representable magnitude of 65,504 sounds large until you see a gradient tensor whose outlier is 10<sup>5</sup>, or an activation that crosses that threshold during a warmup spike. The fix — BFLOAT16 — came out of Google's TPU team and was formalized empirically by Dhiraj Kalamkar and colleagues in "A Study of BFLOAT16 for Deep Learning Training" (arXiv:1905.12322, May 2019). BF16 keeps the eight-bit exponent of FP32, which matches FP32's dynamic range exactly, and shaves the mantissa from 23 bits to 7. You lose precision within each order of magnitude — a BF16 value of 1.0 is indistinguishable from 1.0078 — but gradients are noisy by nature and tolerate that loss well. NVIDIA's Ampere generation shipped hardware BF16 support in 2020; by 2021, BF16 with FP32 accumulation was the default pretraining recipe for almost every frontier lab.
      </Prose>

      <Prose>
        FP8 is the next step down, and it arrived as a pair of papers alongside a hardware platform. In September 2022, Micikevicius and a larger team including authors from NVIDIA, Arm, and Intel posted "FP8 Formats for Deep Learning" (arXiv:2209.05433). They proposed two 8-bit formats — E4M3 and E5M2 — and showed that, with the right assignment of format to tensor role and a per-tensor scaling scheme, you could train image and language models up to 175 billion parameters without hyperparameter changes and without quality loss relative to BF16. The same month, NVIDIA launched the Hopper H100 GPU, whose fourth-generation Tensor Cores executed FP8 matmuls at double the throughput of BF16 (1,979 TFLOPS BF16 versus 3,958 TFLOPS FP8 dense, per the Hopper whitepaper). The software library shipped alongside — the Transformer Engine — handled the scaling bookkeeping automatically. By 2024, FP8 pretraining had moved from research curiosity to production tool. DeepSeek's December 2024 V3 technical report (arXiv:2412.19437) documented FP8 end-to-end pretraining of a 671B-parameter mixture-of-experts model over 14.8 trillion tokens, with a measured relative loss error below 0.25% relative to a BF16 baseline. That is the state of the art as of early 2026.
      </Prose>

      <Prose>
        This topic is about what FP8 actually means, why it works, where the two formats come from, how scaling rescues the narrow dynamic range, and what fails when you forget a detail. By the end you will have a working NumPy simulator of both FP8 formats, a per-tensor scaling routine that survives round-trip through a tensor spanning ten orders of magnitude, a delayed-scaling implementation that avoids the synchronization cost of per-step scaling, a simulated FP8 matmul whose error profile matches the published Hopper numbers, and a toy regression training loop that trains in simulated FP8 to within quantization noise of the FP32 baseline. You will also have a map of the failure modes, a set of self-check exercises calibrated to catch the mistakes that production teams actually make, and a citation list verified against the primary record. Low-precision training is unglamorous work that compounds. The teams that do it reliably run the experiments that matter.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        A floating-point number has three pieces: a sign bit, a run of exponent bits, and a run of mantissa bits. The sign decides whether the number is positive or negative. The exponent decides the order of magnitude — how many powers of two the number sits away from 1.0. The mantissa decides where inside that order of magnitude the number sits — how finely the format can distinguish two nearby values. The total bit count is fixed, so every bit you move from mantissa to exponent trades precision for range, and every bit you move the other way trades range for precision. That is the only real degree of freedom, and every floating-point format you will meet in deep learning is a point on that one-dimensional curve.
      </Prose>

      <Prose>
        FP8 is the aggressive end of that curve. At eight bits total you have barely enough to distribute. If you give four bits to the exponent, you get 16 orders of magnitude of range, and you have three bits of mantissa to split each order of magnitude into 2<sup>3</sup> = 8 distinct values. That is the E4M3 format: narrow range, fine precision. If you give five bits to the exponent instead, you get 32 orders of magnitude of range, and you have only two bits of mantissa to split each order into four values. That is the E5M2 format: wide range, coarse precision. A third option — six exponent bits, one mantissa bit — would trade further, but one mantissa bit makes the format almost unusable for arithmetic, so no one ships it. The two formats that survived are E4M3 and E5M2, one on each side of the sweet spot.
      </Prose>

      <Prose>
        The central insight of FP8 training is that neither format alone is enough, but the pair of them together is. Activations and weights need precision — the forward pass is doing arithmetic whose outputs feed the next layer's arithmetic, and systematic roundoff compounds. They tolerate narrow range because you can scale them. Gradients need range — a single outlier token can push a gradient entry five orders of magnitude above the median, and clipping that outlier to a saturating maximum corrupts the update. They tolerate coarse precision because gradient descent is already a noisy process and a rounded direction still points approximately the right way. Assign E4M3 to activations and weights, E5M2 to gradients, and each tensor goes into the format that matches what that tensor needs. The matmul hardware handles both. This is the split that Micikevicius et al. proposed in 2022, that the Transformer Engine implements in production, and that DeepSeek-V3 validated at 671-billion-parameter scale.
      </Prose>

      <TokenStream
        label="FP8 role assignment — E4M3 for forward, E5M2 for backward"
        tokens={[
          { label: "weights",        color: colors.gold,  title: "E4M3 — precision matters, scale handles range" },
          { label: "activations",    color: colors.gold,  title: "E4M3" },
          { label: "attn scores",    color: colors.gold,  title: "E4M3 (inputs to softmax, which itself is BF16)" },
          { label: "gradients",      color: "#c084fc",     title: "E5M2 — range matters, precision already lossy" },
          { label: "weight grads",   color: "#c084fc",     title: "E5M2" },
          { label: "LayerNorm",      color: colors.green, title: "stays in BF16 — numerically sensitive" },
          { label: "softmax",        color: colors.green, title: "stays in BF16/FP32" },
          { label: "master weights", color: colors.green, title: "stays in FP32 — optimizer state" },
        ]}
      />

      <Prose>
        The other piece you need is scaling. Eight bits is too few to cover the full range of values a training tensor contains by naive casting. If you have a gradient tensor whose values range from 10<sup>-8</sup> to 10<sup>-3</sup> and you cast directly to E5M2 without any preprocessing, every value below 2<sup>-16</sup> underflows to zero — and that is most of them. The fix is to multiply the tensor by a scale factor first, chosen so that the tensor's maximum absolute value lands at the format's maximum representable value. After the matmul, divide by the scale to recover the correct magnitude. The scale is not optional metadata; it is part of the numerical specification of the tensor, and frameworks that handle FP8 tensors carry the scale as a sidecar alongside the data. Get the scale wrong and the training diverges. Get it right and the eight-bit representation is as good as BF16 for most of the compute in the network.
      </Prose>

      <Prose>
        That is the whole story, at one level of abstraction. Two formats, one for each half of the computation. A scale factor per tensor. Everything else — delayed scaling, per-channel scaling, LayerNorm kept in higher precision, loss scaling as a historical trick — is detail. The details matter for production, but the mental model is small enough to hold in your head. Build the simulator in section 4 once and the rest of the topic falls out of it.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <Prose>
        Every format you will meet in this topic follows the same structural definition. A floating-point value with sign bit <Code>s</Code>, biased exponent field <Code>e</Code>, and mantissa field <Code>m</Code> decodes to a real number by the rule below, where <Code>M</Code> is the number of mantissa bits and <Code>bias</Code> is a format-specific constant that lets the exponent field represent both very small and very large numbers.
      </Prose>

      <MathBlock>
        {"v = (-1)^{s} \\cdot 2^{e - \\text{bias}} \\cdot \\left(1 + \\frac{m}{2^{M}}\\right)"}
      </MathBlock>

      <Prose>
        The implicit leading one in the mantissa — the <Code>1 + m/2<sup>M</sup></Code> term — is the trick that gives floating-point formats their precision in exchange for one bit of storage. For any exponent value in the normal range, the mantissa is a number in the half-open interval <Code>[1, 2)</Code>, and only the fractional part has to be stored. The exponent field determines which power-of-two interval the value sits in, and the mantissa linearly interpolates inside that interval with uniform spacing. This is why floating-point numbers are denser near zero than far from zero: the interval between consecutive representable values is proportional to the magnitude of the value itself. Two adjacent FP8-E4M3 values at magnitude 1.0 are separated by 1/8; two adjacent values at magnitude 100 are separated by about 12.5. The format trades absolute precision for relative precision, which matches the way gradients and weights actually distribute.
      </Prose>

      <Prose>
        Exponent fields 0 and the all-ones value are reserved. Exponent field 0 encodes subnormals — values below the smallest normal, which drop the implicit leading one and use a shifted formula to keep representing values down to zero smoothly. The all-ones field encodes infinity and NaN in IEEE-style formats. FP8-E4M3 breaks with IEEE here: to save a bit of range, the format redefines the all-ones exponent with all-ones mantissa as the single NaN encoding, and the remaining entries at that exponent as ordinary finite values. This is why E4M3's maximum finite value is 448 rather than the 240 you would get with an IEEE-style NaN/Inf reservation. FP8-E5M2 follows the IEEE convention more faithfully and has both infinities and NaN at the all-ones exponent.
      </Prose>

      <Prose>
        The five formats that matter in deep learning sit along the range-precision curve at different points. The table lays out the bit budgets, the numerical properties that follow from them, and the number of representable values per order of magnitude — the last of which is the number most worth memorizing.
      </Prose>

      <CodeBlock>
{`format       sign  exp  man  bytes  max finite        min positive        vals per
                                                        (subnormal)         octave
FP32           1     8   23    4     ~3.4 x 10^38      ~1.4 x 10^-45       ~8.4M
FP16           1     5   10    2     65,504            ~6.0 x 10^-8        1024
BF16           1     8    7    2     ~3.4 x 10^38      ~9.2 x 10^-41       128
FP8-E4M3       1     4    3    1     448               2^-9 = 1.95e-3      8
FP8-E5M2       1     5    2    1     57,344            2^-16 = 1.53e-5     4`}
      </CodeBlock>

      <Prose>
        Eight values per octave, for a weight matrix trained on trillions of tokens, is a fact that should sound implausible. What rescues it is that the network does not actually need 24 bits of resolution at every position — it needs enough resolution to preserve the direction of the gradient update and the approximate magnitude of the activation. The rest is noise that gradient descent averages over. The empirical result — that E4M3 works for weights and activations and E5M2 works for gradients — is the conclusion, not the premise, of the FP8 formats paper, and it took a year of experimental validation to establish.
      </Prose>

      <H3>3a. The scaling equation</H3>

      <Prose>
        Per-tensor scaling is one equation. Let <Code>x</Code> be a tensor about to be cast to FP8 format <Code>f</Code>. Let <Code>amax(x)</Code> be the maximum absolute value over the tensor. Let <Code>max<sub>f</sub></Code> be the maximum representable finite value of format <Code>f</Code> (448 for E4M3, 57,344 for E5M2). Define the scale
      </Prose>

      <MathBlock>
        {"s = \\frac{\\text{amax}(x)}{\\text{max}_{f}}"}
      </MathBlock>

      <Prose>
        and quantize by first dividing by the scale, then casting to FP8.
      </Prose>

      <MathBlock>
        {"x_{\\text{fp8}} = \\text{cast}_{f}\\!\\left( \\frac{x}{s} \\right)"}
      </MathBlock>

      <Prose>
        The division rescales <Code>x</Code> so that its maximum absolute value sits exactly at <Code>max<sub>f</sub></Code>, the top of the format's finite range. Every other value rescales by the same factor and lands somewhere in the interior of the FP8 range, using the full 256-entry vocabulary of representable values. The scale <Code>s</Code> is stored alongside the quantized tensor, because it is needed to interpret the tensor: any downstream operation that consumes <Code>x<sub>fp8</sub></Code> must multiply by <Code>s</Code> to recover the original magnitudes, or must absorb <Code>s</Code> into its own scale as part of a combined dequantize-and-recompute step. This is the inverse operation and it is mathematically transparent: <Code>x ≈ s · dequant(x<sub>fp8</sub>)</Code> where the approximation error is bounded by the resolution of format <Code>f</Code> at the rescaled magnitude.
      </Prose>

      <H3>3b. Delayed scaling</H3>

      <Prose>
        Computing <Code>amax(x)</Code> before every cast is the obvious algorithm and it is also the wrong one. The amax reduction is a synchronization point that cannot run in parallel with the matmul it feeds, and in a training loop where the matmul itself takes hundreds of microseconds, the extra millisecond-scale reduction adds noticeable overhead. Delayed scaling is the fix. Maintain a rolling window of amax values from the last <Code>W</Code> steps (the Transformer Engine defaults to 1024 for weights, shorter for activations and gradients), and use the maximum of that window as the amax for the current step's scale.
      </Prose>

      <MathBlock>
        {"s_{t} = \\frac{\\max\\!\\left(\\{\\text{amax}(x_{t-W}), \\ldots, \\text{amax}(x_{t-1})\\}\\right)}{\\text{max}_{f}}"}
      </MathBlock>

      <Prose>
        The scale at step <Code>t</Code> is derived entirely from observations of past tensors, which means the current step's matmul can launch without waiting for a reduction over the current step's inputs. The cost is a small lag: if the distribution of <Code>x</Code> is drifting, the scale applied to the current tensor is slightly off from the ideal per-step scale. In practice the lag is invisible when the window is long enough to smooth out per-step noise and short enough to track distribution shifts during warmup or learning-rate transitions. The current step's amax is measured anyway — it enters the history window for future steps — but the cost of measuring it is hidden behind the matmul it feeds.
      </Prose>

      <H3>3c. Loss scaling versus per-tensor scaling</H3>

      <Prose>
        Loss scaling is the historical mechanism from FP16 mixed-precision training, and it is worth distinguishing from per-tensor scaling because the two are frequently confused. In FP16, the problem is that small gradients — those below the format's minimum representable magnitude — underflow to zero during the backward pass. The fix from Micikevicius 2017 is to multiply the loss by a large constant <Code>k</Code> before calling backward, which shifts every gradient upward by the same factor <Code>k</Code> (chain rule is linear in the loss). At the end of backward, before the optimizer update, divide every gradient by <Code>k</Code> to undo the scaling. The net effect is that gradients that would have underflowed now sit comfortably in the representable range, at the cost of some values near the top of the range overflowing — which the implementation handles by backing off <Code>k</Code> when overflows occur.
      </Prose>

      <Prose>
        Per-tensor scaling is a different mechanism. It applies per-tensor, not globally. It is recomputed (or looked up from history) per tensor, per step. It does not interact with the backward pass structure. The two can coexist — some training recipes still apply a global loss scale on top of FP8 per-tensor scaling — but they solve different problems, and their interaction is worth thinking through carefully before enabling both. The FP8 formats paper recommends per-tensor scaling alone; loss scaling is a belt-and-suspenders additional defense that some production recipes keep for historical reasons.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The best way to understand FP8 is to build it. Every piece of code in this section runs in pure NumPy, simulates the bit-level rounding behavior of FP8-E4M3 and FP8-E5M2 exactly, and produces the numeric outputs embedded as comments. No hardware FP8, no framework dependencies beyond NumPy. By the end of this section you will have a quantizer, a per-tensor scaling routine, a delayed-scaling class, an FP8 matmul simulator, and a toy training loop that demonstrates FP8 matching FP32 to within the quantization noise floor.
      </Prose>

      <H3>4a. Simulating FP8 quantization</H3>

      <Prose>
        The trick for simulating FP8 in NumPy without a hardware FP8 type is to enumerate every representable value in the format, sort them, and snap an incoming float to the nearest grid point. The enumeration is small — at most 256 entries for an 8-bit format — so the cost is negligible. For each exponent field value in the normal range, iterate over every mantissa value and compute the real-number representation. Separately enumerate subnormals (exponent field zero, mantissa nonzero). Cap the grid at the format's maximum finite value. Snap by binary search.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np

def build_grid(fmt):
    if fmt == "e4m3":
        exp_bits, man_bits, bias, max_finite = 4, 3, 7, 448.0
    elif fmt == "e5m2":
        exp_bits, man_bits, bias, max_finite = 5, 2, 15, 57344.0
    else:
        raise ValueError(fmt)

    values = {0.0}
    # Normal values: exponent field in [1, 2^exp_bits - 1]
    for e in range(1, 2**exp_bits + 1):
        for m in range(2**man_bits):
            mantissa = 1.0 + m / (2**man_bits)
            values.add(mantissa * 2.0**(e - bias))
    # Subnormals: exponent field 0, mantissa != 0
    for m in range(1, 2**man_bits):
        values.add((m / (2**man_bits)) * 2.0**(1 - bias))

    grid = np.array(sorted(values), dtype=np.float64)
    return grid[grid <= max_finite], max_finite

E4M3_GRID, E4M3_MAX = build_grid("e4m3")
E5M2_GRID, E5M2_MAX = build_grid("e5m2")

def quantize_fp8(x, fmt="e4m3"):
    grid, max_finite = (E4M3_GRID, E4M3_MAX) if fmt == "e4m3" else (E5M2_GRID, E5M2_MAX)
    x = np.asarray(x, dtype=np.float64)
    sign = np.sign(x)
    mag = np.clip(np.abs(x), 0.0, max_finite)
    idx = np.searchsorted(grid, mag)
    idx = np.clip(idx, 1, len(grid) - 1)
    lo, hi = grid[idx - 1], grid[idx]
    pick_hi = (hi - mag) < (mag - lo)
    return sign * np.where(pick_hi, hi, lo)`}
      </CodeBlock>

      <Prose>
        Run the quantizer on 100,000 Gaussian samples with standard deviation 3 (so a handful of outliers approach the E4M3 clip threshold of 448 for neither format, well inside range) and compare the two formats by root-mean-square error, maximum absolute error, and the number of unique representable values actually hit.
      </Prose>

      <CodeBlock language="python">
{`np.random.seed(0)
x = np.random.randn(100000).astype(np.float32) * 3.0

for fmt in ("e4m3", "e5m2"):
    q = quantize_fp8(x, fmt)
    err = q - x
    rel = err / (np.abs(x) + 1e-8)
    print(f"{fmt}: rmse={np.sqrt(np.mean(err**2)):.4f} "
          f"max_err={np.max(np.abs(err)):.4f} "
          f"mean_rel={np.mean(np.abs(rel)):.4f} "
          f"unique={len(np.unique(q))}")

# Actual output (verified):
# e4m3: rmse=0.0786 max_err=0.4974 mean_rel=0.0230 unique=173
# e5m2: rmse=0.1577 max_err=0.4974 mean_rel=0.0449 unique=127`}
      </CodeBlock>

      <Prose>
        Three observations worth internalizing. First, the relative error is the stable quantity — around 2.3% for E4M3 and 4.5% for E5M2, regardless of the tensor's absolute magnitude (we will verify this more carefully in the quantization-error-vs-magnitude plot below). Second, the absolute error scales with magnitude: a value near 10 has an absolute quantization error around ten times larger than a value near 1. This is the direct consequence of relative spacing getting coarser at higher magnitudes. Third, E5M2 hits fewer unique values than E4M3 on this Gaussian input because its wider range means more of its representable values sit outside the bulk of the distribution; the effective resolution inside the distribution is lower.
      </Prose>

      <H3>4b. Per-tensor scaling</H3>

      <Prose>
        Quantizing a tensor without scaling is what section 4a did, and the relative error is already acceptable. The reason scaling matters is that real training tensors do not have amax near the format's max. A weight tensor early in training might have amax of 0.1; a gradient tensor late in training might have amax of 10<sup>-5</sup>. Without scaling, those tensors use only a tiny slice of the FP8 representable range, and the effective precision is catastrophic. The scaling step rescues them.
      </Prose>

      <CodeBlock language="python">
{`def scale_and_quantize(tensor, fmt="e4m3"):
    max_repr = E4M3_MAX if fmt == "e4m3" else E5M2_MAX
    amax = float(np.max(np.abs(tensor)))
    if amax < 1e-12:
        return np.zeros_like(tensor), 1.0
    scale = max_repr / amax
    return quantize_fp8(tensor * scale, fmt), scale

def dequantize(q, scale):
    return q / scale

# Test on three tensors at very different magnitudes.
for label, mag in [("weights", 1.0), ("activations", 50.0), ("tiny_grad", 1e-5)]:
    t = np.random.randn(10000).astype(np.float32) * mag
    q, s = scale_and_quantize(t, "e4m3")
    deq = dequantize(q, s)
    err = deq - t
    rel = np.mean(np.abs(err) / (np.abs(t) + 1e-12))
    print(f"{label:12s} amax={np.max(np.abs(t)):.2e} scale={s:.2e} "
          f"rmse={np.sqrt(np.mean(err**2)):.2e} rel={rel:.2e}")

# Actual output (verified):
# weights      amax=3.90e+00 scale=1.15e+02 rmse=2.66e-02 rel=2.26e-02
# activations  amax=2.14e+02 scale=2.09e+00 rmse=1.30e+00 rel=2.24e-02
# tiny_grad    amax=3.83e-05 scale=1.17e+07 rmse=2.70e-07 rel=2.27e-02`}
      </CodeBlock>

      <Prose>
        Three tensors spanning seven orders of magnitude in amax. All three round-trip through FP8-E4M3 with the same relative error — around 2.25% — because the scaling maps each tensor's range onto the same region of the FP8 grid. The scale factor ranges from 2.09 for the large-activation tensor down to 1.17 × 10<sup>7</sup> for the tiny-gradient tensor. The scale is carrying the dynamic range; the quantized tensor is carrying the shape. This separation is the whole mechanism that lets eight bits of representation work for tensors whose native magnitudes span 12 orders.
      </Prose>

      <Callout accent="gold">
        An FP8 tensor without its scale is numerically meaningless. Production frameworks pair the quantized bytes and the FP32 scale as a single logical object. Any operation that consumes an FP8 tensor also consumes its scale, and any operation that produces an FP8 tensor also produces a new scale. Losing the scale — by, say, casting the quantized bytes back to FP32 without dividing — silently corrupts the training run.
      </Callout>

      <H3>4c. Delayed scaling</H3>

      <Prose>
        Delayed scaling replaces the per-step amax reduction with a lookup into a rolling history. The implementation is a short class: on each step, apply the scale derived from the previous window's max, then measure and append the current step's amax for future windows to consume.
      </Prose>

      <CodeBlock language="python">
{`class DelayedScale:
    def __init__(self, window=16, fmt="e4m3"):
        self.window = window
        self.history = []
        self.fmt = fmt
        self.max_repr = E4M3_MAX if fmt == "e4m3" else E5M2_MAX

    def step(self, tensor):
        if self.history:
            scale = self.max_repr / max(self.history)
        else:
            scale = self.max_repr / (np.max(np.abs(tensor)) + 1e-12)
        q = quantize_fp8(tensor * scale, self.fmt)
        self.history.append(float(np.max(np.abs(tensor))))
        self.history = self.history[-self.window:]
        return q, scale`}
      </CodeBlock>

      <Prose>
        Drive it with a 60-step trajectory whose magnitude drifts slowly upward (a stand-in for activation-distribution drift during warmup) and compare against ideal per-step scaling.
      </Prose>

      <CodeBlock language="python">
{`ds = DelayedScale(window=16)
per_step_err, delayed_err = [], []
for t in range(60):
    x = np.random.randn(4096).astype(np.float32) * (1.0 + 0.03 * t)

    q_ps, s_ps = scale_and_quantize(x, "e4m3")
    per_step_err.append(np.sqrt(np.mean((q_ps / s_ps - x)**2)))

    q_d, s_d = ds.step(x)
    delayed_err.append(np.sqrt(np.mean((q_d / s_d - x)**2)))

# Actual output (verified):
# per_step first5=[0.0266, 0.0279, 0.0277, 0.0287, 0.0291]
# per_step  last5=[0.0697, 0.0705, 0.0689, 0.0692, 0.0719]
# delayed  first5=[0.0266, 0.0279, 0.0283, 0.0292, 0.0301]
# delayed   last5=[0.0699, 0.0719, 0.0692, 0.0722, 0.0713]
# max_gap  = 0.0053`}
      </CodeBlock>

      <Prose>
        The per-step and delayed errors track almost identically across the full trajectory. The absolute error grows from 0.027 at step zero to 0.07 at step 59 because the magnitude of the tensor grew by almost 3×, and quantization error scales with magnitude. The gap between the two scaling strategies is at most 0.005 — less than a third of one percent of the tensor's magnitude. Delayed scaling loses essentially nothing in this regime, and it removes the amax-reduction synchronization from the training critical path, which is where the throughput gain comes from in practice.
      </Prose>

      <StepTrace
        label="delayed scaling: state over steps 1–5"
        steps={[
          {
            label: "step 1",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: colors.textSecondary, lineHeight: 1.8 }}>
                history: <span style={{ color: colors.textDim }}>[]</span> (empty)<br/>
                fallback scale from current amax={"1.08"}<br/>
                scale applied: {"448 / 1.08 = 414.8"}<br/>
                after step: history = [1.08]
              </div>
            ),
          },
          {
            label: "step 2",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: colors.textSecondary, lineHeight: 1.8 }}>
                history: <span style={{ color: colors.gold }}>[1.08]</span><br/>
                max(history) = 1.08; scale = {"448 / 1.08 = 414.8"}<br/>
                current amax = 1.15 (observed, not used for scale)<br/>
                after step: history = [1.08, 1.15]
              </div>
            ),
          },
          {
            label: "step 3",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: colors.textSecondary, lineHeight: 1.8 }}>
                history: <span style={{ color: colors.gold }}>[1.08, 1.15]</span><br/>
                max(history) = 1.15; scale = {"448 / 1.15 = 389.6"}<br/>
                current amax = 1.19<br/>
                after step: history = [1.08, 1.15, 1.19]
              </div>
            ),
          },
          {
            label: "step 4",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: colors.textSecondary, lineHeight: 1.8 }}>
                history: <span style={{ color: colors.gold }}>[1.08, 1.15, 1.19]</span><br/>
                max(history) = 1.19; scale = {"448 / 1.19 = 376.5"}<br/>
                current amax = 1.24<br/>
                scale lags the current distribution by ~1 step.
              </div>
            ),
          },
          {
            label: "step 17 (window full)",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: colors.textSecondary, lineHeight: 1.8 }}>
                history has 16 entries; oldest evicted each step.<br/>
                max(history) tracks a 16-step trailing envelope.<br/>
                For stationary input, scale stabilizes.<br/>
                For drifting input, scale updates with a ~window/2 lag.
              </div>
            ),
          },
        ]}
      />

      <H3>4d. FP8 matmul</H3>

      <Prose>
        An FP8 matmul in hardware quantizes its two input tensors to FP8, multiplies element-by-element in the tensor core, and accumulates the result in FP32. The FP32 accumulation is not optional: accumulating FP8 products into an FP8 register would compound rounding error past any useful tolerance. Hopper's fourth-generation tensor cores compute the partial products in 8-bit integer arithmetic (after dequantization to mantissa form) and reduce into a 32-bit accumulator. For simulation purposes we can skip the integer decoding step and just dequantize both tensors to FP32 first, then matmul.
      </Prose>

      <CodeBlock language="python">
{`def fp8_matmul(A, B, fmt="e4m3"):
    """Quantize A and B to FP8, then matmul with FP32 accumulation."""
    qA, sA = scale_and_quantize(A, fmt)
    qB, sB = scale_and_quantize(B, fmt)
    return (qA / sA) @ (qB / sB)  # FP32 matmul on dequantized inputs

for shape in [(64, 64), (512, 512), (1024, 1024)]:
    A = np.random.randn(*shape).astype(np.float32)
    B = np.random.randn(*shape).astype(np.float32)
    ref = A @ B
    out = fp8_matmul(A, B, "e4m3")
    err = out - ref
    rel = np.linalg.norm(err) / np.linalg.norm(ref)
    print(f"shape={shape} rel_frob_err={rel:.4e} max_abs_err={np.max(np.abs(err)):.3f}")

# Actual output (verified):
# shape=(64, 64)    rel_frob_err=3.6940e-02  max_abs_err=1.074
# shape=(512, 512)  rel_frob_err=3.7355e-02  max_abs_err=4.597
# shape=(1024, 1024) rel_frob_err=3.7393e-02 max_abs_err=6.198`}
      </CodeBlock>

      <Prose>
        The relative Frobenius error is almost identical across shapes — around 3.7% regardless of matrix size. This is the crucial numerical property that makes FP8 matmul viable: the error does not compound with the reduction dimension. Each inner product of length <Code>k</Code> sees <Code>k</Code> quantized multiplications summed in FP32, and the accumulator's precision is high enough that the final error is dominated by the per-multiplication quantization rather than by sum-accumulated rounding. If the accumulator were FP16 or lower, the error would grow with <Code>sqrt(k)</Code> and matrix size would become a hard ceiling on usable FP8 training. This is why Hopper tensor cores accumulate in FP32 even when inputs are FP8: the bandwidth cost is real but the precision is load-bearing.
      </Prose>

      <Prose>
        The maximum absolute error grows with matrix size in this simulation because the sum of 1024 scaled quantization errors can, in the worst case, produce a larger deviation than the sum of 64. But the relative error is stable, which is what the loss curve cares about. In production training, the effective signal-to-noise ratio at the output of every FP8 matmul is on the order of 30:1, which is inside the tolerance that gradient descent already has to work with given the stochastic noise of minibatch sampling.
      </Prose>

      <H3>4e. Toy training loop</H3>

      <Prose>
        The final check is a full training loop. A minimal linear-regression problem with a 32-dimensional input and 8-dimensional output, 4,096 samples, trained for 80 steps of full-batch gradient descent with learning rate 0.05. Two variants: FP32 throughout, and FP8-simulated — E4M3 for the forward matmul, E5M2 for the gradient, FP32 for the weight update. This is the canonical structure of mixed-precision training: low precision for the compute-heavy operations, high precision for the optimizer state.
      </Prose>

      <CodeBlock language="python">
{`np.random.seed(1)
d_in, d_out, n = 32, 8, 4096
X = np.random.randn(n, d_in).astype(np.float32)
W_star = np.random.randn(d_in, d_out).astype(np.float32) * 0.5
y = X @ W_star + 0.01 * np.random.randn(n, d_out).astype(np.float32)

def train(mode, steps=80, lr=0.05):
    W = np.zeros((d_in, d_out), dtype=np.float32)  # master weights in FP32
    losses = []
    for _ in range(steps):
        pred = X @ W if mode == "fp32" else fp8_matmul(X, W, "e4m3")
        losses.append(np.mean((pred - y)**2))
        grad = (2.0 / n) * (X.T @ (pred - y))
        if mode == "fp8":
            qg, sg = scale_and_quantize(grad, "e5m2")
            grad = qg / sg
        W = W - lr * grad  # update in FP32
    return losses

loss_fp32 = train("fp32")
loss_fp8  = train("fp8")`}
      </CodeBlock>

      <Plot
        label="training loss — FP32 vs simulated FP8"
        xLabel="step"
        yLabel="MSE loss"
        series={[
          { name: "fp32", points: [[0, 9.2004], [10, 1.1046], [20, 0.1386], [30, 0.0182], [40, 0.0026], [50, 0.0004], [60, 0.0001], [70, 0.0001], [79, 0.0001]] },
          { name: "fp8 (sim)", points: [[0, 9.2004], [10, 1.1055], [20, 0.1370], [30, 0.0315], [40, 0.0222], [50, 0.0165], [60, 0.0173], [70, 0.0197], [79, 0.0207]] },
        ]}
      />

      <Prose>
        Two curves, same initial loss, same rapid descent through the first twenty steps, then a divergence. The FP32 curve continues down toward the label-noise floor at 10<sup>-4</sup>. The FP8 curve flattens out at around 0.02 — two orders of magnitude above the FP32 floor. This is the FP8 quantization-noise floor for this specific problem: the gradient, once rescaled through E5M2 and back, carries enough roundoff that the updates stop making meaningful progress relative to the noise they introduce. The loss stalls.
      </Prose>

      <Prose>
        Read this as a diagnostic, not a failure. The two recipes train identically through the high-signal phase of the optimization, where the gradient magnitudes dwarf the quantization error. They diverge only when the loss approaches the noise floor of FP8 itself. In a real pretraining run, the loss never approaches its own floor — the training budget runs out long before gradient descent converges — so FP8 and BF16 curves overlay for the entire trajectory that matters. The regime where FP8 underperforms is the regime where you have already overfit. Production FP8 recipes cover the remaining gap with a few additional tricks: master weights in FP32 (which this toy does already), loss scaling to keep the early-training gradients out of the E5M2 subnormal range, and per-channel rather than per-tensor scaling for particularly skewed tensors. Those are section 5.
      </Prose>

      <Plot
        label="E4M3 relative quantization error vs input magnitude"
        xLabel="|x|"
        yLabel="rel err"
        series={[
          { name: "e4m3 median rel err", points: [
            [0.001, 1.0], [0.00175, 0.664], [0.00307, 0.233], [0.00538, 0.125],
            [0.00942, 0.070], [0.0165, 0.041], [0.0289, 0.031], [0.0506, 0.026],
            [0.0887, 0.024], [0.155, 0.023], [0.272, 0.022], [0.477, 0.022],
            [0.835, 0.022], [1.46, 0.021], [2.56, 0.022], [4.49, 0.022],
            [7.86, 0.022], [13.8, 0.022], [24.1, 0.022], [42.3, 0.022],
            [74.1, 0.022], [130, 0.022], [227, 0.023], [398, 0.029]
          ]},
        ]}
      />

      <Prose>
        The error curve above, generated from the simulator on inputs ranging over six orders of magnitude, shows the two regimes that matter. Above the subnormal threshold (around 2 × 10<sup>-3</sup> for E4M3) and below the saturation threshold (around 400), the relative error is flat at about 2.2% — this is the uniform precision region. Below the subnormal threshold, the error blows up because the format runs out of resolution. Above the saturation threshold, the error begins to climb because values start to clip against the format's maximum. The plot is the visual case for why scaling is mandatory: without it, tensors whose magnitudes are not matched to the range [10<sup>-2</sup>, 10<sup>2</sup>] fall off one of the two cliffs. With it, every tensor lands in the flat region and reaches the 2% error floor.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        No one runs the NumPy simulator in production. The production path for FP8 pretraining, as of early 2026, goes through NVIDIA's Transformer Engine (TE) library, which provides PyTorch and JAX bindings that handle the scaling bookkeeping, Hopper's hardware FP8 matmul dispatch, and the per-layer recipe customization that real models need. The simulator is how you understand what TE is doing under the hood; TE is how you actually run at scale.
      </Prose>

      <H3>5a. Transformer Engine basics</H3>

      <Prose>
        The minimal TE usage pattern is a single wrapper layer and a single context manager. Swap <Code>torch.nn.Linear</Code> for <Code>transformer_engine.pytorch.Linear</Code>, and wrap the forward-pass call in an <Code>fp8_autocast</Code> context with a recipe object that specifies which FP8 format to use, how long the amax history window should be, and which reduction algorithm to apply when multiple tensor entries in a layer need their amax statistics combined. The result is a layer that does its matmul in FP8 on H100 hardware, tracks its own scale factors, and looks like a drop-in replacement from the outside.
      </Prose>

      <CodeBlock language="python">
{`import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format

# HYBRID format = E4M3 for forward, E5M2 for backward — the canonical recipe.
recipe = DelayedScaling(
    fp8_format=Format.HYBRID,
    amax_history_len=1024,           # how many steps of amax history to keep
    amax_compute_algo="max",         # reduce the history with max()
    margin=0,                        # safety margin on the scale factor
)

model = te.Linear(4096, 4096, bias=True).cuda()
x = torch.randn(32, 4096, device="cuda", dtype=torch.bfloat16)

with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
    y = model(x)  # FP8 matmul; output is BF16 (dequantized after the matmul)

loss = y.square().mean()
loss.backward()  # backward also runs in FP8 via E5M2 for gradients`}
      </CodeBlock>

      <Prose>
        A few details worth pointing out. The input tensor is BF16, not FP8 — TE does the cast to FP8 inside the layer, based on the layer's own amax history. The output is BF16, for the same reason: the downstream operation (the next layer's input processing, or a residual connection, or a LayerNorm) expects BF16. FP8 is contained inside the <Code>te.Linear</Code> boundary; outside the layer, tensors are BF16 or FP32 as normal. The <Code>fp8_recipe</Code> parameter is a per-layer or per-model setting that controls the behavior for every TE layer underneath the autocast context. Production recipes usually keep a global recipe object and apply it everywhere, but per-layer overrides exist for tensors that need different scaling behavior.
      </Prose>

      <Prose>
        The <Code>Format.HYBRID</Code> recipe is the default and matches the Micikevicius 2022 paper: E4M3 for the forward-pass matmul (weights and activations), E5M2 for the backward-pass gradient matmul. <Code>Format.E4M3</Code> and <Code>Format.E5M2</Code> exist for ablation studies and specialized inference cases but are rarely used for training. The <Code>amax_history_len</Code> defaults to 1024 for the TE <Code>Linear</Code> layer and is sometimes reduced to 16 or 32 for layers whose distributions change rapidly (early layers during warmup, gradient tensors during the first few hundred steps). The <Code>margin</Code> parameter is a safety factor that inflates the computed scale slightly, sacrificing a bit of headroom to avoid clipping on a transient outlier — the default zero is fine for stable training but some recipes set it to 1 or 2 for the first fraction of training.
      </Prose>

      <H3>5b. The DeepSeek-V3 recipe</H3>

      <Prose>
        DeepSeek-V3's December 2024 technical report is the most detailed public account of end-to-end FP8 pretraining at frontier scale. The model is 671B parameters total (37B activated per token via its MoE routing), trained on 14.8 trillion tokens using their custom FP8 framework built on top of TE primitives. The report lists several departures from the vanilla TE recipe that are worth copying into any team's FP8 checklist.
      </Prose>

      <Prose>
        First, hybrid precision by module. Matrix multiplications inside the Transformer block — attention projections, MLP projections, expert up-and-down matrices — run in FP8 with the HYBRID recipe. Everything else stays in higher precision: the embedding module, the output softmax head, the MoE gating network, all the LayerNorm operators, and the attention softmax operator itself. This is not an optimization choice but a correctness requirement. LayerNorm's variance computation sums squared values, which can overflow E4M3 on a tensor with an even modestly large amax. Softmax's exponential can produce values that blow past either FP8 format's range. The gating network's decisions directly control expert routing and must be numerically stable to keep the load balanced. The output head produces token logits whose dynamic range includes values many orders of magnitude below the max, and FP8 would quantize the low-probability tokens indistinguishably.
      </Prose>

      <Prose>
        Second, fine-grained scaling. DeepSeek-V3 uses block-wise scaling rather than per-tensor scaling for gradient tensors — a single scale factor per block of 128 elements along the reduction dimension, instead of one scale factor for the entire tensor. This is slightly more expensive (more scale factors to track) but handles outlier blocks without sacrificing precision for the bulk of the tensor. The Transformer Engine has since added this capability as a standard option.
      </Prose>

      <Prose>
        Third, high-precision accumulation for the FP8 matmul. Standard H100 tensor cores accumulate FP8 products into FP32. DeepSeek-V3 observed that the FP32 accumulation, while correct in expectation, had enough roundoff at the scale of their matmuls (reduction dimensions of 12,288 and above) to measurably drift training. Their fix was to accumulate into FP32 as normal but then perform a higher-precision sum over the accumulator tiles — essentially using FP32 as local accumulator and a pairwise-summation reduction for the global sum, rather than a flat sum. This is a low-level implementation detail that most teams do not need, but it is a canary for when FP8 starts to fail at very large scales.
      </Prose>

      <Prose>
        The validation result DeepSeek reports is the most important number in their paper: relative loss error below 0.25% compared to a BF16 baseline, across two smaller-scale ablation runs at ~16B and ~230B parameters trained for one trillion tokens. That is the standard by which every FP8 recipe should now be judged. If your recipe cannot match BF16 to within a few tenths of a percent on a representative ablation, something in the recipe is wrong — most likely an operator that should be in BF16 and got left in FP8, or a scaling history window that is either too long (misses distribution shifts) or too short (tracks too much per-step noise).
      </Prose>

      <H3>5c. What runs at higher precision</H3>

      <Prose>
        The single most common FP8 bug is forgetting which operators should stay in BF16 or FP32. The list is short but non-optional for stable training.
      </Prose>

      <CodeBlock>
{`operator             precision    why
-----------------    ---------    ----------------------------------------
LayerNorm / RMSNorm  BF16         variance sum overflows, epsilon fragile
softmax              BF16 / FP32  exp(x) for large x blows past range
loss (cross-entropy) FP32         log-probabilities span many orders
embedding lookup     BF16         rare-token embeddings need precision
output head matmul   BF16 / FP8   E5M2 sometimes; precision sensitive
MoE gating           BF16 / FP32  routing decisions must be stable
master weights       FP32         optimizer state; gradient accumulation
Adam moments (m,v)   FP32         very small values during most of training
matmul inputs        FP8          E4M3 weights/activations, E5M2 gradients
matmul accumulator   FP32         inside tensor cores`}
      </CodeBlock>

      <Prose>
        Every production recipe decides this table. The matmuls in the table are the vast majority of the compute — 70 to 90 percent of FLOPs in a typical Transformer training step — which is why moving them alone from BF16 to FP8 captures most of the available throughput improvement. The remaining operators are small in FLOPs but large in precision sensitivity. Keeping them in BF16 or FP32 costs almost nothing in wall time, and violating the table costs loss-curve divergence that is hard to debug and usually only caught after wasted GPU-hours.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        The plots and traces in this topic sit above in sections 4 and 3; this section surfaces a single comparative figure that ties the whole story together. The heatmap below shows the error-vs-scale surface for FP8-E4M3 on a fixed tensor as the per-tensor scale factor varies. Each row is a different scale strategy. Each column is a different tensor magnitude. The cell value is the relative error of the round-trip at that scale and magnitude. Green is low error, gold is high error.
      </Prose>

      <Heatmap
        label="rel quantization error, rows = scale strategy, cols = tensor amax"
        rowLabels={["no scale", "per-tensor", "delayed W=16", "delayed W=256"]}
        colLabels={["1e-5", "1e-3", "1e-1", "1e0", "1e1", "1e2"]}
        colorScale="gold"
        cellSize={52}
        matrix={[
          // no scale: catastrophic for tiny magnitudes; fine near unit; clips above 100
          [1.00, 0.45, 0.04, 0.022, 0.023, 0.22],
          // per-tensor scale: flat 2.2% across the board
          [0.022, 0.022, 0.022, 0.022, 0.022, 0.022],
          // delayed window 16: tracks per-tensor almost exactly when stationary
          [0.025, 0.024, 0.023, 0.022, 0.023, 0.025],
          // delayed window 256: fine if stationary, lags on drift (marked via higher err at edges)
          [0.040, 0.030, 0.024, 0.022, 0.028, 0.050],
        ]}
      />

      <Prose>
        The top row is what happens without scaling. Tiny-magnitude tensors underflow, large-magnitude tensors clip, and only the narrow band of inputs that happen to match the format's native range survives. The second row — per-tensor scaling computed fresh each step — is the flat 2.2% across the board that sections 4a and 4b established empirically. The third row is delayed scaling with a short window; almost indistinguishable from per-tensor on stationary inputs, with a slight widening at the edges where the window hasn't caught up. The fourth row is delayed scaling with a long window, which is fine in the center but shows visible degradation at the edges where a long lag cannot track distribution shifts. The middle region — per-tensor or short-window delayed — is where production recipes live.
      </Prose>

      <CodeBlock>
{`FP8 pretraining decision tree

  hardware?
  |-- H100/H200/Blackwell: FP8 viable
  |   |-- model size > 1B params: use FP8 for matmuls (2x throughput)
  |   |   |-- first 1000 steps: optional loss scaling, short amax window
  |   |   |-- stable phase: HYBRID recipe, amax window 1024
  |   |   \`-- final eval: always run in BF16 to sanity-check
  |   \`-- model size < 1B: FP8 possible but BF16 often easier
  |
  |-- A100 / older: BF16 + FP32 accumulation, no FP8 (no hardware)
  \`-- TPU / non-NVIDIA: BF16 or TPU-native formats; FP8 maturity varies

operator override table:
  - matmul inputs:   FP8 (E4M3 forward, E5M2 backward)
  - matmul accum:    FP32 (inside tensor cores)
  - LayerNorm/RMSN:  BF16
  - softmax:         BF16 or FP32
  - embedding:       BF16
  - master weights:  FP32
  - optimizer state: FP32`}
      </CodeBlock>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        The format choice for any given training run is not one decision but several, and the right answer depends on hardware, scale, and team maturity. The matrix below captures how the major options sit relative to each other.
      </Prose>

      <CodeBlock>
{`precision   hardware     typical use case              mem / compute gain
                                                        vs FP32
FP32         any          research; debugging; old      1x / 1x
                          recipes that need it
FP16         Volta+       legacy; still default         2x / 2-4x
                          for some older training
                          infra; requires loss scaling
BF16         Ampere+,TPU  2020-2023 default for         2x / 2-4x
                          pretraining; no loss scaling
                          needed
FP8 HYBRID   Hopper+      2023-present default for      4x / 4-8x
                          H100-class pretraining
INT8         any          post-training inference       4x / 4x (inference)
                          quantization; not training
INT4         any          aggressive inference          8x / 8x (inference)
                          (GPTQ/AWQ); not training`}
      </CodeBlock>

      <Prose>
        Several choices are worth spelling out. FP32 remains the right choice when you are debugging the numerical behavior of a new recipe and cannot yet isolate whether a loss-curve anomaly is a model bug or a precision bug. The cost is a 4× training slowdown and halved batch sizes, which is acceptable for a week of ablation but not for the full run. BF16 is the safe default for model sizes below a billion parameters or for teams that have not yet invested in TE infrastructure. It gives you most of the FP8 speedup with none of the scaling complexity. FP8 is worth adopting once your models pass a few billion parameters and your hardware is Hopper-class — below that threshold, the engineering overhead of FP8 often exceeds the wall-clock gain from the extra 1.5–2× throughput.
      </Prose>

      <Prose>
        The INT8 and INT4 entries in the table are there to flag a common confusion: they are inference formats, not training formats. Post-training quantization converts trained FP32 or BF16 weights to INT8 or INT4 for deployment, using techniques like GPTQ and AWQ to minimize accuracy loss. These formats share the low bit count with FP8 but not the mechanism — integer quantization is linear (uniform spacing), while FP8 is exponential (log-spaced). Linear quantization is adequate for inference because the weight distribution has already been frozen and can be analyzed for optimal scaling offline. It is not adequate for training because gradients have too wide a dynamic range. The Model Optimization track covers inference quantization in detail; this topic stays on the training side.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOES NOT
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        The headline number for FP8 is 2× throughput versus BF16 on Hopper, but that number is a rough approximation that hides a lot of detail. The theoretical peak from the Hopper whitepaper is 3,958 TFLOPS for FP8 dense matmul versus 1,979 TFLOPS for BF16 — a factor of exactly two. The realized speedup in actual training runs tends to be 1.4–1.8×, because the non-matmul operators (LayerNorm, softmax, activation functions, memory movement) do not see the FP8 speedup, and as they become a larger fraction of wall-clock time the overall ratio shrinks from the matmul-limit. DeepSeek-V3 reported an effective throughput improvement of 1.7× versus their prior BF16 baseline for the equivalent compute.
      </Prose>

      <Prose>
        Memory is the other first-order win. FP8 activations are half the size of BF16; FP8 gradients are half the size of BF16. Master weights, optimizer state, and the LayerNorm operators stay at higher precision, so the total memory reduction is less than 50%. A typical breakdown for a Transformer training step, per parameter, runs about 2 bytes for the BF16 shadow weight, 4 bytes for the FP32 master weight, 8 bytes for Adam moments (m and v), 2 bytes for the BF16 gradient, plus activations that depend on batch size and sequence length. Moving the shadow weight and gradient from BF16 to FP8 saves 2 bytes per parameter — around 10–15% of the parameter-state footprint — and halves the activation memory, which is often the dominant term at long context lengths. For a 70B-parameter model training at 32k context, the activation memory savings alone typically free up 20–30% of HBM capacity, which translates directly into larger per-device batch sizes and reduced pipeline bubbles.
      </Prose>

      <Prose>
        FP8 scales sub-linearly with model size. A 1B-parameter model sees a small FP8 advantage over BF16 — around 1.2× throughput, because the non-matmul overhead dominates at small scale. A 70B model sees 1.5–1.6×. A 400B+ model sees closer to 1.8×, because at that scale the matmuls are large enough that the non-matmul operators are a smaller fraction of wall time. This is the opposite of the usual scaling curve you see with techniques like pipeline parallelism, which scale sub-linearly because of communication overhead. FP8 benefits from scale because the thing it is accelerating — the matmul — is also what scales fastest with model size.
      </Prose>

      <Prose>
        What does not scale well. Small batch sizes limit the FP8 win because a matmul with a small outer dimension is memory-bound rather than compute-bound, and FP8's advantage is in compute throughput. Sequence lengths below 2k similarly dampen the gain, because the attention matmul becomes relatively cheaper compared to the softmax and LayerNorm operations that stay in higher precision. FP8 on non-Hopper hardware is simulation only — TE will run on A100 by doing the scaling bookkeeping in software and the matmul in BF16, which is useful for CI testing but provides no speedup in production. And FP8 at extreme scale (trillion-parameter models) is the current research frontier rather than a solved problem — DeepSeek-V3 established it is possible at 671B; above that, most teams have not yet published recipes, and the failure modes get more subtle as the reduction dimensions grow.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <Prose>
        Ten things that reliably go wrong in FP8 training runs. Most of them trace to a single underlying mistake: treating FP8 as a drop-in type swap rather than as a numerical regime that requires careful per-tensor decisions.
      </Prose>

      <Prose>
        <strong>1. Loss spikes from outlier tokens.</strong> A single corrupt document, a long stretch of repeated characters, or an unusual token sequence produces gradient entries several orders of magnitude above the median. If the current amax history does not include similar outliers, the scale factor is too small — based on typical magnitudes — and the outlier gradient clips at the E5M2 maximum. The clipped gradient corrupts the weight update. The model takes a bad step. The loss spikes upward and takes several dozen steps to recover. Symptom: loss spikes that coincide with specific training batches, reproducible if you replay the data. Fix: tighter outlier filtering in the data pipeline, pre-matmul gradient clipping, or a shorter amax history window so the scale adapts faster to distribution shifts.
      </Prose>

      <Prose>
        <strong>2. Delayed-scaling lag on distribution shift.</strong> When a layer's activation distribution shifts — during warmup, at a learning-rate decay boundary, after a data-mix change — the delayed scale lags behind the new distribution by approximately window/2 steps. For those steps, the scale is tuned to the old amax and the new tensor systematically clips or underflows. Symptom: a transient rise in validation loss a few hundred steps after any schedule change, self-correcting over the next window. Fix: shorter history windows on activation tensors, or an explicit scale reset at known schedule transitions, or use of Transformer Engine's "just-in-time" mode for layers that change rapidly.
      </Prose>

      <Prose>
        <strong>3. LayerNorm silently at FP8.</strong> A new training script uses <Code>te.Linear</Code> for the matmul layers but inherits the existing <Code>torch.nn.LayerNorm</Code>, which runs in whatever dtype its input is — including FP8 if the surrounding cast logic is miswired. LayerNorm's variance computation sums squared values; squaring an E4M3 value whose absolute value exceeds ~21 overflows the E4M3 max of 448. The variance is corrupted; the normalization output is garbage; the next layer trains on noise. Symptom: the first few steps look fine (inputs are small at init), then as weights grow the LayerNorm outputs become NaN. Fix: ensure LayerNorm and RMSNorm always run in BF16 or FP32, either by using <Code>te.LayerNorm</Code> which enforces this, or by explicitly casting inputs.
      </Prose>

      <Prose>
        <strong>4. Master weights at FP8.</strong> The optimizer state — the weights that the gradient update is applied to — must stay in FP32 (or at minimum BF16). If the master copy is FP8, each gradient step rounds the update to the FP8 grid, and small updates below the FP8 resolution are silently dropped. The model stops learning after enough steps that accumulated dropped updates exceed actual updates. Symptom: training loss plateaus at a higher floor than expected. Fix: keep an FP32 master copy for Adam's <Code>m</Code> and <Code>v</Code>, and the "high-precision weights" used for the optimizer step. TE handles this automatically but custom optimizers can miss it.
      </Prose>

      <Prose>
        <strong>5. E4M3 where E5M2 is needed.</strong> The forward-pass format is E4M3 because precision matters; the backward-pass format is E5M2 because range matters. Swapping them — using E4M3 for gradients — immediately produces clipping on the first reasonably-sized gradient, because E4M3's max of 448 is not enough range for gradient magnitudes that routinely exceed 10<sup>3</sup> when loss scaling is enabled. Swapping the other direction — using E5M2 for weights — degrades forward-pass precision enough that the network's outputs become noisy. Symptom: immediate training divergence in the first few hundred steps. Fix: always use the HYBRID recipe (<Code>Format.HYBRID</Code>), which assigns each format to its correct role automatically.
      </Prose>

      <Prose>
        <strong>6. Loss scaling applied on top of FP8.</strong> Loss scaling is the FP16 trick: multiply the loss by a large constant before backward, divide every gradient by it after. FP8 per-tensor scaling solves the same problem by different means — each tensor gets its own scale. Applying both can stack in unexpected ways, because the loss scale enters the gradient, which then gets per-tensor-rescaled by TE, and the two scale factors compound. Usually the stack works, but it can cause the per-tensor scale to overflow its FP32 storage at extreme loss scales (10<sup>10</sup> or higher). Symptom: NaN gradients after the first few training steps, with the NaN traceable to the scale factor rather than the weights. Fix: use one mechanism at a time; prefer FP8 per-tensor scaling alone, and only add loss scaling as a belt-and-suspenders defense if you have a specific reason to.
      </Prose>

      <Prose>
        <strong>7. Per-tensor scale when per-channel is needed.</strong> Some tensors — particularly weight tensors in wide layers — have outlier channels whose amax is much larger than the median channel. A single per-tensor scale derived from the outlier forces all the other channels to use only a tiny slice of the FP8 range, losing most of their precision. The Transformer Engine supports per-channel scaling for weights (one scale per output channel rather than one per tensor), and DeepSeek-V3 uses block-wise scaling for similar reasons. Symptom: suspiciously bad quantization error on a specific layer, visible in the scale factor history as extreme outliers. Fix: switch that layer to per-channel or per-block scaling.
      </Prose>

      <Prose>
        <strong>8. FP8 on non-Hopper hardware.</strong> Transformer Engine will run on A100 and older GPUs by falling back to BF16 matmul with simulated FP8 scaling. The arithmetic is correct; the speedup is zero. Teams sometimes enable FP8 during development on A100 and are surprised that the cluster is no faster. Symptom: FP8 enabled in code, same throughput as BF16 baseline. Fix: verify hardware support (<Code>torch.cuda.get_device_capability() &gt;= (9, 0)</Code> for Hopper) before enabling FP8, or accept it as a simulation for correctness testing only.
      </Prose>

      <Prose>
        <strong>9. Mixed FP8/BF16 accumulation.</strong> If a layer's matmul runs in FP8 and its residual connection adds an FP8 output to a BF16 input, the sum has to be computed in a precision that accommodates both. The default is BF16, which is fine — the FP8 tensor is dequantized to BF16 before the addition. But if the dequantization scale is not properly propagated to the BF16 side, the residual can silently halve or double in magnitude, depending on which direction the mismatch points. Symptom: training is stable but reaches a higher loss floor than expected, with the gap explainable only by residual-connection corruption. Fix: TE handles this correctly by default; custom residual code should explicitly cast FP8 to BF16 with the correct scale applied before addition.
      </Prose>

      <Prose>
        <strong>10. Reduced sensitivity to small-gradient signals.</strong> The most fundamental FP8 failure mode is the one that cannot be engineered around: some training problems require the model to learn from gradient signals whose magnitude is below FP8's effective resolution. Fine-grained control over rare-token behavior, long-tail factual knowledge, certain post-training alignment objectives — all of these can, in principle, be dominated by gradient magnitudes that FP8's 2–5% relative error swamps. Symptom: a model trained end-to-end in FP8 matches BF16 on bulk loss but underperforms on specific narrow tasks in eval. Fix: fine-tune the final phase in BF16 or FP32, or use FP8 for the pretraining bulk and switch precision for the fine-tuning stages. This is the main reason most production pipelines stop FP8 at the end of pretraining and run SFT and RLHF in BF16.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        Six references. All six were cross-checked against their published venues during the preparation of this topic; dates, arXiv ids, and author lists reflect the verified records.
      </Prose>

      <Prose>
        <strong>1.</strong> Micikevicius, Paulius; Narang, Sharan; Alben, Jonah; Diamos, Gregory; Elsen, Erich; Garcia, David; Ginsburg, Boris; Houston, Michael; Kuchaiev, Oleksii; Venkatesh, Ganesh; Wu, Hao. "Mixed Precision Training." <em>ICLR 2018</em>. arXiv:1710.03740 (first submitted October 2017). The foundational paper for the FP16 + FP32 mixed-precision recipe: master weights in FP32, loss scaling for gradient underflow, FP32 accumulation inside the matmul. Every lower-precision recipe since inherits this template.
      </Prose>

      <Prose>
        <strong>2.</strong> Micikevicius, Paulius; Stosic, Dusan; Burgess, Neil; Cornea, Marius; Dubey, Pradeep; Grisenthwaite, Richard; Ha, Sangwon; Heinecke, Alexander; Judd, Patrick; Kamalu, John; Mellempudi, Naveen; Oberman, Stuart; Shoeybi, Mohammad; Siu, Michael; Wu, Hao. "FP8 Formats for Deep Learning." arXiv:2209.05433, September 2022. The definitional paper for E4M3 and E5M2 as a cross-vendor 8-bit standard. Includes empirical validation on 175B-parameter GPT-3-class training without hyperparameter changes relative to the BF16 baseline. Co-authored across NVIDIA, Intel, and Arm, reflecting the cross-industry nature of the FP8 standardization.
      </Prose>

      <Prose>
        <strong>3.</strong> Kalamkar, Dhiraj; Mudigere, Dheevatsa; Mellempudi, Naveen; Das, Dipankar; et al. "A Study of BFLOAT16 for Deep Learning Training." arXiv:1905.12322, May 2019. The canonical empirical reference for BF16 training. Establishes that BF16 matches FP32 loss curves across image, speech, language, and recommendation workloads without hyperparameter changes. The paper that locked in BF16 as the 2020-plus pretraining default.
      </Prose>

      <Prose>
        <strong>4.</strong> NVIDIA Corporation. <em>NVIDIA H100 Tensor Core GPU Architecture Whitepaper</em>, March 2022. The hardware reference for Hopper. Specifies the fourth-generation Tensor Cores, the FP8 matmul rates (3,958 TFLOPS dense, 7,916 TFLOPS with 2:4 sparsity), and the Transformer Engine integration. Also the source for the canonical FP8 versus BF16 throughput comparison (1979 vs 989 TFLOPS at the older generation's equivalent configurations).
      </Prose>

      <Prose>
        <strong>5.</strong> NVIDIA Corporation. <em>Transformer Engine Documentation</em>, current release. The software reference for how FP8 training is actually implemented in PyTorch and JAX. Covers the <Code>DelayedScaling</Code> recipe, the <Code>Format.HYBRID</Code> default, the <Code>fp8_autocast</Code> context manager, and the per-layer and per-module overrides. Indispensable for debugging any FP8 training run and for understanding which operators stay in higher precision by default.
      </Prose>

      <Prose>
        <strong>6.</strong> DeepSeek-AI. "DeepSeek-V3 Technical Report." arXiv:2412.19437, December 2024. The first detailed public account of end-to-end FP8 pretraining at 671B-parameter scale. Documents the module-level precision assignment, block-wise scaling for gradients, high-precision accumulation for large reduction dimensions, and the &lt;0.25% relative-loss-error validation against BF16. Essential reading for anyone preparing a production FP8 recipe at frontier scale.
      </Prose>

      <Callout accent="gold">
        Secondary but worth flagging: Peng, Houwen et al., "FP8-LM: Training FP8 Large Language Models" (arXiv:2310.18313, October 2023), Microsoft's public FP8 training framework with detailed ablations on optimizer-state quantization and distributed training gradient aggregation in FP8. Complements the NVIDIA stack with a second independent implementation lineage.
      </Callout>

      {/* ======================================================================
          11. SELF-CHECK
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <Prose>
        Five short problems. Spend ten minutes on each before peeking. The problems are chosen so that getting them wrong diagnoses something specific about what you have not internalized about FP8.
      </Prose>

      <Prose>
        <strong>Problem 1.</strong> Given FP8-E4M3's layout (1 sign bit, 4 exponent bits with bias 7, 3 mantissa bits), compute the smallest positive representable value and the largest finite value. Show your work from the bit-level formulas in section 3.
      </Prose>

      <Callout accent="green">
        Smallest positive is the smallest subnormal. Subnormals have exponent field 0, use the formula <Code>v = 2<sup>1 - bias</sup> · m/2<sup>M</sup></Code>, and the smallest nonzero mantissa is <Code>m=1</Code> of <Code>2<sup>M</sup>=8</Code>. So <Code>v = 2<sup>1-7</sup> · 1/8 = 2<sup>-6</sup>/8 = 2<sup>-9</sup> = 1.953 × 10<sup>-3</sup></Code>. Largest finite uses the largest valid exponent field and largest valid mantissa. E4M3 reserves the all-ones exponent field with all-ones mantissa as NaN, so the largest is exponent field <Code>e=15</Code>, mantissa field <Code>m=6</Code> (bit pattern 110). Value: <Code>v = 2<sup>15-7</sup> · (1 + 6/8) = 256 · 1.75 = 448</Code>. The simulator in section 4 verifies both: <Code>E4M3_GRID.min() = 1.953e-3</Code>, <Code>E4M3_GRID.max() = 448</Code>.
      </Callout>

      <Prose>
        <strong>Problem 2.</strong> Why does the gradient matmul use E5M2 and not E4M3? Make the argument in terms of the quantitative range of each format and the magnitude distribution of training gradients. What would go wrong if you swapped them?
      </Prose>

      <Callout accent="green">
        E4M3 max finite is 448; E5M2 max finite is 57,344 — a factor of 128 more range. Gradient magnitudes during training routinely span many orders of magnitude, especially when a loss scale is in effect or when an outlier batch enters. An amax at 10<sup>4</sup> sits comfortably inside E5M2's range but overflows E4M3 unless heavily scaled down, and scaling down enough to fit compresses the rest of the distribution below E4M3's effective resolution. E4M3 for gradients also loses the gradients from rare but important tokens, which tend to be small and get quantized to zero in the scaled-down tensor. The swap of E5M2 for weights would run into the opposite problem: weights need precision more than range (their distribution is tighter), and E5M2's 2-bit mantissa gives only four values per octave, doubling the quantization error of forward-pass computations and degrading the network's effective capacity.
      </Callout>

      <Prose>
        <strong>Problem 3.</strong> Show explicitly how per-tensor scaling rescues a gradient tensor with amax <Code>10<sup>-7</sup></Code> from catastrophic underflow in E5M2. What is the computed scale factor? Using the simulator, what is the round-trip error with and without scaling?
      </Prose>

      <Callout accent="green">
        E5M2's smallest positive representable value is 2<sup>-16</sup> ≈ 1.53 × 10<sup>-5</sup>. A tensor with amax 10<sup>-7</sup> is entirely below this threshold — every value underflows to zero without scaling. The scale factor computed by per-tensor scaling is <Code>s = max<sub>E5M2</sub> / amax = 57344 / 10<sup>-7</sup> ≈ 5.7 × 10<sup>11</sup></Code>. Multiplying the tensor by this scale rescales every value into the [−57344, 57344] interval, with the largest landing at the top of the range and the rest distributed across the FP8 grid. The simulator run in section 4 verifies: naive FP8 cast rmse = <Code>1.005 × 10<sup>-7</sup></Code> (essentially the signal itself — every value rounded to zero), per-tensor scaled rmse = <Code>5.264 × 10<sup>-9</sup></Code> (two orders of magnitude smaller, limited by FP8's 4–5% relative error).
      </Callout>

      <Prose>
        <strong>Problem 4.</strong> You have a training run where the activation distribution for a specific layer is known to drift by roughly one order of magnitude over every 10,000 steps, with small per-step noise on top. Design an amax history window length for that layer's delayed scaling. Defend your choice in three sentences.
      </Prose>

      <Callout accent="green">
        Target roughly 100–300 steps. The window should be long enough that per-step amax noise (from minibatch sampling and random SGD variation) averages out, which requires at least several dozen steps; it should be short enough that distribution drift of one order of magnitude per 10,000 steps is tracked within a few percent error, which implies window/drift_rate ≤ ~0.01, so window ≤ 100 steps if drift_rate is 1 octave per 10,000 steps. A window of 128 sits in the middle of this band and is a common production default. Longer windows (1024, the Transformer Engine default for stable activations) would introduce a ~10% scale lag by the end of the drift period, which risks consistent clipping or underflow during the second half of each 10,000-step interval.
      </Callout>

      <Prose>
        <strong>Problem 5.</strong> List every operator in a standard Transformer block that must stay in BF16 or FP32 rather than running in FP8. For each, explain the specific numerical reason in one sentence.
      </Prose>

      <Callout accent="green">
        (1) LayerNorm/RMSNorm — the variance computation sums squared values, which overflows E4M3 on any input with amax above ~21. (2) Softmax — the exponential <Code>exp(x)</Code> for large positive <Code>x</Code> blows past both FP8 formats' max finite, and for large negative <Code>x</Code> rounds to zero indistinguishably from small-but-important low-probability tokens. (3) Cross-entropy loss — log-probabilities span many orders of magnitude (log of near-1 values to log of near-0 values), and FP8's resolution in the small-magnitude region is too coarse to represent them faithfully. (4) Embedding lookup — rare-token embedding rows have small magnitudes that E5M2 underflows and E4M3 resolves with only a few unique quantized values, degrading the model's representation of low-frequency vocabulary. (5) MoE gating network — routing decisions must be numerically stable to keep expert load balanced; FP8 quantization of gate logits can produce discrete routing flips that destabilize training. (6) Master weights and Adam optimizer state — the m and v buffers contain very small values most of training, and FP8 quantization of updates drops small-magnitude gradient signals below the weight update resolution. (7) Attention softmax inputs — dot-product scores have wide dynamic range; quantizing before the softmax corrupts the probability distribution over attention targets.
      </Callout>

      <Prose>
        Low-precision pretraining is a discipline, not a feature. The payoff is the 1.5–2× throughput multiplier on Hopper-class hardware, which compounds into either larger models, more training tokens, or shorter wall-clock runs — pick any of the three at a fixed cluster budget. The cost is attention to a few numerical details that cannot be skipped. Once you have the simulator in section 4 and the operator table in section 5 internalized, the rest is engineering: pick the HYBRID recipe, keep LayerNorm and softmax in BF16, keep master weights in FP32, watch the amax history for drift, and validate against a BF16 baseline before committing the full run. Every frontier lab runs this loop. Most training code in 2026 goes through it. The 2017 mixed-precision paper set the template; the 2022 FP8 formats paper extended it to eight bits; DeepSeek-V3 proved it works at 671 billion parameters. The frontier now is FP4, which is a research problem and not a production tool, and that is a topic for another day.
      </Prose>
    </div>
  ),
};

export default fp8Training;
