import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";

const fp8Training = {
  title: "FP8 Training & Low-Precision Pre-Training",
  readTime: "12 min",
  content: () => (
    <div>
      <Prose>
        Training a modern LLM in full 32-bit precision would be economic malpractice. You would pay roughly four times the memory and four times the compute for no capability gain — every weight, every activation, every gradient stored and moved as a 32-bit float when the signal they carry does not remotely justify it. Pre-training has been moving steadily to lower precision for a decade: FP32 to FP16 to BF16 to FP8. Each step compresses the representation further, and each step requires more careful numerics to avoid training collapse. FP8 is the current frontier for large-scale pre-training runs.
      </Prose>

      <H2>What a floating-point format actually is</H2>

      <Prose>
        A floating-point number has three parts: a sign bit, an exponent field, and a mantissa (also called the significand). The exponent determines dynamic range — how far from zero the format can reach. The mantissa determines precision within that range — how finely it can distinguish two nearby values. The total bit budget is fixed, so every bit you move from mantissa to exponent trades precision for range, and vice versa.
      </Prose>

      <Prose>
        The five formats relevant to modern training runs sit along that curve. FP32 is the reference: 8 exponent bits and 23 mantissa bits give it both enormous range and fine precision. FP16 cuts to 5 exponent bits and 10 mantissa bits — adequate for many computations, but the narrow range (maximum representable value around 65,504) makes it fragile for training. BF16 keeps the 8-bit exponent of FP32 but uses only 7 mantissa bits; it sacrifices precision to maintain range. FP8 halves the byte count again, leaving almost nothing to distribute. The two FP8 variants, E5M2 and E4M3, make different bets on that tradeoff.
      </Prose>

      <CodeBlock>
{`format    sign  exponent  mantissa    bytes   range        precision
FP32       1       8         23        4       ±3.4e38      ~7 decimal digits
FP16       1       5         10        2       ±65504       ~3 decimal digits
BF16       1       8         7         2       ±3.4e38      ~2 decimal digits
FP8-E5M2   1       5         2         1       ±57344       ~1 decimal digit
FP8-E4M3   1       4         3         1       ±448         ~1 decimal digit`}
      </CodeBlock>

      <Prose>
        The mantissa column tells you how many distinct values the format can represent in any given order of magnitude. FP32's 23 bits give you about 8 million distinct steps between, say, 1.0 and 2.0. FP8-E4M3's 3 bits give you eight. That is not a typo. Eight representable values per power-of-two interval, for weights and activations that have been trained over trillions of tokens. The fact that this works at all is the surprising result, and it works because of scaling.
      </Prose>

      <H2>Why BF16 won the middle</H2>

      <Prose>
        Before FP8 became viable, the industry converged on BF16 as the standard training format, and the reason was range rather than precision. During training, gradients routinely span many orders of magnitude. A gradient early in training might be 0.001; a gradient on a different parameter at the same step might be 10,000. FP16's maximum representable value of 65,504 sounds large, but values above that threshold overflow to infinity — and once you have infinities in your gradient, they propagate and corrupt the update. Conversely, small gradients that fall below FP16's minimum representable magnitude underflow to zero and disappear.
      </Prose>

      <Prose>
        BF16 solved this by keeping the 8 exponent bits from FP32, which maintains the same dynamic range as 32-bit arithmetic while cutting the byte count in half. The mantissa shrinks from 23 bits to 7, which means less precision per representable value — but gradients tolerate that. A gradient rounded to two significant digits still points in approximately the right direction. Google TPUs introduced BF16 for training; NVIDIA followed with Ampere in 2020. By roughly 2021, BF16 was the de facto standard for large pre-training runs, and mixed-precision training — forward pass in BF16, master weights in FP32 — became the default recipe. FP32 accumulation at the end of each matmul kept critical operations numerically clean while most of the bandwidth was saved by the lower-precision representation.
      </Prose>

      <H2>The FP8 recipe — two formats, not one</H2>

      <Prose>
        FP8 for training is not a single format swap. It uses both E4M3 and E5M2 simultaneously, assigning each to the part of the computation it suits. The pattern was introduced publicly with NVIDIA's Transformer Engine on H100 hardware and has become the standard approach for FP8 training.
      </Prose>

      <Prose>
        The split is as follows: E4M3 for forward-pass activations and weights, E5M2 for backward-pass gradients. The logic tracks the table above. In the forward pass, what matters is precision — activations and weights need to represent values accurately enough that the computation produces correct outputs. E4M3 has 3 mantissa bits and the narrower range of ±448; within that range it is the more precise of the two FP8 variants. In the backward pass, what matters is range — gradients span many orders of magnitude and must not overflow. E5M2 has the wider range of ±57344, matching FP16's range, at the cost of only 2 mantissa bits of precision. Gradients are noisy by nature; the loss of precision matters less than the loss of range. The same hardware matmul unit operates on E4M3 inputs in the forward pass and E5M2 inputs in the backward pass, accumulating into FP16 or FP32 at higher precision to prevent rounding error from compounding.
      </Prose>

      <H2>Scaling — the critical numerical trick</H2>

      <Prose>
        FP8's dynamic range is far too narrow to use by naive casting. If a tensor contains values from −100 to +100, and you cast it directly to FP8-E4M3 with a maximum representable value of 448, most values will cluster in the bottom few representable steps and you will lose essentially all precision. The solution is per-tensor scaling: before casting, multiply the tensor by a scale factor chosen so that the distribution of values fills the FP8 range. After computation, divide by the scale to recover the correct magnitude.
      </Prose>

      <Prose>
        Two scaling strategies appear in practice. Per-tensor scaling computes the scale from the current tensor's absolute maximum, casts, computes, then rescales the output. This is mathematically clean but requires a max-reduction over the tensor before every cast — a synchronization point that adds latency. Delayed scaling avoids that synchronization by reusing the maximum value from a recent history window. The scale applied at step <Code>t</Code> is derived from the maximum seen over the last several steps, not from step <Code>t</Code>'s tensor directly. This introduces a small lag, but in practice the distribution of most tensors changes slowly enough that a one-step or sixteen-step history gives adequate scaling with no synchronization overhead.
      </Prose>

      <MathBlock>{"x_{fp8} = \\text{cast}_{FP8}\\left(\\frac{x}{s}\\right), \\quad s = \\frac{\\text{amax}(x)}{\\text{max}_{FP8}}"}</MathBlock>

      <Prose>
        The scale <Code>s</Code> maps the tensor's empirical maximum to the format's maximum, so the full FP8 range is used. The scale is stored alongside the data so that downstream operations can dequantize correctly. The casting itself is performed by hardware — the Transformer Engine kernels do the actual bit manipulation — but the scale management is a software concern, and it is where most FP8 training bugs originate.
      </Prose>

      <CodeBlock language="python">
{`import torch
# Per-tensor FP8 casting (simplified — real impls use Transformer Engine kernels)
def cast_to_fp8(tensor, format="e4m3"):
    fp8_max = 448.0 if format == "e4m3" else 57344.0
    amax = tensor.abs().max()
    scale = fp8_max / (amax + 1e-8)
    scaled = (tensor * scale).clamp(-fp8_max, fp8_max)
    # Hardware actually performs the quantization; here we simulate the bit reduction.
    return scaled, scale  # store scale to dequantize later

# Delayed scaling: reuse previous step's amax to avoid the synchronization on max.
class DelayedScale:
    def __init__(self): self.amax_history = []
    def update(self, tensor):
        self.amax_history.append(tensor.abs().max().item())
        self.amax_history = self.amax_history[-16:]  # rolling window
    def scale(self, fp8_max):
        return fp8_max / (max(self.amax_history) + 1e-8)`}
      </CodeBlock>

      <Callout accent="gold">
        The scale factor is not optional metadata — it is part of the computation. An FP8 tensor without its scale is uninterpretable. Production frameworks store scales in a separate tensor alongside the quantized data, and any operation that consumes the FP8 tensor must also consume and apply the corresponding scale.
      </Callout>

      <H2>What actually goes wrong</H2>

      <Prose>
        The failure modes of FP8 training are well-documented from production runs. They are distinct from the gradient overflow problems that plagued FP16 — FP8 introduces its own, subtler pathologies.
      </Prose>

      <Prose>
        The first is loss spikes. An outlier token — an unusual sequence, a corrupt document, a rare code construct — produces activation values or gradients that exceed the current scale's range. The tensor clips at the FP8 maximum, the gradient that flows back is corrupted, and the loss spikes upward before recovering over subsequent steps. The fix is either tighter outlier filtering in the data pipeline, gradient clipping before the FP8 cast, or a shorter history window in delayed scaling so the scale adapts faster to distribution shifts.
      </Prose>

      <Prose>
        The second is scale lag. When a layer's activation distribution shifts over the course of training — as it does during warmup, during learning rate changes, or when the model encounters a new data domain — the delayed scale derived from old history no longer matches the current distribution. Values systematically overflow or underflow for several steps until the history window catches up. A 16-step history window is more responsive than a 100-step window but introduces more variance. Tuning this window length for each layer is a practical engineering concern that production training frameworks handle via per-layer scale tracking.
      </Prose>

      <Prose>
        The third failure mode is precision loss in numerically sensitive operations. Layer normalization computes running means and variances; softmax computes exponentials and a normalization sum. Both operations are sensitive to the precision of their inputs in ways that most matrix multiplications are not. The standard fix is to keep these operations in BF16 or FP32 regardless of the surrounding FP8 context — a hybrid-precision approach where the bulk of compute (the matmuls) runs in FP8 and the numerically sensitive operations run at higher precision. Well-tuned FP8 training matches BF16 loss curves almost exactly, with throughput improvements of roughly 1.5 to 2 times on H100 hardware.
      </Prose>

      <H3>Who is actually doing this</H3>

      <Prose>
        NVIDIA's Transformer Engine is the default FP8 training library. It ships with PyTorch integration, handles scale management and delayed scaling automatically, and exposes drop-in replacements for standard linear and attention layers. Most recent large-scale training runs on H100 hardware use it, often without the end user needing to reason about scale factors directly — the library handles the bookkeeping.
      </Prose>

      <Prose>
        DeepSeek-V3 is the highest-profile public example of end-to-end FP8 pre-training. Their technical report documents a hybrid-precision recipe where FP8 matmuls dominate the compute and BF16 is retained for layer norms, softmax, and embedding layers. They specifically call out outlier gradient handling as a critical engineering concern, noting that naive FP8 training without outlier mitigation produces visible quality degradation. Llama 3.1 used FP8 for inference quantization but BF16 for training — a conservative choice that reflects the still-maturing state of FP8 training tooling at the time of that release. The frontier is moving: FP4 training is the next open problem, and early results from NVIDIA and Microsoft suggest it is viable for inference already, with training still an open research question.
      </Prose>

      <Prose>
        Low-precision training is one of those unsexy engineering wins that compounds. A two-times throughput gain on an H100 translates directly into either a larger model, more training tokens, or a shorter wall-clock run time — pick any. The capability gains from scale come from the model and the data; the precision format is just the efficiency multiplier that determines how much of each you can afford. Moving from BF16 to FP8 is not a research contribution, but the teams that do it carefully and reliably are the ones that can run the experiments that are.
      </Prose>
    </div>
  ),
};

export default fp8Training;
