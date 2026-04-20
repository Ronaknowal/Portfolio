import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";

const contextWindowExtension = {
  title: "Context Window Extension (RoPE Scaling, YaRN, NTK-Aware)",
  readTime: "12 min",
  content: () => (
    <div>
      <Prose>
        A model trained on 4k-token contexts does not suddenly understand 128k when you feed it longer inputs. The problem is positional: every position the model encounters at inference time must correspond to something it learned during training, and positions 4001 through 128000 were never in that training distribution. Attention patterns break, coherence collapses, and output quality falls off a cliff somewhere between 1× and 2× the training length. Context window extension is the art of stretching a model's effective context beyond its training length — using modified positional encodings, small amounts of continued training, or in some cases neither. RoPE and its scaling variants are what made 128k-to-million-token contexts practical across the open-weight ecosystem.
      </Prose>

      <H2>Why positions matter</H2>

      <Prose>
        Transformers are permutation-invariant without positional encoding: without some signal about where each token sits in the sequence, the model cannot distinguish "cat bites dog" from "dog bites cat." Early schemes simply added a fixed embedding vector per position. RoPE (Su et al. 2021) takes a different approach: instead of adding position to token representations, it rotates query and key vectors in pairs of dimensions by an angle proportional to position. The rotation angles form a geometric series across the dimension pairs, where <Code>d</Code> is the head dimension and <Code>i</Code> indexes the pair:
      </Prose>

      <MathBlock>{"\\theta_i = 10000^{-2i/d}, \\quad \\text{for dimension pair } i"}</MathBlock>

      <Prose>
        Low-indexed dimension pairs rotate quickly (high frequency); high-indexed pairs rotate slowly (low frequency). When two token positions <Code>m</Code> and <Code>n</Code> attend to one another, the dot product of their rotated query and key depends only on the difference <Code>m - n</Code>, not on either absolute position. Relative distance is captured purely through geometry — no lookup table, no learned position embeddings. This structural property is exactly why RoPE extends better than earlier positional schemes: as long as you can define a rotation for any position, the mechanism stays coherent. The question is just which rotations to use beyond the training range.
      </Prose>

      <H2>Why naive scaling fails</H2>

      <Prose>
        The failure mode is simple. If the model was trained at 4k tokens, the rotation angles for positions 0 through 4095 were seen thousands of times in training data. The model learned which attention patterns to form for tokens separated by 10 positions, 100 positions, or 2000 positions. Positions 4001 through 32000 were never in any training example. The angles at those positions are mathematically valid — RoPE has no hard cutoff — but they land in a region of rotation space the model has no experience with. Attention patterns that the model learned to use at medium range do not transfer to these unseen angles.
      </Prose>

      <Prose>
        In practice the collapse is sharp. Models evaluated on sequences 1.5× their training length often still produce passable output because the out-of-distribution rotations are not yet extreme. At 2× training length, output typically becomes noticeably incoherent. At 8× it fails completely. The naive approach — just run the model on longer sequences and hope — is not a strategy. It is a degradation curve with a predictable shape.
      </Prose>

      <H3>Position Interpolation</H3>

      <Prose>
        The first principled fix came from Chen et al. 2023 in a technique called Position Interpolation (PI). The core idea is simple: instead of mapping sequence position 32000 to RoPE angle 32000 — which is out-of-distribution — divide all positions by a scale factor so they remain within the training range. To go from 4k training context to 32k inference context, the scale factor is 8. Every position in the 32k sequence is divided by 8 before computing the rotation angles, so the model still sees angles corresponding to positions 0 through 4095. Eight input tokens now share the positional resolution that one token had before; the position axis is simply compressed.
      </Prose>

      <CodeBlock language="python">
{`def position_interpolation_rope(x, position, scale_factor, base=10000, dim=128):
    """
    Position Interpolation: compress positions into the training range.
    scale_factor = new_context / original_context  (e.g., 32k / 4k = 8)
    """
    # Effective position seen by the rotation
    effective_pos = position / scale_factor

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    angles = effective_pos * freqs
    return apply_rotary_embedding(x, angles)`}
      </CodeBlock>

      <Prose>
        PI works out of the box for modest extensions — 2× to 4× — because the compressed positions are still well within the training distribution and close enough to their original meanings that the model adapts quickly. For larger stretches, a small amount of continued training is necessary: fine-tuning for a few hundred steps on sequences at the extended length is typically enough to recover quality at 8× extension. The limitation of PI is that it treats all RoPE dimension pairs identically. Low-frequency dimensions (which encode coarse, long-range position) benefit most from stretching; high-frequency dimensions (which encode fine-grained local position) are compressed unnecessarily and lose some of the short-range resolution the model was relying on.
      </Prose>

      <H3>NTK-Aware scaling</H3>

      <Prose>
        NTK-aware scaling (named after Neural Tangent Kernel intuitions, first discussed on the LocalLLaMA subreddit in 2023) addresses PI's uniform treatment of all dimension pairs. Instead of dividing positions by a constant, it changes the RoPE base frequency: increase the base from 10,000 to <Code>10000 × scale^(d/(d-2))</Code>. This modification distributes the stretching unevenly across dimensions. High-frequency dimensions (low <Code>i</Code>, which handle short-range relative positions) are perturbed only slightly. Low-frequency dimensions (high <Code>i</Code>, which handle long-range structure) are stretched aggressively to accommodate the extended context. The short-range representational capacity the model built during training is preserved; only the long-range capacity is extended. In practice, NTK-aware scaling works better than PI for zero-shot context extension — no fine-tuning at all — because the high-frequency dimensions that encode local syntax and grammar are left largely intact.
      </Prose>

      <H2>YaRN — the refined version</H2>

      <Prose>
        YaRN (Yet another RoPE extensioN method, Peng et al. 2023) treats the non-uniform approach of NTK-aware scaling as a starting point and formalizes it. Different dimension pairs are assigned to one of three regimes based on their wavelength relative to the training context length. High-frequency pairs — those with wavelengths shorter than the original context length — should not be rescaled at all, because their rotation angles are already fully within the training distribution. Low-frequency pairs — wavelengths longer than the original context — should be rescaled fully, like PI. Mid-frequency pairs receive partial rescaling, with a smooth ramp between the two extremes. YaRN parameterizes this as a piecewise function with continuous boundaries between regimes:
      </Prose>

      <MathBlock>{"\\theta^{\\text{YaRN}}_i = \\begin{cases} \\theta_i & \\text{high freq (local)} \\\\ \\theta_i / s & \\text{low freq (long-range)} \\\\ \\text{ramp between} & \\text{mid freq} \\end{cases}"}</MathBlock>

      <Prose>
        YaRN also corrects for a subtler problem: at long distances, the softmax over attention scores tends to become sharper than it was during training, because more tokens are accumulating tiny additive contributions. YaRN introduces an attention temperature scaling factor — a small multiplier on the pre-softmax scores — that restores the distribution shape to something closer to what the model saw during training. The combination of per-dimension frequency rescaling and attention temperature correction produces strong long-context quality with remarkably little fine-tuning. Empirically, YaRN reaches 128k context with roughly 400 fine-tuning steps. Yi-34B-200K, the Qwen long-context series, and many Llama-3-extended variants use YaRN for their context extension.
      </Prose>

      <H2>What happens to attention at 128k</H2>

      <Prose>
        Even with ideal positional encoding, attention over very long sequences has structural problems that no RoPE variant fully resolves. The softmax at each attention step must normalize over 128,000 positions. Most of those positions receive vanishingly small weight — the distribution becomes extremely peaked, and the long tail is effectively ignored. This is not a precision problem; it is a fundamental consequence of the softmax over a very long sequence. A model attending to 128k tokens is, in practice, attending to a much smaller effective window concentrated around the positions it judges most relevant, with the rest contributing almost nothing.
      </Prose>

      <Prose>
        A related phenomenon, documented by Xiao et al. 2023 under the name attention sinks, adds a second wrinkle. In models trained without explicit long-context data, the first few tokens of a sequence — the initial newline, a BOS token, the start of the system prompt — accumulate disproportionately large attention weights regardless of their content. The model learns to use them as a numerical-stability drain: since softmax must sum to 1 and all the semantically meaningful positions have small weights, the residual weight pools onto early tokens that serve as a consistent anchor. StreamingLLM explicitly preserves attention-sink tokens even when evicting older context for this reason. Interleaved local-global attention schemes, such as those in Gemini's long-context architecture, address the underlying problem more directly by having some attention heads operate on narrow sliding windows and others on globally sampled positions, so no single head is responsible for softmaxing over 128k entries.
      </Prose>

      <H3>The benchmarks and the reality</H3>

      <Prose>
        Nominal context windows — 128k, 1M, 10M tokens — tell you how many tokens a model will accept without throwing an error. They do not tell you how well the model actually uses that span. "Needle in a haystack" evaluations, which test whether a model can retrieve a specific fact inserted at a known position in a long filler document, pass reasonably well at declared context lengths for most current models. Real long-context reasoning — tracking multiple threads across a novel, maintaining causal consistency across a long technical document, synthesizing information spread across 200 pages — often fails beyond 32k to 64k tokens even when the nominal window is much larger. Retrieval and reasoning are different cognitive tasks, and passing the retrieval test is much easier than passing the reasoning one.
      </Prose>

      <Callout accent="gold">
        A model's nominal context window tells you how many tokens it can input. Its effective reasoning window — where it can still follow the thread — is usually smaller, sometimes much smaller.
      </Callout>

      <Prose>
        Model cards and API documentation routinely describe the nominal window as if it were the reasoning window. Benchmarks that test only retrieval give a flattering picture. Independent evaluations that require multi-hop reasoning across long context consistently find that the effective window is 20–50% of the advertised one for most models. The engineering for accurate positional encoding is largely solved; the training for deep long-range reasoning is not.
      </Prose>

      <H2>Where this stands</H2>

      <Prose>
        Context extension is one area where engineering has outrun theory. NTK-aware scaling and YaRN were figured out by practitioners working from geometric intuition and empirical trial before clean mathematical explanations existed. The field moved from 4k to 128k contexts in roughly 18 months, almost entirely through RoPE modifications and targeted continued pretraining rather than architectural overhaul. The next approach to long context works differently: instead of stretching the window to fit more tokens, it asks what tokens are worth keeping in the first place — selective retrieval over large external stores rather than attending to a very long sequence at once.
      </Prose>
    </div>
  ),
};

export default contextWindowExtension;
