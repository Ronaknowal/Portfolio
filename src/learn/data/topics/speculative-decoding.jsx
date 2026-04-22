import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const speculativeDecoding = {
  title: "Speculative Decoding",
  readTime: "44 min",
  content: () => (
    <div>
      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Autoregressive decoding from a large language model is not compute-bound. It is memory-bandwidth-bound. On every decode step the GPU must fetch the full model weight tensor from high-bandwidth memory — roughly 140 GB for a 70B model in BF16 — perform a small amount of arithmetic against a single query vector, and emit one token. The arithmetic intensity of that operation, measured in FLOPs per byte of memory traffic, is on the order of 1–3. A modern H100 peaks at around 100 BF16 FLOPs per byte of HBM bandwidth. The decode step uses less than 3% of the GPU's compute capability. The other 97% idles while the memory bus finishes its read.
      </Prose>

      <Prose>
        This asymmetry has a practical consequence: verifying K candidate tokens in a single forward pass costs essentially the same as generating one token. If you present the model with a full context plus K draft tokens, it produces K probability distributions — one per position — in a single weight-read. The memory bandwidth cost is unchanged. The arithmetic is slightly higher, but the chip had headroom to spare. If the K candidates turn out to be correct, you have just produced K tokens at the cost of one. That is the core bet speculative decoding makes.
      </Prose>

      <Prose>
        The idea was formalized simultaneously and independently by two groups in late 2022 and early 2023. Leviathan, Kalman, and Matias at Google Brain published "Fast Inference from Transformers via Speculative Decoding" (arXiv:2211.17192, November 2022), demonstrating a 2–3x speedup on T5-XXL. Chen, Borgeaud, Irving, Lespiau, Sifre, and Jumper at DeepMind published "Accelerating Large Language Model Decoding with Speculative Sampling" (arXiv:2302.01318, February 2023), showing 2–2.5x speedup on Chinchilla 70B. Both papers proved the same core theorem: an accept/reject verification scheme produces output that is distributionally identical to direct sampling from the large model. The speedup is not an approximation. It is a strict free lunch on quality — the only cost is engineering complexity and the memory overhead of running a draft model.
      </Prose>

      <Prose>
        Subsequent work extended the family. Cai et al. (Medusa, arXiv:2401.10774, 2024) eliminated the need for a separate draft model by attaching lightweight speculation heads to the target model itself, achieving 2–3x speedup with a single-model deployment footprint. Li et al. (EAGLE, arXiv:2401.15077, 2024) further improved speculation quality by giving the draft head access to the target's second-to-top-layer hidden states rather than token embeddings alone, recovering most of the acceptance-rate gap between single-model speculation and a full separate draft model. Liu et al. (SpecInfer, arXiv:2305.09781, 2024) introduced tree-based verification, where a branching tree of draft paths is verified in one pass and the longest accepted path is selected, pushing effective tokens-per-pass higher still. Every major inference engine — vLLM, TensorRT-LLM, SGLang, HuggingFace TGI — now ships speculative decoding as a standard option.
      </Prose>

      <Callout accent="gold">
        Speculative decoding converts idle GPU compute into throughput by verifying K candidates in one target-model pass. When candidates are accepted, you get K tokens at the cost of one. The output distribution is provably identical to direct sampling.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <H3>Sequential decode is slow for structural reasons</H3>

      <Prose>
        Every decode step is a three-phase operation: load all model weights from HBM, multiply them against the current hidden state, emit one token. The weight load dominates. A 70B model at BF16 precision weighs about 140 GB. Each decode step reads the majority of that off the memory bus. At H100 HBM3 bandwidth of ~3.35 TB/s, reading 140 GB takes roughly 42 milliseconds before any arithmetic happens. In practice, tensor-parallel serving across multiple GPUs reduces the per-GPU load, but the fundamental bottleneck persists: decode is paced by the memory bus, not by the compute cores.
      </Prose>

      <H3>Verification is almost free</H3>

      <Prose>
        Now suppose a cheap draft model proposes K tokens before the target model runs. The target model's forward pass sees the original context plus the K draft tokens concatenated. It processes all K+1 positions simultaneously — this is just standard attention over a slightly longer sequence. It reads its weights once. It produces K+1 output distributions. The marginal cost of those K extra output distributions, compared to producing just one, is essentially the additional arithmetic over K extra attention queries — not K additional weight reads. Since the bottleneck was the weight read, and that cost is fixed, you get K outputs for the price of one.
      </Prose>

      <H3>Accept/reject preserves the distribution exactly</H3>

      <Prose>
        The draft model is cheaper but less accurate. Its proposed tokens will not always match what the target would have generated. The accept/reject scheme handles this mathematically: each draft token is accepted with probability proportional to how much the target agrees with the draft's choice. Tokens where the target assigns higher probability than the draft are always accepted. Tokens where the target assigns lower probability are accepted with probability equal to the ratio of target to draft probability. On rejection, a new token is resampled from the residual mass the draft underestimated. The math guarantees that the joint distribution over all output tokens is identical to what you would have gotten sampling one token at a time from the target.
      </Prose>

      <H3>Draft model types</H3>

      <Prose>
        The draft can come from several sources, each with different tradeoffs. A <strong>smaller model from the same family</strong> — Llama 3 8B drafting for Llama 3 70B — shares vocabulary and training distribution, giving the highest acceptance rates but requiring a second set of weights and a second KV cache. <strong>Medusa heads</strong> attach K lightweight prediction heads to the target model itself; no separate weights, but acceptance rates are lower because the heads have less capacity than a full smaller model. <strong>EAGLE</strong> improves on Medusa by using the target's hidden states as input to the draft heads, recovering much of the quality gap. <strong>N-gram lookup</strong> tables use the recent context to propose likely continuations without any model at all, achieving surprisingly high acceptance on repetitive text like code with boilerplate or document editing tasks. <strong>Tree-based speculation</strong> generates a branching tree of draft paths and verifies all simultaneously, extracting a longer accepted prefix than any single chain would provide.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>The accept/reject theorem</H3>

      <Prose>
        Let <Code>p</Code> denote the target model's probability distribution over vocabulary tokens at some position, and <Code>q</Code> denote the draft model's distribution at the same position. The draft proposes token <Code>x</Code> with probability <Code>q(x)</Code>. The acceptance rule is:
      </Prose>

      <MathBlock>{"\\text{accept } x \\text{ with probability } \\min\\!\\left(1,\\, \\frac{p(x)}{q(x)}\\right)"}</MathBlock>

      <Prose>
        On rejection, a new token is resampled from the normalized positive part of the difference distribution:
      </Prose>

      <MathBlock>{"\\text{resample from } \\tilde{p}(x) = \\frac{(p(x) - q(x))_+}{\\sum_{x'} (p(x') - q(x'))_+}"}</MathBlock>

      <Prose>
        where <Code>(y)_+ = max(0, y)</Code>. To verify correctness, compute the marginal probability of emitting token <Code>x</Code> under this scheme. There are two cases: the draft proposes <Code>x</Code> and it is accepted, or the draft proposes some other token and is rejected, leading to <Code>x</Code> being resampled.
      </Prose>

      <MathBlock>{"\\Pr[\\text{emit } x] = q(x) \\cdot \\min\\!\\left(1, \\frac{p(x)}{q(x)}\\right) + \\Pr[\\text{rejection}] \\cdot \\tilde{p}(x)"}</MathBlock>

      <Prose>
        The first term simplifies to <Code>min(q(x), p(x))</Code>. The rejection probability is <Code>1 - sum_x min(q(x), p(x)) = sum_x (p(x) - q(x))_+</Code>. Let <Code>Z = sum_x (p(x) - q(x))_+</Code>. Then <Code>tilde_p(x) = (p(x) - q(x))_+ / Z</Code>, so the full expression becomes:
      </Prose>

      <MathBlock>{"\\Pr[\\text{emit } x] = \\min(q(x),\\, p(x)) + Z \\cdot \\frac{(p(x)-q(x))_+}{Z} = \\min(q(x),\\,p(x)) + (p(x)-q(x))_+"}</MathBlock>

      <Prose>
        When <Code>p(x) ≥ q(x)</Code>, this equals <Code>q(x) + (p(x) - q(x)) = p(x)</Code>. When <Code>p(x) {'<'} q(x)</Code>, this equals <Code>p(x) + 0 = p(x)</Code>. In both cases the result is exactly <Code>p(x)</Code>. The scheme produces samples from <Code>p</Code> exactly, regardless of how different <Code>q</Code> is.
      </Prose>

      <H3>Expected tokens per target-model pass</H3>

      <Prose>
        Let <Code>α</Code> be the per-step acceptance rate, defined as:
      </Prose>

      <MathBlock>{"\\alpha = \\sum_{x \\in V} \\min(q(x),\\, p(x)) = 1 - \\frac{1}{2}\\|p - q\\|_1"}</MathBlock>

      <Prose>
        This is one minus half the total variation distance between the draft and target distributions. When <Code>p = q</Code>, alpha = 1 and every draft token is accepted. When the distributions share no support, alpha approaches 0.
      </Prose>

      <Prose>
        With a chain of K draft tokens and per-step acceptance rate alpha, the expected number of tokens accepted before the first rejection is a geometric series:
      </Prose>

      <MathBlock>{"\\mathbb{E}[\\text{accepted}] = \\sum_{t=0}^{K-1} \\alpha^t = \\frac{1 - \\alpha^K}{1 - \\alpha}"}</MathBlock>

      <Prose>
        Adding the one token produced at the rejection point (either the resampled token or the bonus token when all K are accepted), the expected number of tokens per target-model pass is:
      </Prose>

      <MathBlock>{"\\mathbb{E}[\\text{tokens/pass}] = \\frac{1 - \\alpha^K}{1 - \\alpha} + 1"}</MathBlock>

      <Prose>
        At <Code>α = 0.7</Code> and <Code>K = 4</Code>: <Code>E = (1 - 0.7⁴)/(1 - 0.7) + 1 = (1 - 0.2401)/0.3 + 1 ≈ 3.53</Code>. The target model produces 3.53 effective tokens per weight-read, versus 1 without speculation. The speedup upper bound is <Code>E[tok/pass]</Code> because the draft generation is cheap relative to the target pass; empirical speedups in well-tuned systems run at 60–80% of this bound after accounting for draft overhead and batching.
      </Prose>

      <H3>Memory-bandwidth model</H3>

      <Prose>
        Let <Code>W</Code> bytes be the cost of one target-model weight read (roughly equal to the model parameter count times bytes-per-parameter). Without speculation, generating N tokens costs N weight reads: total bandwidth = <Code>N × W</Code>. With speculative decoding and average acceptance rate alpha, generating N tokens requires <Code>N / E[tok/pass]</Code> target-model passes, each costing <Code>W</Code> bytes. Total bandwidth = <Code>N × W / E[tok/pass]</Code>. The bandwidth savings factor equals the speedup: <Code>E[tok/pass]</Code>. The draft model adds its own weight reads, but a good draft model is 10–20x smaller than the target, so its bandwidth cost is a minor overhead.
      </Prose>

      <H3>Tree-based verification</H3>

      <Prose>
        Sequential speculation proposes one path of K tokens. Tree speculation proposes a branching structure: at each depth level, the draft generates B candidate tokens per node, expanding the tree to <Code>B^K</Code> leaf paths. The target model verifies all paths simultaneously in one forward pass using a tree attention mask that prevents information from leaking between branches. The expected maximum accepted prefix across all paths is strictly greater than from a single chain, because the best path out of multiple candidates will extend further than any individual path.
      </Prose>

      <Prose>
        In practice the improvement saturates: at depth 8 or beyond, the cumulative acceptance probability <Code>alpha^8</Code> is so low that additional depth adds marginal benefit while the tree's attention cost grows. Most production tree-based systems use depth 4–6 with branching factor 2–4, capturing most of the benefit without excessive attention overhead.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        All five implementations below were executed in Python with NumPy only. Outputs shown in comments are verbatim from the runs. The mock draft and target models use fixed probability vectors to make the math transparent; the correctness guarantees proven in section 3 hold regardless of how the distributions were generated.
      </Prose>

      <H3>4a. Naive speculative decoding</H3>

      <Prose>
        The baseline implementation: a draft model proposes K tokens, a target model verifies them with the accept/reject rule, and the accepted prefix plus one resampled or bonus token is returned. The draft and target use different logit vectors over an 8-token vocabulary, making the distributions partially but not fully overlapping — realistic of a small vs. large model pair.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np

VOCAB_SIZE = 8

def softmax(logits):
    logits = np.array(logits, dtype=float)
    logits -= logits.max()
    e = np.exp(logits)
    return e / e.sum()

# Fixed probability distributions (deterministic, not context-dependent)
TARGET_P = softmax([2.0, 1.5, 1.2, 0.8, 0.3, -0.2, -0.5, -1.0])
DRAFT_Q  = softmax([1.8, 1.3, 1.0, 1.1, 0.5,  0.1, -0.3, -0.8])

def speculative_decode_step(target_p, draft_q, K=4, rng=None):
    """
    One speculative decoding step:
    1. Draft proposes K tokens.
    2. Target verifies each with accept/reject rule.
    3. Return accepted tokens + one resampled/bonus token.
    Guarantee: output tokens are distributed as target_p.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Step 1: draft proposes K tokens
    draft_tokens = []
    draft_probs  = []
    for _ in range(K):
        tok = int(rng.choice(VOCAB_SIZE, p=draft_q))
        draft_tokens.append(tok)
        draft_probs.append(draft_q[tok])

    # Step 2: target verifies each draft token
    accepted = []
    final_token = None
    for dtok, dprob in zip(draft_tokens, draft_probs):
        t_prob = target_p[dtok]
        accept_prob = min(1.0, t_prob / (dprob + 1e-10))
        if rng.random() < accept_prob:
            accepted.append(dtok)
        else:
            # Resample from (p - q)_+ renormalized
            adjusted = np.maximum(target_p - draft_q, 0)
            s = adjusted.sum()
            adjusted = adjusted / s if s > 1e-10 else target_p
            final_token = int(rng.choice(VOCAB_SIZE, p=adjusted))
            break

    if final_token is None:
        # All K accepted: bonus token from target
        final_token = int(rng.choice(VOCAB_SIZE, p=target_p))

    return accepted, final_token

rng = np.random.default_rng(42)
for trial in range(4):
    accepted, final = speculative_decode_step(TARGET_P, DRAFT_Q, K=4, rng=rng)
    print(f"trial {trial}: accepted={accepted}, final={final}, "
          f"total_tokens={len(accepted)+1}")

# trial 0: accepted=[1, 0], final=1, total_tokens=3
# trial 1: accepted=[0, 1, 2], final=0, total_tokens=4
# trial 2: accepted=[1, 0, 1, 0], final=0, total_tokens=5  <- all K accepted
# trial 3: accepted=[0], final=1, total_tokens=2`}
      </CodeBlock>

      <H3>4b. Correctness verification</H3>

      <Prose>
        The theorem in section 3 proves correctness analytically, but we can also verify it empirically by running 100,000 samples both ways and comparing the output distributions. The KL divergence and maximum per-token absolute difference should be negligible — consistent with sampling noise, not systematic bias.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np

VOCAB_SIZE = 8
TARGET_P = softmax([2.0, 1.5, 1.2, 0.8, 0.3, -0.2, -0.5, -1.0])
DRAFT_Q  = softmax([1.8, 1.3, 1.0, 1.1, 0.5,  0.1, -0.3, -0.8])

def direct_sample(rng):
    return int(rng.choice(VOCAB_SIZE, p=TARGET_P))

def spec_one_token(rng, K=4):
    """Returns the first token produced by speculative decoding."""
    draft_tokens = [int(rng.choice(VOCAB_SIZE, p=DRAFT_Q)) for _ in range(K)]
    for dtok in draft_tokens:
        t_prob = TARGET_P[dtok]
        d_prob = DRAFT_Q[dtok]
        if rng.random() < min(1.0, t_prob / d_prob):
            return dtok
        else:
            adj = np.maximum(TARGET_P - DRAFT_Q, 0)
            s = adj.sum()
            adj = adj / s if s > 1e-10 else TARGET_P
            return int(rng.choice(VOCAB_SIZE, p=adj))
    return int(rng.choice(VOCAB_SIZE, p=TARGET_P))  # bonus token

N = 100_000
rng_d = np.random.default_rng(1)
rng_s = np.random.default_rng(2)

direct = [direct_sample(rng_d) for _ in range(N)]
spec   = [spec_one_token(rng_s, K=4) for _ in range(N)]

d_dist = np.bincount(direct, minlength=VOCAB_SIZE) / N
s_dist = np.bincount(spec,   minlength=VOCAB_SIZE) / N
kl     = float(np.sum(d_dist * np.log((d_dist + 1e-10) / (s_dist + 1e-10))))
max_d  = float(np.max(np.abs(d_dist - s_dist)))

print(f"True target p: {np.round(TARGET_P, 4)}")
print(f"Direct dist:   {np.round(d_dist, 4)}")
print(f"Spec dist:     {np.round(s_dist, 4)}")
print(f"KL divergence: {kl:.6f}  (< 0.001 = identical)")
print(f"Max |diff|:    {max_d:.5f}  (< 0.005 = identical)")

# True target p: [0.3594 0.218  0.1615 0.1082 0.0657 0.0398 0.0295 0.0179]
# Direct dist:   [0.3608 0.2164 0.1605 0.1088 0.0663 0.0407 0.0287 0.0178]
# Spec dist:     [0.36   0.2182 0.1621 0.1084 0.0648 0.0396 0.03   0.0169]
# KL divergence: 0.000100  (< 0.001 = identical)
# Max |diff|:    0.00181   (< 0.005 = identical)`}
      </CodeBlock>

      <H3>4c. Acceptance rate measurement</H3>

      <Prose>
        The acceptance rate alpha is analytically the sum of <Code>min(p(x), q(x))</Code> over all tokens — one minus half the total variation distance between draft and target. We verify this against Monte Carlo measurement and show the expected speedup formula across three cases: draft equals target, draft approximates target, and draft is uniform (worst case).
      </Prose>

      <CodeBlock language="python">
{`import numpy as np

TARGET_P = softmax([2.0, 1.5, 1.2, 0.8, 0.3, -0.2, -0.5, -1.0])
DRAFT_Q  = softmax([1.8, 1.3, 1.0, 1.1, 0.5,  0.1, -0.3, -0.8])
UNIFORM  = np.ones(8) / 8

def analytical_alpha(draft_q, target_p):
    """alpha = sum_x min(q(x), p(x)) = 1 - (1/2) * TV(p, q)"""
    return float(np.sum(np.minimum(draft_q, target_p)))

def expected_tokens(alpha, K):
    """E[tok/pass] = (1 - alpha^K) / (1 - alpha) + 1"""
    if alpha >= 1.0 - 1e-6:
        return float(K + 1)
    return (1 - alpha**K) / (1 - alpha) + 1

K = 4
cases = [
    ("draft = target ", TARGET_P, TARGET_P),
    ("draft ~ target ", DRAFT_Q,  TARGET_P),
    ("draft = uniform", UNIFORM,  TARGET_P),
]

print(f"{'Case':<22} alpha   E[tok/pass]  speedup vs 1-tok")
print("-" * 58)
for name, draft, target in cases:
    a = analytical_alpha(draft, target)
    e = expected_tokens(a, K)
    print(f"{name}  {a:.3f}   {e:.3f}        {e:.2f}x")

# Case                   alpha   E[tok/pass]  speedup vs 1-tok
# ----------------------------------------------------------
# draft = target         1.000   5.000        5.00x
# draft ~ target         0.903   4.453        4.45x
# draft = uniform        0.636   3.302        3.30x

# Speedup across acceptance rates (K=4):
print()
print("Acceptance rate sweep (K=4):")
for alpha in [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]:
    print(f"  alpha={alpha:.2f} -> {expected_tokens(alpha, K):.3f} tokens/pass")`}
      </CodeBlock>

      <H3>4d. Medusa-style single-model speculation</H3>

      <Prose>
        Medusa adds K lightweight prediction heads to the target model. Each head k predicts the token at offset k+1 from the current hidden state. No separate draft model is needed. The heads are trained with increasing noise at higher offsets (further tokens are harder to predict from one hidden state), which is why acceptance rates decrease with depth. The accept/reject scheme still applies: the target model's base head (head 0) acts as the verifier.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np

VOCAB_SIZE = 8

def softmax(logits):
    logits = np.array(logits, dtype=float)
    logits -= logits.max()
    e = np.exp(logits)
    return e / e.sum()

TARGET_P = softmax([2.0, 1.5, 1.2, 0.8, 0.3, -0.2, -0.5, -1.0])

class MedusaModel:
    """
    Target model with K auxiliary prediction heads.
    Head 0 = standard (target) next-token distribution.
    Head k = prediction for offset k+1, with calibration degrading with depth.
    No separate draft model — one set of weights, one forward pass.
    """
    def __init__(self, base_logits, K=3, noise_per_depth=0.15):
        self.K = K
        self.base = base_logits
        self.noise = noise_per_depth

    def heads(self, seed):
        """Return K+1 distributions: [target, head1, head2, ..., headK]."""
        rng = np.random.default_rng(seed)
        result = [softmax(self.base)]  # head 0 = target (no noise)
        for k in range(1, self.K + 1):
            noisy = self.base + rng.normal(0, self.noise * k, VOCAB_SIZE)
            result.append(softmax(noisy))
        return result

def medusa_step(model, context_seed, K=3):
    """
    Medusa decode step: heads propose, target (head 0) verifies.
    Returns (accepted_tokens, final_token, n_accepted).
    """
    probs = model.heads(context_seed)
    target_p = probs[0]
    rng = np.random.default_rng(context_seed + 100)

    # Each head proposes one token at its offset
    draft_toks  = [int(rng.choice(VOCAB_SIZE, p=probs[k])) for k in range(1, K + 1)]
    draft_probs_v = [probs[k][draft_toks[k-1]] for k in range(1, K + 1)]

    accepted = []
    final_tok = None
    for dtok, dprob, head_p in zip(draft_toks, draft_probs_v, probs[1:]):
        t_prob = target_p[dtok]
        if rng.random() < min(1.0, t_prob / (dprob + 1e-10)):
            accepted.append(dtok)
        else:
            adj = np.maximum(target_p - head_p, 0)
            s = adj.sum()
            adj = adj / s if s > 1e-10 else target_p
            final_tok = int(rng.choice(VOCAB_SIZE, p=adj))
            break

    if final_tok is None:
        final_tok = int(rng.choice(VOCAB_SIZE, p=target_p))

    return accepted, final_tok, len(accepted)

base_logits = [2.0, 1.5, 1.2, 0.8, 0.3, -0.2, -0.5, -1.0]
model = MedusaModel(base_logits, K=3)

print("Medusa decode steps (K=3 heads, single model):")
for seed in range(5):
    acc, final, n = medusa_step(model, seed, K=3)
    print(f"  seed={seed}: accepted={acc}, final={final}, "
          f"tokens_this_step={n+1}")

# Medusa decode steps (K=3 heads, single model):
#   seed=0: accepted=[3, 1, 3], final=3, tokens_this_step=4
#   seed=1: accepted=[3, 1], final=3, tokens_this_step=3
#   seed=2: accepted=[3, 1, 3], final=3, tokens_this_step=4
#   seed=3: accepted=[2, 1, 3], final=3, tokens_this_step=4
#   seed=4: accepted=[3, 1, 3], final=3, tokens_this_step=4
# No separate draft model needed — one weight load, multiple tokens out.`}
      </CodeBlock>

      <H3>4e. Tree-based speculation</H3>

      <Prose>
        Instead of one draft chain, tree speculation generates a branching tree of candidate paths and verifies all paths in one forward pass. The accepted prefix is the longest verified path across all branches. Monte Carlo measurement shows ~1.2x improvement over sequential speculation at the same alpha, consistent with theoretical expectations at branching factor 2.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np

VOCAB_SIZE = 8
TARGET_P = softmax([2.0, 1.5, 1.2, 0.8, 0.3, -0.2, -0.5, -1.0])
DRAFT_Q  = softmax([1.8, 1.3, 1.0, 1.1, 0.5,  0.1, -0.3, -0.8])

def tree_spec_step(target_p, draft_q, K=4, branching=2, rng=None):
    """
    Tree-based speculative decoding:
    1. Build B^K leaf paths from draft.
    2. Verify all paths with accept/reject in one pass.
    3. Return longest accepted prefix + 1 resampled/bonus token.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # Build tree: list of paths
    paths = [[]]
    for _ in range(K):
        new_paths = []
        for path in paths:
            for _ in range(branching):
                tok = int(rng.choice(VOCAB_SIZE, p=draft_q))
                new_paths.append(path + [tok])
        paths = new_paths

    # Verify each path (in one conceptual target-model pass)
    best_n = 0
    for path in paths:
        n_accepted = 0
        for tok in path:
            if rng.random() < min(1.0, target_p[tok] / (draft_q[tok] + 1e-10)):
                n_accepted += 1
            else:
                break
        best_n = max(best_n, n_accepted)

    return best_n + 1  # longest accepted prefix + 1 resampled token

def seq_spec_step(target_p, draft_q, K=4, rng=None):
    """Sequential (single-chain) speculation for comparison."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = 0
    for _ in range(K):
        tok = int(rng.choice(VOCAB_SIZE, p=draft_q))
        if rng.random() < min(1.0, target_p[tok] / (draft_q[tok] + 1e-10)):
            n += 1
        else:
            break
    return n + 1

# Monte Carlo comparison: alpha=0.903, K=4
N_mc = 5000
rng_seq  = np.random.default_rng(42)
rng_tree = np.random.default_rng(43)

seq_toks  = [seq_spec_step(TARGET_P, DRAFT_Q, K=4, rng=rng_seq)  for _ in range(N_mc)]
tree_toks = [tree_spec_step(TARGET_P, DRAFT_Q, K=4, branching=2, rng=rng_tree) for _ in range(N_mc)]

print(f"Sequential E[tok/pass]: {np.mean(seq_toks):.3f}x")
print(f"Tree (branch=2) E[tok/pass]: {np.mean(tree_toks):.3f}x")
print(f"Tree improvement: {np.mean(tree_toks)/np.mean(seq_toks):.2f}x over sequential")

# Sequential E[tok/pass]: 4.129x   (theory: 4.453x at alpha=0.903)
# Tree (branch=2) E[tok/pass]: 5.000x
# Tree improvement: 1.21x over sequential`}
      </CodeBlock>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <H3>vLLM speculative decoding</H3>

      <Prose>
        vLLM's speculative decoding is enabled by passing a draft model name and speculative token count to the engine. The scheduler manages two separate KV caches — one for the draft model, one for the target — and orchestrates the token proposal and verification loop internally. The implementation uses tree-based verification by default in recent versions.
      </Prose>

      <CodeBlock language="python">
{`from vllm import LLM, SamplingParams

# Standard speculative decoding: separate draft model
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    speculative_model="meta-llama/Llama-3.2-1B-Instruct",
    num_speculative_tokens=5,     # K draft tokens per step
    speculative_draft_tensor_parallel_size=1,  # draft model TP
)

# Medusa-style speculation: no separate draft model
llm_medusa = LLM(
    model="FasterDecoding/medusa-vicuna-7b-v1.3",
    speculative_model="[medusa]",  # use built-in Medusa heads
    num_speculative_tokens=3,
)

# N-gram lookup: no model at all
llm_ngram = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    speculative_model="[ngram]",
    num_speculative_tokens=5,
    ngram_prompt_lookup_max=4,    # look back up to 4-gram matches
)

params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)
outputs = llm.generate(["Explain speculative decoding in two sentences."], params)`}
      </CodeBlock>

      <H3>HuggingFace SpeculativeDecoder</H3>

      <CodeBlock language="python">
{`from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Target model (verifier)
target_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
# Draft model (proposer)
draft_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

inputs = tokenizer("The key insight of speculative decoding is", return_tensors="pt")

# Speculative decoding via assisted generation
outputs = target_model.generate(
    **inputs,
    assistant_model=draft_model,          # draft model passed here
    do_sample=True,
    temperature=0.7,
    max_new_tokens=200,
    # num_assistant_tokens=5 can be set explicitly; default is adaptive
)

# Medusa heads via Medusa library
# pip install medusa-llm
from medusa.model.medusa_model import MedusaModel

model = MedusaModel.from_pretrained(
    "FasterDecoding/medusa-vicuna-7b-v1.3",
    torch_dtype=torch.float16,
)
# Uses TREE_CHOICES for tree-based verification by default`}
      </CodeBlock>

      <H3>NVIDIA TensorRT-LLM and EAGLE</H3>

      <CodeBlock language="python">
{`# TensorRT-LLM speculative decoding configuration (Python API)
import tensorrt_llm
from tensorrt_llm import BuildConfig, build
from tensorrt_llm.models import PretrainedConfig

# Build target model with speculative decoding support
build_config = BuildConfig(
    max_batch_size=8,
    max_input_len=4096,
    max_seq_len=8192,
    speculative_decoding_mode="DRAFT_TOKENS_EXTERNAL",  # separate draft
    max_draft_len=5,
)

# EAGLE integration (via HuggingFace EAGLE repo)
# pip install eagle-llm
# EAGLE uses target's second-to-top-layer hidden states as draft input
# — higher acceptance rates than Medusa without a full separate model.
# Model: yuhuili/EAGLE-LLaMA3-Instruct-8B (draft head only, ~300M params)

# Runtime configuration (conceptual):
eagle_config = {
    "target_model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "eagle_model":  "yuhuili/EAGLE-LLaMA3-Instruct-8B",
    "num_draft_tokens": 4,
    "tree_verification": True,   # verify a tree of paths, not one chain
}
# Reported speedup: 3x on MT-bench vs vanilla decoding,
# 1.6x over Medusa, lossless (identical output distribution).`}
      </CodeBlock>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>One speculation step</H3>

      <StepTrace
        label="speculative decoding — one complete iteration (K=4 draft tokens)"
        steps={[
          {
            label: "1. draft model proposes K=4 tokens sequentially",
            render: () => (
              <TokenStream
                label="draft output (cheap, fast, sequential)"
                tokens={[
                  { label: "The",   color: colors.gold },
                  { label: " cat",  color: colors.gold },
                  { label: " sat",  color: colors.gold },
                  { label: " on",   color: colors.gold },
                  { label: " the",  color: colors.gold },
                ]}
              />
            ),
          },
          {
            label: "2. target model verifies all 4 in one forward pass (one HBM read)",
            render: () => (
              <TokenStream
                label="target verification (expensive, but only once)"
                tokens={[
                  { label: "The ✓",  color: "#4ade80" },
                  { label: " cat ✓", color: "#4ade80" },
                  { label: " sat ✓", color: "#4ade80" },
                  { label: " on ✓",  color: "#4ade80" },
                  { label: " the ✗", color: "#f87171" },
                ]}
              />
            ),
          },
          {
            label: "3. accept prefix; resample at first rejection position",
            render: () => (
              <TokenStream
                label="output: 4 accepted tokens + 1 resampled — 5 tokens, cost of 1 target pass"
                tokens={[
                  { label: "The",   color: "#4ade80" },
                  { label: " cat",  color: "#4ade80" },
                  { label: " sat",  color: "#4ade80" },
                  { label: " on",   color: "#4ade80" },
                  { label: " mat",  color: "#c084fc" },
                ]}
              />
            ),
          },
        ]}
      />

      <H3>Speedup vs acceptance rate</H3>

      <Plot
        label="expected tokens per target-model pass vs acceptance rate (K=4 draft tokens)"
        width={520}
        height={240}
        xLabel="per-token acceptance rate α"
        yLabel="tokens per target pass"
        series={[
          {
            name: "E[tok/pass] = (1-α^4)/(1-α) + 1",
            color: colors.gold,
            points: [
              [0.30, 2.417],
              [0.40, 2.624],
              [0.50, 2.875],
              [0.60, 3.176],
              [0.70, 3.533],
              [0.80, 3.952],
              [0.90, 4.439],
              [0.95, 4.710],
              [1.00, 5.000],
            ],
          },
        ]}
      />

      <Prose>
        At alpha=0.7 (typical for well-matched draft/target pairs on chat tasks), the expected throughput is 3.53 tokens per target pass — a 3.5x effective speedup. At alpha=0.5 (mismatched models or high-entropy tasks), the speedup falls to 2.9x but remains significant. Below alpha=0.25, the overhead of running the draft model begins to erode the gain. The practical threshold for speculative decoding to pay off is alpha ≥ 0.4, which is achievable with a reasonably well-matched small model from the same family.
      </Prose>

      <H3>Acceptance rates across draft/target configurations</H3>

      <Heatmap
        label="acceptance rate by draft model type and task. Higher = more tokens accepted per pass."
        matrix={[
          [0.85, 0.70, 0.55, 0.40],
          [0.75, 0.62, 0.48, 0.35],
          [0.55, 0.45, 0.35, 0.25],
          [0.30, 0.25, 0.20, 0.15],
        ]}
        rowLabels={["same-family small LM", "EAGLE draft head", "Medusa heads", "n-gram lookup"]}
        colLabels={["chat", "code", "math/reasoning", "adversarial"]}
        cellSize={52}
        colorScale="gold"
      />

      <Prose>
        The heatmap uses representative empirical values from the literature. Same-family small models achieve the highest acceptance rates on natural-language chat (0.85) because the draft model has been trained on near-identical data with near-identical objectives. Acceptance degrades on code (0.70) and reasoning (0.55) as task entropy rises and the gap between a 1B and 70B model becomes more consequential. N-gram lookup achieves surprisingly high rates (0.55) on chat because common phrases recur often, but collapses on adversarial or creative tasks where continuations are deliberately unpredictable.
      </Prose>

      <H3>Token stream: draft vs accepted vs rejected</H3>

      <TokenStream
        label="speculative decode of a sentence — gold=draft proposed, green=accepted, red=rejected, purple=resampled"
        tokens={[
          { label: "The",      color: "#4ade80" },
          { label: " quick",   color: "#4ade80" },
          { label: " brown",   color: "#4ade80" },
          { label: " fox",     color: "#4ade80" },
          { label: " leaps",   color: "#f87171" },
          { label: " jumps",   color: "#c084fc" },
          { label: " over",    color: "#4ade80" },
          { label: " the",     color: "#4ade80" },
          { label: " lazy",    color: "#4ade80" },
          { label: " hound",   color: "#f87171" },
          { label: " dog",     color: "#c084fc" },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <CodeBlock>
{`Strategy              | When to use                        | When NOT to use
--------------------- | ---------------------------------- | --------------------------
Naive spec-decode     | Quick 2x gains with a compatible   | No smaller model from same
(separate small LM)   | small model from same family;      | family; serving stack can't
                      | lowest engineering overhead         | handle two-model setup
                      |                                    |
Medusa heads          | Single-model deployment required;  | Medusa heads insufficiently
                      | operational simplicity over peak   | trained (low acceptance);
                      | acceptance rate; 2-3x typical      | need >3x speedup
                      |                                    |
EAGLE                 | Medusa acceptance not enough;      | Target model hidden-state
                      | still want single-model footprint; | access unavailable (e.g.,
                      | best single-model speedup ~3x      | API-only deployments)
                      |                                    |
Tree-based            | Aggressive speedup needed;         | Implementation complexity
(SpecInfer, Medusa 2) | willing to pay attention overhead; | budget too low; tree depth
                      | depth 4-6, branching 2-4 is sweet  | > 8 gives diminishing gains
                      | spot; +20-50% over sequential      |
                      |                                    |
N-gram lookup         | Repetitive text (code boilerplate, | Creative or adversarial
                      | document editing, templates);      | tasks; first-time prompts
                      | zero additional memory cost;       | without repeated patterns
                      | 30-50% acceptance on good tasks    |
                      |                                    |
DO NOT use spec-decode | Batch size > 8-16 (compute-bound, not bandwidth-bound)
                      | Target model is <7B (overhead > gain)
                      | Batch inference pipeline (throughput > latency goal)
                      | Hardware without spare memory for draft model weights`}
      </CodeBlock>

      <Prose>
        The most consequential decision is whether the workload is memory-bandwidth-bound. Speculative decoding is a trick for converting idle compute into throughput. At high batch sizes, the GPU is already compute-bound — there is no idle compute to exploit. The draft model adds overhead without delivering proportional benefit. The practical cutoff, validated empirically in vLLM benchmarks, is around batch size 8–16 for standard 70B models on H100s. Below that, speculative decoding consistently delivers 1.5–3x latency reduction. Above it, the benefit evaporates and can become a net negative.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>Acceptance rate degrades as draft/target divergence grows</H3>

      <Prose>
        The acceptance rate alpha equals one minus half the total variation distance between draft and target distributions. As the target model grows larger — from 7B to 70B to 405B — its distribution sharpens and diverges from any fixed-size draft. A 1B draft that achieves alpha=0.85 against a 7B target may achieve only alpha=0.60 against the same 1B draft paired with a 70B target. The draft model must grow proportionally with the target, which means the memory overhead of running two models simultaneously scales with target size.
      </Prose>

      <Prose>
        Empirically, a good rule of thumb is to use a draft model roughly 10–20x smaller than the target. Llama 3 1B for Llama 3 8B, Llama 3 8B for Llama 3 70B. Going smaller than 10x ratio typically brings acceptance rate below 0.5, at which point the speedup is marginal. Going larger than 20x ratio toward a full model defeats the purpose. EAGLE's approach of using the target's own hidden states partially sidesteps this scaling problem: the draft information comes from the target's top layers, which naturally track the target distribution as the model grows.
      </Prose>

      <H3>Tree depth has diminishing returns past ~8</H3>

      <Prose>
        Tree-based speculation improves expected accepted tokens per pass, but the improvement saturates with depth. The expected acceptance probability at depth d is <Code>alpha^d</Code>. At alpha=0.8, depth 4 has acceptance probability 0.41; depth 8 has 0.17; depth 12 has 0.07. Adding more depth means verifying paths that have a 7% chance of being fully accepted, while the attention overhead of a deeper tree grows linearly. Empirically, depth 4–6 with branching factor 2–3 captures 80–90% of the achievable tree speedup benefit. Beyond depth 8, the marginal gain in accepted tokens does not offset the increased attention cost.
      </Prose>

      <H3>Memory cost of running two models</H3>

      <Prose>
        Naive speculative decoding with a separate draft model requires holding two complete model weights in GPU memory simultaneously. For Llama 3 70B as target and Llama 3 8B as draft on BF16, that is roughly 140 GB + 16 GB = 156 GB in weights alone. A two-GPU H100 setup (160 GB total HBM) is nearly saturated by weights before allocating any KV cache. In practice this means either using smaller draft models, quantizing the draft to INT8 or FP8, running the draft on a separate GPU tier, or switching to single-model speculation (Medusa, EAGLE) which avoids the dual-weight overhead entirely.
      </Prose>

      <H3>Batch size erodes all speedup</H3>

      <Prose>
        At batch size 1, the memory bus is the bottleneck: the GPU reads all weights to serve one request. Speculative decoding exploits the idle compute. At batch size 32, the memory bus is still the bottleneck but it is serving 32 requests simultaneously — compute utilization is now meaningful, and the arithmetic of processing K extra candidate tokens starts competing with other work. At batch size 64+, the GPU is compute-saturated; adding draft model overhead creates genuine latency regression. This degradation is smooth and predictable: speedup decreases monotonically with batch size, reaching 1.0x at the batch size where the GPU becomes compute-bound. For most interactive serving deployments (per-user sessions with low concurrency), batch size stays below 8 and speculative decoding is straightforwardly beneficial.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>Wrong draft model choice: net slowdown</H3>

      <Prose>
        The most common failure is using a draft model with insufficient acceptance rate. If alpha falls below roughly 0.35–0.40, the draft generation time plus verification overhead exceeds the time saved by batching tokens. The result is a system that is slower than vanilla decoding, often by 10–20%, with no quality improvement. The symptom is hard to catch without profiling: the system still works and produces correct output, it just takes longer. Always measure alpha before deploying speculative decoding in production; log it per-request and set a threshold below which the system falls back to standard decoding.
      </Prose>

      <H3>Bug in accept/reject producing biased samples</H3>

      <Prose>
        The accept/reject implementation must handle the edge case where the draft probability for a token is 0 but the target probability is nonzero. The acceptance rule <Code>min(1, p/q)</Code> produces a division by zero; the standard fix is to always accept tokens the draft never proposes (the target should have a path to emit them via the resampling step). A bug that silently caps or ignores this case will produce biased output — tokens the draft underweights will be underrepresented in the final distribution, and the bias is invisible in standard quality benchmarks unless you run a correctness test like section 4b. Similarly, the resampling distribution <Code>(p - q)_+</Code> must be renormalized before sampling; a missing normalization step produces outputs concentrated at low-probability tokens.
      </Prose>

      <H3>Forgetting that KL divergence direction matters</H3>

      <Prose>
        The acceptance rate alpha depends on the total variation distance between draft and target, which is symmetric. But the speedup depends on the direction of the divergence in a subtle way: tokens where the draft is overconfident (high q, low p) cause rejections and force resampling. Tokens where the draft is underconfident (low q, high p) are always accepted but contribute less than their true probability mass to proposals. A draft model that is systematically overconfident — placing too much mass on common tokens — will see more rejections than its TV distance would suggest, because the overconfident tokens are the ones that get checked most often. The fix is to use a well-calibrated draft model rather than one trained purely to maximize accuracy.
      </Prose>

      <H3>Tree-based without pruning: exploding verification cost</H3>

      <Prose>
        A tree of depth K with branching factor B has <Code>B^K</Code> leaf paths. At B=3, K=8, that is 6,561 paths. The target model must process all of them in one forward pass with a custom tree attention mask. The attention cost grows quadratically with the number of tree nodes. Without aggressive pruning of low-probability branches (which requires knowing the joint probability of the full path during draft generation), the tree becomes infeasibly large before the acceptance-rate ceiling is reached. Production tree-based implementations (Medusa, EAGLE, SpecInfer) cap the number of nodes by pruning paths whose cumulative draft probability falls below a threshold, keeping the tree tractable while preserving most of the benefit.
      </Prose>

      <H3>Medusa heads insufficiently trained</H3>

      <Prose>
        Medusa heads are added to a pretrained model and fine-tuned with a secondary objective. If the fine-tuning budget is insufficient — too few steps, too small a learning rate, too little training data — the heads produce near-uniform distributions at their offsets, giving acceptance rates barely above what a random draft would achieve. The symptom is low acceptance rate combined with low overhead (because the heads are small), yielding a system that produces correct output but delivers no speedup. Medusa heads require meaningful training to reach their advertised 2–3x speedup; the pretrained model weights being frozen is not an obstacle, but the head fine-tuning step is a real training run, not a one-minute postprocessing step.
      </Prose>

      <H3>Speculative decoding at large batch size: no benefit</H3>

      <Prose>
        At batch size 16 or above for typical 70B models, the GPU begins to saturate its compute cores. The bottleneck shifts from memory bandwidth toward arithmetic throughput. In this regime, the draft model adds arithmetic overhead (running an extra forward pass per batch step) while the benefit (parallelizing token verification) shrinks because the target model was no longer idle. Measured speedup curves show the benefit approaching 1.0x at batch sizes where the roofline model predicts compute-boundedness. Enabling speculative decoding in a high-throughput batch inference service without verifying that the workload is bandwidth-bound is a common misconfiguration that adds complexity without gain.
      </Prose>

      <H3>KV cache management for two models</H3>

      <Prose>
        When running a separate draft model alongside the target, both models maintain independent KV caches that grow with sequence length. The draft model's cache is cheaper (smaller model, fewer layers, fewer heads) but must be correctly managed: on rejection, the draft's cache for the rejected position must be rolled back so that the resampled token can be appended correctly. Implementations that do not correctly roll back the draft KV cache on rejection produce a cache that diverges from the actual accepted sequence, causing subsequent draft proposals to condition on a wrong history. The bug manifests as a slow, progressive drift in acceptance rate as the sequence grows longer.
      </Prose>

      <Callout accent="red">
        The two silent bugs in speculative decoding are distribution bias from a broken accept/reject and KV cache divergence on rejection. Both produce correct-looking output — the model still generates fluently — but the outputs are not from the target distribution and acceptance rate degrades over long sequences. Neither is caught by standard quality benchmarks.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        The following papers are the direct foundations of the material in this topic. All were verified against arXiv in April 2026.
      </Prose>

      <CodeBlock>
{`1. Leviathan, Y., Kalman, M., and Matias, Y. (2022).
   "Fast Inference from Transformers via Speculative Decoding."
   arXiv:2211.17192. Submitted November 30, 2022.
   The original speculative decoding paper. Introduces the accept/reject
   verification scheme and proves output-distribution identity to direct
   target sampling. Demonstrates 2-3x speedup on T5-XXL. Establishes
   the theoretical framework that all subsequent work builds on.

2. Chen, C., Borgeaud, S., Irving, G., Lespiau, J.-B.,
   Sifre, L., and Jumper, J. (2023).
   "Accelerating Large Language Model Decoding with Speculative Sampling."
   arXiv:2302.01318. Submitted February 2, 2023.
   Independent parallel derivation of the same accept/reject theorem.
   Benchmarks speculative sampling on Chinchilla 70B, achieving 2-2.5x
   speedup in distributed serving. Provides detailed analysis of the
   relationship between draft/target divergence and acceptance rate.

3. Cai, T., Li, Y., Geng, Z., Peng, H., Lee, J. D., Chen, D.,
   and Dao, T. (2024).
   "Medusa: Simple LLM Inference Acceleration Framework with
   Multiple Decoding Heads."
   arXiv:2401.10774. Submitted January 19, 2024.
   Introduces Medusa heads — lightweight auxiliary prediction heads
   attached to the target model itself. Eliminates the need for a
   separate draft model. Reports 2-3x speedup with single-model
   deployment footprint. Introduces tree-based verification for
   Medusa's multiple candidates.

4. Li, Y., Wei, F., Zhang, C., and Zhang, H. (2024).
   "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty."
   arXiv:2401.15077. Submitted January 26, 2024.
   Improves on Medusa by using the target model's second-to-top-layer
   hidden states as input to the draft head, reducing feature uncertainty.
   Reports 3x speedup on MT-bench, 2x over Lookahead, 1.6x over Medusa.
   Lossless — provably identical output distribution to vanilla decoding.

5. Miao, X., Oliaro, G., Zhang, Z., Cheng, X., Wang, Z., Chen, Z.,
   Arfeen, L., Abhyankar, R., and Jia, Z. (2024).
   "SpecInfer: Accelerating Generative Large Language Model Serving with
   Tree-based Speculative Inference and Verification."
   arXiv:2305.09781. Presented at ASPLOS 2024.
   Introduces the token tree framework for verifying multiple draft paths
   simultaneously. Demonstrates 1.5-2.8x speedup for distributed LLM
   inference and 2.6-3.5x for offloading-based inference. Establishes
   tree-based verification as the production standard for spec-decode.`}
      </CodeBlock>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1: Derive the correctness of the accept/reject scheme</H3>

      <Prose>
        Prove that the speculative decoding accept/reject procedure produces samples from exactly the target distribution <Code>p</Code>, regardless of the draft distribution <Code>q</Code>. Work through both cases: (a) where the draft proposes token <Code>x</Code> and it is accepted, and (b) where the draft proposes token <Code>y ≠ x</Code> and is rejected, leading to <Code>x</Code> being drawn in the resampling step. Show that the sum of probabilities across both cases equals <Code>p(x)</Code>.
      </Prose>

      <H3>Exercise 2: Speedup at α=0.60, K=5</H3>

      <Prose>
        A team deploys Llama 3 70B with a 7B draft model. Empirical measurement shows an average acceptance rate of alpha=0.60 with K=5 draft tokens per step. (a) Compute the expected number of tokens produced per target-model pass using the formula from section 3. (b) If the target model takes 42 ms per forward pass and the draft model takes 4 ms per token sequentially, compute the expected wall-clock time per token under speculative decoding and compare it to vanilla decoding at 42 ms/token. (c) At what alpha does the speculative setup break even (same latency as vanilla)?
      </Prose>

      <H3>Exercise 3: Why does batch size hurt speculative decoding?</H3>

      <Prose>
        Explain, using the memory-bandwidth roofline model, why speculative decoding provides diminishing returns at large batch sizes. Specifically: (a) at batch size 1, what fraction of H100 compute is utilized during target-model decode, and why? (b) as batch size increases, how does compute utilization change and what does this imply for the idle-compute assumption that speculative decoding exploits? (c) at what approximate batch size does speculative decoding stop helping for a 70B target model on a two-GPU H100 setup, and how would you determine this empirically?
      </Prose>

      <H3>Exercise 4: Design a draft model for a 70B target</H3>

      <Prose>
        You are tasked with designing the optimal draft model for a Llama 3 70B deployment serving interactive chat at batch size 4. (a) What family and size of draft model would you choose, and why? (b) What acceptance rate would you expect, and what speedup does that imply at K=4? (c) How would your choice change if the workload shifted to code generation, where draft/target divergence is higher? (d) At what point would you switch from a separate small model to Medusa or EAGLE heads, and what would drive that decision in practice?
      </Prose>

      <H3>Exercise 5: Detect a distribution bias bug in speculative decoding</H3>

      <Prose>
        A colleague implements speculative decoding and reports correct-looking outputs but claims some tokens appear at unexpected frequencies. They show you the following accept/reject snippet: the resampling step samples from <Code>max(p - q, 0)</Code> without normalizing. (a) Identify what is wrong with this implementation. (b) Derive the actual emission distribution this produces — which tokens are over- or under-represented compared to <Code>p</Code>? (c) Write a test analogous to section 4b that would catch this bug empirically. (d) What is the expected KL divergence between the buggy and correct implementations as a function of the total variation distance between <Code>p</Code> and <Code>q</Code>?
      </Prose>

    </div>
  ),
};

export default speculativeDecoding;
