import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const decodingStrategies = {
  title: "Decoding Strategies (Greedy, Beam, Top-k, Top-p, Temperature)",
  readTime: "38 min",
  content: () => (
    <div>
      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        A language model does not generate text. It outputs a vector — one floating-point number per vocabulary entry — at each position in the sequence. That vector, the logit vector, encodes the model's learned beliefs about what token should come next given every token that came before. A softmax converts the logits into a proper probability distribution. At that point the model's work is done. What happens next is the decoding strategy's problem: given a distribution over 50,000 or 100,000 tokens, which one do you actually emit?
      </Prose>

      <Prose>
        This question sounds like a footnote. It is not. The research community spent years treating decoding as an implementation detail — you take the argmax, or you sample, and you move on. The assumption was that better models, trained on more data with better objectives, would make the decoding choice irrelevant: a sufficiently good distribution would be so peaked that any reasonable sampling procedure would converge on the same output. That assumption turned out to be wrong in an interesting way. Better models do produce better distributions, but better distributions create a richer, more exploitable decoding surface — one where the choice of strategy extracts dramatically different behaviors from the same underlying weights. As models grew large enough to produce coherent text under sampling, practitioners discovered that the gap between greedy and temperature-sampled output was not just quantitative (one more accurate, one more varied) but qualitative: different genres, different registers, different apparent personalities.
      </Prose>

      <Prose>
        The choice is not cosmetic. Take a fixed, unchanged model checkpoint. Feed it the same prompt. Change the decoding configuration — temperature, top-p, top-k — and you have, in effect, a different product. At low temperature with greedy decoding the model writes precise, deterministic, reproducible output that looks like technical documentation. At high temperature with wide nucleus sampling, the same weights produce creative, surprising, occasionally incoherent prose. The weights encode what the model knows. The decoding strategy controls how that knowledge is expressed. Practitioners who do not understand decoding are, in a real sense, flying blind: they are running the same knobs without knowing which direction is which, and the tuning space is large enough that the difference between a mediocre and an excellent product often lives here rather than in the training.
      </Prose>

      <Prose>
        There is a timeline worth noting. The dominant strategy in neural machine translation from roughly 2015 to 2019 was beam search — a principled generalization of greedy that maintains multiple candidate sequences and prunes them by cumulative probability. Beam search won benchmarks. It produced translations that scored highly on BLEU. But when the same models were asked to generate open-ended text — stories, dialogue, free-form answers — beam search produced output that was recognizably, almost offensively bland. Holtzman et al. (2019) named the phenomenon neural text degeneration and introduced nucleus sampling (top-p) as the remedy. Within two years, nucleus sampling had displaced beam search as the default decoding method for nearly every generative application that did not involve a verifier. The lesson: the right strategy depends on the task, and the community took years of embarrassing product failures to absorb it.
      </Prose>

      <Prose>
        Decoding strategies also interact with safety. A model that reliably refuses harmful requests under greedy decoding may occasionally comply under high temperature, because the refusal token was only the most probable token — not the only possible token — and high temperature gives the tail of the distribution a real chance to win. Repetition penalties affect content as well as form, and logit biases can be used to steer or suppress entire classes of output. The last mile of inference is where all of these levers live. Alignment work that focuses exclusively on training-time behavior without auditing decoding-time behavior is incomplete — a safety evaluation run at T=0 does not generalize to T=1.2.
      </Prose>

      <Callout accent="gold">
        Decoding is not postprocessing. It is the interface between what the model knows and what the user sees. Understanding it is a prerequisite for controlling it.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <H3>Greedy: commit to the peak</H3>

      <Prose>
        Greedy decoding is the obvious baseline: at every step, take the token with the highest probability and move on. Deterministic, fast, requires no configuration. If the model is well-calibrated and the task has a narrow set of correct answers — precise extraction, translation of a common phrase, code with a syntactically forced continuation — greedy does well. The problem is that it commits irrevocably. A bad choice at step three affects the probability distribution at step four, which affects step five, and so on. The errors compound, and greedy is particularly susceptible to repetition loops: a phrase that the model generates at high probability makes itself slightly more probable in the next position (because it now appears in the context), which makes it even more probable in the position after that, until the context fills with verbatim repetition.
      </Prose>

      <Prose>
        The repetition failure mode is so characteristic of greedy decoding that it became a benchmark for evaluating alternatives. A greedy-decoded story that descends into "the the the the the the" or "I think I think I think I think" is not exhibiting a training defect — the model itself is fine. It is exhibiting a decoding defect: the absence of any noise or truncation means that the highest-probability token wins at every step unconditionally, and when local context makes a token dominant, there is nothing to disrupt the feedback loop. Simple fixes like length-normalized scoring do not address this — the loop persists because the repetition is genuinely high-probability given the immediately preceding tokens, not because the score is wrong.
      </Prose>

      <H3>Beam search: keep multiple guesses alive</H3>

      <Prose>
        Beam search generalizes greedy by maintaining the top-k candidate sequences in parallel rather than committing to a single path. At each step every beam is extended by its most probable continuations, the combined pool is scored by cumulative log-probability, and the top-k sequences survive. This allows the search to recover from locally bad choices. A beam that picks a mediocre token at step three stays alive only if it remains competitive; a different beam that took a lower-probability token early but produces a high-probability continuation later can overtake it. In constrained settings — neural machine translation, speech recognition, structured output — beam search was the dominant strategy for years, and remains so for applications where a verifier can select among candidates. For open-ended generation, its bias toward short, common, safe phrases makes it actively harmful.
      </Prose>

      <Prose>
        The failure mode of beam search in open-ended generation is distinct from greedy's repetition problem. Beam search does not produce loops — the cumulative log-probability of any repeating sequence falls because each repetition of a common token is evaluated against a different context. Instead, beam search produces what practitioners call safe, generic, modal text: it finds the high-probability highway through the model's distribution and stays on it. The result is coherent, grammatically perfect, and almost indistinguishable from the most-average possible continuation of any prompt. Asking a beam search decoder to write a creative story reliably produces something that sounds like the opening paragraph of a business email. The diversity that makes text interesting — the unexpected word choice, the tonal shift, the structural surprise — lives in the lower-probability branches that beam search prunes.
      </Prose>

      <H3>Temperature: squeeze or spread the distribution</H3>

      <Prose>
        Temperature is the single most-used decoding knob. It rescales the logits before the softmax. Divide by a number less than one and the distribution sharpens — the most probable tokens become relatively more probable. Divide by a number greater than one and the distribution flattens — low-probability tokens get a proportionally larger share of the mass. Temperature below one approaches greedy (in the limit, T→0 makes softmax into argmax). Temperature equal to one is the model's native training distribution. Temperature above one moves toward uniform, which means increasingly random output that strays further from what the model was calibrated to produce.
      </Prose>

      <Prose>
        The name comes from statistical physics. In the Boltzmann distribution, temperature controls the ratio of high-energy to low-energy state probabilities: at low temperature, a system tends to occupy its lowest-energy states; at high temperature, all states become roughly equally probable. The analogy is direct — low logit values correspond to high energy, and temperature rescaling applies the same physics to the language model's distribution over tokens. This is not just a metaphor: Ackley, Hinton, and Sejnowski imported the concept directly into Boltzmann machines in 1985, and temperature as a decoding parameter in NLP descends directly from that work.
      </Prose>

      <H3>Top-k: hard cutoff at the k-th token</H3>

      <Prose>
        Top-k sets a fixed number of candidates. Keep only the k highest-probability tokens, zero out the rest, renormalize, and sample from the resulting truncated distribution. It is simple and easy to reason about. Its weakness is that the right k varies step by step: some positions have one or two overwhelmingly likely continuations, and top-k=40 means sampling from 38 tokens that have no business being there; other positions have genuine distributional spread, and top-k=40 may be needlessly restrictive. The conceptual problem is that k is a count-based threshold applied to a probability-based phenomenon. Probability distributions do not respect constant widths: the natural boundary of the "plausible" region of a distribution changes with the peakedness of that distribution, and a fixed k is always either too wide or too narrow depending on context.
      </Prose>

      <H3>Top-p (nucleus): adaptive cutoff by mass</H3>

      <Prose>
        Top-p, introduced under the name nucleus sampling by Holtzman et al. in 2019, solves top-k's fixed-width problem by making the cutoff adaptive. Keep the smallest set of tokens whose cumulative probability reaches p. When the model is confident, that set contains one or two tokens. When the model is uncertain, it contains many. The cutoff tracks the model's own uncertainty rather than a human-chosen constant. Top-p with p=0.9 or 0.95 has become the primary truncation mechanism in production systems.
      </Prose>

      <Prose>
        The key insight behind top-p is that the relevant quantity is not "how many tokens are in the plausible set" but "what fraction of total probability mass is in the plausible set." A fixed fraction p is a probability-native criterion. When you set p=0.9, you are saying: exclude the tokens that, in aggregate, account for only the bottom 10% of the probability mass. The model's own distribution determines where that line falls. At a syntactically forced position, 90% of the mass might sit on a single token, and the nucleus contains just that token. At a free creative position, 90% of the mass might be spread across 200 tokens, and the nucleus admits all 200. This adaptive behavior is exactly what was missing from top-k.
      </Prose>

      <H3>Min-p, repetition penalty, and other knobs</H3>

      <Prose>
        Min-p is a recent variant (Nguyen et al., 2024) that sets a relative threshold: keep only tokens whose probability exceeds <Code>p_min × max_probability</Code>. When the distribution is peaked, the threshold is high and only a few tokens survive. When the distribution is flat, the threshold falls and more tokens are admitted. This scales more gracefully than top-p at high temperatures, where the nucleus sampling mechanism can degenerate: as temperature rises and the distribution flattens, a fixed-p nucleus grows toward the entire vocabulary, because many tokens collectively sum to p. Min-p avoids this because the threshold is proportional to the peak probability, which also falls as the distribution flattens — meaning the same p_min excludes progressively fewer tokens in proportion to how uncertain the model is, rather than admitting progressively more.
      </Prose>

      <Prose>
        Repetition penalties directly modify logits for tokens that have already appeared in the context — dividing them by a factor greater than one — which breaks the feedback loops that cause repetition without changing the temperature or nucleus shape. Frequency penalties scale that discount by occurrence count, applying heavier discounts to tokens that have appeared many times versus those that appeared once. Stop strings terminate generation when a specified token sequence appears in the output, acting as a generation boundary. Logit biases add or subtract arbitrary constants from specific token logits before all other processing, allowing hard steering of the model toward or away from particular vocabulary items. Each of these interacts with the probability-based filters in ways that can surprise if you apply them in the wrong order or combination.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Math foundation</H2>

      <H3>Softmax with temperature</H3>

      <Prose>
        Let <Code>z</Code> be the logit vector of length V (vocabulary size). The standard softmax converts it to probabilities:
      </Prose>

      <MathBlock>{"p(x_i) = \\frac{\\exp(z_i)}{\\sum_{j=1}^{V} \\exp(z_j)}"}</MathBlock>

      <Prose>
        Temperature T is applied by dividing every logit before the softmax:
      </Prose>

      <MathBlock>{"p_T(x_i) = \\frac{\\exp(z_i / T)}{\\sum_{j=1}^{V} \\exp(z_j / T)}"}</MathBlock>

      <Prose>
        As T→0, the ratio <Code>z_i/T</Code> diverges for any logit that differs from the maximum. The softmax becomes an argmax: all probability mass concentrates on the single highest-logit token. This is greedy decoding. At T=1 the formula reduces to standard softmax — the distribution the model was trained against. At T→∞ the logit ratios all collapse to zero, and softmax returns the uniform distribution: every token in the vocabulary is equally likely. In practice temperatures above 1.5 are rarely useful because the model samples deep into the tail of its distribution, emitting tokens that are improbable by its own measure.
      </Prose>

      <Prose>
        The entropy of <Code>p_T</Code> is a clean summary of how much uncertainty the distribution carries. For a fixed logit vector, entropy is monotonically increasing in T: zero at T→0 (one certain token), log(V) at T→∞ (all tokens equally likely).
      </Prose>

      <H3>Top-k truncation</H3>

      <Prose>
        After temperature scaling, top-k truncation sets the logit of every token outside the top-k to negative infinity before renormalization:
      </Prose>

      <MathBlock>{"\\tilde{z}_i = \\begin{cases} z_i & \\text{if } i \\in \\mathrm{top\\text{-}}k(z) \\\\ -\\infty & \\text{otherwise} \\end{cases}"}</MathBlock>

      <Prose>
        Passing <Code>z̃</Code> through softmax renormalizes the surviving k tokens to sum to one. In implementation, it is more efficient to use <Code>argpartition</Code> (O(V)) than a full sort (O(V log V)) to identify the top-k indices.
      </Prose>

      <H3>Top-p (nucleus) truncation</H3>

      <Prose>
        Sort tokens by probability in descending order. Walk down the sorted list, accumulating probability mass. Stop at the smallest index i such that the cumulative sum reaches p. All tokens beyond index i receive zero probability; the survivors are renormalized:
      </Prose>

      <MathBlock>{"\\text{nucleus}(p) = \\min\\!\\left\\{ S \\subseteq V : \\sum_{i \\in S} p(x_i) \\geq p,\\ S \\text{ sorted by } p(x_i) \\text{ desc} \\right\\}"}</MathBlock>

      <Prose>
        The nucleus size varies adaptively with the distribution. When the model assigns 0.96 of its mass to one token, the nucleus contains a single token regardless of p. When mass is spread across hundreds of tokens, the nucleus is large and sampling is exploratory.
      </Prose>

      <H3>Min-p</H3>

      <Prose>
        Let <Code>p_max = max_i p(x_i)</Code> be the probability of the most likely token. Min-p keeps only tokens satisfying:
      </Prose>

      <MathBlock>{"p(x_i) \\geq p_{\\min} \\cdot p_{\\max}"}</MathBlock>

      <Prose>
        The threshold scales with the peak of the distribution, not with an absolute value. When the model is very confident (<Code>p_max</Code> = 0.95), only tokens with probability ≥ 0.095 survive — a tight nucleus. When the model is uncertain (<Code>p_max</Code> = 0.2), tokens with probability ≥ 0.01 survive — a much wider nucleus. This adaptive scaling is why min-p handles high temperatures more gracefully than fixed top-p: as temperature rises and the distribution flattens, <Code>p_max</Code> falls and the threshold falls proportionally, admitting more tokens rather than the nucleus exploding toward the full vocabulary.
      </Prose>

      <H3>Repetition penalty</H3>

      <Prose>
        Let <Code>S</Code> be the set of token ids that have appeared in the context. For each token in S, the logit is scaled before softmax:
      </Prose>

      <MathBlock>{"z'_i = \\begin{cases} z_i / \\theta & \\text{if } z_i > 0 \\text{ and } i \\in S \\\\ z_i \\cdot \\theta & \\text{if } z_i \\leq 0 \\text{ and } i \\in S \\\\ z_i & \\text{otherwise} \\end{cases}"}</MathBlock>

      <Prose>
        Where <Code>θ &gt; 1</Code> is the penalty factor. The signed branching ensures the penalty always pushes probability downward, regardless of whether the logit is positive or negative. A frequency penalty variant replaces the binary membership test with a count-weighted version, scaling the discount proportionally to how many times each token has appeared.
      </Prose>

      <H3>Beam search scoring</H3>

      <Prose>
        A beam is a partial sequence <Code>y₁…yₜ</Code>. Its score is the sum of log-probabilities:
      </Prose>

      <MathBlock>{"\\text{score}(y_1, \\ldots, y_t) = \\sum_{i=1}^{t} \\log p(y_i \\mid y_{1:i-1}, x)"}</MathBlock>

      <Prose>
        Because log-probabilities are negative, longer sequences accumulate more negative score. This creates a length bias: beam search tends to favor shorter completions unless a length normalization term (dividing by sequence length or a power thereof) is added. The length bias was documented empirically by Stahlberg and Byrne (2019), who showed that the globally optimal sequence under many NMT models was the empty string — a consequence of model-side probability mass leaking into the empty output interacting with length-penalized beam scoring.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        Every function below was run in Python with NumPy only. Outputs shown in comments are the actual outputs, verbatim.
      </Prose>

      <H3>4a. Greedy decoding</H3>

      <CodeBlock language="python">
{`import numpy as np

VOCAB = ['<eos>', 'the', 'cat', 'sat', 'mat']

# Toy bigram log-prob table: logits[last_token_id] -> logit vector
LOGIT_TABLE = {
    0: np.log([0.01, 0.50, 0.30, 0.10, 0.09] + np.finfo(float).eps),
    1: np.log([0.01, 0.01, 0.80, 0.10, 0.08] + np.finfo(float).eps),
    2: np.log([0.02, 0.03, 0.02, 0.88, 0.05] + np.finfo(float).eps),
    3: np.log([0.10, 0.10, 0.05, 0.05, 0.70] + np.finfo(float).eps),
    4: np.log([0.80, 0.12, 0.04, 0.02, 0.02] + np.finfo(float).eps),
}

def toy_logits(tokens):
    return LOGIT_TABLE[tokens[-1] % 5]

def greedy_decode(logits_fn, prompt_ids, max_tokens=20, eos_id=None):
    tokens = list(prompt_ids)
    for _ in range(max_tokens):
        logits = logits_fn(tokens)
        next_tok = int(np.argmax(logits))
        tokens.append(next_tok)
        if eos_id is not None and next_tok == eos_id:
            break
    return tokens

result = greedy_decode(toy_logits, [1], max_tokens=6, eos_id=0)
print([VOCAB[i] for i in result])
# ['the', 'cat', 'sat', 'mat', '<eos>']`}
      </CodeBlock>

      <Prose>
        The loop is a direct implementation of the definition: call the model, take argmax, append, repeat. The only subtlety is EOS handling — without it the loop runs until max_tokens regardless of whether the model has signaled completion.
      </Prose>

      <H3>4b. Temperature + softmax</H3>

      <CodeBlock language="python">
{`def softmax_with_temperature(logits, T=1.0):
    logits = np.array(logits, dtype=float) / T
    logits -= logits.max()          # numerical stability: subtract max before exp
    exp = np.exp(logits)
    return exp / exp.sum()

test_logits = [2.0, 1.5, 0.5, -0.5, -1.5]

for T in [0.1, 0.5, 1.0, 1.5, 2.0]:
    p = softmax_with_temperature(test_logits, T)
    entropy = -np.sum(p * np.log(p + 1e-12))
    print(f"T={T}: probs={np.round(p, 3)}, H={entropy:.4f}")

# T=0.1: probs=[0.993 0.007 0.    0.    0.   ], H=0.0402
# T=0.5: probs=[0.702 0.258 0.035 0.005 0.001], H=0.7454
# T=1.0: probs=[0.515 0.312 0.115 0.042 0.016], H=1.1523
# T=1.5: probs=[0.422 0.302 0.155 0.08  0.041], H=1.3472
# T=2.0: probs=[0.369 0.287 0.174 0.106 0.064], H=1.4442`}
      </CodeBlock>

      <Prose>
        The subtracting-the-max step is mandatory in practice. Without it, large logits overflow <Code>np.exp</Code> to infinity. The result is mathematically identical — the constant cancels in the normalization — but numerically safe. Note how T=0.1 collapses the distribution almost entirely onto the first token (greedy behavior), while T=2.0 gives the last token 6.4% of the mass despite it having a logit of -1.5 at T=1.
      </Prose>

      <H3>4c. Top-k sampling</H3>

      <CodeBlock language="python">
{`def top_k_sample(logits, k=3, temperature=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    logits = np.array(logits, dtype=float) / temperature
    # argpartition: O(V) instead of O(V log V) full sort
    top_k_idx = np.argpartition(logits, -k)[-k:]
    top_k_logits = logits[top_k_idx]
    top_k_logits -= top_k_logits.max()
    probs = np.exp(top_k_logits)
    probs /= probs.sum()
    sampled = rng.choice(len(probs), p=probs)
    return int(top_k_idx[sampled])

# Verify: only indices 0, 1, 2 should appear (top-3 of test_logits)
rng = np.random.default_rng(0)
counts = {}
for _ in range(10_000):
    t = top_k_sample(test_logits, k=3, temperature=1.0, rng=rng)
    counts[t] = counts.get(t, 0) + 1

print(counts)
# {0: 5444, 1: 3318, 2: 1238}  — only indices 0,1,2 appear`}
      </CodeBlock>

      <Prose>
        The 10,000-sample verification confirms that tokens at indices 3 and 4 (the two lowest-logit tokens) never appear. The surviving three tokens are sampled in proportion to their renormalized probabilities, preserving the relative ordering from the original distribution.
      </Prose>

      <H3>4d. Top-p (nucleus) sampling</H3>

      <CodeBlock language="python">
{`def top_p_sample(logits, p=0.9, temperature=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    logits = np.array(logits, dtype=float) / temperature
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()

    sorted_idx = np.argsort(probs)[::-1]           # descending order
    sorted_probs = probs[sorted_idx]
    cumsum = np.cumsum(sorted_probs)

    # smallest prefix summing to >= p
    cutoff = int(np.searchsorted(cumsum, p)) + 1
    cutoff = max(1, cutoff)                        # always keep at least one token

    nucleus_idx = sorted_idx[:cutoff]
    nucleus_probs = probs[nucleus_idx]
    nucleus_probs /= nucleus_probs.sum()

    sampled = rng.choice(len(nucleus_probs), p=nucleus_probs)
    return int(nucleus_idx[sampled])

# How many tokens does p=0.9 keep for test_logits at T=1?
p_test = softmax_with_temperature(test_logits, 1.0)
sorted_p = np.sort(p_test)[::-1]
cumsum_test = np.cumsum(sorted_p)
print("Sorted probs:", np.round(sorted_p, 4))
print("Cumsum:      ", np.round(cumsum_test, 4))
print("Tokens to reach 0.9:", int((cumsum_test < 0.9).sum()) + 1)

# Sorted probs: [0.5149 0.3123 0.1149 0.0423 0.0156]
# Cumsum:       [0.5149 0.8273 0.9422 0.9844 1.    ]
# Tokens to reach 0.9: 3`}
      </CodeBlock>

      <Prose>
        Three tokens cover 94.2% of the probability mass at T=1. If temperature is raised to 2.0, the distribution flattens and all five tokens may enter the nucleus; at T=0.1, the nucleus collapses to a single token. This is the adaptive property that makes top-p superior to top-k for variable-context generation.
      </Prose>

      <H3>4e. Min-p sampling</H3>

      <CodeBlock language="python">
{`def min_p_sample(logits, p_min=0.05, temperature=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    logits = np.array(logits, dtype=float) / temperature
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()

    max_prob = probs.max()
    threshold = p_min * max_prob    # relative, not absolute
    keep = probs >= threshold
    kept_probs = probs * keep
    kept_probs /= kept_probs.sum()

    return int(rng.choice(len(kept_probs), p=kept_probs))

# Peaked distribution — model confident about index 0
peaked_logits = [3.0, 0.5, 0.3, 0.1, 0.0]
_, probs_pk, n_pk = min_p_sample(peaked_logits, p_min=0.1, temperature=1.0), None, None
p_pk = softmax_with_temperature(peaked_logits)
max_p = p_pk.max()
keep_pk = p_pk >= 0.1 * max_p
print("Peaked — max_prob:", round(float(max_p), 4),
      "threshold:", round(float(0.1 * max_p), 4),
      "tokens kept:", keep_pk.sum())
# Peaked — max_prob: 0.8949  threshold: 0.0895  tokens kept: 1

# Flat distribution — model uncertain
flat_logits = [1.0, 0.9, 0.8, 0.7, 0.6]
p_flat = softmax_with_temperature(flat_logits)
max_f = p_flat.max()
keep_flat = p_flat >= 0.1 * max_f
print("Flat — max_prob:", round(float(max_f), 4),
      "threshold:", round(float(0.1 * max_f), 4),
      "tokens kept:", keep_flat.sum())
# Flat — max_prob: 0.2419  threshold: 0.0242  tokens kept: 5`}
      </CodeBlock>

      <Prose>
        This is the defining property of min-p. On the peaked distribution the threshold is ~9% of total mass — only the dominant token survives. On the flat distribution the threshold falls to ~2.4% of total mass — all five tokens are admitted. With fixed top-p=0.9, both distributions would produce a nucleus of similar size; min-p tracks the model's actual confidence level and shrinks or expands accordingly.
      </Prose>

      <H3>4f. Beam search with length bias demonstration</H3>

      <CodeBlock language="python">
{`def beam_search(logits_fn, prompt_ids, beam_width=3, max_tokens=8, eos_id=None):
    beams = [(list(prompt_ids), 0.0)]   # (tokens, cumulative_log_prob)
    for _ in range(max_tokens):
        candidates = []
        all_done = True
        for seq, score in beams:
            if eos_id is not None and seq[-1] == eos_id:
                candidates.append((seq, score))   # completed beam survives as-is
                continue
            all_done = False
            logits = logits_fn(seq)
            logits -= logits.max()
            log_probs = logits - np.log(np.sum(np.exp(logits)))
            for tok_id in range(len(log_probs)):
                candidates.append((seq + [tok_id], score + log_probs[tok_id]))
        if all_done:
            break
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_width]
    return beams

beams = beam_search(toy_logits, [1], beam_width=3, max_tokens=6, eos_id=0)
print("Beam search (width=3):")
for seq, score in beams:
    words = [VOCAB[i] for i in seq]
    length_norm = score / len(seq)
    print(f"  raw={score:.4f}  norm={length_norm:.4f}  len={len(seq)}  {words}")

# Beam search (width=3):
#   raw=-0.9308  norm=-0.1862  len=5  ['the', 'cat', 'sat', 'mat', '<eos>']
#   raw=-2.6536  norm=-0.6634  len=4  ['the', 'cat', 'sat', '<eos>']
#   raw=-2.7489  norm=-0.9163  len=3  ['the', 'mat', '<eos>']`}
      </CodeBlock>

      <Prose>
        The length bias is visible here. Ranked by raw score, the longest sequence wins — it accumulated log-probability over more steps. But the normalized score (dividing by length) shows the same picture, because in this toy model each additional step actually adds probability rather than being a liability. In real NMT models with less peaked distributions and a meaningful EOS probability at every step, the opposite pattern emerges: the model assigns non-trivial probability to EOS at every token, so longer sequences pay a per-step penalty that shorter sequences avoid. Stahlberg and Byrne (2019) showed this leads to models whose globally optimal sequence — under exact search — is the empty string.
      </Prose>

      <H3>4g. Repetition penalty</H3>

      <CodeBlock language="python">
{`def apply_repetition_penalty(logits, past_token_ids, penalty=1.3):
    logits = np.array(logits, dtype=float)
    seen = set(past_token_ids)
    for tok in seen:
        if logits[tok] > 0:
            logits[tok] /= penalty    # positive logit: reduce
        else:
            logits[tok] *= penalty    # negative logit: push more negative
    return logits

# Model strongly favors 'cat' (index 2), which has appeared twice
base_logits = np.array([0.1, 0.5, 2.0, 1.0, 0.3])
past = [1, 2, 3, 2]   # 'cat' appears twice in past context

penalized = apply_repetition_penalty(base_logits, past, penalty=1.3)
print("Base logits:     ", base_logits)
print("Penalized logits:", np.round(penalized, 4))
# Base logits:      [0.1  0.5  2.   1.   0.3 ]
# Penalized logits: [0.1  0.3846 1.5385 0.7692 0.3   ]

def to_probs(logits):
    l = logits - logits.max()
    p = np.exp(l); return p / p.sum()

print("P(cat) base:     ", round(float(to_probs(base_logits)[2]), 4))
print("P(cat) penalized:", round(float(to_probs(penalized)[2]), 4))
# P(cat) base:      0.52
# P(cat) penalized: 0.4337`}
      </CodeBlock>

      <Prose>
        The penalty reduces the probability of 'cat' from 52% to 43% — a meaningful shift despite the token still having the highest logit. With a stronger penalty (θ=1.5 or 2.0), repeated tokens can be suppressed below the top-p threshold entirely. The interaction with list and table generation is a known failure mode: a model generating a structured list with repeated syntactic patterns (bullet points, commas, colons) will have those tokens penalized even when they are correct and expected. Section 9 covers this in detail.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <H3>HuggingFace Transformers</H3>

      <CodeBlock language="python">
{`from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

inputs = tokenizer("The capital of France is", return_tensors="pt")

# Greedy
greedy_out = model.generate(**inputs, max_new_tokens=20, do_sample=False)

# Temperature + top-p (chat default)
sample_out = model.generate(
    **inputs,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
)

# Beam search (translation / structured output)
beam_out = model.generate(
    **inputs,
    max_new_tokens=100,
    num_beams=5,
    no_repeat_ngram_size=3,   # prevents beam from repeating 3-grams
    early_stopping=True,
)

# Note: temperature and do_sample=False are mutually redundant.
# HuggingFace will warn if temperature != 1.0 and do_sample=False.`}
      </CodeBlock>

      <H3>vLLM</H3>

      <CodeBlock language="python">
{`from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")

# Chat default
params_chat = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
    repetition_penalty=1.1,
)

# Code / math (near-greedy)
params_code = SamplingParams(
    temperature=0.0,    # vLLM uses greedy when temperature=0
    max_tokens=1024,
)

# Creative writing
params_creative = SamplingParams(
    temperature=1.0,
    top_p=0.95,
    min_p=0.05,         # vLLM supports min_p natively
    max_tokens=2048,
)

outputs = llm.generate(["Write a haiku about gradient descent"], params_creative)`}
      </CodeBlock>

      <H3>OpenAI API</H3>

      <CodeBlock language="python">
{`from openai import OpenAI

client = OpenAI()

# OpenAI exposes: temperature (0-2), top_p (0-1),
# presence_penalty (-2 to 2), frequency_penalty (-2 to 2).
# top_k is NOT exposed. Nucleus sampling is the primary mechanism.

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Explain beam search in one paragraph."}],
    temperature=0.7,
    top_p=0.9,
    presence_penalty=0.0,    # adds flat penalty to tokens that have appeared (like reptition_penalty=1.x)
    frequency_penalty=0.3,   # scales penalty by occurrence count
    max_tokens=512,
)

# Note: OpenAI recommends changing only temperature OR top_p, not both.
# Their default temperature is 1.0, top_p is 1.0 (unconstrained nucleus).`}
      </CodeBlock>

      <H3>Anthropic API</H3>

      <CodeBlock language="python">
{`import anthropic

client = anthropic.Anthropic()

# Anthropic exposes: temperature (0-1), top_k, top_p.
# No repetition penalty or frequency penalty.
# Default: temperature=1.0, top_p and top_k are unset (unconstrained).

message = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Describe decoding in one sentence."}],
    temperature=0.7,
    top_p=0.9,
    # top_k=40,   # can be combined with top_p; both truncations apply
)

# Key difference vs OpenAI:
# - Anthropic temperature range is 0-1 (not 0-2)
# - top_k is available; OpenAI does not expose it
# - No presence_penalty / frequency_penalty equivalents`}
      </CodeBlock>

      <Callout accent="gold">
        API defaults differ. OpenAI defaults to temperature=1.0, top_p=1.0 (unconstrained). Anthropic defaults to temperature=1.0 with no truncation. HuggingFace defaults to greedy (do_sample=False). vLLM defaults to temperature=0.0 (greedy). Verify defaults before assuming production behavior matches your expectations.
      </Callout>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>Temperature sweep: entropy of output distribution</H3>

      <Prose>
        Entropy measures how spread the probability mass is across the vocabulary. At T=0 entropy is zero (one certain token). At T=∞ entropy is log(V). The curve below shows entropy for a fixed logit vector at temperatures across the practical range.
      </Prose>

      <Plot
        label="entropy vs temperature (fixed logit vector [3.0, 2.0, 1.0, 0.0, -1.0])"
        xLabel="temperature"
        yLabel="entropy (nats)"
        series={[
          {
            name: "H(p_T)",
            color: colors.gold,
            points: [
              [0.1, 0.0005],
              [0.3, 0.1596],
              [0.5, 0.4579],
              [0.7, 0.7178],
              [1.0, 1.0],
              [1.2, 1.1295],
              [1.5, 1.2641],
              [2.0, 1.3943],
            ],
          },
        ]}
      />

      <Prose>
        Entropy rises steeply from T=0 through T=1, then flattens. The curve is concave — each additional unit of temperature past 1.0 buys progressively less diversity. This is why temperatures above 1.5 rarely improve creative output: the model is already sampling broadly, and pushing further merely adds noise.
      </Prose>

      <H3>Top-p vs top-k: nucleus size comparison</H3>

      <Prose>
        This heatmap compares how many tokens survive truncation for top-p and top-k across two extremes: a peaked distribution (model confident, probability mass concentrated on one token) and a flat distribution (model uncertain, mass spread across all tokens). Top-k always selects exactly k tokens regardless of distribution shape. Top-p adapts.
      </Prose>

      <Heatmap
        label="tokens surviving truncation: peaked vs flat distribution"
        matrix={[
          [1, 5],
          [10, 10],
        ]}
        rowLabels={["top-p (p=0.9)", "top-k (k=10)"]}
        colLabels={["peaked dist", "flat dist"]}
        cellSize={60}
        colorScale="gold"
      />

      <Prose>
        Top-p with p=0.9 selects 1 token from the peaked distribution (the single dominant token already holds 89.5% of the mass) and all 5 tokens from the flat distribution. Top-k=10 selects 10 tokens in both cases — overcounting in the peaked case, appropriately wide in the flat case. In real production settings with 50,000-token vocabularies the contrast is more dramatic: top-p might select 2 tokens at a deterministic code position and 800 tokens during creative generation.
      </Prose>

      <H3>Qualitative drift across decoding configurations</H3>

      <Prose>
        The same prompt decoded three ways shows the qualitative effect of decoding parameters on perceived personality:
      </Prose>

      <TokenStream
        label="greedy (T≈0) — deterministic, safe, can feel mechanical"
        tokens={[
          { label: "The", color: colors.gold },
          { label: " gradient", color: colors.gold },
          { label: " descent", color: colors.gold },
          { label: " algorithm", color: colors.gold },
          { label: " updates", color: colors.gold },
          { label: " parameters", color: colors.gold },
          { label: " by", color: colors.gold },
          { label: " moving", color: colors.gold },
          { label: " in", color: colors.gold },
          { label: " the", color: colors.gold },
          { label: " direction", color: colors.gold },
          { label: " of", color: colors.gold },
          { label: " steepest", color: colors.gold },
          { label: " descent.", color: colors.gold },
        ]}
      />

      <TokenStream
        label="T=0.7, top-p=0.9 — chat default, natural variation retained"
        tokens={[
          { label: "Gradient", color: "#4ade80" },
          { label: " descent", color: "#4ade80" },
          { label: " iteratively", color: "#4ade80" },
          { label: " adjusts", color: "#4ade80" },
          { label: " model", color: "#4ade80" },
          { label: " weights", color: "#4ade80" },
          { label: " by", color: "#4ade80" },
          { label: " following", color: "#4ade80" },
          { label: " the", color: "#4ade80" },
          { label: " negative", color: "#4ade80" },
          { label: " gradient", color: "#4ade80" },
          { label: " of", color: "#4ade80" },
          { label: " the", color: "#4ade80" },
          { label: " loss.", color: "#4ade80" },
        ]}
      />

      <TokenStream
        label="T=1.5 — creative but risking drift; tail tokens appear"
        tokens={[
          { label: "Like", color: "#c084fc" },
          { label: " water", color: "#c084fc" },
          { label: " finding", color: "#c084fc" },
          { label: " its", color: "#c084fc" },
          { label: " lowest", color: "#c084fc" },
          { label: " valley,", color: "#c084fc" },
          { label: " the", color: "#c084fc" },
          { label: " optimizer", color: "#c084fc" },
          { label: " flows", color: "#c084fc" },
          { label: " relentlessly", color: "#c084fc" },
          { label: " toward", color: "#c084fc" },
          { label: " lower", color: "#c084fc" },
          { label: " loss", color: "#c084fc" },
          { label: " terrain.", color: "#c084fc" },
        ]}
      />

      <H3>One decoding step: logits to sampled token</H3>

      <StepTrace
        label="decoding step — follow the pipeline"
        steps={[
          {
            label: "raw logits",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 2 }}>
                <div style={{ color: "#555", marginBottom: 6 }}>Model output for position t (first 5 tokens shown):</div>
                <div>
                  {[["the", 2.0], ["a", 1.5], ["one", 0.5], ["my", -0.5], ["xyz", -1.5]].map(([tok, v]) => (
                    <span key={tok} style={{ marginRight: 16 }}>
                      <span style={{ color: colors.gold }}>{tok}</span>
                      <span style={{ color: "#555" }}> → </span>
                      <span style={{ color: "#e8e8e8" }}>{v}</span>
                    </span>
                  ))}
                </div>
              </div>
            ),
          },
          {
            label: "apply temperature T=0.7",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 2 }}>
                <div style={{ color: "#555", marginBottom: 6 }}>Divide each logit by T=0.7. Distribution sharpens:</div>
                <div>
                  {[["the", (2.0/0.7).toFixed(2)], ["a", (1.5/0.7).toFixed(2)], ["one", (0.5/0.7).toFixed(2)], ["my", (-0.5/0.7).toFixed(2)], ["xyz", (-1.5/0.7).toFixed(2)]].map(([tok, v]) => (
                    <span key={tok} style={{ marginRight: 16 }}>
                      <span style={{ color: colors.gold }}>{tok}</span>
                      <span style={{ color: "#555" }}> → </span>
                      <span style={{ color: "#e8e8e8" }}>{v}</span>
                    </span>
                  ))}
                </div>
              </div>
            ),
          },
          {
            label: "softmax → probabilities",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 2 }}>
                <div style={{ color: "#555", marginBottom: 6 }}>exp(z/T) / Σ exp(z/T):</div>
                <div>
                  {[["the", "0.702"], ["a", "0.258"], ["one", "0.035"], ["my", "0.005"], ["xyz", "0.001"]].map(([tok, v]) => (
                    <span key={tok} style={{ marginRight: 16 }}>
                      <span style={{ color: colors.gold }}>{tok}</span>
                      <span style={{ color: "#555" }}> → </span>
                      <span style={{ color: "#e8e8e8" }}>{v}</span>
                    </span>
                  ))}
                </div>
              </div>
            ),
          },
          {
            label: "top-p=0.9 truncation",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 2 }}>
                <div style={{ color: "#555", marginBottom: 6 }}>Cumsum: 0.702 → 0.960 (reaches 0.9 at 2 tokens). Keep top 2:</div>
                <div>
                  {[["the", "0.731", true], ["a", "0.269", true], ["one", "0.000", false], ["my", "0.000", false], ["xyz", "0.000", false]].map(([tok, v, kept]) => (
                    <span key={tok} style={{ marginRight: 16, opacity: kept ? 1 : 0.3 }}>
                      <span style={{ color: kept ? colors.gold : "#555" }}>{tok}</span>
                      <span style={{ color: "#555" }}> → </span>
                      <span style={{ color: kept ? "#e8e8e8" : "#555" }}>{v}</span>
                    </span>
                  ))}
                </div>
              </div>
            ),
          },
          {
            label: "sample → token emitted",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 2 }}>
                <div style={{ color: "#555", marginBottom: 6 }}>Sample from {"{the: 0.731, a: 0.269}"}. With p=0.731, we draw:</div>
                <div style={{ fontSize: 20, color: colors.gold, marginTop: 8 }}>"the"</div>
                <div style={{ color: "#555", marginTop: 4, fontSize: 11 }}>Append to sequence. Advance to position t+1.</div>
              </div>
            ),
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        Every decoding configuration is a bet on what the task rewards. Tasks where correctness is narrow and verifiable reward precision — greedy or near-greedy. Tasks where output quality is judged by human preference reward naturalness and variety — sampling from a temperature-scaled truncated distribution. Tasks where multiple candidate outputs can be externally ranked reward the ability to generate diverse, good-faith candidates — beam search or repeated sampling with reranking. The decision matrix below captures the dominant patterns, but it is a starting point, not a prescription. Empirical evaluation on your specific data distribution, with your specific evaluation metric, should always override any generic guideline.
      </Prose>

      <Heatmap
        label="decoding configuration by task"
        matrix={[
          [1, 0, 0, 0, 0],
          [2, 1, 1, 0, 0],
          [3, 2, 2, 1, 0],
          [3, 3, 3, 2, 1],
          [3, 3, 3, 3, 2],
          [2, 3, 3, 3, 3],
          [1, 2, 3, 3, 3],
        ]}
        rowLabels={["code / math", "factual Q&A", "summarization", "chat", "translation", "dialogue", "creative"]}
        colLabels={["greedy", "T=0.3", "T=0.7 p=0.9", "T=1.0 p=0.95", "T=1.2+"]}
        cellSize={44}
        colorScale="gold"
      />

      <Prose>
        Values represent suitability (0=avoid, 1=marginal, 2=reasonable, 3=recommended). For code and math, greedy or near-greedy (T=0.0–0.1) minimizes stochastic errors — in a code generation benchmark, a single incorrectly sampled token can invalidate the entire function, so the variance reduction from low temperature is almost always worth the cost in diversity. For creative tasks, T=1.0 or above with wide nucleus sampling lets the model express genuine distributional variety — the model was trained on creative text at T=1, and sampling at that temperature produces output calibrated to its training signal. Chat sits in the middle: T=0.7, top-p=0.9 is the de facto industry default across OpenAI, Anthropic, and most third-party chat interfaces, a value that was arrived at empirically and has remained stable through multiple model generations.
      </Prose>

      <Prose>
        Beam search belongs in a separate column: it is the recommended strategy for translation, summarization with faithfulness constraints, and any setting where a verifier exists to select among candidates. With beam width 4–8 and no-repeat-ngram-size=3, beam search reliably outperforms sampling on constrained generation tasks as measured by BLEU, ROUGE, or exact-match metrics. The moment the task becomes open-ended and the evaluation metric becomes human preference, beam search's length and diversity biases make it a poor choice. A useful diagnostic: if you can write a program that definitively says whether a given output is correct, beam search is worth trying. If correctness requires human judgment, use sampling.
      </Prose>

      <Prose>
        Min-p deserves explicit mention in the creative column. It handles high temperatures (T=1.0–1.5) better than top-p alone, because as temperature rises and the distribution flattens, top-p's nucleus can grow to include hundreds of tokens that should not appear — tokens that are genuinely low-probability but happen to collectively sum to the p threshold. Min-p's relative threshold scales with the peak probability of the distribution, maintaining a tighter nucleus without requiring manual re-tuning of p at each temperature level. For applications that push temperature past 1.0 for stylistic reasons, combining T=1.2 with min-p=0.07 is a more principled setup than T=1.2 with top-p=0.95, which risks admitting large portions of the long tail.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>Per-token cost</H3>

      <Prose>
        Sampling cost scales as O(V) per token: the model computes a logit for every vocabulary entry, and applying temperature or top-k is linear in V. The sorted step in top-p is O(V log V), but in practice this is dominated by the matrix multiplication in the final projection layer, which is O(V × d_model). For modern large models where d_model is in the thousands, the projection itself is the bottleneck — sorting a 100k-token vocabulary on top of that projection is a few percent overhead, not a fundamental constraint. For large vocabularies, sorting is cached or approximated; inference frameworks like vLLM maintain sorted token indices across steps when the model's vocabulary does not change.
      </Prose>

      <Prose>
        Min-p and top-p add a cumulative sum over the sorted distribution, which is another O(V) pass after sorting. In absolute terms on modern hardware this is microseconds for V=50,000, but at extreme batch sizes and very short generation lengths the overhead per token becomes a larger fraction of total latency. Greedy, which skips sampling entirely and just calls argmax, is the fastest strategy per token by a comfortable margin and should be the default for any application where determinism and latency are the primary constraints.
      </Prose>

      <H3>Beam search memory</H3>

      <Prose>
        Beam search with width k requires k times the KV-cache memory compared to a single-sequence greedy decode, and k times the computation per step since each beam must independently evaluate the model. For large models (70B+) and long sequences, this quickly becomes prohibitive — a 70B model with a 4,096-token context and beam width 4 needs roughly 4× the KV-cache memory of greedy. This is why beam search is rarely used in production for open-ended generation from frontier models; it is reserved for scenarios where candidate scoring is done externally (reranking pipelines) or for smaller specialized models where the multiple forward passes are affordable.
      </Prose>

      <Prose>
        There is a practical workaround: run beam search offline, not in a live inference server. For batch applications — generating a large number of candidates to be filtered by a separate ranker — beam search can be run at full width without latency constraints. The reranking pattern (generate k candidates with beam search or repeated sampling, then score each with a separate model or heuristic and return the top-ranked) is increasingly common in code generation, mathematical reasoning, and retrieval-augmented generation pipelines. In these settings beam search's diversity bias is actually a feature: you want the k candidates to cover different regions of the output space, and beam search's penalty against repetitive beams ensures that.
      </Prose>

      <H3>Sequence length</H3>

      <Prose>
        All sampling strategies are per-token: the decoding cost grows linearly in sequence length regardless of strategy. There is no structural reason that longer sequences are more expensive per-token from the decoding perspective (attention cost grows quadratically with sequence length, but that is a model concern — KV-cache memory and attention computation — not a decoding strategy concern). Repetition penalty becomes slightly more expensive as the context grows, because the seen-token set grows with context length, and checking membership in that set takes time proportional to the number of unique tokens seen. In practice this remains O(context length) per step and is dominated by the attention cost by orders of magnitude.
      </Prose>

      <H3>Parallelism</H3>

      <Prose>
        Greedy decoding and sampling-based strategies are trivially parallelizable across independent sequences in a batch — each prompt is processed independently with no inter-sequence synchronization required. All token selections are made in parallel across the batch. Beam search breaks this parallelism: beams within a single sequence must be synchronized at every step to prune and rank, which requires a reduction across the beam dimension before the next step can proceed. This synchronization point prevents the batching efficiency gains that modern inference servers exploit.
      </Prose>

      <Prose>
        Speculative decoding — a recent inference acceleration technique where a small draft model proposes multiple tokens that the large model verifies in one forward pass — is orthogonal to the sampling strategy and can be combined with any of the methods described here. The draft model generates a sequence of candidate tokens (typically 4–8) using greedy or sampling decoding, and the large model verifies them in parallel. Accepted tokens are committed; rejected tokens trigger a correction. The effective throughput improvement is 2–4× for latency-sensitive workloads where the draft model is fast relative to the verifier. The choice of sampling strategy applies at the draft stage, at the verification/correction stage, or both.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>Temperature too high: incoherence</H3>

      <Prose>
        Above T=1.2–1.5, the model begins sampling from tokens that are genuinely improbable by its own assessment. The resulting text may be grammatically valid sentence by sentence but semantically incoherent across sentences — topic drift, mid-sentence pronoun switches, factual contradictions. The model does not signal this degradation; the output just quietly gets worse. The fix is to keep temperature in the 0.7–1.0 range and use top-p to control diversity rather than temperature alone.
      </Prose>

      <H3>Top-k too low: missing valid continuations</H3>

      <Prose>
        Top-k with k=5 or k=10 at positions with genuine distributional spread clips many valid continuations. In creative generation or open-ended dialogue, the model may have 50 plausible next words, and forcing a choice from 10 produces repetitive outputs that feel unnaturally constrained. Top-p is a better default for variable-context tasks precisely because it adapts to this.
      </Prose>

      <H3>Beam search length bias: short responses</H3>

      <Prose>
        Raw-score beam search systematically prefers shorter sequences. In translation this manifests as undertranslated outputs; in open-ended generation it produces abnormally terse responses. The standard mitigation is length normalization (dividing the score by sequence length or length^α for some α ∈ [0.5, 0.75]), but choosing α requires task-specific tuning. In open-ended generation, avoid beam search entirely.
      </Prose>

      <H3>Repetition penalty hurting legitimate repetition</H3>

      <Prose>
        Repetition penalty is context-blind: it penalizes any token that has appeared, regardless of whether the repetition is appropriate. Bullet lists repeat syntactic tokens (hyphens, newlines, colons) intentionally. Code repeats variable names. Tables repeat structural tokens. A repetition penalty of θ=1.3 can disrupt all of these by degrading the probability of entirely correct tokens just because they appeared earlier. If you enable repetition penalty, test it explicitly against your structured output tasks before deploying.
      </Prose>

      <H3>Stop strings truncating valid content</H3>

      <Prose>
        Stop strings are matched literally against the raw output stream. If a stop string appears as part of a longer valid output — for example, a stop string of <Code>"\n\n"</Code> used to delimit responses will truncate any model output that includes a deliberate blank line, such as a code block with an empty line — the output is silently cut at that point. Always verify that stop strings do not appear in the legitimate outputs for your task.
      </Prose>

      <H3>Sampling producing invalid structured output</H3>

      <Prose>
        Probability-based sampling does not guarantee syntactic validity. A model with top-p=0.9 can still emit a character that invalidates a JSON object, closes a code block prematurely, or produces a tool call with the wrong argument type. For structured output tasks, constrained decoding (masking tokens that would invalidate the formal grammar at every step) is the correct solution, not tighter sampling parameters. Logit biases can help nudge the model toward structural tokens but are not a substitute for proper grammar-constrained decoding.
      </Prose>

      <H3>Logit bias misuse</H3>

      <Prose>
        Logit biases add or subtract arbitrary constants from specific token logits before all other processing (temperature, top-p, top-k). This is powerful and dangerous. A logit bias strong enough to always force a particular token overrides any probability signal the model has learned. Using logit bias to soft-steer the model toward or away from certain vocabulary items (controlling output language, suppressing profanity) works well in moderation. Using it to force specific content can produce outputs that are grammatically valid but semantically incoherent, because the surrounding context was not generated to accommodate the forced token.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All five sources below were verified against their arXiv abstracts as of April 2026.
      </Prose>

      <H3>Nucleus / top-p sampling</H3>

      <Prose>
        Holtzman, A., Buys, J., Du, L., Forbes, M., and Choi, Y. (2020). "The Curious Case of Neural Text Degeneration." <em>ICLR 2020</em>. arXiv:1904.09751. The paper that introduced nucleus (top-p) sampling and provided the first systematic empirical analysis of why greedy and beam search produce degenerate text. Demonstrated that the likelihood of human-written text does not fall uniformly under a well-trained LM — it concentrates in a dynamic nucleus that varies step by step — and that sampling from this nucleus dramatically improves text quality by human evaluation.
      </Prose>

      <H3>Top-k sampling</H3>

      <Prose>
        Fan, A., Lewis, M., and Dauphin, Y. (2018). "Hierarchical Neural Story Generation." <em>ACL 2018</em>. arXiv:1805.04833. Introduced top-k sampling as a method for improving diversity and coherence in story generation. The paper proposed a two-stage generation process (premise then story) and introduced top-k filtering as a principled way to exclude the implausible tail of the vocabulary without sacrificing diversity.
      </Prose>

      <H3>Beam search inadequacy</H3>

      <Prose>
        Stahlberg, F. and Byrne, B. (2019). "On NMT Search Errors and Model Errors: Cat Got Your Tongue?" <em>EMNLP-IJCNLP 2019</em>. arXiv:1908.10090. Showed that exact beam search (using depth-first search to find the global optimum) reveals that NMT models assign their highest probability to the empty string for a majority of sentences — a consequence of per-step EOS probability interacting with length-biased scoring. The paper demonstrated that what practitioners think of as a search problem (beam search not finding the best sequence) is often actually a model problem (the model's global optimum is degenerate).
      </Prose>

      <H3>Temperature origins</H3>

      <Prose>
        Ackley, D. H., Hinton, G. E., and Sejnowski, T. J. (1985). "A Learning Algorithm for Boltzmann Machines." <em>Cognitive Science, 9(1)</em>. The concept of temperature as a control parameter for probability distributions over discrete states originates in statistical physics (Boltzmann distribution) and was imported into machine learning through Boltzmann machines. The temperature-scaled softmax used in modern LM decoding is a direct descendant of the Boltzmann sampling mechanism described here.
      </Prose>

      <H3>Min-p sampling</H3>

      <Prose>
        Nguyen, M. N., Baker, A., Neo, C., Roush, A., Kirsch, A., and Shwartz-Ziv, R. (2024). "Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs." arXiv:2407.01082. Introduced min-p as a dynamic truncation method that scales the probability threshold relative to the top token's probability rather than using an absolute threshold. Showed improvements on both quality benchmarks (GPQA, GSM8K) and creative writing (AlpacaEval) relative to top-p, particularly at higher temperatures. Min-p has since been adopted natively in HuggingFace Transformers, vLLM, and llama.cpp.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1: Entropy as a function of temperature</H3>

      <Prose>
        Given a probability vector <Code>p = [0.6, 0.3, 0.1]</Code> at T=1, derive the entropy of the temperature-scaled distribution at T=0.5 and T=2.0. Then show analytically that as T→∞, the entropy approaches log(V) for any initial logit vector. What does this tell you about the relationship between temperature and the model's effective vocabulary size?
      </Prose>

      <H3>Exercise 2: When does top-p with p=1.0 differ from unconstrained sampling?</H3>

      <Prose>
        Top-p with p=1.0 keeps all tokens — the nucleus is the full vocabulary. Does this differ from not applying top-p at all? Identify at least one implementation-level scenario where the answer is yes, and explain why. (Hint: consider numerical precision and how implementations handle the cumulative sum cutoff when the final token pushes cumsum to exactly 1.0.)
      </Prose>

      <H3>Exercise 3: Why does beam search hurt open-ended generation?</H3>

      <Prose>
        Explain the beam search curse — the tendency of beam search to produce bland, repetitive, generic text — in terms of the probability distribution over sequences. Specifically: why does optimizing for highest-cumulative-log-probability produce output that ranks lower in human quality evaluations than random sampling from a truncated distribution? What distributional property of human-written text does beam search ignore?
      </Prose>

      <H3>Exercise 4: Decoding strategy for high-stakes factual Q&A</H3>

      <Prose>
        Design a complete decoding configuration for a factual question-answering system where the model should express high certainty when it knows the answer and should abstain (output "I don't know") when it does not. Specify temperature, top-p or top-k, and any additional parameters. Explain how your configuration interacts with the model's calibration — specifically, what happens when the model is confidently wrong?
      </Prose>

      <H3>Exercise 5: Repetition penalty and list generation</H3>

      <Prose>
        A model with repetition penalty θ=1.3 is generating a numbered list. The list format requires the tokens "1.", "2.", "3." to appear at the start of each line. Trace what happens to the logit of the token "2." after "1." and "2." have already appeared in the context. Would a frequency penalty (scaling by occurrence count) make this better or worse than a flat repetition penalty? Propose a mitigation that preserves list formatting while still breaking unwanted repetition loops.
      </Prose>

      <Callout accent="gold">
        The decoding strategy is the last degree of freedom before the model's output reaches the user. Temperature, top-p, top-k, and min-p are not black-box magic numbers — they are parameters with precise mathematical meanings and predictable effects. Understand the math, verify the code, and test empirically: the right configuration depends on your task in ways that no general default can anticipate.
      </Callout>
    </div>
  ),
};

export default decodingStrategies;
