import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Plot } from "../../components/viz";
import { colors } from "../../styles";

const dynamicTokenization = {
  title: "Dynamic Tokenization (ADAT, BoundlessBPE, LiteToken)",
  readTime: "24 min",
  content: () => (
    <div>
      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Every tokenizer this section has covered — BPE, WordPiece, Unigram, byte-level BPE,
        ViT patches, VQ-VAE codebooks, RVQ for audio — shares one property: its vocabulary is
        decided once, at training time, and then frozen for the life of the model. That is
        not a detail. It is the single assumption that every piece of serving infrastructure
        downstream of the tokenizer depends on. Embedding tables are allocated at a fixed
        size. KV caches index by token ID. API billing meters bytes per token at a rate
        fixed at model release. The vocabulary is load-bearing in a way that a schema is
        load-bearing for a database: you can change it, but you have to reindex the world.
      </Prose>

      <Prose>
        Dynamic tokenization is the name for a small cluster of research directions that
        push on whether the vocabulary has to be that static. The motivation is a failure
        mode easy to demonstrate. Consider a codebase that imports a new library published
        last week — <Code>from pyrelic.retrieval import BM42Index</Code>. A production
        tokenizer trained in 2024 has never seen <Code>pyrelic</Code> or <Code>BM42Index</Code>.
        Its merge table knows nothing about them. When the line reaches the model, it will
        come apart into a dozen pieces —
        <Code>['p', 'yre', 'lic', '.', 'retrieval', ' import', ' BM', '42', 'Index']</Code>
        is a plausible segmentation, and the exact split doesn't matter; the point is that
        a name the programmer uses as an atomic identifier has been shredded into character-
        class fragments. The model has to learn, at inference time, that these fragments
        refer to one thing. Every repeated occurrence in the same file pays the same tax.
      </Prose>

      <Prose>
        Generalize that failure: any static tokenizer is a bet, at training time, about what
        text will look like. That bet can only age. Domain drift (new jargon, new libraries,
        new product names, emerging slang, scientific vocabulary in fields that did not
        exist during training) turns previously-compressive tokens into noise and turns
        previously-rare bigrams into common ones. There is no feedback loop. The tokenizer
        cannot update. The model's vocabulary literally could not be older than it is.
      </Prose>

      <Prose>
        Three lines of recent work push on this. Meta's Byte-Latent Transformer (Pagnoni et
        al., arXiv:2412.09871) asks whether the concept of a fixed vocabulary is necessary
        at all — it operates directly on bytes and learns dynamically-sized patches whose
        boundaries depend on the input. Kensho's BoundlessBPE (Schmidt et al., COLM 2025,
        arXiv:2504.00178) relaxes the pre-tokenization constraint that forces BPE merges to
        respect whitespace boundaries, producing a vocabulary of "superwords" that span
        pre-tokens. A 2024 NeurIPS paper on adaptive LLM tokenization (ADAT, Zheng et al.)
        refines the vocabulary itself using downstream-model feedback during training.
        LiteToken (arXiv:2602.04706) analyzes intermediate-merge residues in existing
        tokenizers and prunes them. None of these fit the naive "tokenizer that rebuilds
        itself at inference" picture perfectly — and that discrepancy is itself the point.
      </Prose>

      <Callout accent="gold">
        Honest framing. There is no production LLM as of this writing that uses a fully
        dynamic, inference-time-mutable tokenizer. Dynamic tokenization is a frontier
        topic. The papers exist, the experiments work, the engineering path to production
        is not yet walked.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Three angles are worth separating cleanly, because they are different ideas and
        the literature muddles them. Each corresponds to a different thing you might mean
        by "dynamic."
      </Prose>

      <H3>Adaptive boundaries</H3>

      <Prose>
        The unit of tokenization can be a function of the input rather than a lookup. A
        learnable boundary predictor — a small network that reads characters or bytes and
        outputs, for each gap between two bytes, a probability of inserting a token
        boundary there — lets segmentation condition on context. The same three bytes
        might form one token in prose and three tokens in code, depending on what
        surrounds them. Charformer (Tay et al. 2021, arXiv:2106.12672) trains such a
        predictor end-to-end with the downstream language-modeling loss, using a soft
        gradient-based block scoring module (GBST). Byte-Latent Transformer extends the
        idea: patches are segmented based on the entropy of the next byte, so the
        tokenizer spends more granularity on information-dense regions and coarsens
        through predictable runs. MambaByte (Wang et al. 2024, arXiv:2401.13660) takes
        the dual position — operate on raw bytes, let the state-space model's gating
        dynamics do the work a tokenizer would otherwise do.
      </Prose>

      <H3>Vocabulary growth</H3>

      <Prose>
        The vocabulary itself could be mutable. Most concretely: an inference-time
        mechanism that watches a stream of tokens, notices that a particular adjacent
        pair repeats enough times to be worth a single token, and adds it to the vocab.
        BoundlessBPE does not do this at inference — it does it at training, by relaxing
        the pre-tokenization barrier — but it points at the same idea: BPE's merge list
        is artificially constrained by a one-time whitespace pre-split, and lifting that
        constraint produces measurably better compression (at least 19.7% more bytes per
        token in their reported results). The conceptual extrapolation, which no mainstream
        system ships, is to let the merge table grow at inference when the input stream
        justifies it.
      </Prose>

      <H3>Compression-aware, budget-bounded</H3>

      <Prose>
        Tokenization can be framed as rate-distortion under a fixed budget. Given a cap of
        <Code>k</Code> tokens per unit text, which segmentation best preserves downstream
        task performance? Schmidt et al. 2024 ("Tokenization Is More Than Compression,"
        arXiv:2402.18376) explored this directly with PathPiece — a tokenizer that
        segments into the minimum number of tokens for a given vocabulary — and reported
        the non-obvious finding that fewer tokens is not always better. The more
        interesting frame is content-adaptive: a scientific document full of molecular
        notation should spend its token budget differently than an email. LiteToken
        (arXiv:2602.04706) is related but narrower — it removes "residue" tokens that
        appear during BPE training but are rarely emitted at inference, cutting vocab
        size by roughly 10% without loss.
      </Prose>

      <Prose>
        All three angles share one structural property. They move some decision that
        static tokenization makes at training time to a later point — either to training
        time but conditioned on the downstream model, or to inference time conditioned on
        the input. Every gain compounds with the scale of that deferral, and every cost
        is the engineering problem of making the new decision fast and deterministic.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <Prose>
        For learnable boundary prediction, the question is how gradients flow through a
        discrete decision. Let <Code>x = (x₁, ..., xₙ)</Code> be a byte sequence and
        <Code>b = (b₁, ..., bₙ₋₁)</Code> be the boundary indicators (<Code>bᵢ = 1</Code>
        if there is a token boundary between <Code>xᵢ</Code> and <Code>xᵢ₊₁</Code>).
        A boundary scorer outputs logits <Code>ℓᵢ</Code>, and we want a soft relaxation
        that is differentiable. Charformer's GBST uses a convolution across candidate
        block widths and takes a softmax over widths per position.
      </Prose>

      <MathBlock>
        {"p(b_i = 1 \\mid x) = \\sigma(\\ell_i), \\qquad \\ell_i = f_\\theta(x_{i-c:i+c})"}
      </MathBlock>

      <Prose>
        Rather than committing to a hard boundary, the representation at each position is a
        soft mixture over possible block widths. Let <Code>h⁽ᵂ⁾</Code> denote the pooled
        representation over width <Code>w</Code> centered at position <Code>i</Code>, and
        let <Code>αᵢ⁽ᵂ⁾</Code> be the softmax-weighted attention over widths.
      </Prose>

      <MathBlock>
        {"\\tilde{h}_i = \\sum_{w \\in W} \\alpha_i^{(w)} h_i^{(w)}, \\qquad \\alpha_i^{(w)} = \\mathrm{softmax}_w(\\mathrm{score}(h_i^{(w)}))"}
      </MathBlock>

      <Prose>
        Gradients on the downstream loss flow through this soft mixture back to
        <Code>f_θ</Code>. At inference, one can either keep the soft mixture (expensive)
        or argmax the width (cheap, but loses the gradient that justified the design).
        BLT replaces this with an entropy-based boundary criterion: insert a boundary
        when the next-byte entropy under a small byte-level language model crosses a
        threshold. That criterion is not itself learned, but the patch-encoder that
        consumes the resulting bytes is.
      </Prose>

      <Prose>
        For inference-time vocabulary extension, the math is simpler but worth making
        explicit. Suppose the current vocabulary <Code>V</Code> has probability
        distribution <Code>p</Code> over tokens, and a pair <Code>(a, b)</Code> with
        joint frequency <Code>f(a, b)</Code> in a sliding window of length <Code>L</Code>
        has been observed. Adding <Code>ab</Code> as a new vocab entry increases
        probability mass on that specific composition.
      </Prose>

      <MathBlock>
        {"\\Delta \\log P(\\text{window}) = f(a, b) \\cdot \\left[\\log p(ab) - \\log p(a) - \\log p(b)\\right]"}
      </MathBlock>

      <Prose>
        The bracket is the pointwise mutual information of the pair under the unigram
        model. It is positive exactly when <Code>(a, b)</Code> co-occurs more often than
        independence would predict — which is the condition under which adding the merge
        is a net win for compression. The trigger rule used by a typical pedagogical
        implementation (cross a count threshold inside a sliding window) is a cheap
        approximation of that PMI test.
      </Prose>

      <Prose>
        For the rate-distortion framing, let <Code>R</Code> be the number of tokens
        (rate) and <Code>D</Code> be a downstream-task loss (distortion). The
        Pareto frontier <Code>D*(R)</Code> is the minimum distortion achievable at rate
        <Code>R</Code>. A static tokenizer fixes one operating point; a content-adaptive
        tokenizer chooses a different point per input.
      </Prose>

      <MathBlock>
        {"D^*(R) = \\min_{T : |T(x)| \\le R} \\; \\mathbb{E}\\bigl[\\mathcal{L}(f_\\theta(T(x)), y)\\bigr]"}
      </MathBlock>

      <Prose>
        No tokenizer in practice solves this optimization directly — the expectation is
        intractable and the minimum over segmentations is combinatorial. But the frame
        clarifies what experiments like PathPiece's actually tested: they minimized
        <Code>R</Code> at a fixed vocabulary and measured <Code>D</Code>, and found the
        relationship is not monotone.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        Because these methods are research-stage and lack canonical open implementations,
        the code here is deliberately simplified. Each section builds a pedagogical
        analog of one of the three ideas in section 2 — not the full paper, but the
        core mechanism stripped to fit on one screen. Every implementation was run; the
        outputs embedded as comments are the actual outputs.
      </Prose>

      <Callout accent="gold">
        These are pedagogical simplifications. The real papers go further — they train
        end-to-end, they use larger models for the scoring network, they benchmark on
        downstream tasks — but they use the same core ideas shown here.
      </Callout>

      <H3>4a. Learnable segmentation (Charformer / BLT analog)</H3>

      <Prose>
        The simplest learnable segmenter reads pairs of adjacent characters and outputs
        a boundary score. A real system would train this scorer end-to-end; here we use
        two hand-designed scorers so the mechanism is visible. The first, <Code>
        transition_scorer</Code>, places a boundary whenever the character class changes
        (letter → digit, letter → punct, and so on) — a stripped-down heuristic analog of
        what a learned scorer tends to converge on for code and scientific text. The
        second, <Code>frequency_scorer</Code>, uses a tiny bigram frequency table and
        breaks on rare bigrams — an analog of BLT's entropy-based boundary insertion.
      </Prose>

      <CodeBlock language="python">
{`from collections import Counter

def char_class(c):
    if c.isalpha() and c.islower(): return "lower"
    if c.isalpha() and c.isupper(): return "upper"
    if c.isdigit():                 return "digit"
    if c.isspace():                 return "space"
    return "punct"

def transition_scorer(left, right):
    """Break on character-class transitions."""
    return 1.0 if char_class(left) != char_class(right) else 0.0

def frequency_scorer(pair_counts, threshold):
    """Break on rare bigrams (entropy-like heuristic)."""
    def score(left, right):
        return 1.0 if pair_counts[(left, right)] < threshold else 0.0
    return score

def segment(text, scorer):
    if not text: return []
    out, cur = [], text[0]
    for i in range(1, len(text)):
        if scorer(text[i - 1], text[i]) >= 0.5:
            out.append(cur); cur = text[i]
        else:
            cur += text[i]
    out.append(cur)
    return out

samples = [
    "the quick brown fox",
    "compute_attention_scores(x, 42)",
    "H2SO4 reacts with NaOH",
    "TX-4891",
]
for s in samples:
    print(f"{s!r:>40}  ->  {segment(s, transition_scorer)}")

# Actual output (verified by running this code):
#              'the quick brown fox' -> ['the', ' ', 'quick', ' ', 'brown', ' ', 'fox']
#  'compute_attention_scores(x, 42)' -> ['compute', '_', 'attention', '_', 'scores',
#                                        '(', 'x', ',', ' ', '42', ')']
#           'H2SO4 reacts with NaOH' -> ['H', '2', 'SO', '4', ' ', 'reacts', ' ',
#                                        'with', ' ', 'N', 'a', 'OH']
#                          'TX-4891' -> ['TX', '-', '4891']`}
      </CodeBlock>

      <Prose>
        Notice what changes between samples. The same scorer produces seven tokens on the
        prose sample, eleven on the code sample, and splits the chemical formula at
        digit–letter boundaries. No lookup table is doing this. The segmentation is a
        function of the input, recomputed from scratch each time. That is the property
        Charformer and BLT have in common, stripped of everything else. A learned scorer
        would replace the hand-written <Code>char_class</Code> logic with a trained
        network, but the flow is identical.
      </Prose>

      <Prose>
        Swap in the frequency-based scorer and the segmentation changes again — this
        time aggressively, because a bigram table learned from nine words of training
        data thinks almost every byte-pair is rare.
      </Prose>

      <CodeBlock language="python">
{`corpus = "the quick brown fox jumps over the lazy dog the the the"
pair_counts = Counter()
for i in range(len(corpus) - 1):
    pair_counts[(corpus[i], corpus[i + 1])] += 1

scorer = frequency_scorer(pair_counts, threshold=2)
print(segment("the quick brown fox", scorer))
print(segment("TX-4891", scorer))

# Actual output:
# ['the ', 'q', 'u', 'i', 'c', 'k', ' ', 'b', 'r', 'o', 'w', 'n', ' ',
#  'f', 'o', 'x']
# ['T', 'X', '-', '4', '8', '9', '1']`}
      </CodeBlock>

      <Prose>
        The word <Code>the</Code> survives as a unit because <Code>t-h</Code> and
        <Code>h-e</Code> and <Code>e-(space)</Code> all appear at least twice in the
        corpus — they pass the threshold. Everything else fragments. In production, the
        scorer would be a neural net trained on billions of tokens rather than nine
        words, and the result would land between these two extremes: some parts of the
        input stay coarse, the novel parts fragment finely.
      </Prose>

      <H3>4b. Inference-time vocabulary growth (BoundlessBPE-style conceptual)</H3>

      <Prose>
        BoundlessBPE proper does its merging at training time and emits a fixed
        vocabulary. The conceptual extrapolation — letting the merge table grow at
        inference — is useful to build because it exposes the engineering problems
        (caching, determinism) directly. The implementation below starts with a simple
        base vocabulary of single ASCII characters, greedy-encodes each incoming text
        chunk, tracks pair frequencies inside a sliding window, and promotes a pair to
        a vocab entry once it crosses a count threshold.
      </Prose>

      <CodeBlock language="python">
{`from collections import Counter, deque

BASE_VOCAB = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789_-. ,:;()[]{}/"
)

def greedy_encode(text, vocab):
    """Longest-match encode against the current vocab."""
    i, out = 0, []
    max_len = max(len(t) for t in vocab)
    while i < len(text):
        matched = None
        for L in range(min(max_len, len(text) - i), 0, -1):
            if text[i:i + L] in vocab:
                matched = text[i:i + L]; break
        if matched is None:
            out.append(text[i]); i += 1
        else:
            out.append(matched); i += len(matched)
    return out

class GrowingTokenizer:
    def __init__(self, base_vocab, window=400, threshold=3):
        self.vocab = set(base_vocab)
        self.window, self.threshold = window, threshold
        self.history = deque()
        self.pair_counts = Counter()
        self.added = []

    def _observe(self, pair):
        self.history.append(pair)
        self.pair_counts[pair] += 1
        if len(self.history) > self.window:
            old = self.history.popleft()
            self.pair_counts[old] -= 1
            if self.pair_counts[old] <= 0:
                del self.pair_counts[old]

    def process(self, text):
        tokens = greedy_encode(text, self.vocab)
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            self._observe(pair)
            if (self.pair_counts[pair] >= self.threshold
                    and (pair[0] + pair[1]) not in self.vocab):
                self.vocab.add(pair[0] + pair[1])
                self.added.append(pair[0] + pair[1])
        return greedy_encode(text, self.vocab)`}
      </CodeBlock>

      <Prose>
        Run it on a stream where the string <Code>TX-4891</Code> recurs across six
        chunks of a long document. The static baseline re-fragments it every time; the
        growing tokenizer builds up merges for <Code>TX</Code>, <Code>-4</Code>,
        <Code>891</Code>, and eventually <Code>TX-4</Code> and <Code>-489</Code>, and
        the sequence length drops.
      </Prose>

      <CodeBlock language="python">
{`stream = [
    "order TX-4891 shipped. ",
    "status of TX-4891 is pending. ",
    "TX-4891 was delayed. ",
    "rebill TX-4891 next week. ",
    "TX-4891 TX-4891 TX-4891 ",
    "final: TX-4891 closed.",
]
tok = GrowingTokenizer(BASE_VOCAB, window=400, threshold=3)

for i, chunk in enumerate(stream, 1):
    s = len(greedy_encode(chunk, BASE_VOCAB))
    d = len(tok.process(chunk))
    print(f"chunk {i}: static={s:>3}  dynamic={d:>3}  vocab=+{len(tok.added)}")

# Actual output (verified by running this code):
# chunk 1: static= 23  dynamic= 23  vocab=+0
# chunk 2: static= 30  dynamic= 30  vocab=+0
# chunk 3: static= 21  dynamic= 15  vocab=+9
# chunk 4: static= 26  dynamic= 21  vocab=+9
# chunk 5: static= 24  dynamic=  6  vocab=+12
# chunk 6: static= 22  dynamic= 15  vocab=+14
#
# totals:  static=146  dynamic=110  compression=1.33x  merges added=14
# final additions: ['TX','X-','-4','48','89','91','1 ','s ','. ',
#                   'TX-4','-489','891 ','ed','d.']`}
      </CodeBlock>

      <Prose>
        The dynamic tokenizer compresses the stream to 75% of its static length by the
        end, with the savings concentrated in chunk 5 where <Code>TX-4891</Code> repeats
        three times back-to-back and gets absorbed into a two-token encoding instead of
        seven. The new merges it picks up are telling: it discovers not just
        <Code>TX-4</Code> and <Code>891</Code> but also <Code>{"'s '"}</Code>,
        <Code>{"'. '"}</Code>, <Code>ed</Code> — unit-less pairs that happen to recur
        often enough in English to be worth coalescing. A real system would be smarter
        about PMI versus raw count, would cap vocab growth, and would have to solve the
        problem that chunk 3's token IDs are not the same as chunk 4's token IDs because
        the vocab grew in between. That problem is section 9.
      </Prose>

      <H3>4c. Rate-distortion-aware tokenization</H3>

      <Prose>
        The third implementation is the most directly new. Given a fixed token budget
        <Code>k</Code> per input, find a segmentation into exactly <Code>k</Code> pieces
        that preserves information best. The proxy we use for "information" is the
        unigram surprisal of each character — more-common characters (like a run of
        <Code>a</Code>) carry few bits, rare characters (a parenthesis or a digit in a
        letter-heavy string) carry many. A static strategy cuts the string into equal
        character widths, which wastes tokens on boring runs. An adaptive strategy
        solves for cuts that equalize bits-per-token.
      </Prose>

      <CodeBlock language="python">
{`import math
from collections import Counter

def char_logp(text):
    c = Counter(text); n = sum(c.values())
    return {ch: math.log2(n / v) for ch, v in c.items()}

def surprisal(seg, lp):
    return sum(lp.get(c, 0.0) for c in seg)

def static_segmentation(text, k):
    n = len(text); cuts = [round(i * n / k) for i in range(k + 1)]
    return [text[cuts[i]:cuts[i + 1]] for i in range(k)]

def adaptive_segmentation(text, k, lp):
    """DP minimizing the variance of bits across k segments."""
    n = len(text)
    ps = [0.0]
    for c in text:
        ps.append(ps[-1] + lp.get(c, 0.0))
    target = ps[n] / k

    INF = float("inf")
    dp = [[INF] * (k + 1) for _ in range(n + 1)]
    back = [[-1] * (k + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(1, n + 1):
        for m in range(1, min(k, i) + 1):
            for j in range(m - 1, i):
                bits = ps[i] - ps[j]
                c = dp[j][m - 1] + (bits - target) ** 2
                if c < dp[i][m]:
                    dp[i][m] = c; back[i][m] = j
    cuts = [n]; i, m = n, k
    while m > 0:
        j = back[i][m]; cuts.append(j); i, m = j, m - 1
    cuts.reverse()
    return [text[cuts[i]:cuts[i + 1]] for i in range(len(cuts) - 1)]

text = "aaaaaaaaaaaaTX-4891bbbbbbbbbbC(=O)OHccccccccccccc"
lp = char_logp(text)
for name, segs in [("static", static_segmentation(text, 6)),
                   ("adaptive", adaptive_segmentation(text, 6, lp))]:
    bits = [surprisal(s, lp) for s in segs]
    m = sum(bits) / len(bits)
    var = sum((b - m) ** 2 for b in bits) / len(bits)
    print(f"{name}: segments={segs}  var={var:.2f}")

# Actual output (verified by running this code):
# static:   ['aaaaaaaa','aaaaTX-4','891bbbbb','bbbbbC(=O',')OHccccc','cccccccc']
#           var=45.89
# adaptive: ['aaaaaaaaaaaa','TX-4','891bbbb','bbbbbbC(','=O)OH','ccccccccccccc']
#           var=1.47`}
      </CodeBlock>

      <Prose>
        The variance of bits-per-segment drops by 31x under the adaptive strategy. The
        boring <Code>aaa...</Code> prefix gets a single 12-character token because each
        of those characters carries almost no information. The high-surprisal region
        <Code>=O)OH</Code> gets a dedicated 5-character token because squeezing it into
        a larger one would have put too many bits into a single slot. Both strategies
        produce exactly six tokens. The difference is entirely in where the cuts go.
      </Prose>

      <Prose>
        This is what "rate-distortion-aware" means in practice. Not free-lunch compression
        — both outputs have the same token count — but better use of that budget for
        downstream prediction. A language model consuming the adaptive output has the
        high-information regions cleanly delimited; a model consuming the static output
        has to learn to handle tokens that straddle meaningful boundaries like
        <Code>bbbbbC(=O</Code>, where the relevant structure is cut in two.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        Be honest: no major production LLM as of this writing uses a fully-dynamic
        tokenizer in the sense implied by section 4. The closest production-scale
        examples are partial, and worth naming precisely so the gap to full dynamism is
        visible.
      </Prose>

      <H3>Byte-level BPE (production, partial)</H3>

      <Prose>
        GPT-2/3/4, Llama's byte-level fallback, and every modern tokenizer that handles
        arbitrary Unicode uses byte-level BPE. It is not dynamic in the sense of
        adapting after training, but it is graceful in one specific way: anything
        representable as UTF-8 bytes has a valid (if long) tokenization, so the "OOV"
        failure mode never occurs. That counts as the most production-mature form of
        dynamism available today — zero-shot handling of any input, with degraded
        compression on the tail.
      </Prose>

      <H3>Byte-Latent Transformer (Meta, research-scale but frontier)</H3>

      <Prose>
        Meta published BLT in December 2024 with a FLOP-controlled scaling study up to
        8B parameters and 4T training bytes. The patch encoder segments bytes based on
        next-byte entropy (computed by a small auxiliary LM), producing patches whose
        boundaries depend on the input. Code and model weights are on the
        <Code>facebookresearch/blt</Code> repo. This is the closest thing to
        production-grade dynamic tokenization as of 2026 — "production-grade" in the
        sense that it has been trained at scale with published results, not that it is
        serving user traffic. Whether Meta has moved Llama in this direction at scale is
        not publicly confirmed.
      </Prose>

      <H3>BoundlessBPE (research code, training-time)</H3>

      <Prose>
        Kensho released <Code>boundlessbpe</Code> (GitHub: <Code>kensho-technologies/
        boundlessbpe</Code>) alongside the COLM 2025 paper. The algorithm runs at
        training time, produces a fixed output vocabulary, and plugs into standard
        serving stacks — so it is "production-compatible" but not "dynamic at inference."
        It demonstrates the training-time variant of the vocabulary-growth idea: lift
        the pre-tokenization constraint that forces merges to respect whitespace, get a
        measurably better final vocabulary, ship it the normal way.
      </Prose>

      <H3>LiteToken (research, post-hoc)</H3>

      <Prose>
        LiteToken (arXiv:2602.04706) runs on an already-trained tokenizer and removes
        intermediate merge residues — tokens that appeared during BPE training but are
        rarely emitted at inference. The paper reports ~10% of tokens in major
        tokenizers (Qwen-3, Llama-3.1) are such residues. Removing them reduces
        parameters and improves robustness on misspelled inputs without retraining.
        This is a post-hoc simplification, not a dynamic adaptation — but it highlights
        that the static vocabulary a model ships with is not always well-chosen, and
        that it is possible to improve it after the fact.
      </Prose>

      <H3>Tokenizer retraining (slow and offline)</H3>

      <Prose>
        The closest thing to adaptation in practice is the process of retraining a
        tokenizer alongside a new model generation. Llama 2 → Llama 3 shipped a
        substantially different tokenizer, re-estimated on updated training data.
        GPT-4o's tokenizer handles non-English text better than GPT-4's by design. This
        is a slow form of adaptation — on the order of model-release cadence — and it
        does not help with within-inference or within-document distribution shift. It
        is the version of dynamism that has always existed and will always exist; the
        research question is whether there is value in anything faster.
      </Prose>

      <Callout accent="blue">
        If you are building an LLM application today and wondering whether to adopt a
        dynamic tokenizer — don't. Use byte-level BPE. The frontier is worth
        understanding; it is not yet worth integrating.
      </Callout>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        The abstract claim of section 2a — "the same bytes can become different tokens
        in different contexts" — is easier to see on concrete inputs. Step through the
        examples below. Each one shows how an adaptive segmenter redraws its boundaries
        based on the content type.
      </Prose>

      <StepTrace
        label="adaptive boundaries shift with content"
        steps={[
          { label: "English prose", render: () => (
            <TokenStream tokens={["The", " ", "quick", " ", "brown", " ", "fox", " ", "jumps"]} />
          ) },
          { label: "Python code", render: () => (
            <TokenStream tokens={["def", " ", "compute", "_", "scores", "(", "x", ",", " ", "42", ")"]} />
          ) },
          { label: "Chemical notation", render: () => (
            <TokenStream tokens={["H", "2", "SO", "4", " ", "+", " ", "Na", "OH"]} />
          ) },
          { label: "Identifier in long context", render: () => (
            <TokenStream tokens={[{ label: "TX-4891", color: "#f59e0b" }, " ", "shipped"]} />
          ) },
          { label: "Same identifier, static tokenizer", render: () => (
            <TokenStream tokens={["T", "X", "-", "48", "91", " ", "shipped"]} />
          ) },
        ]}
      />

      <Prose>
        The fourth and fifth steps show the same seven-character identifier under two
        different tokenizers. An adaptive tokenizer that has seen <Code>TX-4891</Code>
        enough times in a document treats it as a single token; a static tokenizer
        fragments it into five. In a thousand-token document where the identifier
        appears forty times, that difference is 160 tokens of context-window savings —
        roughly one full code block's worth.
      </Prose>

      <Prose>
        The second visualization: how sequence length on a recurring-pattern stream
        evolves as the growing tokenizer from section 4b accumulates merges. Static
        stays flat; dynamic drops sharply once the repeating pattern crosses the
        threshold and gets promoted.
      </Prose>

      <Plot
        label="tokens per chunk — static vs adaptive on a recurring-identifier stream"
        xLabel="chunk index"
        yLabel="tokens"
        series={[
          { name: "static", color: colors.textMuted, points: [[1, 23], [2, 30], [3, 21], [4, 26], [5, 24], [6, 22]] },
          { name: "adaptive", color: colors.gold, points: [[1, 23], [2, 30], [3, 15], [4, 21], [5, 6], [6, 15]] },
        ]}
      />

      <Prose>
        The sharp drop at chunk 5 is where <Code>TX-4891 TX-4891 TX-4891</Code> repeats
        three times in a row; by that point the merge table has absorbed enough of the
        prefix that each occurrence compresses to two tokens instead of seven. Chunk 6
        rebounds because a new substring — <Code>ed.</Code> — has not yet been merged.
        This is the characteristic dynamic-tokenization curve: static is flat, adaptive
        is step-function-like with a lag proportional to the merge threshold.
      </Prose>

      <Prose>
        Third visualization: static vs adaptive on identical input. The bytes are the
        same; the tokens are not.
      </Prose>

      <TokenStream
        label="static byte-level BPE — fragments the novel identifier"
        tokens={["from", " ", "py", "rel", "ic", ".", "retrieval", " import", " BM", "42", "Index"]}
      />

      <TokenStream
        label="adaptive tokenizer — has absorbed the identifier into units"
        tokens={[
          "from",
          " ",
          { label: "pyrelic", color: "#f59e0b" },
          ".",
          "retrieval",
          " import",
          " ",
          { label: "BM42Index", color: "#f59e0b" },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        The honest answer for almost every production application today is: use static
        byte-level BPE. The ecosystem is built around it; the tooling is mature; the
        costs are known. Dynamic techniques are worth the engineering tax only in
        specific cases.
      </Prose>

      <H3>Use static byte-level BPE when</H3>

      <Prose>
        You are shipping a general-purpose LLM, a chat product, an API, or any system
        where the vocabulary distribution is roughly stable over the model's deployed
        lifetime. You use mainstream serving infrastructure (vLLM, TGI, Triton) and
        value KV caching and request batching. You want token costs to be predictable
        across runs of the same input. You have not measured a specific failure mode
        that dynamism would fix. This covers probably 95% of current LLM deployments.
      </Prose>

      <H3>Consider BoundlessBPE-style training-time fixes when</H3>

      <Prose>
        You are training a new model from scratch and care about tokenizer efficiency
        at the margin — bytes per token on your target corpus is a KPI. The target
        corpus has patterns that cross whitespace boundaries systematically (function
        calls with common argument shapes, templated log lines, compound scientific
        terms). You are willing to run tokenizer training as a one-time offline step.
        You want the output to be a standard fixed vocabulary that ships normally.
      </Prose>

      <H3>Consider BLT / learnable boundaries when</H3>

      <Prose>
        You are doing research on LLM architecture, particularly at Meta scale. You
        want to evaluate whether removing tokenization as a hyperparameter improves
        scaling laws. You are willing to pay the engineering cost of a custom serving
        stack. You value robustness to noise and non-Latin-script performance. You have
        the compute budget to train a byte-level model from scratch.
      </Prose>

      <H3>Consider inference-time vocab growth (hypothetical) when</H3>

      <Prose>
        Your corpus distribution shifts dramatically over time — a conversational
        agent accumulating user-specific jargon across sessions, a code assistant
        working inside a single repository with its own naming conventions, a
        long-document RAG system where the same named entities recur thousands of
        times per document. Your serving stack is bespoke enough to tolerate the
        cache-invalidation and determinism problems. You are willing to treat this
        as an open research problem in your own production system. Very few teams
        should be in this bucket.
      </Prose>

      <H3>Consider LiteToken-style residue pruning when</H3>

      <Prose>
        You have an already-deployed model and an already-trained tokenizer, and you
        want robustness to misspelled inputs without retraining. This is the cheapest
        option on the table — post-hoc analysis, no training, ships as a different
        tokenizer config.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn&apos;t</H2>

      <Prose>
        Every dynamic technique trades something away that static tokenization gives for
        free. Enumerate the axes explicitly.
      </Prose>

      <H3>Inference latency</H3>

      <Prose>
        A lookup-table tokenizer is effectively free — a few microseconds per thousand
        characters, dominated by memory access. A learnable boundary predictor adds a
        forward pass through a small network for every input: BLT's entropy LM, for
        instance, is a distilled byte-level model that is not free but is small. In
        published BLT numbers the inference efficiency gain from coarser patches
        outweighs the cost of the boundary predictor at scale, but the crossover point
        depends on model size. An inference-time merge-growth tokenizer pays even more:
        every merge decision requires a pair-count update and a possible vocabulary
        write, and the re-encoding step runs every time the vocab changes. Allowance
        for this cost is what separates research code from serving-ready code.
      </Prose>

      <H3>Training compatibility</H3>

      <Prose>
        Most training frameworks assume a fixed vocabulary size at dataloader time,
        because the embedding matrix and output head need to be allocated at known
        shape. A dynamic-vocab approach requires either pre-allocating a maximum vocab
        size (wasteful memory, especially for the output head which is often tied to
        the embedding) or rebuilding the embedding at runtime (latency-hostile,
        gradient-hostile). BLT works around this by not having a vocab at all — patches
        are summarized into continuous vectors without an embedding-table lookup —
        which is a sharp architectural departure, not a drop-in change.
      </Prose>

      <H3>Memory overhead of growing vocabulary</H3>

      <Prose>
        Assuming each new entry is ~10 bytes for the string plus overhead, an
        inference-time tokenizer that grows unboundedly across a long session can
        accumulate megabytes of vocabulary state per user. This is tractable at
        per-request scale; it is hostile to the shared-state model of most serving
        stacks, where user data should not persist across requests. A session-scoped
        vocabulary is one reasonable middle ground — grow during a conversation, discard
        at the end — but it surrenders the cross-session learning that would motivate
        the whole scheme.
      </Prose>

      <H3>KV cache implications</H3>

      <Prose>
        This is the single hardest scaling problem, and it has its own subsection in
        failure modes (section 9). The one-line summary: KV caches index by token
        position and implicitly by token identity. If two requests tokenize the same
        input differently — even at one position — the caches are not interchangeable.
        At serving scale where cache hit rates in the 60-90% range are the margin
        between viable and unviable, a tokenizer that produces different outputs on
        the same input is dead on arrival. Session-scoped dynamics are more viable
        here because the cache can be scoped to the session too; cross-session
        dynamics are the hard version.
      </Prose>

      <H3>Batching</H3>

      <Prose>
        Batched inference assumes requests in the batch share a vocabulary for the
        output head. If request A has added merges that request B has not, the output
        logits for A have a different shape than for B, and efficient batching breaks.
        Workarounds exist (always pre-allocate max vocab, mask unused rows) but they
        give back most of the compression gains that motivated the dynamism.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes &amp; gotchas</H2>

      <H3>KV cache invalidation</H3>

      <Prose>
        A shared prefix between two requests is the single largest source of cache
        reuse in modern serving. If request 1 tokenizes "you are a helpful assistant"
        to one sequence of IDs and request 2 — arriving five seconds later, after the
        vocabulary has grown — tokenizes the same prefix to a different sequence, the
        cache does not hit. This is not a performance regression measured in percent.
        It is potentially a 10-100x slowdown on workloads that relied on prefix
        caching, which is most of them.
      </Prose>

      <H3>Token ID drift across inference sessions</H3>

      <Prose>
        In a naive implementation, the integer ID assigned to a merged pair depends on
        the order in which merges were added. Two servers running the same model may
        assign different IDs to "TX-4891" if they encountered different input streams.
        For single-node serving this is cosmetic; for distributed inference, model
        snapshots, fine-tuning on serving-era logs, or reproducing a bug reported by a
        user three weeks later, it is a correctness hazard. Every ID must be
        deterministic from the state that produced it — which in practice means logging
        the full merge sequence, which in practice means giving up most of the storage
        savings dynamism offered.
      </Prose>

      <H3>Serving infra assumes fixed embedding table size</H3>

      <Prose>
        vLLM, TGI, Triton, and most production stacks allocate the embedding matrix
        and output projection at model-load time to a known shape. Dynamic vocab
        growth requires either pre-allocating a ceiling (wasteful, and the ceiling
        must be picked correctly) or adding runtime allocation inside the hot path
        (a category of bug that every ML engineer has spent a week debugging). Any
        approach that merely "grows the vocab" without solving this is not deployable.
      </Prose>

      <H3>Unbounded growth on adversarial input</H3>

      <Prose>
        A simple adversary feeds the tokenizer a long string of unique, repeating
        byte patterns designed to trigger merges: <Code>ABAB ABAB ABAB...</Code>,
        followed by <Code>ABABAB ABABAB...</Code>, followed by the three-tier version,
        and so on. Each generation passes the threshold and gets promoted. Vocabulary
        grows unboundedly. Memory OOMs. Any production system must cap growth, which
        re-introduces a version of the static-vocab constraint the scheme was meant
        to avoid.
      </Prose>

      <H3>Privacy leaks through vocabulary state</H3>

      <Prose>
        Under-examined in the literature but immediately concerning. A tokenizer that
        has adapted to one user's conversation contains, in its merge table, a
        compressed summary of that user's recurring language — names, project
        identifiers, domain jargon. If the vocabulary state persists across user
        boundaries, that state is a data-leakage vector. Any shared-tokenizer design
        needs explicit scoping (session, user, organization) and a strong story about
        how state leaves when it is supposed to. Shared-across-users cross-session
        vocabulary growth is almost certainly a GDPR violation waiting to happen.
      </Prose>

      <H3>Evaluation inconsistency</H3>

      <Prose>
        Perplexity, bits-per-byte, and most token-level metrics assume a fixed
        tokenization of the evaluation set. If the same text is tokenized to different
        numbers of tokens across runs, perplexity numbers are not comparable. This is
        why BLT's published results are denominated in bits-per-byte rather than
        perplexity-per-token: the denominator is invariant to the tokenization scheme.
        Any team evaluating a dynamic tokenizer must adopt byte-level metrics or
        accept that their numbers are not directly comparable to published static-vocab
        results.
      </Prose>

      <H3>Debugging becomes harder</H3>

      <Prose>
        A serving bug at 2 AM where a particular user's prompt produces bad output is
        already hard to reproduce. Add "and the tokenizer has evolved since training"
        and reproduction requires snapshotting the full tokenizer state alongside the
        prompt. This is possible but adds operational burden that every team considering
        dynamism must budget for.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        The naming in the original topic title deserves a correction. ADAT, BoundlessBPE,
        and LiteToken all correspond to real published work, but they are not all
        inference-time-adaptive tokenizers in the sense the title suggests. ADAT
        (Zheng et al., NeurIPS 2024) refines vocabulary during training using
        downstream-model feedback; BoundlessBPE (Schmidt et al., COLM 2025) relaxes the
        pre-tokenization barrier at training time; LiteToken (arXiv:2602.04706) prunes
        residue tokens from existing tokenizers. The closest work to actual
        inference-time dynamism is Meta's BLT. The taxonomy in section 2 reflects this
        corrected understanding.
      </Prose>

      <ul style={{ color: colors.textSecondary, lineHeight: 1.7, fontSize: 14 }}>
        <li>
          <strong>Byte-Latent Transformer (BLT)</strong> — Pagnoni, Pasunuru, Rodriguez et
          al. Meta, December 2024. arXiv:2412.09871. "Patches Scale Better Than Tokens."
          First FLOP-controlled scaling study of byte-level models up to 8B/4T. Closest
          production-scale dynamic tokenization work.
        </li>
        <li>
          <strong>Charformer</strong> — Tay, Tran, Ruder et al., Google. arXiv:2106.12672
          (2021, revised 2022). "Fast Character Transformers via Gradient-based Subword
          Tokenization." The GBST module is the canonical example of learnable boundary
          prediction.
        </li>
        <li>
          <strong>MambaByte</strong> — Wang, Gangavarapu, Yan, Rush. Cornell.
          arXiv:2401.13660 (2024). "Token-free Selective State Space Model." Byte-level
          alternative to tokenization via SSM gating, with 2.6x speedup over naive
          implementation via speculative decoding.
        </li>
        <li>
          <strong>BoundlessBPE</strong> — Schmidt, Reddy, Tanner, Pinter. COLM 2025.
          arXiv:2504.00178. "Breaking the Pre-tokenization Barrier." Relaxes the whitespace
          constraint in BPE; 19.7%+ bytes-per-token improvement. Code:
          <Code>kensho-technologies/boundlessbpe</Code>.
        </li>
        <li>
          <strong>ADAT (Adaptive Tokenizers via LLM Feedback)</strong> — Zheng et al.
          NeurIPS 2024. "Enhancing Large Language Models through Adaptive Tokenizers."
          Iteratively refines vocabulary based on downstream-model losses during training.
        </li>
        <li>
          <strong>LiteToken</strong> — arXiv:2602.04706. "Removing Intermediate Merge
          Residues From BPE Tokenizers." Finetuning-free pipeline that prunes ~10% of
          tokens in Qwen-3 and Llama-3.1 without performance loss.
        </li>
        <li>
          <strong>Tokenization Is More Than Compression</strong> — Schmidt, Reddy, Zhang et
          al. EMNLP 2024. arXiv:2402.18376. Introduces PathPiece (minimum-token segmenter)
          and empirically refutes the hypothesis that fewer tokens ⇒ better downstream
          performance.
        </li>
      </ul>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <Prose>
        <strong>1.</strong> A serving stack uses prefix caching — if two requests share
        the first N tokens, the KV activations for those N positions are reused. Explain
        precisely why an inference-time-growing tokenizer breaks this, and describe one
        mitigation that preserves at least some cache reuse.
      </Prose>

      <Prose>
        <strong>2.</strong> Design an adaptive tokenizer scoped to a single long
        conversation with one user. What input signal would you use to trigger a new
        merge? What would the throwaway policy be at session end, and what
        implications does that have for cross-session behavior?
      </Prose>

      <Prose>
        <strong>3.</strong> Research papers on adaptive tokenization tend to benchmark on
        code, scientific text, or non-English languages — not on general web text. Why?
        What does that imply about the upper bound of gains on general-purpose LLM
        workloads?
      </Prose>

      <Prose>
        <strong>4.</strong> The implementation in section 4b observes pair frequencies in
        a sliding window and promotes a pair once it crosses a count threshold. What goes
        wrong if you replace "count threshold" with "PMI threshold" naively? (Hint:
        consider the PMI of a rare character pair that co-occurs every time.) How would
        you fix it?
      </Prose>

      <Prose>
        <strong>5.</strong> BLT uses next-byte entropy from a small auxiliary language
        model as its boundary criterion. Why entropy rather than, say, raw next-byte
        probability? What does entropy capture about the local segmentation decision
        that probability alone does not?
      </Prose>

      <Prose>
        Dynamic tokenization is the frontier frontier. The ideas are clear; the papers
        are published; the engineering cost of shipping them is large and mostly
        unpaid. Over the next few years either the cost comes down — through better
        serving infrastructure, better APIs for KV management, cheaper learned scorers
        — or the field concludes that static byte-level BPE plus periodic retraining is
        a local optimum worth staying at. Both outcomes are plausible. What is not
        plausible is that the current arrangement — tokenizers designed in 2023 running
        on inputs the world produces in 2026 — stays stable forever.
      </Prose>
    </div>
  ),
};

export default dynamicTokenization;
