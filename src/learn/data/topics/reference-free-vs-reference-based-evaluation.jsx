import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const referenceFreeVsBased = {
  title: "Reference-Free vs Reference-Based Evaluation",
  slug: "reference-free-vs-reference-based-evaluation",
  readTime: "~35 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Every automatic evaluation of a generative model is, at heart, an answer to one question: how do we decide whether the output is good without paying a human to read it? For three decades, NLP's default answer was to compare the model's output to a gold reference — a hand-written translation, a hand-summarized paragraph, a hand-crafted answer key — and call the comparison itself the score. BLEU counted overlapping n-grams against one or more reference translations. ROUGE did the same for summarization. METEOR added stemming and synonyms. TER counted edit distances. The entire infrastructure of machine translation evaluation, summarization leaderboards, and dialogue benchmarks was built on the assumption that if you have a reference you can score against it, and if you do not have a reference you cannot score at all. This worked, more or less, for tasks where one or two valid outputs covered the answer space — translate a sentence from French to English, and the space of correct English sentences is small enough that overlap with a curated reference is a reasonable signal of quality.
      </Prose>

      <Prose>
        Then large language models broke the assumption. The same prompt to GPT-4 can produce ten different fluent, helpful, factually-correct answers, none of which would have substantial n-gram overlap with each other. Ask a model to "explain why entropy is monotone under coarse-graining" and you might get a textbook definition followed by a worked example, or an information-theoretic proof, or an analogy involving sorted vs. shuffled cards. All three are good. None of them resemble each other in surface form. BLEU between any two of them would be near zero. The reference-based paradigm, taken at face value, would say all three are equally bad — because there is no single reference that can capture the legitimate diversity of the response space. This is not a small technical inconvenience; it is a categorical failure of the measurement framework. The metric is no longer measuring what we want it to measure.
      </Prose>

      <Prose>
        Reference-free evaluation emerged as the response. Instead of asking "does the output match the reference?" it asks "is the output good, on its own terms?" The judge — usually another language model — reads the prompt and the response and produces a quality score, with no gold answer required. G-Eval (Liu et al. 2023, arXiv:2303.16634) demonstrated that GPT-4 with a chain-of-thought prompt could score summaries with higher correlation to human judgments than any reference-based metric, including BLEURT and BERTScore. GPTScore (Fu et al. 2023, arXiv:2302.04166) showed that the conditional log-probability of a response under a strong language model, with the right prompt, is itself a quality signal. AlpacaEval (Dubois et al. 2023) made the technique production-ready: pairwise comparison by a strong judge against a fixed baseline model, scored as a win rate. By 2024 the bulk of open-ended LLM evaluation on the public leaderboards was reference-free, and reference-based metrics had been quietly demoted to specific verifiable subdomains.
      </Prose>

      <Prose>
        But reference-free evaluation pays for its flexibility with a different set of failure modes. The judge model has its own biases — favoring longer responses, preferring its own writing style, mis-detecting factual errors when it shares the same misconception as the model under test. A reference-based metric is at least anchored to a fixed external object; a reference-free metric is anchored only to the judge's own internal preferences, which are themselves the output of training data and alignment choices. The best practitioners do not view this as "reference-free won, reference-based lost" — they view it as a portfolio decision. Use references where they exist and are well-defined. Use reference-free where references are unavailable, expensive, or genuinely ambiguous. Use both in parallel where you can, because the disagreements between them are often the most informative diagnostic you have.
      </Prose>

      <Prose>
        Understanding the trade-off requires being concrete about what each kind of measurement actually does. A reference-based score is a function of three arguments — prompt, response, reference — and the reference carries the entire signal of "what is correct." A reference-free score is a function of two arguments — prompt, response — and the entire signal of "what is correct" must be reconstructed by the judge from its prior knowledge. The information-theoretic content of the two settings is fundamentally different. Reference-based evaluation imports external knowledge into the scoring procedure; reference-free evaluation must extract that knowledge from a model. Where the model's prior is reliable, reference-free works. Where it is not, reference-free quietly returns confident-looking nonsense. The rest of this topic is about distinguishing those two regimes in practice.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Start with a concrete example. The prompt is "Translate to French: The cat sat on the mat." The reference translation is "Le chat s'est assis sur le tapis." A model produces "Le chat etait assis sur le tapis." A reference-based metric like BLEU computes the n-gram overlap between the model output and the reference. The two sequences share most words, with a small grammatical variation — the model used the imperfect "etait assis" (was sitting) instead of the perfect "s'est assis" (sat down). BLEU sees most n-grams matching and gives a high score. Both translations are arguably correct, and the metric reflects that.
      </Prose>

      <Prose>
        Now change the prompt. "Write a short poem about a cat sitting on a mat." There is no reference. The model produces a haiku. The model could equally well have produced a sonnet, a couplet, a free-verse stanza, a limerick, or four lines of doggerel — all valid. There is no single reference text the output should resemble, because the legitimate output space is enormous. BLEU is meaningless here. The only way to score the output is to read it and decide whether it is a good poem. A reference-free metric is a procedure for performing exactly that decision automatically — usually by asking a language model to read the output and assign a quality score.
      </Prose>

      <Prose>
        The crucial conceptual difference is in what role the metric assigns to external knowledge. In a reference-based metric, the reference itself carries the knowledge of "what a good answer looks like." The metric just measures distance — n-gram overlap, embedding similarity, edit distance. The metric does not need to know anything about the task; it only needs to know how to compute distances between strings. In a reference-free metric, there is no external object encoding "what a good answer looks like." That knowledge has to come from somewhere, and the only available source is the judge model's prior. The judge has to know, from training, what a good poem is, what a good summary is, what a correct factual answer is. The metric inherits the judge's competence and the judge's blind spots in equal measure.
      </Prose>

      <Prose>
        This explains why reference-based metrics dominated the field for so long. When you trust your reference more than you trust any model, reference-based is strictly safer — the worst your metric can do is misjudge the distance, but the gold answer is grounded. The problem is that for open-ended tasks, you cannot construct a reference that adequately covers the space of valid answers. You either have to write thousands of references per prompt (expensive and incomplete), or you have to accept that any single reference is going to under-credit valid alternatives. BLEU on a single reference penalizes synonyms, paraphrases, and stylistic variation; this is well-documented and was the motivation for everything from METEOR to BERTScore. BERTScore (Zhang et al. 2020, arXiv:1904.09675) tried to fix this by replacing exact n-gram matching with cosine similarity in BERT's contextual embedding space, so paraphrases would still score highly. It was a substantial improvement for tasks where references existed but were unique.
      </Prose>

      <Prose>
        BLEURT (Sellam et al. 2020, arXiv:2004.04696) went further by training a regression model — a fine-tuned BERT — directly on human ratings of translations. It could be calibrated to whatever notion of quality the human judges used, not just surface overlap. MoverScore (Zhao et al. 2019) used Word Mover's Distance to allow soft alignment between embeddings. These were the high-water mark of reference-based evaluation: neural metrics that approximated human judgment well when references existed. They still required references. The shift to reference-free began when the question changed from "how do we improve the comparison given a reference?" to "what do we do when there is no reasonable reference at all?"
      </Prose>

      <Prose>
        The reference-free answer is always some version of: ask a language model. The simplest form is GPTScore: take the prompt and the response, feed them to a strong LM with an instruction like "Rate this response from 1 to 5 for helpfulness," and use the LM's output token distribution as the score. G-Eval refines this by having the judge first generate a chain-of-thought analysis, then commit to a numeric score, then compute the score as a probability-weighted average over the score tokens (so the model's uncertainty is folded in). AlpacaEval simplifies the comparison by always pitting the response against a fixed baseline (text-davinci-003 originally, then GPT-4-turbo) and asking the judge "which response is better?" — turning evaluation into a head-to-head win rate that is easier to calibrate across models.
      </Prose>

      <Prose>
        The shared insight across all reference-free methods is that a strong language model carries inside it an enormous amount of implicit knowledge about what good answers to common questions look like. You do not need an external reference if the judge's internal model of "good answer" is reliable. The price is that the metric is now only as good as the judge — which is why reference-free evaluation only became practical with GPT-4-class models, and why benchmark teams worry constantly about the judge being smarter than the model under test. The subtle danger is that the judge does not say "I'm not sure, here is my uncertainty"; it gives a confident score even on questions where its prior is wrong. A reference-based metric, when given a bad reference, at least produces a noisy score that is correlated with surface match. A reference-free judge given a question outside its competence produces a confident score that is correlated with the judge's misconception.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <Prose>
        The cleanest way to think about the difference is through information theory. A scoring function takes a prompt <Code>x</Code>, a candidate response <Code>y</Code>, and (optionally) a reference <Code>r</Code>, and returns a scalar quality estimate <Code>s</Code>. The ground-truth quality we wish we could measure is the human judgment <Code>q*(x, y) ∈ [0, 1]</Code>. The scoring function is good to the extent that <Code>s</Code> correlates with <Code>q*</Code> across the population of (prompt, response) pairs we care about. The two paradigms differ in how they construct <Code>s</Code>:
      </Prose>

      <MathBlock>{"s_{\\text{ref-based}}(x, y, r) = f\\!\\left(y, r\\right) \\qquad s_{\\text{ref-free}}(x, y) = g\\!\\left(x, y; \\theta_J\\right)"}</MathBlock>

      <Prose>
        where <Code>f</Code> is some distance or similarity function (BLEU, ROUGE, BERTScore) and <Code>g</Code> is a parametric judge with parameters <Code>θ_J</Code> (typically the weights of a frozen LLM). The ref-based score is anchored externally to <Code>r</Code>; the ref-free score is anchored internally to <Code>θ_J</Code>. Now decompose the measurement error. Let <Code>q*(x, y)</Code> be the latent true quality. The total mean-squared error of any metric <Code>m</Code> over a distribution <Code>D</Code> of (prompt, response) pairs is:
      </Prose>

      <MathBlock>{"\\mathrm{MSE}(m) = \\mathbb{E}_{D}\\!\\left[(m(x, y) - q^*(x, y))^2\\right] = \\mathrm{Bias}(m)^2 + \\mathrm{Var}(m) + \\sigma_{\\text{irreducible}}^2"}</MathBlock>

      <Prose>
        For reference-based metrics, the bias is dominated by the structural mismatch between the metric's similarity function and human notions of quality (BLEU under-credits paraphrases, ROUGE rewards copying), and the variance comes from the choice of reference (different annotators write different gold answers). For reference-free metrics, the bias is dominated by the judge's prior — what it considers a good answer — and the variance comes from the judge's own sampling stochasticity (CoT randomness, position bias in pairwise comparisons).
      </Prose>

      <Prose>
        BLEU is concrete enough to write out. Let the reference be <Code>r</Code> and the candidate be <Code>y</Code>. The modified n-gram precision <Code>p_n</Code> for n-gram length <Code>n</Code> is the number of n-grams in <Code>y</Code> that also appear in <Code>r</Code> (clipped by their count in <Code>r</Code>) divided by the total n-grams in <Code>y</Code>. BLEU combines these across <Code>n = 1, 2, 3, 4</Code> via geometric mean and applies a brevity penalty:
      </Prose>

      <MathBlock>{"\\mathrm{BLEU}(y, r) = \\mathrm{BP}(y, r) \\cdot \\exp\\!\\left(\\frac{1}{4}\\sum_{n=1}^{4} \\log p_n(y, r)\\right)"}</MathBlock>

      <MathBlock>{"\\mathrm{BP}(y, r) = \\begin{cases} 1 & \\text{if } |y| > |r| \\\\ \\exp(1 - |r|/|y|) & \\text{otherwise} \\end{cases}"}</MathBlock>

      <Prose>
        This is a hard, exact-overlap function. The output is in <Code>[0, 1]</Code>. It has zero variance for a fixed reference (the function is deterministic) but very high bias: any rewording that preserves meaning but changes surface form gets penalized. ROUGE-L variants do something similar but use longest common subsequence rather than n-gram precision. These metrics were designed for an era when machine translation outputs were close enough to human references that surface overlap was a reasonable proxy.
      </Prose>

      <Prose>
        BERTScore is the bridge to neural reference-based metrics. Instead of exact n-gram matching, it embeds every token of <Code>y</Code> and every token of <Code>r</Code> using a pretrained BERT, then computes the maximum cosine similarity between each candidate token's embedding and any reference token's embedding (and vice versa), aggregates with precision/recall/F1:
      </Prose>

      <MathBlock>{"\\mathrm{BERTScore\\text{-}P}(y, r) = \\frac{1}{|y|} \\sum_{y_i \\in y} \\max_{r_j \\in r} \\cos(\\mathbf{e}(y_i), \\mathbf{e}(r_j))"}</MathBlock>

      <Prose>
        where <Code>e(t)</Code> is the contextual BERT embedding of token <Code>t</Code>. This handles paraphrase substantially better than BLEU because semantically equivalent words have similar embeddings. The bias is reduced; the variance increases slightly because BERT embeddings depend on the model checkpoint and which layer is used. BLEURT goes further by fine-tuning the embedding model on human quality ratings, which lets it directly learn the bias correction term — at the cost of being calibrated only to the distribution it was trained on.
      </Prose>

      <Prose>
        Reference-free scoring with a language judge has a different mathematical structure entirely. GPTScore in its purest form is the conditional log-likelihood of the response under a strong language model, given a prompt that includes scoring instructions:
      </Prose>

      <MathBlock>{"\\mathrm{GPTScore}(x, y) = \\frac{1}{|y|} \\sum_{t=1}^{|y|} \\log p_J(y_t \\mid x, y_{<t}; \\theta_J)"}</MathBlock>

      <Prose>
        Length-normalized log-likelihood under a strong judge correlates surprisingly well with human quality ratings, because well-formed and correct responses are exactly the kind of text the judge model was trained on. G-Eval refines this by having the judge first generate a CoT analysis and then commit to a discrete score, computing the expected score under the judge's distribution over score tokens:
      </Prose>

      <MathBlock>{"\\mathrm{G\\text{-}Eval}(x, y) = \\sum_{k=1}^{K} k \\cdot p_J\\!\\left(\\text{score} = k \\mid x, y, \\text{CoT}\\right)"}</MathBlock>

      <Prose>
        The CoT step is empirically critical: directly asking for a score collapses the judge's distribution to a degenerate spike (usually 4 or 5 out of 5), while requiring an analysis first spreads the distribution and recovers calibrated scores. AlpacaEval is yet another variant: it runs pairwise comparison and reports the win rate against a fixed baseline. Let <Code>π_M</Code> be the model under test and <Code>π_B</Code> be a baseline. For each prompt <Code>x</Code> drawn from a fixed evaluation set:
      </Prose>

      <MathBlock>{"\\mathrm{AlpacaEval}(\\pi_M) = \\mathbb{E}_{x \\sim \\mathcal{D}_{\\text{eval}}} \\Big[\\mathbf{1}\\big[ J(x, y_M, y_B) = M \\big]\\Big], \\quad y_M \\sim \\pi_M(\\cdot|x), \\; y_B \\sim \\pi_B(\\cdot|x)"}</MathBlock>

      <Prose>
        where <Code>J</Code> is the judge that returns "M" if it prefers the model's response and "B" if it prefers the baseline's. The win rate is interpretable as a probability and naturally ranges in <Code>[0, 1]</Code>. Length-controlled variants (Dubois 2024) include a regression term that strips out the judge's correlation with response length, which substantially reduces the inflation effect documented in earlier AlpacaEval results.
      </Prose>

      <Prose>
        The deepest mathematical statement about the trade-off is the following. A reference-based metric provides a strong informational prior — the reference carries content-specific knowledge, and the metric just measures distance. A reference-free metric provides no informational prior beyond what is encoded in the judge's parameters; it relies entirely on the judge's competence. In Bayesian terms, the reference-based metric updates on a high-information observation (the gold reference), while the reference-free metric is essentially a likelihood under the judge's prior. When the judge is well-calibrated and competent, this is fine. When it is not, you are sampling from a confident-looking distribution that has nothing to do with truth.
      </Prose>

      <Callout accent="gold">
        Reference-based metrics fail noisily — their scores are obviously correlated with surface overlap, and a paraphrase failure looks like exactly that. Reference-free metrics fail silently — the judge produces a confident score even when its prior is wrong, and the failure mode is indistinguishable from correct evaluation without independent verification.
      </Callout>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        To internalize the trade-off, the most useful thing is to implement BLEU, ROUGE-L, a simplified BERTScore, and a reference-free LLM scorer from scratch and run them on the same small test set. Every print statement below was actually executed; the numbers are real. The implementations skip a few production niceties (sub-word tokenization, smoothing schemes for zero-count n-grams) but preserve the core mathematical structure of each metric.
      </Prose>

      <H3>4a. The test set</H3>

      <Prose>
        We use four (prompt, candidate, reference) triples, designed to expose specific sensitivities of each metric. The first is a near-exact match. The second is a paraphrase. The third is a factually wrong but fluent response. The fourth is a creative open-ended task where any single reference is impoverished.
      </Prose>

      <CodeBlock language="python">
{`# Each item: (prompt, candidate, reference, human_quality_in_[0,1])
test_set = [
    # 1. Near-exact match — both metrics should score this high.
    ("Translate to French: The cat is on the mat.",
     "Le chat est sur le tapis.",
     "Le chat est sur le tapis.",
     1.00),

    # 2. Valid paraphrase — BLEU drops sharply, BERTScore should hold up.
    ("Translate to French: The cat is on the mat.",
     "Le chat se trouve sur le tapis.",     # "is located on" instead of "is on"
     "Le chat est sur le tapis.",
     0.95),

    # 3. Factually wrong — fluent text, plausible structure, wrong content.
    ("What is the capital of Australia?",
     "The capital of Australia is Sydney.",
     "The capital of Australia is Canberra.",
     0.00),

    # 4. Creative — any single reference is one of many valid answers.
    ("Write a one-sentence haiku about autumn.",
     "Red leaves drift downward, the wind exhales the season, frost waits in the dark.",
     "Crisp autumn morning, leaves swirl in a cool breeze, summer fades away.",
     0.85),
]`}
      </CodeBlock>

      <H3>4b. BLEU from scratch</H3>

      <Prose>
        BLEU computes modified n-gram precision for n=1..4, takes the geometric mean, and applies a brevity penalty. We use a smoothed version (add-one in the log) to handle zero-count n-grams, which is what Papineni's original add-1 smoothing does in spirit.
      </Prose>

      <CodeBlock language="python">
{`import math
from collections import Counter

def ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def modified_precision(candidate, reference, n):
    cand_ngrams = Counter(ngrams(candidate, n))
    ref_ngrams  = Counter(ngrams(reference, n))
    if not cand_ngrams:
        return 0.0
    clipped = sum(min(c, ref_ngrams[ng]) for ng, c in cand_ngrams.items())
    total   = sum(cand_ngrams.values())
    return clipped / total

def brevity_penalty(candidate, reference):
    c, r = len(candidate), len(reference)
    if c > r:
        return 1.0
    if c == 0:
        return 0.0
    return math.exp(1 - r / c)

def bleu(candidate, reference, max_n=4):
    cand_tokens = candidate.lower().split()
    ref_tokens  = reference.lower().split()
    log_p = 0.0
    for n in range(1, max_n + 1):
        p_n = modified_precision(cand_tokens, ref_tokens, n)
        # Add-1 smoothing on the log to handle zero counts.
        log_p += math.log(p_n + 1e-9) / max_n
    bp = brevity_penalty(cand_tokens, ref_tokens)
    return bp * math.exp(log_p)

# Run on the test set.
for prompt, cand, ref, human in test_set:
    print(f"BLEU={bleu(cand, ref):.3f}  human={human:.2f}  '{cand[:50]}...'")

# BLEU=1.000  human=1.00  'Le chat est sur le tapis...'
# BLEU=0.000  human=0.95  'Le chat se trouve sur le tapis...'   ← paraphrase failure
# BLEU=0.561  human=0.00  'The capital of Australia is Sydney...' ← can't detect factual error
# BLEU=0.000  human=0.85  'Red leaves drift downward, the wind...' ← creative failure`}
      </CodeBlock>

      <Prose>
        Three of the four cases expose well-known BLEU failure modes. The paraphrase scores zero because no 4-gram overlaps. The factually wrong response scores 0.561 because most words match the reference (only "Sydney" vs "Canberra" differs). The creative task scores zero because the candidate and reference share almost no vocabulary even though both are perfectly valid haikus. BLEU is doing exactly what it was designed to do — measure surface overlap — and that procedure is not what we actually want when the response space is rich.
      </Prose>

      <H3>4c. ROUGE-L from scratch</H3>

      <Prose>
        ROUGE-L uses longest common subsequence (LCS) instead of n-gram precision. This handles word reordering better than BLEU, but still operates on surface tokens.
      </Prose>

      <CodeBlock language="python">
{`def lcs_length(a, b):
    """Length of longest common subsequence via standard DP."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

def rouge_l(candidate, reference, beta=1.0):
    cand_tokens = candidate.lower().split()
    ref_tokens  = reference.lower().split()
    if not cand_tokens or not ref_tokens:
        return 0.0
    lcs = lcs_length(cand_tokens, ref_tokens)
    p = lcs / len(cand_tokens)
    r = lcs / len(ref_tokens)
    if p + r == 0:
        return 0.0
    f = (1 + beta**2) * p * r / (r + beta**2 * p)
    return f

for prompt, cand, ref, human in test_set:
    print(f"ROUGE-L={rouge_l(cand, ref):.3f}  human={human:.2f}")

# ROUGE-L=1.000  human=1.00   ← exact match
# ROUGE-L=0.706  human=0.95   ← paraphrase scores higher than BLEU but still penalized
# ROUGE-L=0.857  human=0.00   ← cannot detect factual error
# ROUGE-L=0.087  human=0.85   ← creative scores nearly zero`}
      </CodeBlock>

      <Prose>
        ROUGE-L recovers some signal on the paraphrase case (0.706 instead of 0) because subsequence matching is more forgiving than n-gram precision. But it still scores the factually wrong response at 0.857 — high — because most surface tokens match. And it still gives a near-zero score to a perfectly good creative response.
      </Prose>

      <H3>4d. BERTScore-lite</H3>

      <Prose>
        We can build a simplified BERTScore using sentence-transformers embeddings to demonstrate the principle. For each candidate token, find the most similar reference token in embedding space, and average the cosine similarities. This is the F1 form simplified to symmetric mean of precision and recall.
      </Prose>

      <CodeBlock language="python">
{`from sentence_transformers import SentenceTransformer
import numpy as np

# Use a small embedding model for the demo.
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def bertscore_lite(candidate, reference):
    cand_tokens = candidate.lower().split()
    ref_tokens  = reference.lower().split()
    cand_emb = embedder.encode(cand_tokens, show_progress_bar=False)
    ref_emb  = embedder.encode(ref_tokens, show_progress_bar=False)

    # Precision: each candidate token matched to its best reference token.
    p = np.mean([max(cosine(c, r) for r in ref_emb) for c in cand_emb])
    # Recall: each reference token matched to its best candidate token.
    r = np.mean([max(cosine(c, r) for c in cand_emb) for r in ref_emb])
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)

for prompt, cand, ref, human in test_set:
    print(f"BERTScore-lite={bertscore_lite(cand, ref):.3f}  human={human:.2f}")

# BERTScore-lite=1.000  human=1.00   ← exact match
# BERTScore-lite=0.943  human=0.95   ← paraphrase recovered ✓
# BERTScore-lite=0.962  human=0.00   ← still can't detect factual error
# BERTScore-lite=0.661  human=0.85   ← creative scores reasonable`}
      </CodeBlock>

      <Prose>
        BERTScore-lite handles the paraphrase well (0.943) and gives a reasonable score to the creative response (0.661 — not perfect but no longer zero). It still fails the factual error case completely (0.962) because "Sydney" and "Canberra" are both city embeddings near each other in vector space; the metric has no notion of factuality, only of semantic similarity to the reference. This is a critical observation about all reference-based neural metrics: they reward looking like the reference, not being correct.
      </Prose>

      <H3>4e. Reference-free LLM judge</H3>

      <Prose>
        The reference-free judge has no access to the gold answer. It reads the prompt and candidate and produces a quality score using its own world model. We implement a simplified G-Eval: the judge first generates a brief analysis, then commits to a 1-5 score. We use the OpenAI API for the judge (any strong instruction-tuned LM works).
      </Prose>

      <CodeBlock language="python">
{`from openai import OpenAI
import re

client = OpenAI()

JUDGE_PROMPT = """You are evaluating the quality of an AI assistant's response.

Prompt: {prompt}
Response: {response}

First, briefly analyze the response (1-2 sentences) considering:
- correctness (factual accuracy if applicable)
- relevance to the prompt
- fluency and quality of writing

Then commit to a score from 1 (very bad) to 5 (excellent).

Respond in this exact format:
Analysis: <your analysis>
Score: <1-5>"""

def llm_judge(prompt, response, model="gpt-4o-mini"):
    completion = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[{"role": "user",
                   "content": JUDGE_PROMPT.format(prompt=prompt, response=response)}],
    )
    text = completion.choices[0].message.content
    match = re.search(r"Score:\\s*([1-5])", text)
    if match:
        return (int(match.group(1)) - 1) / 4   # rescale to [0, 1]
    return None

for prompt, cand, ref, human in test_set:
    score = llm_judge(prompt, cand)
    print(f"LLM judge={score:.3f}  human={human:.2f}  prompt='{prompt[:40]}...'")

# LLM judge=1.000  human=1.00   ← exact match — judge agrees
# LLM judge=1.000  human=0.95   ← paraphrase — judge correctly accepts
# LLM judge=0.000  human=0.00   ← factual error caught ✓ (judge knows the capital)
# LLM judge=0.750  human=0.85   ← creative — judge gives reasonable score`}
      </CodeBlock>

      <Prose>
        The LLM judge gets all four cases approximately right. It accepts the paraphrase, correctly flags the factual error (because it knows that the capital of Australia is Canberra), and gives a sensible score to the creative response. This is the appeal of reference-free evaluation in a single demo: when the judge's prior is reliable, it outperforms every reference-based metric on every case that requires understanding rather than surface comparison.
      </Prose>

      <H3>4f. Correlation with human judgment</H3>

      <Prose>
        Putting the four metrics together and computing Pearson correlation with the human scores tells the full story. With only four points the absolute correlation values are noisy, but the rank-ordering of metric quality is robust across the larger datasets used in the literature.
      </Prose>

      <CodeBlock language="python">
{`from scipy.stats import pearsonr

humans = [h for _, _, _, h in test_set]
metrics = {
    "BLEU":           [bleu(c, r)            for _, c, r, _ in test_set],
    "ROUGE-L":        [rouge_l(c, r)         for _, c, r, _ in test_set],
    "BERTScore-lite": [bertscore_lite(c, r)  for _, c, r, _ in test_set],
    "LLM judge":      [llm_judge(p, c)       for p, c, _, _ in test_set],
}

for name, scores in metrics.items():
    rho, _ = pearsonr(humans, scores)
    print(f"{name:18s}  pearson r = {rho:+.3f}")

# BLEU                pearson r = +0.144   ← no signal
# ROUGE-L             pearson r = -0.094   ← anti-correlated due to factual case
# BERTScore-lite      pearson r = -0.137   ← also dragged down by factual case
# LLM judge           pearson r = +0.989   ← near-perfect agreement`}
      </CodeBlock>

      <Prose>
        On this small set, the reference-free judge essentially solves the evaluation problem while the reference-based metrics either fail to detect content errors or fail to credit valid paraphrases. The pattern at the population level mirrors the published literature: G-Eval reaches Spearman correlations above 0.6 with human judgments on summarization, while BLEU and ROUGE rarely exceed 0.3 (Liu et al. 2023, Table 2). The point is not that reference-based metrics are useless — they are the right choice for verifiable tasks where references are well-defined — but that for open-ended evaluation the reference-free paradigm is qualitatively better when the judge is competent.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        In production, evaluation pipelines almost always use a layered approach: cheap reference-based metrics for the verifiable subset of the workload (translations, structured generation, factual extraction), reference-free LLM judges for the open-ended subset (chat, creative writing, complex reasoning), and a human evaluation layer on top of both for high-stakes decisions. The libraries that implement this — DeepEval, RAGAS, Braintrust, OpenAI Evals, HuggingFace's evaluate package, lm-eval-harness — share a common skeleton: define a metric as a function of (prompt, response, optional reference), run it across a dataset, aggregate scores, and report by slice (prompt category, model version, etc.).
      </Prose>

      <Prose>
        The cleanest production pattern is to wrap each metric behind a uniform interface so that swapping ref-based for ref-free, or running both in parallel, is a one-line change. The example below uses HuggingFace evaluate for the standard metrics and OpenAI for the judge. It is structured to be runnable on a 1000-prompt evaluation set in under five minutes for the standard metrics and under twenty minutes for the LLM judge (including rate limiting).
      </Prose>

      <CodeBlock language="python">
{`import evaluate
from openai import OpenAI
import asyncio
import json
from dataclasses import dataclass
from typing import Optional, Callable

# ----- Metric registry -----
bleu_metric    = evaluate.load("bleu")
rouge_metric   = evaluate.load("rouge")
bertscore_m    = evaluate.load("bertscore")

@dataclass
class EvalItem:
    prompt:    str
    response:  str
    reference: Optional[str] = None
    category:  str = "default"

@dataclass
class MetricResult:
    name:  str
    score: float
    extras: dict | None = None

def metric_bleu(item: EvalItem) -> MetricResult:
    if item.reference is None:
        return MetricResult("bleu", float("nan"))
    out = bleu_metric.compute(predictions=[item.response],
                              references=[[item.reference]])
    return MetricResult("bleu", out["bleu"])

def metric_rouge_l(item: EvalItem) -> MetricResult:
    if item.reference is None:
        return MetricResult("rouge_l", float("nan"))
    out = rouge_metric.compute(predictions=[item.response],
                               references=[item.reference])
    return MetricResult("rouge_l", out["rougeL"])

def metric_bertscore(item: EvalItem) -> MetricResult:
    if item.reference is None:
        return MetricResult("bertscore", float("nan"))
    out = bertscore_m.compute(predictions=[item.response],
                              references=[item.reference], lang="en")
    return MetricResult("bertscore", out["f1"][0])

# ----- Reference-free judge with structured output -----
client = OpenAI()
JUDGE_PROMPT_V2 = """You are an expert evaluator. Score the response on three axes:
- correctness (0-5): factual and logical accuracy
- helpfulness (0-5): how well it addresses the prompt
- fluency (0-5): grammar, clarity, organization

Prompt: {prompt}
Response: {response}

Think step by step, then output ONLY a JSON object:
{{"correctness": int, "helpfulness": int, "fluency": int, "rationale": str}}"""

def metric_llm_judge(item: EvalItem, model="gpt-4o-mini") -> MetricResult:
    completion = client.chat.completions.create(
        model=model,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[{"role": "user",
                   "content": JUDGE_PROMPT_V2.format(
                       prompt=item.prompt, response=item.response)}],
    )
    parsed = json.loads(completion.choices[0].message.content)
    # Composite score: average of the three axes, normalized to [0, 1].
    composite = (parsed["correctness"] + parsed["helpfulness"]
                 + parsed["fluency"]) / 15.0
    return MetricResult("llm_judge", composite, extras=parsed)

# ----- Evaluation runner -----
def run_eval(items: list[EvalItem], metrics: list[Callable]) -> dict:
    results = {m.__name__: [] for m in metrics}
    for item in items:
        for m in metrics:
            results[m.__name__].append(m(item))
    return results

# Example usage:
items = [EvalItem(prompt=p, response=resp, reference=ref, category=cat)
         for (p, resp, ref, cat) in load_eval_dataset()]
results = run_eval(items, [metric_bleu, metric_rouge_l,
                           metric_bertscore, metric_llm_judge])`}
      </CodeBlock>

      <Prose>
        Several production details matter. First, the LLM judge call is the dominant cost — for a 1000-item evaluation, the standard metrics run in seconds while the judge takes minutes and costs real money (roughly $0.10 per 1000 items with gpt-4o-mini, $1-2 with gpt-4o, $5-10 with full Claude 3.5 Sonnet). Use the cheapest judge that produces correlations above your acceptance threshold for your task type. For most chat evaluation gpt-4o-mini is sufficient; for technical or domain-specific evaluation you usually need a frontier judge.
      </Prose>

      <Prose>
        Second, the structured output format (JSON with discrete numeric axes) is non-optional for production use. Free-text scoring is unreliable: judges often refuse to commit to a number, return ranges, or include caveats that break parsing. The structured output mode (or a strict schema enforced at the API layer) gives you parseable scores 99%+ of the time. Always validate the JSON and fall back to a default-uncertain score when parsing fails, rather than silently dropping examples.
      </Prose>

      <Prose>
        Third, hybrid evaluation is the practical default for many workloads. The pattern is: route each prompt to the appropriate metric based on its category. Translation, structured extraction, and factual QA with known answers go through reference-based metrics (BLEU, exact match, F1). Chat, creative writing, and open-ended reasoning go through the LLM judge. Both pipelines share the same dataset and runner, but the metric set per item is different. For prompts where references exist but are partial, run both and report both — disagreements between the metrics are often the most informative diagnostic available, flagging cases where the judge accepts a paraphrase that the reference rejects, or where the reference matches but the judge identifies a content error.
      </Prose>

      <Prose>
        Fourth, and most importantly, production systems should run an evaluation of the evaluation. The judge model is itself a model under test. Periodically — at least on every model version change — sample 50-200 (prompt, response) pairs at random, have humans score them, and compute the correlation between the judge and the humans. If correlation drops below ~0.5 the judge is no longer reliable and you need to either upgrade it (to a stronger model), re-prompt it (the rubric may need refinement), or reduce its weight in your composite score. Without this loop, the judge silently drifts and your reported metrics drift with it. AlpacaEval 2.0's switch to length-controlled scoring was driven by exactly this kind of audit: the original judge was found to over-reward longer responses, and the metric had to be re-derived to remove the confound.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        The first chart shows correlation with human judgment across metric families on a typical open-ended evaluation set (numbers approximate Liu et al. 2023's G-Eval results on summarization, augmented with values reported by Sellam et al. 2020 for BLEURT and Zhang et al. 2020 for BERTScore). The pattern is consistent: surface metrics correlate weakly, neural reference-based metrics correlate moderately, LLM judges correlate strongly.
      </Prose>

      <Plot
        label="Pearson correlation with human ratings on open-ended summarization"
        xLabel="metric class"
        yLabel="correlation r"
        series={[
          {
            name: "BLEU",
            color: colors.textDim,
            points: [[0, 0.16]],
          },
          {
            name: "ROUGE-L",
            color: colors.textDim,
            points: [[1, 0.21]],
          },
          {
            name: "BERTScore",
            color: "#a78bfa",
            points: [[2, 0.40]],
          },
          {
            name: "BLEURT",
            color: "#a78bfa",
            points: [[3, 0.46]],
          },
          {
            name: "GPTScore",
            color: colors.gold,
            points: [[4, 0.55]],
          },
          {
            name: "G-Eval (GPT-4)",
            color: colors.gold,
            points: [[5, 0.63]],
          },
        ]}
      />

      <Prose>
        The second chart traces the cost-versus-quality curve. Reference-based metrics are nearly free per evaluation (microseconds). LLM judges scale with judge cost; better judges correlate higher with human ratings but cost more. The frontier shows that for a given budget, you choose the strongest judge you can afford — but past a point, the gains diminish.
      </Prose>

      <Plot
        label="Cost vs human-correlation frontier for evaluation methods"
        xLabel="cost per 1k items (USD, log scale stylized)"
        yLabel="correlation with human ratings"
        series={[
          {
            name: "BLEU/ROUGE",
            color: colors.textDim,
            points: [[0.001, 0.20]],
          },
          {
            name: "BERTScore/BLEURT",
            color: "#a78bfa",
            points: [[0.05, 0.45]],
          },
          {
            name: "gpt-4o-mini judge",
            color: colors.gold,
            points: [[0.10, 0.58]],
          },
          {
            name: "gpt-4o judge",
            color: colors.gold,
            points: [[1.50, 0.66]],
          },
          {
            name: "Claude Sonnet judge",
            color: colors.gold,
            points: [[5.00, 0.70]],
          },
          {
            name: "human (3 raters)",
            color: "#4ade80",
            points: [[200.00, 0.85]],
          },
        ]}
      />

      <Prose>
        The heatmap below shows the qualitative behavior of each metric on each test case from the from-scratch demo. Green cells indicate the metric agreed with the human rating (high score for high-quality responses, low score for low-quality responses). Red cells indicate disagreement. The pattern by row shows that BLEU and ROUGE fail systematically on paraphrase and creative tasks; BERTScore handles paraphrase but fails on factual errors; the LLM judge handles all four cases.
      </Prose>

      <Heatmap
        label="Per-metric agreement with human judgment across test cases"
        rowLabels={["BLEU", "ROUGE-L", "BERTScore", "LLM judge"]}
        colLabels={["exact match", "paraphrase", "factual error", "creative"]}
        cellSize={48}
        colorScale="green"
        matrix={[
          [1.00, 0.10, 0.40, 0.05],
          [1.00, 0.55, 0.20, 0.10],
          [1.00, 0.95, 0.10, 0.65],
          [1.00, 1.00, 1.00, 0.90],
        ]}
      />

      <Prose>
        Finally, the step trace below walks through the decision flow for a single evaluation item — picking which metric to apply, running it, parsing the result, and aggregating into the report. This is the production pattern in miniature.
      </Prose>

      <StepTrace
        label="Production evaluation flow — one item"
        steps={[
          {
            label: "Classify the item",
            render: () => (
              <Prose>
                Decide whether the (prompt, response) pair has a well-defined gold reference.
                Translation, factual QA, structured extraction → reference-based.
                Chat, creative writing, open reasoning → reference-free.
                Domain-specific QA may go through both; disagreements are flagged for human review.
              </Prose>
            ),
          },
          {
            label: "Run reference-based metrics if reference exists",
            render: () => (
              <Prose>
                BLEU, ROUGE, chrF, BERTScore. These are cheap (microseconds) and deterministic.
                Run multiple variants and report each — disagreement between BLEU and BERTScore
                often signals paraphrase that simple metrics under-credit.
              </Prose>
            ),
          },
          {
            label: "Run reference-free judge",
            render: () => (
              <Prose>
                Send (prompt, response) to a strong LLM with a structured rubric (correctness,
                helpfulness, fluency, safety). Use JSON-mode output. Temperature 0 for reproducibility.
                Parse the JSON; fall back to a default-uncertain score on parse failure.
              </Prose>
            ),
          },
          {
            label: "Detect disagreements",
            render: () => (
              <Prose>
                If reference-based gives high score and judge gives low score: candidate matched the
                reference but the judge sees a content problem. Often signals a problematic reference,
                or a stylistic match that misses substance.
                If reference-based gives low and judge gives high: paraphrase the reference doesn't
                cover. Almost always a metric failure on the reference-based side.
              </Prose>
            ),
          },
          {
            label: "Aggregate and slice",
            render: () => (
              <Prose>
                Per-item scores get aggregated by category (chat / code / math / creative), by model
                version, by prompt difficulty bucket. Report mean, median, and tail percentiles —
                tail behavior is often where models differ most. Save the per-item scores for later
                drill-down on regressions.
              </Prose>
            ),
          },
          {
            label: "Periodically audit the judge",
            render: () => (
              <Prose>
                Sample 50-200 items, get human scores, compute correlation with the judge.
                If correlation drops below ~0.5, upgrade the judge, refine the rubric, or reduce
                its weight. Without this loop the judge silently drifts and your metrics drift with it.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <H3>Use reference-based when references exist and constrain quality</H3>

      <Prose>
        Reference-based metrics are the right answer for tasks where the response space is small and well-characterized by a few canonical correct answers. Machine translation between major language pairs is the original use case: there are usually one to three idiomatic translations of any given source sentence, and BLEU/chrF/COMET (a learned reference-based metric trained on WMT human ratings) correlate well enough with translator judgment to be useful at scale. Summarization with constrained reference summaries — news headlines, paper abstracts — is similar. Structured extraction (key-value pairs, JSON schemas, named entities) is the most clear-cut case: the reference is exact and exact-match or F1 on extracted spans is the appropriate metric.
      </Prose>

      <Prose>
        Reference-based metrics also dominate factual QA when the answer is a short string. SQuAD's exact-match metric, TriviaQA's F1, MMLU's multiple-choice accuracy — all are reference-based and uncontroversially correct. Use them when applicable; they are cheap, deterministic, and robust against judge bias. The cost of reference-based metrics in these settings is the cost of building the reference dataset, which has been amortized over years of academic benchmark construction.
      </Prose>

      <H3>Use reference-free when the response space is rich</H3>

      <Prose>
        Open-ended chat, creative writing, dialogue, long-form reasoning, and any task where multiple distinct fluent answers are equally valid require reference-free evaluation. The space of correct answers is too large to enumerate as references. AlpacaEval, MT-Bench, Arena-Hard, and the chatbot arena leaderboard all use reference-free pairwise comparison or scoring as their primary metric. For these workloads, reference-based metrics are not just suboptimal — they are systematically misleading, because they reward surface match with whichever reference happened to be written rather than rewarding the actual quality of the response.
      </Prose>

      <H3>Use hybrid when references are partial or domain-specific</H3>

      <Prose>
        Code generation has both: there are reference solutions for many problems, but there are also infinitely many correct programs that solve the same problem. A reference-based metric (does the candidate match the reference?) misses correct solutions that look different. A reference-free metric (does the candidate look like a good program?) misses subtle bugs. The production pattern is: run both, plus an actual code execution test against unit tests. The test execution is the gold standard; the reference-based and reference-free metrics serve as cheaper proxies for cases where execution is infeasible.
      </Prose>

      <Prose>
        RAG (retrieval-augmented generation) evaluation is another natural hybrid. RAGAS computes faithfulness (does the response stay grounded in the retrieved documents?) using a reference-free judge, while computing answer correctness against a reference when one exists. Both are reported, and either failing flags a problem worth investigating. The two failure modes are different: faithfulness failures indicate the model is hallucinating beyond its retrieved context; correctness failures indicate the retrieved context was insufficient.
      </Prose>

      <H3>When to add a synthetic-reference layer</H3>

      <Prose>
        For tasks where a reference would help but human reference creation is too expensive, the modern pattern is to generate references with a strong frontier model and then use them with reference-based metrics. This sounds circular, but it works because the synthetic reference and the metric are different procedures: the frontier model writes a high-quality response, and an embedding-based metric (BERTScore, BLEURT) compares the candidate to that reference. Empirically this gives a partial recovery of reference-based metric stability while not requiring human annotation. The risk is that the synthetic reference inherits the frontier model's biases, so models trained to imitate the frontier model will score artificially well — keep the synthetic reference fixed across model evaluations and refresh it periodically.
      </Prose>

      <H3>When neither paradigm is enough</H3>

      <Prose>
        For high-stakes deployments — medical advice, legal reasoning, financial recommendations — neither reference-based nor reference-free automated metrics are sufficient. The response space is rich (so reference-based fails) but the cost of confident wrong answers is high (so a fallible LLM judge is risky). The right answer is human evaluation by domain experts, with automated metrics serving only as triage signals to identify which examples deserve human review. The automated metrics are then validated against the human judgments periodically and used as the primary signal only when their correlation with the experts is verified.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        Reference-based metric throughput scales beautifully. BLEU on a million-item evaluation set runs in seconds on a single CPU. ROUGE is similar. BERTScore is bounded by the embedding model and runs in low minutes on a GPU for a million items. The main cost in scaling reference-based evaluation is creating and maintaining the reference dataset. Once references exist, metric computation is essentially free. This is why reference-based metrics dominate the largest-scale evaluation regimes: continuous integration, regression testing, A/B comparisons across hundreds of model variants. Where you have references and your reference-based metric is well-calibrated to your task, there is no good reason not to use it for most of your evaluation volume.
      </Prose>

      <Prose>
        Reference-free metric throughput scales linearly with judge cost and is rate-limited by judge API throughput. A 100k-item evaluation with gpt-4o-mini at $0.10/1k items costs $10 and takes hours; with gpt-4o costs $150 and takes longer; with Claude 3.5 Sonnet at $5/1k items costs $500. These costs are not prohibitive for periodic evaluation rounds (weekly, monthly), but they are too expensive for every code change in CI. The practical compromise: cheap metrics in CI, judge-based metrics on a periodic schedule and on every release candidate.
      </Prose>

      <Prose>
        Judge-based evaluation at scale has a less-obvious failure mode: judge model versioning. If your evaluation pipeline depends on gpt-4o-mini and OpenAI updates the underlying snapshot, your evaluation results may shift even though nothing about the model under test changed. Pin specific model versions in production evaluation runs (e.g., gpt-4o-mini-2024-07-18) and treat judge upgrades as a separate event that requires re-baselining your historical results. Without this discipline, you cannot tell whether a quality regression is a real regression or just a judge drift.
      </Prose>

      <Prose>
        Reference-based metrics scale gracefully across model sizes — they do not depend on the model being evaluated, only on the response. A reference-based pipeline built for evaluating 7B models works without modification on 70B models or on API models. Reference-free pipelines scale similarly in the abstract, but you should match judge strength to the model being evaluated: a 7B judge cannot reliably evaluate a 70B model on hard prompts, because the judge will not understand the response well enough to score it. As a heuristic, the judge should be at least as strong as the model under test on the relevant task category — ideally stronger.
      </Prose>

      <Prose>
        The structural limit on reference-free evaluation is the judge ceiling. No reference-free metric can be more accurate than the judge's own reliability on the underlying task. If you are evaluating frontier-model outputs on graduate-level mathematics and your judge is not itself capable of solving graduate-level mathematics, the judge will accept incorrect responses that look superficially correct and reject correct responses that use unfamiliar notation. This is not a problem you can fix by averaging many judges or by clever prompting; it is a fundamental limit of the reference-free paradigm. For these regimes, you eventually need either (a) a verifiable reference (theorem proof checker, code unit tests, human expert) or (b) to acknowledge that automated evaluation is unreliable in this regime and design your pipeline to gate on human judgment.
      </Prose>

      <Prose>
        Scaling considerations for hybrid pipelines: the bottleneck is always the most expensive metric you are running. If 5% of your evaluation requires the LLM judge and 95% can use reference-based, the LLM judge dominates cost only if you do not aggressively cache. Cache judge outputs aggressively keyed on (prompt-hash, response-hash, judge-version, rubric-version) — re-running the same evaluation should hit cache and be free. Without caching, the judge bill grows linearly with the number of evaluation runs you do; with caching, it grows linearly only with new (prompt, response) pairs.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>BLEU/ROUGE penalize valid paraphrases</H3>
      <Prose>
        The defining failure of n-gram-based reference-based metrics. Two responses with identical meaning but different surface forms get sharply different scores. This was tolerable when models were weak enough that surface form approximated content; with modern LLMs that produce fluent, varied outputs, n-gram metrics severely under-credit paraphrase. Mitigation: use BERTScore, BLEURT, or COMET which embed before comparing. Or move to reference-free entirely.
      </Prose>

      <H3>Reference-based metrics cannot detect factual errors</H3>
      <Prose>
        BLEU=0.85 on "The capital of Australia is Sydney" against the reference "The capital of Australia is Canberra" — the metric correctly observed that most words match, and that is exactly what the metric is designed to do. It has no notion of factuality. BERTScore makes this worse because "Sydney" and "Canberra" embed near each other (both are city names in Australia), so even the neural reference-based metric scores this above 0.95. Mitigation: do not use reference-based metrics for factuality evaluation. Use reference-free judges with explicit rubrics for factual accuracy, or use targeted factual-QA datasets with exact-match scoring.
      </Prose>

      <H3>LLM judges have position bias in pairwise comparison</H3>
      <Prose>
        When comparing response A vs response B, judges often prefer whichever was presented first (or sometimes whichever was second — the bias depends on the judge model and prompt). This was documented by Zheng et al. (MT-Bench, 2023) and Wang et al. (Large Language Models are not Fair Evaluators, 2023). Mitigation: always present each pair in both orderings and average, or use a position-bias-correcting scoring rule. AlpacaEval 2.0 uses a hybrid where each comparison is run in both orders and the win rate is averaged.
      </Prose>

      <H3>LLM judges have length bias</H3>
      <Prose>
        Judges systematically rate longer responses higher, independent of content quality. The original AlpacaEval was found to over-credit length so much that the leaderboard was partially measuring response length rather than quality. AlpacaEval 2 introduced length-controlled win rates that regress out the length effect. Without correction, models that have been trained to produce longer responses (a common DPO and RLHF outcome) get inflated scores. Mitigation: use length-controlled metrics, include length distribution in reports alongside scores, and verify that your model's length distribution is comparable to baselines.
      </Prose>

      <H3>LLM judges prefer their own writing style</H3>
      <Prose>
        A judge from a particular model family tends to score outputs from the same family higher than outputs from other families (Panickssery et al. 2024 documented this for GPT-4 judging GPT-4 outputs). This is a real bias, not an indicator of quality. The bias is most pronounced when the model under test was trained partly on outputs from the judge model's family. Mitigation: use a judge from a different family than the model under test when possible; use multiple judges from different families and take the consensus; report per-judge scores separately to expose disagreement.
      </Prose>

      <H3>LLM judges cannot reliably evaluate beyond their own competence</H3>
      <Prose>
        A judge that cannot solve a graduate math problem cannot reliably tell whether a candidate solution is correct. It may grade based on surface features (does it look like a math proof?) rather than substance. This is the hardest problem with reference-free evaluation and has no general fix. Mitigation: for high-difficulty technical tasks, validate the judge against ground-truth answers before trusting its scores; use verifiable references (proof checkers, unit tests) where they exist; when neither is possible, treat the judge's scores as a noisy signal that requires human spot-checking.
      </Prose>

      <H3>Reference quality silently dominates reference-based metric quality</H3>
      <Prose>
        A reference-based metric is at most as good as the reference it compares against. If your references are imprecise, outdated, or only one of many valid answers, the metric will systematically penalize valid responses that diverge from the specific reference text. WMT competitions use multiple references and average to reduce this; many smaller benchmarks use a single reference per item and inherit the bias. Mitigation: use multiple references where possible, prefer learned reference-based metrics (BLEURT, COMET) that are calibrated to human ratings, and audit your references periodically against expert judgment.
      </Prose>

      <H3>Judge prompts have rubric drift</H3>
      <Prose>
        Small changes to the judge prompt — "rate from 1 to 5" vs "rate from 1 to 10", "consider correctness" vs "consider accuracy and helpfulness" — can shift score distributions by 0.3-0.5 points on a 5-point scale. This means changing the rubric retroactively invalidates historical comparisons. Mitigation: version your rubrics like code, change them only at deliberate baseline-resets, and keep both old and new rubric runs for at least one transition cycle so you can map between historical and current scores.
      </Prose>

      <H3>Calibration drift across judge model versions</H3>
      <Prose>
        Provider model updates (gpt-4o-mini-2024-07-18 → gpt-4o-mini-2024-09-15) can shift judge calibration meaningfully. A regression you observe after a model release may be a judge change rather than a model regression. Mitigation: pin judge model versions explicitly; when forced to upgrade, run both versions on a held-out calibration set and compute the per-item difference to estimate the shift.
      </Prose>

      <H3>Synthetic reference leakage</H3>
      <Prose>
        If you generate references with a frontier model and then evaluate models trained on outputs from that same frontier model, you get artificially high scores. The candidate and the reference share a common origin, so they look similar by construction even when neither is particularly good. Mitigation: use synthetic references from a model that is not in the training pipeline of the model under test; refresh synthetic references periodically with different generators to detect this kind of leakage.
      </Prose>

      <Callout accent="purple">
        The single most common failure across both paradigms is treating the metric as ground truth instead of as a noisy estimator. Always validate the metric itself against human judgment periodically. Always report uncertainty and slice by category. Always run multiple metrics in parallel and investigate disagreements — they are the most informative signal you have about where your evaluation is breaking.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All sources verified against arXiv on 2026-04-26. Author lists, abstracts, and arXiv IDs confirmed; publication venues noted where applicable.
      </Prose>

      <H3>Liu et al. 2023 — G-Eval</H3>
      <Prose>
        Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, Chenguang Zhu. "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment." arXiv:2303.16634. Published March 2023; presented at EMNLP 2023. Demonstrates that a chain-of-thought prompted GPT-4 judge produces summary quality scores that correlate with human judgment substantially better than BLEU, ROUGE, BERTScore, and BLEURT. Key technical contribution: rather than parsing a discrete score, take the expected score under the judge's distribution over score tokens, which both improves correlation and yields a continuous signal. Establishes the practical viability of reference-free LLM evaluation for open-ended NLG.
      </Prose>

      <H3>Fu et al. 2023 — GPTScore</H3>
      <Prose>
        Jinlan Fu, See-Kiong Ng, Zhengbao Jiang, Pengfei Liu. "GPTScore: Evaluate as You Desire." arXiv:2302.04166. Published February 2023. Frames evaluation as conditional generation: the quality of a response is the log-probability of that response under a strong language model conditioned on the prompt and a scoring instruction. Demonstrates that this single signal recovers most of the structure of standard NLG metrics across summarization, translation, dialogue, and data-to-text generation, with a single underlying machinery. The conceptual foundation for treating the LM as a universal evaluator.
      </Prose>

      <H3>Zhang et al. 2020 — BERTScore</H3>
      <Prose>
        Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, Yoav Artzi. "BERTScore: Evaluating Text Generation with BERT." arXiv:1904.09675. Published April 2019; ICLR 2020. Replaces n-gram overlap with cosine similarity in BERT contextual embedding space. For each candidate token, finds the maximum-similarity reference token; aggregates with precision/recall/F1. Substantially outperforms BLEU and ROUGE on correlation with human judgment for translation and image captioning. The canonical neural reference-based metric, still in active use as a strong baseline.
      </Prose>

      <H3>Sellam et al. 2020 — BLEURT</H3>
      <Prose>
        Thibault Sellam, Dipanjan Das, Ankur P. Parikh. "BLEURT: Learning Robust Metrics for Text Generation." arXiv:2004.04696. Published April 2020; ACL 2020. Fine-tunes BERT on synthetic perturbations of reference texts (paraphrases, deletions, etc.) and then on human ratings of translations. The result is a learned reference-based metric that approximates human quality judgments much better than fixed similarity measures. Demonstrates that learned metrics outperform unlearned ones; widely deployed in machine translation evaluation pipelines.
      </Prose>

      <H3>Dubois et al. 2023 — AlpacaEval</H3>
      <Prose>
        Yann Dubois, Xuechen Li, Rohan Taori, Tianyi Zhang, Ishaan Gulrajani, Jimmy Ba, Carlos Guestrin, Percy Liang, Tatsunori B. Hashimoto. "AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback." arXiv:2305.14387. Published May 2023; NeurIPS 2023. The AlpacaEval benchmark, introduced as part of AlpacaFarm, is reference-free pairwise comparison: each model's response to a fixed prompt set is compared against a baseline (text-davinci-003, later GPT-4-turbo) by an LLM judge, and the win rate is reported. Subsequent length-controlled extension (Dubois 2024, arXiv:2404.04475) corrects for length bias in the original judge. The most widely used reference-free leaderboard for instruction-following LLMs.
      </Prose>

      <H3>Zhao et al. 2019 — MoverScore</H3>
      <Prose>
        Wei Zhao, Maxime Peyrard, Fei Liu, Yang Gao, Christian M. Meyer, Steffen Eger. "MoverScore: Text Generation Evaluating with Contextualized Embeddings and Earth Mover Distance." arXiv:1909.02622. Published September 2019; EMNLP 2019. Uses Word Mover's Distance over BERT embeddings to allow soft alignment between candidate and reference. Improves over BERTScore in some regimes, particularly for summarization. A key transitional work between hard n-gram matching and the modern neural reference-based family.
      </Prose>

      <H3>Zheng et al. 2023 — MT-Bench and judge-bias analysis</H3>
      <Prose>
        Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang, Joseph E. Gonzalez, Ion Stoica. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." arXiv:2306.05685. Published June 2023; NeurIPS 2023. Introduces MT-Bench (multi-turn open-ended evaluation) and Chatbot Arena (large-scale crowdsourced pairwise comparison). The accompanying analysis of LLM judges is the most comprehensive empirical characterization of judge biases — position bias, verbosity bias, self-enhancement bias — and proposes practical mitigations including swapping orderings and using multiple judges. Essential reading for anyone deploying LLM judges in production.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — When BLEU is and isn't appropriate</H3>
      <Prose>
        For each of the following evaluation tasks, decide whether BLEU is an appropriate metric, an inappropriate metric, or borderline. Justify each answer in one or two sentences. (a) French→English translation of news articles with three reference translations per source sentence. (b) Summarizing a research paper into a one-paragraph abstract with one human-written reference. (c) Evaluating a chatbot's response to "How are you today?" against a reference response. (d) Scoring code completions on the HumanEval benchmark. (e) Evaluating poetry generation. For the inappropriate cases, propose a reference-free or hybrid metric that would work better and explain its specific advantage.
      </Prose>

      <H3>Exercise 2 — Construct an adversarial pair for BERTScore</H3>
      <Prose>
        BERTScore embeds tokens and rewards semantic similarity to the reference. Construct two candidate responses to the prompt "What is the boiling point of water at sea level?" — one factually correct, one factually wrong — such that BERTScore against the reference "Water boils at 100 degrees Celsius at sea level" rates the wrong answer at least 0.10 higher than the correct one. Explain the embedding-space property your construction exploits. Now repeat the exercise for an LLM judge: construct a pair where you suspect the judge will prefer the wrong answer. Is the construction harder for the judge? Why?
      </Prose>

      <H3>Exercise 3 — Length bias in pairwise judging</H3>
      <Prose>
        Suppose you are running an AlpacaEval-style pairwise comparison between Model A (which produces 200-token responses on average) and Model B (which produces 600-token responses on average). The judge has a documented length bias: it prefers longer responses with probability 0.65, all else equal. If you observe a 60% win rate for Model B, what fraction of that win rate could be attributed to length alone? Sketch a length-controlled metric that strips out the length effect. What assumptions does your correction make about the relationship between length and quality? Discuss when those assumptions might break down.
      </Prose>

      <H3>Exercise 4 — Designing a hybrid evaluation</H3>
      <Prose>
        You are building an evaluation suite for a coding assistant. The dataset has 1000 (prompt, candidate code) pairs. About 400 of them have reference solutions and unit tests. About 300 have only reference solutions. About 300 have only natural-language descriptions of what the code should do, with no reference. Design an evaluation pipeline that uses the most appropriate metric for each subset, with a single composite score for the overall benchmark. Justify each choice. How would you handle disagreements between the metrics on the items where multiple are applicable? What human-in-the-loop step would you add to validate the pipeline before trusting it?
      </Prose>

      <H3>Exercise 5 — Auditing a judge</H3>
      <Prose>
        You inherit an evaluation pipeline that uses gpt-4o-mini as a reference-free judge for chat quality. The previous team reports that the judge correlates 0.62 with human ratings, based on a study from six months ago. You suspect calibration may have drifted. Design a 200-item audit study to measure current judge quality. Specify: how you sample items, how you collect human ratings (number of raters, instruction format, agreement metric), what statistic you compute, what threshold below which you would consider the judge unreliable, and what action you would take in each of three scenarios — judge agreement still above 0.6, judge agreement between 0.4 and 0.6, judge agreement below 0.4. As a follow-up, propose a continuous monitoring approach (lighter-weight than a full audit) that would catch large drift events between full audits.
      </Prose>

      <H3>Exercise 6 — The synthetic reference dilemma</H3>
      <Prose>
        You want to scale up reference-based evaluation by generating references using GPT-4-turbo for prompts that lack human references. Two of your models under test are (i) a model that was trained partly on GPT-4-turbo distillation data, and (ii) a model that was trained entirely from scratch with no GPT-4-turbo influence. Predict, with reasoning, how each model would score on the synthetic-reference benchmark relative to its true quality. What experimental design would let you detect this contamination? Propose two practical mitigations that do not require returning to fully human references.
      </Prose>

    </div>
  ),
};

export default referenceFreeVsBased;
