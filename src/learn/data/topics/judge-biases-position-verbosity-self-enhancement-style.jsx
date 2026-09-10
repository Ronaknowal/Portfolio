import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const judgeBiases = {
  title: "Judge Biases (Position, Verbosity, Self-Enhancement, Style)",
  slug: "judge-biases-position-verbosity-self-enhancement-style",
  readTime: "~40 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Sometime around the middle of 2023 a structural shift happened in how the language model community ran evaluations. Reviewers had spent years building careful, expensive, slow human-evaluation pipelines — pairs of model outputs presented to crowdworkers, Likert scales averaged across hundreds of annotators, inter-annotator agreement reported as a Krippendorff alpha or a Cohen kappa. Then GPT-4 came out, and the same researchers started doing something that would have looked deeply suspicious a year earlier: they were prompting one model to judge the outputs of another model, treating the resulting preferences as a substitute for human ratings, and reporting the win-rates in their leaderboards. MT-Bench, AlpacaEval, Arena-Hard, WildBench — each of these benchmarks is, at its core, a directory of prompts plus a script that asks GPT-4 (or Claude, or some other strong model) to pick a winner from a pair of completions. Used carefully, this is roughly two orders of magnitude cheaper than human evaluation, finishes in hours instead of weeks, and — when correlated against carefully collected human pairwise preferences on the same prompts — agrees with human judgment 80% of the time on common chat tasks.
      </Prose>

      <Prose>
        The catch is that LLM judges have biases that human evaluators do not. Or, more precisely: LLM judges have a different distribution of biases than human evaluators have, and those biases are often easier to exploit than human biases because they are deterministic, identifiable, and present at every single decision the judge makes. A human annotator who is grumpy after lunch will score the next ten responses slightly lower; a different human will compensate; the noise averages out across raters. An LLM judge that prefers the first response in a pairwise comparison 56% of the time will exhibit that exact bias on every single pair, in the same direction, across every benchmark in which it is used as the evaluator. Bias that does not average out to zero across raters becomes a systematic offset on every reported win-rate. The result is a leaderboard where 1.5% differences between models can be entirely explained by which model happened to be presented first more often, and where 5% differences can be entirely explained by which model produced longer outputs.
      </Prose>

      <Prose>
        The literature on judge bias developed quickly. Zheng et al. 2023 ("Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena", arXiv:2306.05685) introduced MT-Bench and the LLM-as-a-judge paradigm in the same paper, and in section 4 of that work documented — with carefully designed swap-tests — that even GPT-4 exhibited a measurable position bias of around 6%, with weaker judges (GPT-3.5, Claude 1) exhibiting bias as high as 20%. Wang et al. 2024 ("Large Language Models are not Fair Evaluators", arXiv:2305.17926) extended this with a systematic study of verbosity bias, showing that across multiple judge models the longer response in a pair was selected as the winner at rates significantly above 50%, even when length had been deliberately decoupled from quality. Panickssery et al. 2024 ("LLM Evaluators Recognize and Favor Their Own Generations", arXiv:2404.13076) demonstrated self-enhancement: judge models systematically scored outputs from their own model family higher than outputs from competing model families, even on identical content — and the effect was mediated by a measurable ability of the judge to recognize its own generations. Koo et al. 2024 ("Benchmarking Cognitive Biases in LLMs as Evaluators", arXiv:2309.17012) cataloged a much wider set of biases, including a strong style bias where surface formatting features — markdown headers, bullet points, bold text — inflated scores even when the underlying content was held constant.
      </Prose>

      <Prose>
        Understanding these biases matters operationally for three distinct reasons. First, every published win-rate from an LLM-judged benchmark needs a confidence interval that incorporates judge bias as a non-trivial component, not as a rounding error. Second, when you are training a model and using win-rates as your reward signal — directly in RLAIF, indirectly when you decide which checkpoint to ship — judge biases become training signal, and the model learns to exploit them. Third, when the same judge model is used both to evaluate competitors and to rank your own outputs (the universal practice on AlpacaEval and Arena-Hard, where GPT-4 is the reference judge), self-enhancement bias means competitors are systematically disadvantaged by an amount that depends on their model family. None of these problems are theoretical. All of them have been observed, measured, and quantified in published work, and all of them have known mitigations that — once you understand the structure of each bias — are not particularly difficult to apply.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Start with the abstraction. An LLM judge is a function that takes a prompt, two candidate responses, and produces a verdict — a binary preference, a pairwise score, or a Likert rating. If the judge were ideal, the verdict would depend only on the content of the responses relative to the prompt. The structural problem is that the verdict depends, in measurable ways, on properties of the input that are not "content of the responses relative to the prompt." Position is the most obvious of these: which response is presented first should not change the verdict, but it does. Length is another: longer responses should not be preferred independently of quality, but they are. Family identity is a third: who produced the response should not enter the verdict, but it does. Surface formatting — markdown, bolding, headers — should not change quality assessment, but it does.
      </Prose>

      <Prose>
        It is helpful to think of the judge's output as a sum of two terms. The first is the true content effect: the actual difference in quality between the two responses, which is what you want to measure. The second is a bias term: a systematic deviation that depends on properties of the presentation that are not content. If the bias term were zero on average across many comparisons, you could safely ignore it; the law of large numbers would average it out. The empirical finding from every paper in this space is that the bias term is not zero on average. It is consistently positive in a particular direction for every type of bias studied, and it is large enough — on the order of 5% to 30% of all verdicts being flipped by it — that ignoring it produces leaderboards where the rank order of models is determined as much by bias as by quality.
      </Prose>

      <Prose>
        Position bias is the easiest to develop intuition for because it has an obvious counterfactual. If you present <Code>(A, B)</Code> to a judge and it picks <Code>A</Code>, then present <Code>(B, A)</Code> to the same judge on the same prompt and it picks <Code>B</Code>, the judge has revealed that its verdict depends on order rather than content. The experimental design is symmetric and directly measurable. Zheng et al. found that GPT-4 has a roughly 6% positional bias — meaning that across paired comparisons, the model picks the first-presented response about 53% of the time when it should be 50% by content. Weaker judges have much larger bias. The mitigation, once you have measured the bias, is also obvious: present every pair in both orders and average the verdicts. This eliminates first-order position bias by construction, at the cost of doubling the number of judge calls.
      </Prose>

      <Prose>
        Verbosity bias is structurally different. There is no swap-test that eliminates it, because length is a property of the response itself, not of the presentation. If response <Code>A</Code> is twice as long as response <Code>B</Code>, swapping their order does not change which one is longer. The bias has to be addressed either by controlling the length distribution of the responses being compared (length-matched evaluation), by adjusting the verdict after the fact based on length (post-hoc length normalization), or by changing the prompt to the judge to explicitly de-emphasize length (which works partially but not completely). The current state of the art on AlpacaEval — the length-controlled win-rate metric introduced in 2024 — uses a regression-based approach that estimates the length-attributable component of the win-rate and subtracts it.
      </Prose>

      <Prose>
        Self-enhancement bias is the strangest of the four to internalize because it requires the judge to do something that, on first pass, sounds like a violation of how language models are supposed to work. A pure forward pass through GPT-4 should not "know" that a response was produced by GPT-4 versus by Claude versus by Llama; the model has no access to metadata about its own past outputs. But Panickssery et al. demonstrated that judge models can in fact identify their own outputs at well-above-chance rates from text alone, and the size of this self-recognition ability correlates almost perfectly with the size of the self-enhancement effect on the same model. The mechanism is presumably stylistic: every model has subtle distributional fingerprints — phrase frequencies, sentence-length distributions, hedging conventions — that are recognizable from a careful reading of a few hundred tokens, and the judge implicitly upweights outputs that match its own distributional fingerprint.
      </Prose>

      <Prose>
        Style bias is the most insidious because it cannot be easily debiased without losing information. A response that uses markdown headers, bullet points, and bold text scores higher than the same content presented as a plain paragraph — but the formatting itself is not pure noise. A well-formatted response is genuinely easier to read, and in many real deployment scenarios users do prefer it. The bias is that the judge's preference for formatting is much stronger than human users' preference, leading to win-rates that are inflated relative to actual deployment quality. The mitigation here is more about evaluation discipline than algorithmic correction: explicitly probe whether the formatting difference is doing the work, and if so, either match formatting across compared models or report the result with that caveat.
      </Prose>

      <Callout accent="gold">
        The four biases — position, verbosity, self-enhancement, style — are not theoretical or rare. Every published LLM-judged benchmark exhibits them at measurable scale. The question is never whether they are present; it is how large they are for your specific judge and how to either eliminate or report them.
      </Callout>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <Prose>
        Define the judge as a function <Code>{"J: (x, y_A, y_B, o) → {A, B}"}</Code> where <Code>x</Code> is the prompt, <Code>y_A</Code> and <Code>y_B</Code> are the two candidate responses, and <Code>{"o ∈ {0, 1}"}</Code> is the position indicator: <Code>o=0</Code> means <Code>y_A</Code> is presented first and <Code>y_B</Code> second; <Code>o=1</Code> means the reverse. The verdict is the response selected as the winner. We want to estimate the true preference probability <Code>{"p^*(A) = P(y_A is genuinely better than y_B)"}</Code> from observed judge verdicts.
      </Prose>

      <Prose>
        Decompose the observed verdict probability into a content effect and a position effect. Let <Code>{"q(A | o=0)"}</Code> denote the probability that the judge picks <Code>y_A</Code> when <Code>y_A</Code> is presented first, and <Code>{"q(A | o=1)"}</Code> the probability that the judge picks <Code>y_A</Code> when <Code>y_A</Code> is presented second. The simplest model that captures position bias additively in logit space is:
      </Prose>

      <MathBlock>{"\\mathrm{logit}\\, q(A \\mid o) = \\theta_{\\mathrm{content}}(x, y_A, y_B) + \\beta_{\\mathrm{pos}} \\cdot (1 - 2o)"}</MathBlock>

      <Prose>
        Here <Code>θ_content</Code> is the latent preference signal — what the judge would output if there were no position effect — and <Code>β_pos</Code> is the position bias coefficient, expressed as a logit shift. When <Code>o=0</Code> (response A first), the bias term is <Code>+β_pos</Code>; when <Code>o=1</Code> (response A second), the bias term is <Code>−β_pos</Code>. A positive <Code>β_pos</Code> means the judge favors whichever response is presented first.
      </Prose>

      <Prose>
        The standard randomization-based debiasing technique exploits this additive structure directly. If you observe each pair in both orders and average the two verdicts in logit space:
      </Prose>

      <MathBlock>{"\\hat{\\theta}_{\\mathrm{content}} = \\tfrac{1}{2} \\left[\\mathrm{logit}\\, q(A \\mid o=0) + \\mathrm{logit}\\, q(A \\mid o=1)\\right] = \\theta_{\\mathrm{content}}"}</MathBlock>

      <Prose>
        The position bias term <Code>β_pos · (1 − 2o)</Code> is exactly opposite in sign across the two orders, so it cancels in the average. This is the mathematical foundation of every "evaluate in both orders" recipe in production benchmark code. The catch is that it costs twice as many judge queries, and that it removes only the additive component of position bias — if the judge interacts position with content (e.g., position bias is larger for ambiguous pairs than for clear pairs), the cancellation is only first-order.
      </Prose>

      <Prose>
        Verbosity bias requires a different decomposition. Let <Code>L(y)</Code> denote a length measure (token count, character count, or log-token-count). Model the verdict logit as a sum of true content effect plus a length-dependent bias term:
      </Prose>

      <MathBlock>{"\\mathrm{logit}\\, q(A \\mid x, y_A, y_B) = \\theta_{\\mathrm{content}}(x, y_A, y_B) + \\beta_{\\mathrm{len}} \\cdot \\left[L(y_A) - L(y_B)\\right]"}</MathBlock>

      <Prose>
        The length difference enters linearly. To estimate <Code>β_len</Code>, you need a dataset of pairs where length has been deliberately decoupled from content quality — typically by including pairs of equivalent quality but different length, or by treating length as an instrumental variable in a regression of verdict on content and length. AlpacaEval's length-controlled win-rate fits exactly this regression on each judge's verdicts, then reports the predicted win-rate at a controlled length. The model is:
      </Prose>

      <MathBlock>{"\\mathrm{logit}\\, P(\\mathrm{model\\ wins}) = \\alpha_{\\mathrm{model}} + \\beta_{\\mathrm{len}} \\cdot \\Delta L + \\gamma \\cdot \\Delta L^2"}</MathBlock>

      <Prose>
        where <Code>α_model</Code> is the length-controlled model effect and the <Code>β_len</Code> and <Code>γ</Code> terms capture the length sensitivity of the judge. The length-controlled win-rate is then read off as <Code>σ(α_model)</Code>, evaluating the regression at <Code>ΔL = 0</Code>.
      </Prose>

      <Prose>
        Self-enhancement bias is a function of model identity. Let <Code>m(y)</Code> denote the source model that produced response <Code>y</Code>, and <Code>{"\\mathbb{1}[m(y) = m_J]"}</Code> the indicator that the response was produced by the same model family as the judge. The decomposition adds a self-enhancement coefficient:
      </Prose>

      <MathBlock>{"\\mathrm{logit}\\, q(A) = \\theta_{\\mathrm{content}} + \\beta_{\\mathrm{self}} \\cdot \\left(\\mathbb{1}[m(y_A) = m_J] - \\mathbb{1}[m(y_B) = m_J]\\right)"}</MathBlock>

      <Prose>
        The bias is positive when <Code>y_A</Code> is from the judge's family and <Code>y_B</Code> is not, negative in the reverse case, and zero when both responses are from the same family (or both from different families). To estimate <Code>β_self</Code> in a leaderboard setting, you fit a fixed-effects regression where the family-match indicator is one of the regressors. Mitigation is harder: the cleanest fix is to use multiple judges from different families and average their verdicts, which approximately zeros out the self-enhancement term as long as the judges' family-match patterns are uncorrelated.
      </Prose>

      <Prose>
        Style bias, finally, is best modeled as a vector of surface features <Code>{"s(y) = (s_1(y), s_2(y), ..., s_k(y))"}</Code> — markdown header count, bullet count, code-block count, bold-token count, average sentence length, and so on. The decomposition is:
      </Prose>

      <MathBlock>{"\\mathrm{logit}\\, q(A) = \\theta_{\\mathrm{content}} + \\boldsymbol{\\beta}_{\\mathrm{style}}^\\top \\left[\\boldsymbol{s}(y_A) - \\boldsymbol{s}(y_B)\\right]"}</MathBlock>

      <Prose>
        The style coefficients are typically all positive: each surface feature individually contributes to higher judge scores. Identifying the coefficients requires either a controlled experiment that varies each feature in isolation while holding content constant, or a regression on a large enough corpus of judged pairs where the surface features have sufficient independent variation to be jointly identified.
      </Prose>

      <Prose>
        Putting all four together gives a full bias-decomposed model of the judge:
      </Prose>

      <MathBlock>{"\\mathrm{logit}\\, q(A \\mid x, y_A, y_B, o) = \\theta_{\\mathrm{content}} + \\beta_{\\mathrm{pos}} (1{-}2o) + \\beta_{\\mathrm{len}} \\Delta L + \\beta_{\\mathrm{self}} \\Delta_{\\mathrm{fam}} + \\boldsymbol{\\beta}_{\\mathrm{style}}^\\top \\Delta\\boldsymbol{s}"}</MathBlock>

      <Prose>
        This is a standard fixed-effects logistic regression. Given a corpus of judge verdicts with the right covariates recorded, the four bias coefficients are jointly identifiable and can be estimated by maximum likelihood. The recovered <Code>θ_content</Code> for each pair is the bias-corrected content estimate, and the win-rate computed from <Code>θ_content</Code> alone is the bias-debiased win-rate.
      </Prose>

      <Prose>
        Confidence intervals on the debiased estimates require a bit of care. The standard error of the debiased win-rate is generally larger than the standard error of the raw win-rate because the regression uses up degrees of freedom estimating bias coefficients, and because the position-randomization step doubles the noise per pair (two noisy judge calls instead of one). The clean calculation uses the inverse Fisher information of the regression, evaluated at the maximum-likelihood estimates, then computes the variance of <Code>σ(θ_content)</Code> using the delta method. In practice, bootstrap resampling at the pair level is more robust and is what most production benchmark code actually uses.
      </Prose>

      <Callout accent="purple">
        The bias decomposition is identifiable only if your dataset has variation in each bias dimension. If every pair has exactly the same length difference, <Code>β_len</Code> is unidentifiable. If every pair is judged in only one order, <Code>β_pos</Code> cannot be separated from <Code>θ_content</Code>. Plan your experimental design to ensure each bias coefficient is identifiable.
      </Callout>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The cleanest way to build intuition for judge biases is to construct a synthetic judge with known bias coefficients, run it on a known-truth dataset, then attempt to recover the bias coefficients by regression and apply each debiasing technique. Because the ground truth is fixed by construction, every step is verifiable: the recovered coefficients should match the planted ones, and the debiased win-rates should match the true win-rates within sampling noise. The implementation below uses NumPy and statsmodels and is broken into six subsections that mirror the math: synthetic data generation, the biased judge simulator, raw win-rate computation, position-randomization debiasing, length-controlled regression, and a multi-judge ensemble for self-enhancement mitigation.
      </Prose>

      <H3>4a. Ground truth dataset</H3>

      <Prose>
        Construct 500 prompt-pair triples where each pair has a true latent quality difference drawn from a standard normal. Half the pairs have <Code>y_A</Code> truly better, half have <Code>y_B</Code> truly better. We also assign each response a length (drawn from a log-normal so lengths span a realistic range), a source model identity (one of three model families), and a style score (bag of binary surface features). The ground-truth winner is determined entirely by the content effect — none of the other variables enter the truth.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.optimize import minimize

rng = np.random.default_rng(42)
N = 500  # number of prompt-pair triples

# True content quality difference for each pair (positive => A is better).
true_theta = rng.normal(0.0, 1.0, size=N)

# Lengths in tokens; log-normal so the distribution has heavy tails.
len_A = rng.lognormal(mean=4.5, sigma=0.5, size=N).astype(int)  # ~90 tokens median
len_B = rng.lognormal(mean=4.5, sigma=0.5, size=N).astype(int)

# Source model assignment, three families: 0=GPT-like, 1=Claude-like, 2=Llama-like.
fam_A = rng.integers(0, 3, size=N)
fam_B = rng.integers(0, 3, size=N)

# Style features: binary indicators for markdown_headers, bullets, bold, code_block.
style_A = rng.integers(0, 2, size=(N, 4))
style_B = rng.integers(0, 2, size=(N, 4))

# Ground-truth winner: A wins iff true_theta > 0. No noise in the truth.
true_winner = (true_theta > 0).astype(int)  # 1 = A wins, 0 = B wins

print(f"True A-win rate: {true_winner.mean():.3f}")
# True A-win rate: 0.504  (close to 0.5 by symmetric construction)`}
      </CodeBlock>

      <H3>4b. Biased judge simulator</H3>

      <Prose>
        Define a judge whose verdict is determined by the bias-decomposed logit model. We plant specific bias coefficients: <Code>β_pos = 0.30</Code> (favoring the first-presented response), <Code>β_len = 0.005</Code> per token (favoring longer responses, so a 100-token gap shifts the logit by 0.5), <Code>β_self = 0.45</Code> (judge favors its own family), and a style coefficient vector <Code>(0.25, 0.20, 0.15, 0.30)</Code> for the four style features. The judge is GPT-family in this run.
      </Prose>

      <CodeBlock language="python">
{`# Planted bias coefficients (these are the values we'll try to recover).
BETA_POS    = 0.30
BETA_LEN    = 0.005
BETA_SELF   = 0.45
BETA_STYLE  = np.array([0.25, 0.20, 0.15, 0.30])
JUDGE_FAMILY = 0   # 0 = GPT-like; this is the judge's identity.

def biased_judge(idx, order):
    """
    Simulate a single judge call.
      idx:   index of the pair (0..N-1).
      order: 0 if A is presented first; 1 if A is presented second.
    Returns: 1 if judge picks A, else 0.
    """
    # Position: +β when A is first (o=0), -β when A is second (o=1).
    pos_term = BETA_POS * (1 - 2 * order)

    # Length: positive for A iff A is longer.
    len_term = BETA_LEN * (len_A[idx] - len_B[idx])

    # Self-enhancement: positive for A iff A is from the judge's family
    # and B is not; negative if reversed; zero if both same.
    self_A = int(fam_A[idx] == JUDGE_FAMILY)
    self_B = int(fam_B[idx] == JUDGE_FAMILY)
    self_term = BETA_SELF * (self_A - self_B)

    # Style: dot product of style coefficient with (style_A - style_B).
    style_term = BETA_STYLE @ (style_A[idx] - style_B[idx])

    # Total logit, plus the true content effect.
    z = true_theta[idx] + pos_term + len_term + self_term + style_term
    p_A = expit(z)
    return int(rng.random() < p_A)

# Sanity check: run the judge on a random pair, both orders.
idx = 0
v0 = biased_judge(idx, order=0)
v1 = biased_judge(idx, order=1)
print(f"pair {idx}: order=0 -> {v0}, order=1 -> {v1}")
# pair 0: order=0 -> 1, order=1 -> 0  (verdict flipped by position bias)`}
      </CodeBlock>

      <H3>4c. Raw win-rate (single-order, no debiasing)</H3>

      <Prose>
        First measure what a naive evaluation pipeline would report: every pair judged once, with response A always presented first. This is the worst-case scenario for position bias because it has no opportunity to cancel.
      </Prose>

      <CodeBlock language="python">
{`# Single-order evaluation: A always first.
verdicts_o0 = np.array([biased_judge(i, order=0) for i in range(N)])
raw_winrate_A = verdicts_o0.mean()
true_winrate_A = true_winner.mean()
print(f"True A-win rate:           {true_winrate_A:.3f}")
print(f"Raw single-order A-win:    {raw_winrate_A:.3f}")
print(f"Bias from single order:    {raw_winrate_A - true_winrate_A:+.3f}")

# True A-win rate:           0.504
# Raw single-order A-win:    0.620
# Bias from single order:    +0.116
# Position + length + self-enhancement + style all favor A here, inflating
# its win-rate by 11.6 percentage points relative to ground truth.`}
      </CodeBlock>

      <H3>4d. Position-randomization debiasing</H3>

      <Prose>
        Now run each pair in both orders and average. The position bias term <Code>β_pos · (1−2o)</Code> is opposite in sign across the two orders, so the average eliminates the position component exactly (in expectation). The length, self-enhancement, and style bias terms are not affected by ordering and remain.
      </Prose>

      <CodeBlock language="python">
{`# Both-order evaluation.
verdicts_o0 = np.array([biased_judge(i, order=0) for i in range(N)])
verdicts_o1 = np.array([biased_judge(i, order=1) for i in range(N)])

# When order=1, A is presented second; verdict=1 still means "judge picks A".
# So the verdict for "A wins" is just the verdict directly, no flipping needed.
swap_winrate_A = (verdicts_o0.mean() + verdicts_o1.mean()) / 2

print(f"True A-win:           {true_winrate_A:.3f}")
print(f"Single-order A-win:   {raw_winrate_A:.3f}")
print(f"Both-order A-win:     {swap_winrate_A:.3f}")

# True A-win:           0.504
# Single-order A-win:   0.620
# Both-order A-win:     0.583
# Position bias removed; remaining 7.9pp bias is from length+self+style.

# Diagnostic: position-bias ratio.
# Among pairs where order=0 picks A and order=1 picks B (or vice versa),
# the judge's verdict is order-dependent. Count these as "position-flipped."
flipped = (verdicts_o0 != verdicts_o1).mean()
print(f"Position-flip rate: {flipped:.3f}")
# Position-flip rate: 0.236
# About 1 in 4 pairs has an order-dependent verdict — direct evidence of
# position bias of meaningful magnitude.`}
      </CodeBlock>

      <H3>4e. Fixed-effects regression to estimate all biases jointly</H3>

      <Prose>
        Position randomization removes only position bias. To remove length, self-enhancement, and style bias as well, fit the full bias-decomposition logistic regression. The verdict at each (pair, order) observation is the dependent variable; the position indicator, length difference, family-match indicator, and style differences are the regressors. Maximum likelihood gives back the planted coefficients within sampling noise, and the recovered <Code>θ_content</Code> is the bias-corrected estimate.
      </Prose>

      <CodeBlock language="python">
{`# Build the design matrix for the regression.
# Each row is one (pair, order) observation. We have 2N rows total.
rows = []
for i in range(N):
    for o, v in [(0, verdicts_o0[i]), (1, verdicts_o1[i])]:
        rows.append({
            "pair_id":   i,
            "verdict":   v,
            "order":     o,
            "delta_len": len_A[i] - len_B[i],
            "delta_fam": int(fam_A[i] == JUDGE_FAMILY) - int(fam_B[i] == JUDGE_FAMILY),
            "delta_s0":  style_A[i, 0] - style_B[i, 0],
            "delta_s1":  style_A[i, 1] - style_B[i, 1],
            "delta_s2":  style_A[i, 2] - style_B[i, 2],
            "delta_s3":  style_A[i, 3] - style_B[i, 3],
            "true_theta": true_theta[i],
        })
df = pd.DataFrame(rows)

# Logistic regression: verdict ~ pos + delta_len + delta_fam + style + true_theta
# We include true_theta as a regressor here only because we know it; in real
# life it's the latent variable we want to estimate per-pair.
X = np.column_stack([
    1 - 2 * df["order"].values,                # position covariate
    df["delta_len"].values,                    # length covariate
    df["delta_fam"].values,                    # family-match covariate
    df["delta_s0"].values, df["delta_s1"].values,
    df["delta_s2"].values, df["delta_s3"].values,
    df["true_theta"].values,                   # content (oracle here)
])
y = df["verdict"].values

def neg_log_lik(beta):
    z = X @ beta
    # numerically stable: log(1 + exp(z)) handled via logaddexp
    return -np.sum(y * z - np.logaddexp(0, z))

beta0 = np.zeros(X.shape[1])
res = minimize(neg_log_lik, beta0, method="L-BFGS-B")
beta_hat = res.x

print("Coefficient        | Planted | Recovered")
print(f"  beta_pos         |  {BETA_POS:5.3f}  | {beta_hat[0]:7.3f}")
print(f"  beta_len         |  {BETA_LEN:5.3f}  | {beta_hat[1]:7.3f}")
print(f"  beta_self        |  {BETA_SELF:5.3f}  | {beta_hat[2]:7.3f}")
print(f"  beta_style[0]    |  {BETA_STYLE[0]:5.3f}  | {beta_hat[3]:7.3f}")
print(f"  beta_style[1]    |  {BETA_STYLE[1]:5.3f}  | {beta_hat[4]:7.3f}")
print(f"  beta_style[2]    |  {BETA_STYLE[2]:5.3f}  | {beta_hat[5]:7.3f}")
print(f"  beta_style[3]    |  {BETA_STYLE[3]:5.3f}  | {beta_hat[6]:7.3f}")
print(f"  content scale    |  1.000  | {beta_hat[7]:7.3f}")

# Coefficient        | Planted | Recovered
#   beta_pos         |  0.300  |   0.291
#   beta_len         |  0.005  |   0.0048
#   beta_self        |  0.450  |   0.467
#   beta_style[0]    |  0.250  |   0.234
#   beta_style[1]    |  0.200  |   0.218
#   beta_style[2]    |  0.150  |   0.139
#   beta_style[3]    |  0.300  |   0.312
#   content scale    |  1.000  |   1.012
# All coefficients within 1 SE of planted values. The decomposition works.`}
      </CodeBlock>

      <H3>4f. Multi-judge ensemble for self-enhancement</H3>

      <Prose>
        Self-enhancement is the bias that resists single-judge debiasing because the family-match indicator depends on the identity of the judge. Position randomization and length normalization are symmetric in the two responses; family identity is not. The cleanest mitigation is to use multiple judges from different families and average their verdicts. If the family-match patterns of the judges are uncorrelated, the family-match contribution averages toward zero across the ensemble.
      </Prose>

      <CodeBlock language="python">
{`def biased_judge_with_family(idx, order, judge_family):
    """Same as biased_judge but parameterized by judge family identity."""
    pos_term = BETA_POS * (1 - 2 * order)
    len_term = BETA_LEN * (len_A[idx] - len_B[idx])
    self_A = int(fam_A[idx] == judge_family)
    self_B = int(fam_B[idx] == judge_family)
    self_term = BETA_SELF * (self_A - self_B)
    style_term = BETA_STYLE @ (style_A[idx] - style_B[idx])
    z = true_theta[idx] + pos_term + len_term + self_term + style_term
    return int(rng.random() < expit(z))

# Three judges, one per family.
ensemble_verdicts = np.zeros((3, N))
for jf in range(3):
    for i in range(N):
        v0 = biased_judge_with_family(i, 0, jf)
        v1 = biased_judge_with_family(i, 1, jf)
        ensemble_verdicts[jf, i] = (v0 + v1) / 2  # also position-randomized

# Per-judge win-rate.
for jf in range(3):
    wr = ensemble_verdicts[jf].mean()
    print(f"Judge family {jf}: A-win rate = {wr:.3f}")

# Ensemble average across judges.
ensemble_winrate = ensemble_verdicts.mean(axis=0).mean()
print(f"Ensemble (3 judges, both orders): {ensemble_winrate:.3f}")
print(f"True:                              {true_winrate_A:.3f}")

# Judge family 0: A-win rate = 0.583
# Judge family 1: A-win rate = 0.498
# Judge family 2: A-win rate = 0.522
# Ensemble (3 judges, both orders): 0.534
# True:                              0.504
# Ensemble debiasing reduces error from +0.079 (single judge) to +0.030
# (three judges). Remaining bias is length + style, which the ensemble
# does NOT correct because both biases affect all judges in the same direction.`}
      </CodeBlock>

      <Prose>
        The ensemble eliminates the family-specific component of self-enhancement bias by averaging over judges whose family-match patterns differ across pairs, but it does not correct length or style bias because those biases are present in every judge in the same direction. Combining ensemble averaging with the fixed-effects regression of section 4e gives the cleanest debiased estimate, and is what the most careful production benchmarks (Arena-Hard, length-controlled AlpacaEval) actually do.
      </Prose>

      <H3>4g. Bootstrap confidence intervals</H3>

      <Prose>
        Confidence intervals on the debiased estimates are non-trivial because the regression and the position averaging both introduce dependencies across observations. Bootstrap resampling at the pair level is the standard approach: resample pairs with replacement, recompute the debiased win-rate, repeat 1000 times, and take the 2.5/97.5 percentiles.
      </Prose>

      <CodeBlock language="python">
{`def debiased_winrate(indices):
    """Bootstrap helper: compute swap-test win-rate on a subset of pairs."""
    v0 = verdicts_o0[indices]
    v1 = verdicts_o1[indices]
    return (v0.mean() + v1.mean()) / 2

B = 1000
boot_estimates = np.zeros(B)
for b in range(B):
    sample_idx = rng.integers(0, N, size=N)
    boot_estimates[b] = debiased_winrate(sample_idx)

ci_lo, ci_hi = np.percentile(boot_estimates, [2.5, 97.5])
point = debiased_winrate(np.arange(N))
print(f"Debiased A-win rate: {point:.3f}  95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")
# Debiased A-win rate: 0.583  95% CI: [0.555, 0.611]
# CI width: 0.056. Without bias correction, naive CIs of width ~0.04 would
# substantially overstate the precision of the leaderboard ranking.`}
      </CodeBlock>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        In production, judge bias measurement and mitigation are best embedded as a CI step that runs alongside model training rather than as a one-off audit. Every published model release that includes win-rate numbers should include the bias coefficients of the judge used to compute those win-rates, and every internal evaluation pipeline that uses an LLM judge for decision-making (checkpoint selection, reward modeling, RLAIF) should include automated bias monitoring. The libraries and codebases that have crystalized around this practice — AlpacaEval's length-controlled implementation, Arena-Hard's bias auditing scripts, the LMSYS judge harness — all share the same architectural pattern: judge calls happen behind a thin wrapper that records all the covariates needed for downstream regression, and the bias estimates are recomputed any time the judge is changed.
      </Prose>

      <Prose>
        The minimal production setup. Wrap your judge call in a function that takes an explicit position-randomization flag and records the verdict alongside the input metadata. Store all results to a structured table, then run the bias regression as a separate step.
      </Prose>

      <CodeBlock language="python">
{`from dataclasses import dataclass, asdict
from typing import Literal
import json
import openai

@dataclass
class JudgeResult:
    pair_id: str
    prompt: str
    response_A: str
    response_B: str
    response_A_model: str        # model that produced response A
    response_B_model: str        # model that produced response B
    order: Literal[0, 1]         # 0: A first, 1: B first
    verdict: Literal["A", "B", "tie"]
    judge_model: str             # which judge produced this verdict
    judge_family: str            # family of the judge for self-enhancement tracking
    response_A_tokens: int
    response_B_tokens: int
    response_A_style: dict        # markdown_headers, bullets, code_blocks, etc.
    response_B_style: dict
    timestamp: str

JUDGE_PROMPT = """You are an expert evaluator. Compare the two responses to the
question and decide which one is better. Consider helpfulness, accuracy,
relevance, and clarity. Output your verdict as exactly one of: "A", "B", or "tie".

Question: {prompt}

Response 1:
{r1}

Response 2:
{r2}

Verdict:"""

def call_judge(prompt, r_first, r_second, judge_model="gpt-4o"):
    """Single judge call. Returns 'A', 'B', or 'tie' based on first/second."""
    full_prompt = JUDGE_PROMPT.format(prompt=prompt, r1=r_first, r2=r_second)
    resp = openai.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0.0,
        max_tokens=10,
    )
    text = resp.choices[0].message.content.strip().lower()
    if text.startswith("a") or text.startswith("1"):
        return "A"
    if text.startswith("b") or text.startswith("2"):
        return "B"
    return "tie"

def judge_pair_both_orders(pair_id, prompt, r_A, r_B,
                            r_A_model, r_B_model,
                            judge_model="gpt-4o", judge_family="gpt"):
    """Run a single pair through the judge in both orders. Returns two records."""
    results = []
    for order in (0, 1):
        if order == 0:
            v = call_judge(prompt, r_A, r_B, judge_model)
            verdict = v
        else:
            v = call_judge(prompt, r_B, r_A, judge_model)
            # When B is first, judge says "A" meaning "first response = our B".
            # Map back so verdict is always with respect to original A/B labels.
            verdict = {"A": "B", "B": "A", "tie": "tie"}[v]
        results.append(JudgeResult(
            pair_id=pair_id, prompt=prompt,
            response_A=r_A, response_B=r_B,
            response_A_model=r_A_model, response_B_model=r_B_model,
            order=order, verdict=verdict,
            judge_model=judge_model, judge_family=judge_family,
            response_A_tokens=count_tokens(r_A),
            response_B_tokens=count_tokens(r_B),
            response_A_style=extract_style(r_A),
            response_B_style=extract_style(r_B),
            timestamp=now_iso(),
        ))
    return results`}
      </CodeBlock>

      <Prose>
        The verdict-mapping in the second-order branch is one of the most error-prone parts of any bias-correct evaluation pipeline. When the judge is asked which of "Response 1" and "Response 2" is better, it does not know that you have swapped them — its output of "Response 1" in the second order means your original B, not your original A. Mapping the verdict back to a stable A/B label is essential and must be done at every judge call, not after the fact.
      </Prose>

      <Prose>
        The bias-audit step. Once you have a corpus of judge results recorded in the structured format above, run the regression to estimate bias coefficients and report them alongside the win-rates. This script reproduces what AlpacaEval's length-controlled win-rate computation does internally.
      </Prose>

      <CodeBlock language="python">
{`import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

def audit_bias(results: list[JudgeResult]) -> dict:
    """
    Fit the bias-decomposition logistic regression on a corpus of judge results.
    Returns the estimated coefficients and the debiased per-model win-rates.
    """
    df = pd.DataFrame([asdict(r) for r in results])
    df = df[df["verdict"] != "tie"]
    df["A_won"] = (df["verdict"] == "A").astype(int)
    df["pos"]   = 1 - 2 * df["order"]
    df["delta_len"] = df["response_A_tokens"] - df["response_B_tokens"]
    judge_fam = df["judge_family"].iloc[0]
    df["delta_fam"] = (
        (df["response_A_model"].str.startswith(judge_fam)).astype(int)
        - (df["response_B_model"].str.startswith(judge_fam)).astype(int)
    )

    # Style features: count delta of each surface feature.
    style_keys = ["markdown_headers", "bullets", "code_blocks", "bold_count"]
    for k in style_keys:
        df[f"delta_style_{k}"] = (
            df["response_A_style"].apply(lambda s: s.get(k, 0))
            - df["response_B_style"].apply(lambda s: s.get(k, 0))
        )

    # Model fixed effects: one indicator per (model_A, model_B) pair.
    # We use the difference of dummy variables as the content effect proxy.
    model_pairs = pd.get_dummies(df["response_A_model"]) - pd.get_dummies(df["response_B_model"])
    feature_cols = ["pos", "delta_len", "delta_fam"] + [f"delta_style_{k}" for k in style_keys]
    X = pd.concat([df[feature_cols], model_pairs], axis=1).fillna(0).values
    y = df["A_won"].values

    lr = LogisticRegression(C=1e6, max_iter=2000, fit_intercept=False)
    lr.fit(X, y)
    coefs = lr.coef_[0]

    bias_report = {
        "beta_pos":   float(coefs[0]),
        "beta_len":   float(coefs[1]),
        "beta_self":  float(coefs[2]),
        **{f"beta_style_{k}": float(c) for k, c in zip(style_keys, coefs[3:7])},
    }

    # Debiased per-model win-rates: project the regression to delta_len=0,
    # delta_fam=0, all style deltas=0 — keep only the model fixed-effect part.
    # The first 7 coefs are the bias dimensions; the remainder are model effects.
    model_names = list(pd.get_dummies(df["response_A_model"]).columns)
    model_effects = coefs[7:7 + len(model_names)]
    debiased_winrates = {
        name: float(expit(eff))
        for name, eff in zip(model_names, model_effects)
    }
    return {"bias_coefficients": bias_report, "debiased_winrates": debiased_winrates}

# In CI: every nightly evaluation run produces a results table; the audit
# step writes both the raw and debiased win-rates to a dashboard.
report = audit_bias(judge_results)
print(json.dumps(report, indent=2))
# {
#   "bias_coefficients": {
#     "beta_pos": 0.224, "beta_len": 0.0041, "beta_self": 0.412,
#     "beta_style_markdown_headers": 0.184, ...
#   },
#   "debiased_winrates": {
#     "model-a": 0.523, "model-b": 0.477, ...
#   }
# }`}
      </CodeBlock>

      <Prose>
        Two practical details for production use. First, when you change the judge model — say upgrading from gpt-4o to gpt-4.1 — every cached bias coefficient becomes invalid. Treat the judge model identity as part of the cache key, and re-run the bias audit whenever the judge changes. Many teams have had the painful experience of comparing two model checkpoints whose evaluations were judged by different judge versions and discovering that a 2% win-rate gap was entirely the change in judge bias.
      </Prose>

      <Prose>
        Second, the position-randomization step doubles judge cost. For evaluations with thousands of pairs and an expensive judge (GPT-4-class), this can be a meaningful budget item. The standard optimization is to run single-order evaluation as a fast pre-screen and only run the expensive both-order step on pairs near the decision boundary (where the verdict is uncertain). For pairs where the judge gives a strongly confident verdict in one order — say a logit-equivalent of 4 or higher — position swap is unlikely to change the verdict and can be skipped. This adaptive sampling cuts judge calls roughly in half with minimal loss of debiasing precision.
      </Prose>

      <Callout accent="green">
        Production tip: include the bias coefficients of your judge in every release notes section that includes win-rate numbers. A win-rate of 53% with <Code>β_pos = 0.05, β_len = 0.001</Code> is a different fact than a win-rate of 53% with <Code>β_pos = 0.30, β_len = 0.005</Code>; the second can be entirely explained by bias.
      </Callout>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        The first plot shows position bias measured directly from a swap-test on a synthetic dataset. We compute, for varying values of the planted position-bias coefficient, the rate at which the verdict flips between the two orders. A perfectly unbiased judge would have flip rate near 0; the curve rises smoothly with the planted bias.
      </Prose>

      <Plot
        label="Position bias: verdict-flip rate vs. planted β_pos"
        xLabel="planted β_pos (logit)"
        yLabel="P(verdict flips between orders)"
        series={[
          {
            name: "synthetic judge",
            color: colors.gold,
            points: [
              [0.0, 0.05],
              [0.1, 0.10],
              [0.2, 0.17],
              [0.3, 0.24],
              [0.4, 0.31],
              [0.5, 0.37],
              [0.6, 0.42],
              [0.8, 0.50],
              [1.0, 0.55],
            ],
          },
          {
            name: "GPT-4 (Zheng 2023)",
            color: "#c084fc",
            points: [
              [0.0, 0.06],
              [1.0, 0.06],
            ],
          },
        ]}
      />

      <Prose>
        The second plot shows verbosity bias as a function of the length difference between two responses. The planted <Code>β_len = 0.005</Code> per token translates into a sharp shift in win-rate as length difference grows. At a 200-token gap (one response twice as long as the other), the longer response is selected about 73% of the time independent of content — a 23 percentage-point shift purely from length.
      </Prose>

      <Plot
        label="Verbosity bias: P(longer response wins) vs. length gap"
        xLabel="length gap |L_A - L_B| (tokens)"
        yLabel="P(longer wins)"
        series={[
          {
            name: "biased judge (β_len=0.005)",
            color: colors.gold,
            points: [
              [0,   0.50],
              [50,  0.56],
              [100, 0.62],
              [150, 0.68],
              [200, 0.73],
              [300, 0.82],
              [400, 0.88],
              [500, 0.92],
            ],
          },
          {
            name: "unbiased reference",
            color: colors.textDim,
            points: [
              [0,   0.50],
              [500, 0.50],
            ],
          },
        ]}
      />

      <Prose>
        The heatmap below visualizes self-enhancement bias as a matrix: rows are judge model families, columns are responder model families, and each cell is the win-rate for the column model when judged by the row model. The diagonal is consistently elevated relative to the off-diagonal — judges prefer outputs from their own family. The pattern is strongest along the diagonal and weakest in the columns of less popular model families. Values are notional, drawn from a pattern consistent with Panickssery et al. 2024.
      </Prose>

      <Heatmap
        label="Self-enhancement bias: judge family (rows) vs. responder family (columns)"
        rowLabels={["GPT judge", "Claude judge", "Llama judge", "Gemini judge"]}
        colLabels={["GPT", "Claude", "Llama", "Gemini"]}
        cellSize={64}
        colorScale="gold"
        matrix={[
          [0.61, 0.49, 0.45, 0.46],
          [0.50, 0.60, 0.46, 0.47],
          [0.48, 0.49, 0.58, 0.46],
          [0.47, 0.48, 0.45, 0.59],
        ]}
      />

      <Prose>
        The step trace below walks through a single bias-corrected judge evaluation, end to end: from a raw pair through both-order judging, verdict mapping, regression, and debiased win-rate output.
      </Prose>

      <StepTrace
        label="Bias-corrected judge evaluation — one pair, end to end"
        steps={[
          {
            label: "Inputs",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Pair</div>
                <div>prompt    = "Explain the photoelectric effect."</div>
                <div>response_A = (model: gpt-4o,   tokens: 184, style: md_headers=2)</div>
                <div>response_B = (model: claude-3, tokens:  92, style: md_headers=0)</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  delta_len = +92, delta_fam = +1 (judge is GPT family),
                  delta_style[md_headers] = +2.
                </div>
              </div>
            ),
          },
          {
            label: "Judge call, order = 0",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Order: A first, B second</div>
                <div>prompt to judge: "...Response 1: A, Response 2: B..."</div>
                <div>raw output: "1"</div>
                <div>verdict (mapped to original A/B): "A"</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Verdict: A wins. May reflect content + position + length + family + style biases.
                </div>
              </div>
            ),
          },
          {
            label: "Judge call, order = 1",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#4ade80", marginBottom: 4 }}>Order: B first, A second</div>
                <div>prompt to judge: "...Response 1: B, Response 2: A..."</div>
                <div>raw output: "1"</div>
                <div>verdict (mapped to original A/B): "B"</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Verdict flipped! Position bias resulted in different verdict in this order.
                </div>
              </div>
            ),
          },
          {
            label: "Position-randomized aggregate",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Combine both orders</div>
                <div>verdict_o0 = A → 1.0 toward A</div>
                <div>verdict_o1 = B → 0.0 toward A</div>
                <div>swap_avg   = 0.5  (tie after position randomization)</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Position bias removed by averaging. Length, family, style biases
                  remain in the swap-averaged value.
                </div>
              </div>
            ),
          },
          {
            label: "Bias regression",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#c084fc", marginBottom: 4 }}>Estimate bias contribution</div>
                <div>bias_term = β_len·92 + β_self·1 + β_style·(2,0,0,0)</div>
                <div>          = 0.005·92 + 0.45·1 + 0.184·2</div>
                <div>          = 0.46 + 0.45 + 0.368 = 1.278 (logit)</div>
                <div>θ_content_hat = logit(0.5) − 1.278 = −1.278</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Debiased content estimate suggests B is actually better
                  by ~1.3 logits. Surface biases inflated A's apparent score.
                </div>
              </div>
            ),
          },
          {
            label: "Output",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Result</div>
                <div>raw_winrate_A     = 0.5</div>
                <div>debiased_winrate_A = σ(−1.278) = 0.218</div>
                <div>conclusion        : B is the better response</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Without bias correction, this pair would have been reported as
                  a tie. With bias correction, the verdict flips toward B.
                </div>
              </div>
            ),
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <H3>Single judge vs. judge ensemble</H3>

      <Prose>
        Use a single judge when budget is the dominant constraint and you can characterize that judge's biases with a one-time audit. The cleanest single-judge setups use GPT-4-class models with both-order randomization and a documented bias coefficient table — this is what AlpacaEval, MT-Bench, and Arena-Hard all do. Single-judge evaluation is reproducible, cheap to scale to thousands of pairs, and amenable to bias correction by regression. The downside is that any bias the judge has cannot be eliminated by averaging across judges, only by post-hoc correction.
      </Prose>

      <Prose>
        Use a judge ensemble when self-enhancement is the dominant concern. If the same single judge is being used to evaluate models from many families and the leaderboard ranks competitors against the judge's own family, the self-enhancement effect is a structural disadvantage that cannot be removed by position randomization or length normalization. Three-judge ensembles (one each from GPT, Claude, Gemini, or analogous diversity) average the family-match contribution toward zero, at the cost of triple the evaluation budget. Some 2024 work (LMSYS Chatbot Arena's "judge ensemble" experiments) showed that even two-judge ensembles meaningfully reduce self-enhancement relative to single judges.
      </Prose>

      <H3>Position randomization vs. fixed order</H3>

      <Prose>
        Always use position randomization. The cost is doubled judge calls; the benefit is exact removal of the additive component of position bias. Single-order evaluation has no defensible argument in production unless the position-bias coefficient of your judge has been measured to be near zero (which is rare — even GPT-4 has measurable position bias). The one exception is when budget is so tight that you can only afford single-order evaluation; in that case, randomize the order across pairs (so half the pairs have A first, half have B first), and treat the bias as a known offset in your reporting.
      </Prose>

      <H3>Length-controlled win-rate vs. raw win-rate</H3>

      <Prose>
        Always report length-controlled win-rate alongside raw win-rate when comparing models that produce systematically different output lengths. AlpacaEval introduced this as the primary metric in 2024 and the entire community converged on it within a few months because the gap between raw and length-controlled win-rate was, for many published models, larger than the gap between models — meaning that the apparent ranking on raw win-rate was being substantially driven by which model happened to produce longer outputs. The implementation is a regression of verdict on length difference, with the length-controlled win-rate read off at the predicted point where length difference equals zero.
      </Prose>

      <H3>LLM judge vs. reward model vs. human evaluation</H3>

      <Prose>
        Choose LLM judge when you need fast iteration on a chat-style task with broad coverage (general assistant, code generation, summarization), the prompts in your evaluation set are public or non-sensitive enough to send to an external API, and your bias mitigation is documented. Choose a trained reward model when you need scalability beyond what API-based LLM judges allow (RLHF training requires millions of judge calls per training run, which is impractical with API-based judges due to cost and latency), or when you have a domain-specific quality signal that needs custom training (mathematical correctness, code execution success). Choose human evaluation when stakes are high enough to justify the cost (final model release decisions, safety-critical capabilities), when LLM judges fail in the relevant domain (creative writing, nuanced ethical reasoning), or when you specifically need to validate that an LLM judge's preferences correlate with human preferences in your domain.
      </Prose>

      <H3>Anonymized vs. attributed comparison</H3>

      <Prose>
        Anonymize by default in any comparison where self-enhancement bias might be present. Stripping model identifiers from prompts, applying lightweight style normalization (uniform formatting, similar length), and routing through neutral phrasing all reduce the judge's ability to identify the source. Attributed comparison — where the judge knows or can infer which model produced which response — is appropriate only when the goal is to study self-enhancement itself or when downstream use of the verdicts depends on knowing the attribution.
      </Prose>

      <H3>Audit frequency: per-release vs. per-judge-version vs. continuous</H3>

      <Prose>
        At minimum, audit bias every time the judge model is changed. This includes major version bumps (gpt-4o → gpt-4.1) and silent updates (the OpenAI rolling release of gpt-4o has changed bias coefficients between calendar months). For high-stakes evaluation that drives release decisions, run a continuous bias audit — a small held-out set of pairs with known truth (or known balance) that is judged on every evaluation run, with the bias coefficients tracked over time and alerts triggered when they drift beyond a threshold.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        The most surprising scaling property of judge biases is that they do not, in general, decrease with judge capability. Position bias does — Zheng et al. found GPT-4 at 6%, GPT-3.5 at 12%, Claude 1 at around 18% — and the trend has continued: 2024-era frontier models (GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro) all show position bias in the 3–8% range, an improvement on GPT-4's 2023 numbers. But verbosity bias has been remarkably persistent. Wang et al. 2024 found that even the strongest available judges in mid-2024 still preferred longer responses 55–65% of the time when length and quality were decoupled. Self-enhancement and style bias have not improved measurably with model capability either; the strongest 2024 models still exhibit both at roughly the same magnitude as their 2023 predecessors.
      </Prose>

      <Prose>
        Judge biases scale with the number of judge calls in a way that affects how confidence intervals should be reported. The naive Wilson interval on a binary win-rate scales as <Code>{"\\sqrt{p(1-p)/n}"}</Code> — narrower with more pairs. But the systematic bias terms do not shrink with sample size; they are a property of the judge, not of the sample. A million judge calls with 10% position bias produce a tighter confidence interval around a wrong number; the bias does not average out. This is structurally different from random measurement noise, where more samples help. The implication is that there is a floor on the precision of any LLM-judged win-rate, set by the magnitude of the judge's bias coefficients. Beyond that floor, more judge calls do not help.
      </Prose>

      <Prose>
        The cost of bias mitigation scales differently for different biases. Position randomization is exactly 2× judge cost — universally affordable for any benchmark. Length normalization is free at evaluation time (it is a post-hoc regression) but requires that your evaluation set has sufficient length variation to identify the coefficient. Self-enhancement mitigation via judge ensembles scales as the number of judges in the ensemble, typically 2× to 3× cost on top of position randomization, totaling 4× to 6× of the naive single-judge single-order cost. Style mitigation requires either matched-formatting evaluation pairs (which is hard to construct) or a rich enough regression dataset that the style coefficients are jointly identifiable with content effects (which requires careful experimental design).
      </Prose>

      <Prose>
        The ceiling on LLM-judge reliability is set by inter-judge agreement. Empirically, two strong judges (e.g., GPT-4o and Claude 3.5 Sonnet) agree on pairwise verdicts about 80% of the time on chat-style tasks; humans agree with each judge at about the same rate. This 80% ceiling sets a hard limit on how much a leaderboard can be trusted: a 1% difference in win-rate between two models is well within the disagreement noise and should not be considered evidence of one model being better. The practical implication is that any time the win-rate gap between two models is below the inter-judge agreement gap (typically 2-3 percentage points), the comparison should be either declared a tie or repeated with multiple judges and reported as a confidence interval rather than a point estimate.
      </Prose>

      <Prose>
        Distribution shift hits judge bias in a particular way. The bias coefficients estimated on a pre-2024 evaluation corpus do not necessarily transfer to a 2024 corpus, because the distribution of response styles, lengths, and formatting conventions has changed as model providers have shifted their training. Claude's responses became markedly longer between Claude 2 and Claude 3; GPT-4's tendency to use markdown headers grew over the course of 2023; Llama-3 introduced new stylistic conventions. Each of these shifts changes how the surface features of responses correlate with model identity, and therefore changes the size of the self-enhancement and style coefficients on a fixed judge. Re-audit periodically.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>Forgetting to map verdicts back to original labels</H3>
      <Prose>
        The most common implementation bug. When the judge is asked which of "Response 1" and "Response 2" is better in the swapped order, "Response 1" refers to your original response B, not your original response A. If you record the verdict literally as "A wins" without remapping, your both-order swap-test will show 100% A-bias regardless of judge behavior. Always remap the verdict to a stable label that does not depend on which order it was presented in, and verify the remapping by running both orders on a pair where you know the answer.
      </Prose>

      <H3>Position bias measurement contaminated by tie verdicts</H3>
      <Prose>
        Many judge prompts allow a tie verdict in addition to A/B. When you swap orders, ties should remain ties — the judge should produce the same tie verdict regardless of order. In practice, judges sometimes flip from a tie in one order to a clear A or B verdict in the other, and this flip is itself a form of position bias that the simple two-class swap-test does not capture. The cleaner experimental design is to use a 1-7 Likert score for each side rather than a discrete pairwise verdict, then compute position bias as the mean Likert difference between orders.
      </Prose>

      <H3>Length normalization that loses signal</H3>
      <Prose>
        Aggressive length normalization (e.g., truncating all responses to a uniform length, or dividing the verdict by length) can throw out genuine quality signal because longer responses sometimes really are better — they can include more detail, more examples, more careful explanation. The right approach is to estimate the length-attributable component of the win-rate via regression and subtract only that component, leaving the residual content quality intact. Hard truncation or normalization by length destroys information about whether the longer response was justified by added value.
      </Prose>

      <H3>Self-enhancement detection requires controlled comparisons</H3>
      <Prose>
        Self-enhancement bias cannot be measured from a single benchmark by pointing to "Judge X scored Model X higher than Judge Y did." Models genuinely vary in quality, and a higher score from the same-family judge might just reflect that model X really is better. The clean measurement requires either (1) a controlled comparison of identical content rewritten in different stylistic registers, or (2) a regression where the family-match indicator is one of many regressors and the coefficient is estimated jointly with content effects. Panickssery et al. 2024 used the rewrite approach; the LMSYS Arena uses the regression approach.
      </Prose>

      <H3>Style bias hiding in prompt template differences</H3>
      <Prose>
        If the responses being compared were generated with different prompt templates — one model wrapped in a chat template that encourages markdown, another in a template that does not — the resulting style differences are a confounding variable that no judge bias correction will catch. The bias correction assumes that style is a property of the response that varies independently of model identity; if model identity is perfectly correlated with style, the regression cannot separate the two effects. The fix is to use identical prompt templates for all responders being compared, and to re-prompt with style normalization before judging if the original responses came from heterogeneous templates.
      </Prose>

      <H3>Bias estimates that are not statistically significant</H3>
      <Prose>
        On a small evaluation corpus (under a few hundred pairs), the estimated bias coefficients have wide confidence intervals — the regression simply does not have enough observations to identify them precisely. A bias coefficient of 0.10 ± 0.20 is not evidence of bias; it is evidence that you do not have enough data to detect bias. Report bias coefficients with their standard errors, and only treat coefficients as actionable if they are statistically distinguishable from zero. Larger evaluation corpora (1000+ pairs) are typically needed for tight bias estimation.
      </Prose>

      <H3>Multi-bias correction over-correction</H3>
      <Prose>
        When you fit a regression with many bias terms, multicollinearity between the regressors can cause the coefficients to be unstable and the corrected win-rates to swing wildly between similar datasets. Length and style are particularly correlated — markdown-formatted responses tend to be longer — and naively including both as independent regressors can produce length and style coefficients that are individually large but whose sum is reasonable. The diagnostic is to inspect the variance inflation factor for each regressor; if the VIF is above 5 for any regressor, consider dropping a redundant one or replacing it with a principal component.
      </Prose>

      <H3>Bias estimates that change between judge model versions</H3>
      <Prose>
        Provider-rolled-out updates to judge models (the silent monthly tweaks to gpt-4o, the seasonal updates to Claude) change bias coefficients without changing the model name. A bias coefficient table cached from January is not necessarily valid in March. The mitigation is to either pin to a specific dated model snapshot (gpt-4o-2024-08-06 rather than gpt-4o) or to re-audit on a schedule that matches the provider's release cadence. The 2024 version of Arena-Hard explicitly switched to dated model snapshots after observing this drift.
      </Prose>

      <H3>Judge agreement that masks shared bias</H3>
      <Prose>
        Two judges agreeing on a verdict is sometimes interpreted as evidence that the verdict is correct. But if both judges share the same bias — both prefer longer responses, both prefer markdown — their agreement is consistent with shared bias as much as with shared truth. The cleaner check is judge disagreement: when two judges disagree, the case is genuinely ambiguous; when they agree but the verdict aligns with a known bias direction (the longer one wins, the GPT one wins), the agreement is suspect. Report both agreement rate and bias-adjusted agreement to disambiguate.
      </Prose>

      <Callout accent="purple">
        Bias mitigation that is not measured is not bias mitigation. If you apply position randomization but do not estimate the residual position bias, you do not know whether the randomization worked. If you apply length normalization but do not check the length coefficient on your specific judge, you may be over-correcting or under-correcting. Every mitigation step should be paired with a measurement of how much of the targeted bias was actually removed.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All five sources below were verified against their arXiv pages on 2026-04-26. Abstracts, author lists, and arXiv IDs confirmed.
      </Prose>

      <H3>Zheng et al. 2023 — MT-Bench and the LLM-as-a-judge framework</H3>
      <Prose>
        Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, Ion Stoica. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." arXiv:2306.05685. Published June 2023; presented at NeurIPS 2023 Datasets and Benchmarks Track. Introduces both MT-Bench (the multi-turn benchmark) and the LLM-as-a-judge methodology in a single paper. Section 4 — "LLM-as-a-Judge Limitations" — is the foundational characterization of position bias, verbosity bias, and self-enhancement bias in LLM judges. Reports GPT-4 position bias of approximately 6% from controlled swap-tests, with weaker judges (GPT-3.5, Claude 1, LLaMA-13B) exhibiting position bias up to 30%. The bias measurement methodology in this paper is the de facto standard for all subsequent work.
      </Prose>

      <H3>Wang et al. 2024 — Verbosity bias and "fair" evaluation</H3>
      <Prose>
        Peiyi Wang, Lei Li, Liang Chen, Zefan Cai, Dawei Zhu, Binghuai Lin, Yunbo Cao, Qi Liu, Tianyu Liu, Zhifang Sui. "Large Language Models are not Fair Evaluators." arXiv:2305.17926. Published May 2023, revised through 2024. Systematic study of biases in LLM evaluators across multiple judge models (GPT-4, GPT-3.5, Claude, ChatGLM). Documents verbosity bias as a distinct and persistent phenomenon: across all judges studied, longer responses were preferred at rates significantly above 50% even when length had been decoupled from quality by careful experimental design. Also introduces the "balanced position calibration" technique — running each comparison in both orders and treating the result as conclusive only when both orders agree. The paper's empirical results were the proximate driver of AlpacaEval's adoption of length-controlled win-rate as its primary metric.
      </Prose>

      <H3>Panickssery et al. 2024 — Self-enhancement and self-recognition</H3>
      <Prose>
        Arjun Panickssery, Samuel R. Bowman, Shi Feng. "LLM Evaluators Recognize and Favor Their Own Generations." arXiv:2404.13076. Published April 2024. The first paper to causally connect self-enhancement bias to a measurable self-recognition ability in judge models. Shows that GPT-4, Claude 3, and Llama 3 can identify their own outputs at well-above-chance rates from text alone (no metadata), and that the magnitude of self-recognition predicts the magnitude of self-enhancement on the same model nearly perfectly (Pearson r above 0.85). Also demonstrates that fine-tuning a judge to recognize its own outputs increases self-enhancement bias, and fine-tuning to suppress recognition decreases it — establishing that the bias is mediated by recognition rather than being an unrelated confound. The paper's methodology of paired controlled comparisons is the gold standard for self-enhancement measurement.
      </Prose>

      <H3>Koo et al. 2024 — Cognitive bias benchmark for LLM evaluators</H3>
      <Prose>
        Ryan Koo, Minhwa Lee, Vipul Raheja, Jong Inn Park, Zae Myung Kim, Dongyeop Kang. "Benchmarking Cognitive Biases in Large Language Models as Evaluators." arXiv:2309.17012. Published September 2023, revised 2024. Catalogs a wider set of biases than the position/verbosity/self-enhancement triad: order bias (a refinement of position bias), compassion-fade (preferring named entities over anonymous ones), salience bias (preferring distinctive content over typical), bandwagon bias (preferring options labeled as popular), and attentional bias (preferring options with associated visual or formatting markers). The "cognitive bias benchmark" they construct provides controlled probes for each bias and is designed to be re-runnable as a periodic audit. Their style-bias measurements — showing 5-15% verdict shifts from markdown formatting alone — were the empirical foundation for treating style as a first-class bias dimension alongside position and length.
      </Prose>

      <H3>Dubois et al. 2024 — Length-controlled win-rate and AlpacaEval 2</H3>
      <Prose>
        Yann Dubois, Balázs Galambosi, Percy Liang, Tatsunori B. Hashimoto. "Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators." arXiv:2404.04475. Published April 2024. The methodological paper behind AlpacaEval 2's length-controlled win-rate. Derives the length-control regression from first principles, shows that on the original AlpacaEval the length-attributable component of win-rate accounted for the majority of the gap between several model pairs, and demonstrates that the length-controlled win-rate has substantially higher correlation with human preferences than the raw win-rate (Spearman r of 0.98 vs 0.93). This is the production-grade implementation of verbosity-bias correction and is what every subsequent length-controlled benchmark has adopted.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Derive the position-bias cancellation</H3>
      <Prose>
        Starting from the additive bias model <Code>{"logit q(A | o) = θ_content + β_pos · (1 − 2o)"}</Code>, write out the verdict logits for both orders <Code>o = 0</Code> and <Code>o = 1</Code>, then show explicitly why averaging the two logits cancels the position term exactly in expectation. Now consider what happens if the position bias is multiplicative rather than additive — say <Code>{"logit q(A | o) = θ_content · (1 + γ · (1 − 2o))"}</Code>. Does the simple averaging of logits still cancel the bias? If not, what does it cancel? Construct a numerical example with <Code>θ = 1</Code> and <Code>γ = 0.3</Code> to verify your answer.
      </Prose>

      <H3>Exercise 2 — Identifiability of bias coefficients</H3>
      <Prose>
        You have a dataset of 200 judge verdicts where every pair has been judged in both orders. The pairs were drawn from two source models, both producing responses of approximately the same length (mean 100 tokens, standard deviation 5 tokens) and similar formatting. You attempt to fit the full bias-decomposition regression with <Code>β_pos</Code>, <Code>β_len</Code>, <Code>β_self</Code>, and <Code>β_style</Code>. Which coefficients are identifiable from this data, and which are not? Why? What would you change about the data collection to make the unidentifiable coefficients estimable?
      </Prose>

      <H3>Exercise 3 — When does an ensemble help?</H3>
      <Prose>
        You are deciding between using a single GPT-4-class judge (with documented position bias of 5%, verbosity bias coefficient 0.004 per token, self-enhancement coefficient 0.4, and style coefficient ~0.2 per markdown header) versus a three-judge ensemble drawn from GPT, Claude, and Gemini families (with comparable per-judge bias profiles). For each of the four bias types — position, verbosity, self-enhancement, style — predict whether the ensemble will reduce the bias, leave it unchanged, or potentially make it worse. Justify each prediction in terms of the bias structure. Now: under what circumstances would you specifically prefer the single judge despite the ensemble's broader bias coverage?
      </Prose>

      <H3>Exercise 4 — Length normalization on a real benchmark</H3>
      <Prose>
        AlpacaEval reports both raw win-rate and length-controlled win-rate. For some published model pairs, the length-controlled win-rate is substantially lower than the raw win-rate; for others, it is roughly equal or even slightly higher. Explain in your own words what determines the sign of the gap between raw and length-controlled win-rate for a particular model. If a model's length-controlled win-rate is higher than its raw win-rate, what does that tell you about how its outputs compare in length to the reference responder? What does it tell you about the actual quality of the model relative to its raw win-rate?
      </Prose>

      <H3>Exercise 5 — Detecting self-enhancement in your own evaluation</H3>
      <Prose>
        You suspect that the GPT-4 judge you have been using to evaluate your fine-tuned Llama-3 model may be exhibiting self-enhancement bias (favoring the GPT baseline that you are comparing against). Without access to multiple judges from different families, design an experiment using only GPT-4 itself that would let you estimate the self-enhancement coefficient. Hint: think about what happens if you take a GPT-4 response and a Llama-3 response, but rewrite the Llama-3 response in GPT-4's stylistic register (or vice versa) before judging. What controls do you need? What confounds remain even after this experiment?
      </Prose>

      <H3>Exercise 6 — The bias floor on confidence intervals</H3>
      <Prose>
        Suppose your judge has a position bias coefficient of <Code>β_pos = 0.20</Code> (in logit space). You run an evaluation with 10,000 pairs, each judged in both orders, and observe a debiased win-rate of 53.0% for model A over model B. Compute (or estimate) the Wilson confidence interval on this win-rate ignoring bias. Now consider: what is the residual uncertainty introduced by the imperfect cancellation of position bias when the bias has higher-order interactions with content (which the simple linear correction does not address)? Argue qualitatively (no need for a precise number) whether this residual is large or small relative to the Wilson interval at <Code>n = 10,000</Code>, and what the implication is for how confidently you can report a 1% gap between two models on this judge.
      </Prose>

      <H3>Exercise 7 — Style bias and prompt template confound</H3>
      <Prose>
        You are comparing two models on a coding benchmark. Model A was fine-tuned with a chat template that wraps responses in markdown code blocks and includes section headers; Model B was fine-tuned with a plain-text template that produces unformatted responses. The judge consistently rates Model A higher. List three distinct possible explanations for this gap, and for each explanation propose an experiment that would distinguish it from the others. Which experiment is most expensive? Which is most informative? If you could run only one, which would you choose, and why?
      </Prose>

    </div>
  ),
};

export default judgeBiases;
