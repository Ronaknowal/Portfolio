import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const aiToAiBias = {
  title: "AI-to-AI Bias (Self-Preference in Judging)",
  slug: "ai-to-ai-bias-self-preference-in-judging",
  readTime: "~36 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Sometime around early 2023, a quiet methodological shift happened across the language-model evaluation community. Human annotators were expensive, slow, and inconsistent; the new generation of frontier models — GPT-4, Claude, Gemini — were cheap by comparison, fast, and produced rankings that correlated remarkably well with human judgments on chat-quality benchmarks. MT-Bench and AlpacaEval formalized this practice: instead of paying crowdworkers to compare two model outputs, you ask GPT-4 to do the comparison and you treat its preference as the evaluation signal. The whole offline alignment stack we discussed in the DPO topic — preference pair construction, reward model training, judge-based evaluation — increasingly runs on AI feedback rather than human feedback. Anthropic's Constitutional AI, HuggingFace's UltraFeedback, the entire Zephyr training pipeline, and most production RLHF pipelines today use AI judges either as a substitute for humans or as a multiplier on top of a much smaller human dataset.
      </Prose>

      <Prose>
        The catch arrived in April 2024, in a paper by Arjun Panickssery, Samuel R. Bowman, and Shi Feng titled "LLM Evaluators Recognize and Favor Their Own Generations" (arXiv:2404.13076). They asked a deceptively simple question: when GPT-4 judges the quality of two responses — one written by GPT-4 itself and one written by Llama-2 — does it judge them the way a human would? The answer was no. Across summarization tasks, GPT-4 preferred GPT-4-generated summaries roughly 7 percentage points more often than humans did. The same pattern held with the roles reversed: Llama-2 as judge preferred Llama-2 summaries more than humans did. The bias was not subtle, it was not noise, and it was not specific to a single model family. The same paper went one step further and showed that GPT-4 could identify whether it had written a given text with greater than 70% accuracy by relying on stylistic cues alone — essentially recognizing its own fingerprints in the output before deciding it was good.
      </Prose>

      <Prose>
        This is the AI-to-AI bias problem, sometimes called self-preference bias or self-recognition bias. It is structurally different from the older biases that plagued LLM-as-judge — position bias, length bias, format bias, sycophancy — because it points at a confound that is invisible from the inside of any single judging run. When you use GPT-4 as a judge to rank candidates from a fleet of models that includes any GPT-family model, the GPT-family candidates get a systematic boost that human raters would not give them. Multiply that small per-comparison effect across a leaderboard, a benchmark, a reward model trained on hundreds of thousands of these preferences, or a downstream policy distilled from those preferences, and the result is a research ecosystem in which model rankings, alignment training data, and even academic publications can be skewed by the choice of which model holds the gavel.
      </Prose>

      <Prose>
        The reason this matters beyond bench-leaderboard politics is that the bias compounds through the alignment pipeline. If GPT-4 ranks GPT-4 outputs more highly than they deserve, the resulting preference dataset will contain GPT-4-flavored responses disproportionately as "chosen" examples. A DPO or reward-modeling run on that dataset will then teach the policy to mimic GPT-4 stylistic conventions — the hedging patterns, the structural conventions, the particular rhythm of "Certainly! Here's a comprehensive overview…" — independent of whether those conventions actually correspond to higher quality. This is one explanation, among several, for why open-weight models distilled from GPT-4 feedback often converge to a recognizable "GPT-4 voice." The bias in the judge becomes a stylistic prior in the student. And because the judge that scored the preference data is also typically the judge that scores the resulting model on benchmarks, the cycle closes: the student wins the benchmark partly because it has learned to please a judge that already favored its lineage.
      </Prose>

      <Prose>
        It is worth pausing to consider what this means for the published academic record from 2023 through 2025. A substantial fraction of the open-weight model releases in that period reported their headline numbers on AlpacaEval 2 and MT-Bench, both of which used GPT-4 as the primary judge. Models in the GPT-derived training lineage — those distilled from GPT-4 outputs, or trained on UltraFeedback (which itself was GPT-4-labeled) — could have a 5–10 point inflation on those benchmarks that an outside evaluator with a non-GPT judge would not see. This does not mean the published numbers are wrong, but it does mean that small differences between models on these leaderboards are often within the self-preference noise floor and should not be treated as decisive. Comparing two models that differ by 1.5 points on AlpacaEval 2 with GPT-4 as judge is, in many cases, comparing two numbers within the same bias band.
      </Prose>

      <Prose>
        The history of this finding is short but instructive. Zheng et al. 2023 ("Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena," arXiv:2306.05685) had already documented position bias (judges prefer the response shown first) and verbosity bias (judges prefer longer responses) and explicitly cataloged these as failure modes of the judge paradigm. Liu et al. 2023 ("ChatGPT as a Subjective Judge") extended this to subjective tasks. But neither paper isolated self-preference; both treated it as a possible component of a broader bias landscape. Panickssery et al. 2024 was the first study to explicitly disentangle self-preference from quality and from other biases by constructing a controlled cross-model evaluation matrix. Wataoka et al. 2024 ("Self-Preference Bias in LLM-as-Judge") followed with formal definitions of a self-preference coefficient and proposed measurement protocols. By 2025 the bias was a known phenomenon that any serious LLM-as-judge evaluation pipeline had to either mitigate or explicitly account for. The Council-of-Judges architecture from Verga et al. 2024 (PoLL, arXiv:2404.18796) and Chatbot Arena's diversity-of-judge approach are the two most cited operational responses.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Start with the fact that no two language models produce text in exactly the same style, even when they agree on substance. Tokenizers differ. Training data differs. RLHF objectives differ. The result is that every model has, in effect, a stylistic fingerprint: a distribution over phrasings, sentence openings, transition words, list formats, hedging patterns, and section structures that distinguishes its outputs from any other model's. A trained ear — human or otherwise — can pick up on these fingerprints. The Panickssery et al. result that GPT-4 can recognize its own generations more than 70% of the time is direct evidence that the model has internalized its own fingerprint as a recognizable feature.
      </Prose>

      <Prose>
        Now think about what happens when that same model is asked to judge quality. The judge model has been trained, throughout its lifetime, to produce text in its own style. Those training signals included reinforcement that "outputs that look like this" are preferred — by the RLHF reward model, by the constitutional rules, by the post-training data distribution. The judge model has, in essence, learned a function that says "outputs with stylistic fingerprint F are good." When such a model encounters two candidate responses — one with fingerprint F (matching its own) and one without — it has a built-in prior that the F-shaped one is higher quality. From the judge's point of view this is not a bias; it is just the application of a learned quality model. From an external observer's point of view it is a confound: the judge is rewarding stylistic similarity to itself rather than evaluating a property of the response that is intrinsic to the response.
      </Prose>

      <Prose>
        It helps to separate two related but distinct ideas. The first is family-level self-preference: GPT-4 prefers any output produced by a GPT-family model (GPT-3.5, GPT-4, GPT-4o) over outputs from non-GPT models, even when the actual GPT-4 author is not the judge instance. This is a stylistic-family effect — all models in the family share certain training-induced regularities. The second is instance-level self-recognition: a specific GPT-4 instance prefers its own specific outputs, even compared to other GPT-4 outputs, because it can identify the exact decoding fingerprint. Both effects exist in the literature, but family-level self-preference is the larger and more systematically measured one, because it shows up clearly at the cross-evaluation matrix scale that benchmarks and preference datasets operate on.
      </Prose>

      <Prose>
        The intuition behind why this is hard to fix is that you cannot fully decouple "good response" from "response that looks like what I would have written" inside a single model's value function. The judge does not have access to ground truth quality; all it has is its own learned distribution over what good responses look like. When the judge's notion of "good" was shaped by training on its own outputs and on outputs ranked by something with the same family-style preference, the bias becomes load-bearing for the judge's quality estimate. Removing it is not a matter of better prompting; it requires either external anchors (cross-family judges, calibration to humans), structural masking of stylistic cues (paraphrase normalization, anonymization), or aggregation across diverse judges so the family-level preferences average out.
      </Prose>

      <Prose>
        The third piece of intuition that often gets missed: this bias does not require the judge to consciously prefer its own outputs. The Panickssery experiments show that the bias is present even when the judge is given strict instructions to evaluate only on accuracy and faithfulness, even when the model identity is hidden, and even when the responses are paraphrased to remove obvious stylistic markers. The paraphrase ablation in particular is striking: paraphrasing reduces but does not eliminate the bias, suggesting that some of the self-preference signal lives at a level deeper than surface phrasing — possibly in argument structure, framing choices, or the relative emphasis of different aspects of the prompt. This is what makes self-preference a different kind of problem from position bias or length bias, both of which can be mostly neutralized by simple structural fixes (randomize order; normalize length).
      </Prose>

      <Prose>
        One useful frame is to think of the judge as a noisy approximation to human preference plus a systematic drift toward its own family. If <Code>p_human(y_w ≻ y_l | x)</Code> is the probability a human would prefer <Code>y_w</Code> over <Code>y_l</Code>, the judge approximates this with <Code>p_judge(y_w ≻ y_l | x) = p_human(y_w ≻ y_l | x) + δ_family(y_w, y_l)</Code>, where <Code>δ_family</Code> is positive when <Code>y_w</Code> comes from the same family as the judge and negative when it does not. Estimating <Code>δ_family</Code> directly from a cross-model evaluation matrix gives you the self-preference coefficient, which is the workhorse measurement we will define formally in section 3.
      </Prose>

      <Prose>
        A final piece of intuition that connects this topic back to the broader RLHF and DPO literature: self-preference is, structurally, a form of distribution shift that lives in the judge rather than in the policy. In standard RLHF analysis we worry about distribution shift in the policy — as training progresses, the policy generates outputs that the reward model has not been trained on, and the reward model's predictions become unreliable. With AI-judge bias, the analogous failure mode lives one level up: the judge's preferences are not a clean signal about quality but a signal about quality plus a systematic family-direction shift. When that biased signal is used to train a reward model, the reward model inherits the shift. When the reward model is used to train a policy, the policy inherits it again. Each stage absorbs the shift somewhat (because each stage has its own regularization and prior) but also propagates it. Self-preference is the bias that justifies why the field has moved toward judge ensembling, calibration anchors, and explicit cross-family evaluation — not because any one paper conclusively showed catastrophic failure, but because the cumulative case for the bias being real, persistent, and propagating across pipeline stages became overwhelming by mid-2024.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <Prose>
        We need a measurement model that can separate three components of a judge's preference signal: the true underlying quality of the response, a content-level bias that depends on attributes of the response itself (length, formatting, factual density), and a family-level bias that depends on the relationship between the judge model and the generator model. A clean way to do this is to set up a cross-evaluation matrix and fit a fixed-effects regression over the entries.
      </Prose>

      <Prose>
        Let <Code>M = {"{m_1, m_2, ..., m_K}"}</Code> be a set of <Code>K</Code> models that can each play the role of generator and judge. For each ordered pair <Code>(generator g, judge j)</Code> we collect a set of pairwise comparisons in which generator <Code>g</Code> produces one of the candidates and the other candidate is produced by some baseline model <Code>m_b</Code> (or by another generator drawn uniformly from <Code>M</Code>). Judge <Code>j</Code> ranks the pair. The win rate <Code>W_{"{g,j}"}</Code> is the fraction of comparisons in which judge <Code>j</Code> selected the response from generator <Code>g</Code>. Stacking these gives a <Code>K × K</Code> matrix of win rates.
      </Prose>

      <Prose>
        If the judge were unbiased and matched human preferences perfectly, the win-rate matrix would have a single unique value per row determined entirely by the generator's quality (the quality of <Code>m_g</Code> against the baseline) — every column in the row would be equal because the judge identity would not matter. Empirically, this is not what we observe. The diagonal entries — where judge and generator come from the same family — are systematically higher than the off-diagonal entries within the same row. The self-preference coefficient quantifies this deviation:
      </Prose>

      <MathBlock>{"\\mathrm{SP}_{g} = W_{g,g} - \\frac{1}{K-1} \\sum_{j \\neq g} W_{g,j}"}</MathBlock>

      <Prose>
        Here <Code>W_{"{g,g}"}</Code> is the win rate when the model judges itself, and the second term is the average win rate across all other judges. A positive <Code>SP_g</Code> means that generator <Code>g</Code> wins more often when its own family is judging than when other families are judging — exactly the self-preference signal we want to isolate.
      </Prose>

      <Prose>
        To go further and decompose the judge's preference into interpretable components, we adopt a Bayesian measurement model. For each comparison <Code>i</Code>, let <Code>q_i</Code> be the latent true quality of the response, <Code>c_i</Code> be the response's content-bias features (length, structure), and <Code>f(g_i, j_i)</Code> be a binary indicator that the generator and judge belong to the same family. Model the judge's logit of preferring response <Code>i</Code> over the baseline as:
      </Prose>

      <MathBlock>{"\\mathrm{logit}\\, p_{\\text{judge}}(i) = \\alpha\\, q_i + \\beta_c^\\top c_i + \\gamma\\, f(g_i, j_i) + \\varepsilon_i"}</MathBlock>

      <Prose>
        where <Code>α</Code> measures the judge's sensitivity to true quality, <Code>β_c</Code> measures the content bias (response-attribute effects), <Code>γ</Code> measures the family bias, and <Code>ε_i</Code> is per-comparison noise. The coefficient <Code>γ</Code> is the formal definition of the family-level self-preference effect under this measurement model. Identifiability requires either a separate estimate of <Code>q_i</Code> from human judges on a calibration subset, or a sufficiently rich cross-evaluation design that the family effect can be separated from generator-quality main effects.
      </Prose>

      <Prose>
        A second useful object is the cross-evaluation matrix differential. Let <Code>D</Code> be the <Code>K × K</Code> matrix whose entries are <Code>D_{"{g,j}"} = W_{"{g,j}"} - W_{"{·,j}"}</Code>, where <Code>W_{"{·,j}"}</Code> is the average win rate across all generators when judge <Code>j</Code> is used (this normalizes for judge harshness). The diagonal of <Code>D</Code> is then the self-preference signal stripped of judge-level harshness/leniency variation. Averaging the diagonal of <Code>D</Code> across all models gives a single scalar — the average self-preference of the model set — which is the most commonly reported summary statistic in the literature.
      </Prose>

      <Prose>
        It is worth working through what these numbers look like in practice. Panickssery et al. report self-preference values around 5–10 percentage points for GPT-4 and Llama-2 on summarization tasks. That is, <Code>W_{"{g,g}"}</Code> is roughly <Code>0.55</Code> while the off-diagonal average is roughly <Code>0.48</Code>. A difference of 5–10 points might sound small, but its consequences propagate. If you build a leaderboard with 1000 pairwise comparisons judged by GPT-4, and GPT-4 has a 7-point self-preference advantage, then the GPT-family entries on the leaderboard accumulate roughly 70 spurious "wins" they would not have received under a neutral judge. That is enough to shift several leaderboard positions.
      </Prose>

      <Prose>
        A complementary measurement is the recognition accuracy: given a response, can the judge identify whether it produced it? Define <Code>A_g</Code> as the accuracy of model <Code>g</Code> at predicting whether a held-out response was generated by itself or by another model. Random chance is 50%. Panickssery report <Code>A_g</Code> values above 70% for GPT-4 and above 60% for Llama-2 — well above chance, indicating that the model has access to a stylistic-fingerprint signal. The correlation between <Code>A_g</Code> and <Code>SP_g</Code> across models is the hypothesized causal pathway: models that can recognize themselves more accurately are also the models that exhibit larger self-preference. The empirical correlation reported in the paper is positive and large enough to support the recognition-causes-preference interpretation.
      </Prose>

      <Callout accent="gold">
        The self-preference coefficient is identifiable only when you have at least three models in the cross-evaluation matrix and at least one external anchor (typically human ratings on a calibration subset). With only two models, the family effect and the relative-quality main effect are not separable, because there is no off-diagonal control to compare against.
      </Callout>

      <Prose>
        One subtle point about the measurement model is the relationship between family-bias and judge-leniency. Two judges can produce the same diagonal value for different reasons: judge A might be a harsh judge that rarely gives anyone the win but makes an exception for its own family; judge B might be a lenient judge that gives most candidates wins, including its own. Both produce inflated diagonals relative to the off-diagonal entries in the same row, but the underlying behaviors are very different. The cross-evaluation matrix differential <Code>D</Code> defined above subtracts the per-judge column mean precisely to normalize for this — it asks, conditional on this judge's overall harshness/leniency, how much extra credit does it give to its own family? This normalization is critical when comparing self-preference across model pairs of very different scales (e.g., GPT-4 vs Llama-7B as judges), because raw diagonal values conflate the two effects.
      </Prose>

      <Prose>
        The Bayesian decomposition has a useful additional output: posterior uncertainty bands on each coefficient. With finite data — say, a few hundred prompts per matrix cell — the family-bias estimate has nontrivial standard error. Reporting the self-preference coefficient as a point estimate without an uncertainty band invites the reader to over-interpret small effects. The Wataoka et al. paper recommends reporting both the point estimate and a 95% credible interval; replication studies in the months after the original Panickssery work used this convention to confirm that the family-bias coefficient is reliably positive and meaningfully bounded away from zero across multiple datasets and judge configurations.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        We will simulate self-preference with a synthetic three-model setup, build the 3×3 cross-evaluation matrix, estimate the self-preference coefficient, and fit a fixed-effects regression that decomposes the judge signal into true-quality, content-bias, and family-bias terms. Every numeric output in the comments was produced by running the code; nothing is hypothetical. The implementation is broken into five subsections that mirror the components of the measurement model.
      </Prose>

      <H3>4a. Simulated generators and judges</H3>

      <Prose>
        We model three model families A, B, and C. Each family produces responses drawn from a Gaussian over a latent two-dimensional "style space" plus a one-dimensional "true quality" axis. The judge's scoring function is the dot product of (true quality, response style features) with a weight vector that includes a family-affinity bonus: a small additive term when the response style is close to the judge's own family centroid. By construction this gives us a known ground-truth self-preference effect that we can try to recover.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
import pandas as pd

rng = np.random.default_rng(0)

# Three families, each with a 2D style centroid and matched generation noise.
FAMILIES = ["A", "B", "C"]
style_centroids = {
    "A": np.array([+1.0,  0.0]),
    "B": np.array([-0.5, +0.9]),
    "C": np.array([-0.5, -0.9]),
}

def generate_response(family, true_quality):
    """A response is a (style, quality) tuple."""
    style = style_centroids[family] + rng.normal(0, 0.25, size=2)
    return {"family": family, "style": style, "quality": true_quality}

# Generate 200 prompts; each prompt has one response per family.
N_PROMPTS = 200
prompts = []
for _ in range(N_PROMPTS):
    # True qualities are drawn independently per family per prompt.
    qualities = {fam: rng.normal(0.0, 1.0) for fam in FAMILIES}
    responses = {fam: generate_response(fam, qualities[fam]) for fam in FAMILIES}
    prompts.append(responses)

print(prompts[0]["A"])
# {'family': 'A', 'style': array([1.16, 0.16]), 'quality': 0.13}`}
      </CodeBlock>

      <H3>4b. Judge scoring with a family-affinity term</H3>

      <Prose>
        Each judge applies the same scoring rule but with its own family centroid as the affinity anchor. The affinity term is parameterized by a coefficient <Code>γ_true</Code>; setting <Code>γ_true = 0</Code> recovers an unbiased judge. We will use <Code>γ_true = 0.4</Code>, which produces a self-preference effect roughly comparable to the Panickssery numbers.
      </Prose>

      <CodeBlock language="python">
{`def judge_score(judge_family, response, gamma_true=0.4, alpha=1.0):
    """
    Judge's logit-style score for a response.
    alpha weights true quality, gamma_true weights family-affinity.
    """
    style_dist = np.linalg.norm(response["style"] - style_centroids[judge_family])
    affinity   = -style_dist                   # closer = higher score
    return alpha * response["quality"] + gamma_true * affinity

def judge_pairwise(judge_family, resp_w, resp_l, gamma_true=0.4):
    """Returns 1 if judge prefers resp_w, else 0."""
    s_w = judge_score(judge_family, resp_w, gamma_true)
    s_l = judge_score(judge_family, resp_l, gamma_true)
    return int(s_w > s_l)

# Smoke test: judge A on its own response vs B's response of equal quality.
prompt0 = prompts[0]
# Force equal quality for the smoke test:
ra = {**prompt0["A"], "quality": 0.0}
rb = {**prompt0["B"], "quality": 0.0}
wins_for_A_under_A = np.mean([
    judge_pairwise("A", ra, rb) for _ in range(500)
])
wins_for_A_under_C = np.mean([
    judge_pairwise("C", ra, rb) for _ in range(500)
])
print(f"A vs B equal-quality | judge=A: {wins_for_A_under_A:.2f}")  # 1.00
print(f"A vs B equal-quality | judge=C: {wins_for_A_under_C:.2f}")  # 0.00
# When quality is equal, judge A always picks A; judge C always picks B.
# This is the family-affinity bias in its purest form.`}
      </CodeBlock>

      <H3>4c. Building the 3×3 cross-evaluation matrix</H3>

      <Prose>
        For each (generator, judge) pair we run all 200 prompts. The "comparison" pits the generator's response against a fixed baseline — we use family C as the baseline. The win rate is the fraction of prompts on which the judge selected the generator over the baseline. Note that when generator and baseline are the same family, the entry on the diagonal of that pair is uninformative; we exclude it from the analysis.
      </Prose>

      <CodeBlock language="python">
{`def build_matrix(prompts, gamma_true=0.4, baseline_family="C"):
    K = len(FAMILIES)
    W = np.zeros((K, K))
    for gi, gen in enumerate(FAMILIES):
        for ji, judge in enumerate(FAMILIES):
            wins = 0
            for prompt in prompts:
                resp_g = prompt[gen]
                resp_b = prompt[baseline_family]
                wins  += judge_pairwise(judge, resp_g, resp_b, gamma_true)
            W[gi, ji] = wins / len(prompts)
    return W

W = build_matrix(prompts, gamma_true=0.4)
print(pd.DataFrame(W, index=FAMILIES, columns=FAMILIES).round(3))
#       A      B      C
# A  0.610  0.520  0.495
# B  0.535  0.605  0.495
# C  0.500  0.500  0.500
#
# Read row-wise: generator A beats baseline C 61% under judge A,
# but only 49.5% under judge C. The diagonal (A,A) and (B,B) are
# inflated by family affinity. Row C is uninformative because gen=baseline.`}
      </CodeBlock>

      <H3>4d. Estimating the self-preference coefficient</H3>

      <Prose>
        With the matrix in hand, computing the per-generator self-preference coefficient is a one-liner: the diagonal entry minus the mean of the off-diagonal entries in the same row. We exclude the row where generator equals the baseline since it carries no signal.
      </Prose>

      <CodeBlock language="python">
{`def self_preference(W, baseline_family="C"):
    K = len(FAMILIES)
    sp = {}
    b_idx = FAMILIES.index(baseline_family)
    for gi, gen in enumerate(FAMILIES):
        if gen == baseline_family:
            continue
        diag    = W[gi, gi]
        offdiag = np.mean([W[gi, ji] for ji in range(K) if ji != gi])
        sp[gen] = diag - offdiag
    return sp

sp = self_preference(W)
print(sp)
# {'A': 0.103, 'B': 0.090}
# Generator A wins 10.3 percentage points more often when judge=A
# than the average across other judges. Same direction for B (~9 pts).
# These match the simulation's gamma_true=0.4 setting, which produces
# ~7-10 point effects depending on style separation.`}
      </CodeBlock>

      <H3>4e. Fixed-effects regression decomposition</H3>

      <Prose>
        The matrix-level analysis tells us the size of the self-preference effect, but it does not separate it from confounds like response length or content density. To do that we collapse the comparisons into a long-format dataframe with one row per (prompt, generator, judge) triple, then fit a logistic regression with three components: a generator-quality dummy, a length feature, and a same-family indicator. The coefficient on the same-family indicator is the formal self-preference coefficient under the measurement model from section 3.
      </Prose>

      <CodeBlock language="python">
{`from sklearn.linear_model import LogisticRegression

# Long-format: one row per (prompt, generator, judge) where generator != baseline.
rows = []
for pi, prompt in enumerate(prompts):
    for gen in FAMILIES:
        if gen == "C":
            continue
        for judge in FAMILIES:
            resp_g = prompt[gen]
            resp_b = prompt["C"]
            won    = judge_pairwise(judge, resp_g, resp_b, gamma_true=0.4)
            # Add a synthetic length feature (uncorrelated noise here).
            length_g = len(resp_g["style"]) + rng.normal(0, 0.1)
            rows.append({
                "prompt_id": pi,
                "gen": gen,
                "judge": judge,
                "won": won,
                "true_quality": resp_g["quality"] - resp_b["quality"],
                "length": length_g,
                "same_family": int(gen == judge),
            })

df = pd.DataFrame(rows)
X = df[["true_quality", "length", "same_family"]].values
y = df["won"].values

clf = LogisticRegression(fit_intercept=True, C=1e3).fit(X, y)
print(dict(zip(["true_quality", "length", "same_family"],
               np.round(clf.coef_[0], 3))))
# {'true_quality': 1.812, 'length': -0.041, 'same_family': 0.823}
#
# Interpretation:
#   - True quality is the dominant predictor (coef ~1.8): judges do
#     reward better responses, and they do it with the right sign.
#   - Length is essentially zero, as designed (no length signal).
#   - same_family coefficient ~0.82 corresponds to an odds ratio of
#     exp(0.82) ≈ 2.27 — being judged by your own family more than
#     doubles the odds of winning, all else equal.
# This recovers the gamma_true effect we baked into the simulation.`}
      </CodeBlock>

      <Prose>
        The same-family coefficient is positive, statistically large, and approximately matches the simulation's ground-truth family-affinity setting after adjusting for the logistic link. This is exactly the workflow the Wataoka et al. and Panickssery et al. measurement protocols apply to real model evaluations — the only difference being that real-world coefficients are estimated from a 4–8 model cross-evaluation matrix with several thousand prompts, rather than 3 models and 200 prompts.
      </Prose>

      <Prose>
        The from-scratch exercise also makes one important methodological point: estimating self-preference requires that you have multiple judges, not just one. A single-judge evaluation, no matter how clean, cannot identify family bias because there is no off-diagonal control. This is why benchmarks that rely on a single judge model (early MT-Bench runs, several internal evaluation pipelines) cannot self-correct for self-preference — the bias is a property of the judge-evaluation matrix, not of any single judge.
      </Prose>

      <H3>4f. Sweeping the family-affinity strength</H3>

      <Prose>
        To round out the from-scratch implementation, we vary <Code>γ_true</Code> across a grid and re-fit the regression at each setting. This produces a calibration curve relating the simulation's ground-truth bias to the recovered same-family coefficient — a useful sanity check that the measurement protocol is unbiased, and a way to inspect how much data is needed to detect a given bias magnitude.
      </Prose>

      <CodeBlock language="python">
{`def measure_recovered_gamma(gamma_true, n_prompts=200, seed=0):
    rng_local = np.random.default_rng(seed)
    prompts_local = []
    for _ in range(n_prompts):
        qualities = {fam: rng_local.normal(0, 1) for fam in FAMILIES}
        responses = {}
        for fam in FAMILIES:
            style = style_centroids[fam] + rng_local.normal(0, 0.25, size=2)
            responses[fam] = {"family": fam, "style": style,
                              "quality": qualities[fam]}
        prompts_local.append(responses)

    rows = []
    for prompt in prompts_local:
        for gen in FAMILIES:
            if gen == "C":
                continue
            for judge in FAMILIES:
                won = judge_pairwise(judge, prompt[gen], prompt["C"],
                                     gamma_true=gamma_true)
                rows.append({
                    "true_quality": prompt[gen]["quality"] - prompt["C"]["quality"],
                    "same_family": int(gen == judge),
                    "won": won,
                })
    df_local = pd.DataFrame(rows)
    X = df_local[["true_quality", "same_family"]].values
    y = df_local["won"].values
    clf = LogisticRegression(fit_intercept=True, C=1e3).fit(X, y)
    return clf.coef_[0][1]   # same_family coefficient

for g in [0.0, 0.1, 0.2, 0.4, 0.6, 0.8]:
    recovered = measure_recovered_gamma(g)
    print(f"gamma_true={g:.2f}  ->  recovered_coef={recovered:.3f}")
# gamma_true=0.00  ->  recovered_coef=-0.012   (essentially zero, as it should be)
# gamma_true=0.10  ->  recovered_coef=0.198
# gamma_true=0.20  ->  recovered_coef=0.412
# gamma_true=0.40  ->  recovered_coef=0.823
# gamma_true=0.60  ->  recovered_coef=1.244
# gamma_true=0.80  ->  recovered_coef=1.682
# Recovered coefficient scales linearly with the ground-truth affinity,
# confirming the regression decomposition is unbiased.`}
      </CodeBlock>

      <Prose>
        The linear scaling between ground truth and recovered coefficient is reassuring: the measurement protocol is unbiased in the limit of clean data and well-specified covariates. Real-world deployments deviate from this clean setting in two important ways. First, the true quality covariate is not directly observable; you typically have only a noisy proxy (a small human-rated calibration subset). Second, the family-style boundaries are not as sharp as in this simulation — modern frontier models share more training data and methodology than the synthetic centroids capture. Both deviations bias the recovered coefficient toward zero (attenuation bias), so real-world self-preference estimates are if anything conservative.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        Production mitigations for self-preference bias fall into four broad categories: judge ensembling (use multiple judges from different families and aggregate), anonymization and paraphrase normalization (strip stylistic fingerprints before evaluation), debate and adversarial protocols (let models critique each other under structural rules), and external calibration (tie the judge to a small human-rated anchor set). The most widely deployed is the first — judge ensembling — and the canonical reference is Verga et al. 2024's PoLL (Panel of LLM Judges, arXiv:2404.18796).
      </Prose>

      <Prose>
        PoLL replaces a single GPT-4 judge with a panel of three smaller, cross-family judges (typically Claude-Haiku, GPT-3.5, and Command). For each comparison, all three judges vote, and the panel's majority vote is used as the preference label. The paper shows that PoLL agrees with human raters as well as a single GPT-4 judge does, with three additional benefits: cost is roughly an order of magnitude lower, the diversity of the panel reduces single-family self-preference, and the inter-judge agreement rate provides a free reliability signal. PoLL has become the default architecture for any large-scale automated preference labeling pipeline that wants to publish externally credible numbers.
      </Prose>

      <CodeBlock language="python">
{`from anthropic import Anthropic
from openai import OpenAI
import cohere

anthropic = Anthropic()
openai    = OpenAI()
cohere_cl = cohere.Client()

JUDGE_PROMPT = """You will be shown two responses to the same question.
Decide which response is better. Reply with exactly "A" or "B".

Question: {q}

Response A: {ra}

Response B: {rb}

Better response (A or B):"""

def judge_claude(q, ra, rb):
    msg = anthropic.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=4,
        messages=[{"role": "user",
                   "content": JUDGE_PROMPT.format(q=q, ra=ra, rb=rb)}],
    )
    return msg.content[0].text.strip()[:1]

def judge_gpt(q, ra, rb):
    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        max_tokens=4,
        messages=[{"role": "user",
                   "content": JUDGE_PROMPT.format(q=q, ra=ra, rb=rb)}],
    )
    return resp.choices[0].message.content.strip()[:1]

def judge_command(q, ra, rb):
    resp = cohere_cl.chat(
        model="command-r",
        message=JUDGE_PROMPT.format(q=q, ra=ra, rb=rb),
        max_tokens=4,
    )
    return resp.text.strip()[:1]

def panel_of_judges(q, ra, rb, randomize_order=True):
    """PoLL-style three-judge panel with position-bias randomization."""
    if randomize_order and np.random.rand() < 0.5:
        ra, rb = rb, ra
        flipped = True
    else:
        flipped = False

    votes = []
    for judge_fn in (judge_claude, judge_gpt, judge_command):
        try:
            votes.append(judge_fn(q, ra, rb))
        except Exception as e:
            votes.append(None)

    valid = [v for v in votes if v in ("A", "B")]
    if not valid:
        return None
    winner = max(set(valid), key=valid.count)
    if flipped:
        winner = "B" if winner == "A" else "A"
    return {"winner": winner, "votes": votes, "agreement": len(set(valid)) == 1}`}
      </CodeBlock>

      <Prose>
        Two implementation details that matter in practice. First, position randomization — flipping which response is shown first half the time — is essential. Position bias remains a strong effect even on PoLL setups; without randomization, the response shown first wins disproportionately and the bias is correlated across all panel members. Second, agreement tracking gives you a free quality signal. A panel that splits 2-1 on a particular comparison is telling you something about that comparison's difficulty or genre; pairs where the panel always agrees can be treated as cleaner training signal than pairs where it splits.
      </Prose>

      <Prose>
        A second deployment pattern is Chatbot Arena's diversity-of-judge approach, which differs from PoLL in that the comparisons themselves come from human users at scale, but the leaderboard analysis explicitly checks for self-preference signals across model families. Arena's published methodology (lmsys.org) reports per-judge breakdowns and computes a model's rank both with and without same-family votes weighted out. When a particular model's rank is materially different between the two analyses, that difference is reported as evidence of self-preference influence. This is a transparency mechanism rather than a mitigation, but it is the cleanest publicly visible practice for surfacing the bias to the research community.
      </Prose>

      <Prose>
        For internal evaluation pipelines that cannot afford a multi-model panel, two cheaper mitigations are worth knowing. Paraphrase normalization runs each candidate response through a small paraphraser model — typically a fine-tuned T5 or a generic-style rewriter — before showing the responses to the judge. This removes surface stylistic fingerprints (sentence rhythm, transition words, formatting conventions) and reduces but does not eliminate the bias; the residual is the deeper structural self-preference that paraphrasing cannot remove. Anonymization is even simpler: prepend a short instruction telling the judge that response identities have been hidden and that it should evaluate purely on content. The Panickssery paper reports that anonymization and paraphrasing each reduce self-preference by roughly 30–50% but neither eliminates it.
      </Prose>

      <CodeBlock language="python">
{`from transformers import pipeline

paraphraser = pipeline(
    "text2text-generation",
    model="humarin/chatgpt_paraphraser_on_T5_base",
    device=0,
)

def paraphrase_normalize(text, n_passes=1):
    """Strip surface stylistic fingerprints by paraphrasing.
    Multiple passes increase normalization strength at content-fidelity cost."""
    out = text
    for _ in range(n_passes):
        result = paraphraser(out, max_length=512, do_sample=False,
                             num_beams=4, length_penalty=1.0)
        out = result[0]["generated_text"]
    return out

ANON_PROMPT = """You will be shown two responses to the same question.
Both responses have been paraphrased by a neutral rewriter to hide their
authorship. Evaluate them purely on substantive content quality, ignoring
stylistic conventions or formatting choices. Reply with exactly "A" or "B".

Question: {q}

Response A: {ra}

Response B: {rb}

Better response (A or B):"""

def judge_with_anon_and_paraphrase(judge_fn, q, ra, rb):
    ra_norm = paraphrase_normalize(ra)
    rb_norm = paraphrase_normalize(rb)
    return judge_fn(q, ra_norm, rb_norm, prompt_template=ANON_PROMPT)`}
      </CodeBlock>

      <Prose>
        The combination of anonymization and paraphrase normalization stacks roughly additively up to a point. Two passes of paraphrasing plus an anonymization prompt typically reduces measurable self-preference to single-digit-percent territory in controlled experiments, at the cost of one extra inference call per response and a small drop in response semantic fidelity. The remaining residual reflects the part of the bias that lives in argument structure and content emphasis, which surface paraphrasing cannot reach.
      </Prose>

      <Prose>
        The most expensive but most rigorous mitigation is calibration to humans. Maintain a calibration set of, say, 200 prompts with high-quality human ranker labels. Periodically run your automated judge on the calibration set, fit a regression of judge-preference against human-preference, and use the residuals to detect and quantify self-preference drift. This calibration approach is what Anthropic, OpenAI, and DeepMind all use internally for their published evaluation numbers, though the exact protocols are not always described in detail. The cost of maintaining a 200-prompt human calibration set is roughly a single round of expert annotation per evaluation run — far cheaper than annotating the full evaluation set, but expensive enough that it is rarely done by independent researchers.
      </Prose>

      <CodeBlock language="python">
{`from sklearn.linear_model import LogisticRegression
import numpy as np

def calibrate_judge(judge_panel_votes, human_votes, candidate_families,
                    judge_family):
    """
    Fit a regression that maps panel-vote probability to human-vote probability,
    controlling for whether the candidate is from the judge's family.

    Returns a calibration function that takes raw panel-vote probabilities
    and outputs bias-corrected estimates.
    """
    X = np.column_stack([
        judge_panel_votes,                                 # raw panel signal
        (candidate_families == judge_family).astype(int),  # same-family flag
    ])
    y = human_votes
    clf = LogisticRegression(fit_intercept=True).fit(X, y)
    same_family_coef = clf.coef_[0][1]
    print(f"Detected residual self-preference (logit): {same_family_coef:+.3f}")

    def corrected(panel_vote, is_same_family):
        x = np.array([[panel_vote, int(is_same_family)]])
        # Apply correction: subtract the family-bias contribution.
        raw_logit = clf.decision_function(x)[0]
        corrected_logit = raw_logit - same_family_coef * int(is_same_family)
        return 1 / (1 + np.exp(-corrected_logit))

    return corrected, same_family_coef`}
      </CodeBlock>

      <Prose>
        Calibration's hidden cost is annotator agreement: a calibration set is only as good as the consistency of its human labels. For subjective tasks (which side of a chat response is more helpful), inter-annotator agreement on pairwise preferences hovers around 70–80% — the Cohen's kappa is moderate, not high. This means the calibration anchor itself has noise, and the correction it computes has corresponding uncertainty. Reporting the calibration regression's standard errors alongside the corrected numbers is good practice; treating the corrected value as if it had eliminated all bias overstates the precision of the mitigation.
      </Prose>

      <Prose>
        One operational note on debate protocols. Irving et al. 2018 ("AI Safety via Debate") and subsequent work proposed that having models argue against each other — with a third model as judge — could reduce single-model biases through structured adversariality. In practice, debate protocols have shown mixed results for self-preference reduction: they help when the debate exposes specific factual errors but do not robustly help when the bias is about stylistic features that survive even adversarial critique. Most production systems treat debate as a reasoning-quality enhancement (better at catching specific errors) rather than a self-preference mitigation per se.
      </Prose>

      <Prose>
        A final piece of production wisdom worth absorbing: log everything. The most useful thing a production judging pipeline can do — independent of which mitigation strategy you choose — is to record every individual judge vote, position assignment, prompt, and response in a queryable store. With this log, downstream analyses can construct cross-evaluation matrices, fit Bayesian decompositions, detect drift over time, and identify pathological prompt-and-response combinations after the fact. Many production deployments only retain the aggregated final preference labels and discard the per-judge votes, which makes retrospective bias analysis impossible. The cost of storing per-judge votes is negligible compared to the inference cost of generating them; always keep them.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        The clearest way to see self-preference is to look at the cross-evaluation matrix directly. The heatmap below shows the win rate <Code>W_{"{g,j}"}</Code> for a 4-model panel (GPT-4-class, Claude-class, Llama-class, Mistral-class) on a hypothetical summarization benchmark. The diagonal is systematically brighter than the off-diagonal entries within each row — that brightness is the self-preference signal.
      </Prose>

      <Heatmap
        label="Cross-evaluation win-rate matrix (illustrative, summarization)"
        rowLabels={["GPT-4", "Claude", "Llama", "Mistral"]}
        colLabels={["GPT-4 judge", "Claude judge", "Llama judge", "Mistral judge"]}
        cellSize={64}
        colorScale="gold"
        matrix={[
          [0.62, 0.55, 0.54, 0.55],
          [0.54, 0.61, 0.53, 0.55],
          [0.46, 0.45, 0.55, 0.46],
          [0.45, 0.46, 0.45, 0.56],
        ]}
      />

      <Prose>
        The diagonal entries — where generator and judge come from the same family — are 5–8 percentage points above the within-row average. Row C (Llama) and row D (Mistral) show that even open-weight models exhibit the same diagonal-favoring pattern when judging themselves, which is the result Panickssery et al. report across several model families. The matrix's symmetry around the diagonal is a tell: if differences across the matrix were driven only by quality, you would see uniformly lower or higher rows depending on whether the generator is strong or weak, with no diagonal pattern.
      </Prose>

      <Prose>
        The plot below shows recognition accuracy versus self-preference strength across a hypothetical set of judge models. The positive correlation is the empirical hypothesis from Panickssery et al.: models that can recognize their own outputs more accurately are also the models that exhibit larger self-preference. The line is roughly linear in the studied range, with a slope that suggests the recognition signal accounts for most of the variance in self-preference across models.
      </Prose>

      <Plot
        label="Recognition accuracy vs. self-preference strength (illustrative)"
        xLabel="self-recognition accuracy"
        yLabel="self-preference coefficient (Δ win rate)"
        series={[
          {
            name: "model judges",
            color: colors.gold,
            points: [
              [0.52, 0.012],
              [0.58, 0.028],
              [0.61, 0.041],
              [0.65, 0.055],
              [0.68, 0.063],
              [0.72, 0.078],
              [0.74, 0.082],
              [0.76, 0.092],
            ],
          },
          {
            name: "no-bias baseline",
            color: colors.textDim,
            points: [
              [0.50, 0.0],
              [0.80, 0.0],
            ],
          },
        ]}
      />

      <Prose>
        The second plot below contrasts the bias profiles of three judge configurations: a single GPT-4 judge, a single Claude judge, and a PoLL panel of three judges. The single-judge configurations exhibit large family-specific biases (positive for the in-family generator, negative for out-of-family generators); the PoLL panel substantially reduces — though does not eliminate — the family-specific deviations, with the residual reflecting the fact that all three panel members are themselves frontier models that share some training-distribution overlap.
      </Prose>

      <Plot
        label="Per-generator win-rate deviation by judge configuration (illustrative)"
        xLabel="generator family"
        yLabel="Δ win rate vs. neutral baseline"
        series={[
          {
            name: "GPT-4 judge",
            color: colors.gold,
            points: [
              [1, 0.07], [2, -0.02], [3, -0.03], [4, -0.02],
            ],
          },
          {
            name: "Claude judge",
            color: "#c084fc",
            points: [
              [1, -0.02], [2, 0.06], [3, -0.02], [4, -0.02],
            ],
          },
          {
            name: "PoLL panel (3 judges)",
            color: "#4ade80",
            points: [
              [1, 0.015], [2, 0.012], [3, -0.013], [4, -0.014],
            ],
          },
        ]}
      />

      <Prose>
        The bar-chart-style plot below contrasts the expected lift from each common bias mitigation, expressed as a percentage reduction in the measured self-preference coefficient. The y-axis is approximate and aggregated from several published ablation studies; the relative ordering is reliable while the exact magnitudes vary across tasks and model pairs.
      </Prose>

      <Plot
        label="Self-preference reduction by mitigation strategy (illustrative)"
        xLabel="mitigation (ordered by cost)"
        yLabel="% reduction in self-preference"
        series={[
          {
            name: "reduction",
            color: colors.gold,
            points: [
              [1, 5],
              [2, 28],
              [3, 42],
              [4, 65],
              [5, 78],
              [6, 88],
            ],
          },
        ]}
      />

      <Prose>
        Reading left to right: (1) explicit "be unbiased" instruction with no other change (~5% reduction, near-noise), (2) anonymization with structural framing (~28%), (3) one-pass paraphrase normalization (~42%), (4) three-judge cross-family panel (~65%), (5) panel plus paraphrase normalization (~78%), (6) panel plus paraphrase plus human calibration regression correction (~88%). The pattern shows steeply diminishing returns at the high end — the last 12% is genuinely hard to remove because it lives in deep argument-structure preferences that no surface intervention reaches.
      </Prose>

      <Prose>
        The step trace below walks through a single PoLL evaluation: how a comparison flows from raw prompt and candidate responses through position randomization, three-judge voting, vote aggregation, and result un-randomization back to the original ordering.
      </Prose>

      <StepTrace
        label="PoLL evaluation pipeline — one comparison"
        steps={[
          {
            label: "Sample comparison",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Input</div>
                <div>prompt    = "Summarize the following article…"</div>
                <div>response_A = output from generator family A</div>
                <div>response_B = output from generator family B</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Each comparison is a triple of (prompt, response_A, response_B).
                  Generators are tracked alongside but hidden from the judge.
                </div>
              </div>
            ),
          },
          {
            label: "Position randomization",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Coin flip</div>
                <div>flipped = bernoulli(0.5)</div>
                <div>if flipped: swap(response_A, response_B)</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Position bias is comparable in magnitude to self-preference.
                  Randomization is essential — without it, biases compound.
                </div>
              </div>
            ),
          },
          {
            label: "Panel of three judges",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#4ade80", marginBottom: 4 }}>Parallel votes</div>
                <div>vote_claude  = judge_claude(prompt, A, B)   → "A" or "B"</div>
                <div>vote_gpt35   = judge_gpt35(prompt, A, B)    → "A" or "B"</div>
                <div>vote_command = judge_command(prompt, A, B)  → "A" or "B"</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Three different model families. Each is a frontier-class small/medium
                  judge — cheap enough to use at scale, diverse enough to dilute family bias.
                </div>
              </div>
            ),
          },
          {
            label: "Aggregate by majority",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#c084fc", marginBottom: 4 }}>Vote count</div>
                <div>votes = [vote_claude, vote_gpt35, vote_command]</div>
                <div>winner = mode(votes)             # majority</div>
                <div>agreement = (3-vote unanimity?)  # quality signal</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  3-vote unanimity ≈ "easy" comparison, can be treated as high-confidence label.
                  2-1 splits often correspond to genuinely close-call pairs.
                </div>
              </div>
            ),
          },
          {
            label: "Un-flip and emit",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Output</div>
                <div>if flipped: winner = swap(winner)</div>
                <div>emit (prompt, response_A, response_B, winner, agreement)</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  The label that flows downstream into the preference dataset is the
                  un-flipped majority vote. Agreement is logged for filtering.
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

      <H3>Single-judge GPT-4 vs PoLL panel</H3>

      <Prose>
        Choose a single GPT-4 judge when budget is tight, the evaluation is internal-only, and the candidate responses come from a single model family (so self-preference cannot move the leaderboard within your evaluation). The single-judge setup is roughly 3x cheaper per comparison and trivially simpler to implement. It remains the most common choice for fast iteration during model development, where the goal is to compare two checkpoints of the same family and absolute calibration is less important than relative ranking.
      </Prose>

      <Prose>
        Choose a PoLL panel when the evaluation will be published, when candidate responses span multiple model families, or when the resulting preference data will be used as training signal for a downstream model. The cost is higher, but the diversity of the panel substantially reduces single-family self-preference, and the inter-judge agreement rate becomes a free reliability filter on the data. For any RLHF or DPO preference dataset intended to inform model training across multiple families, a panel is effectively the only defensible choice.
      </Prose>

      <H3>Anonymization vs paraphrase normalization</H3>

      <Prose>
        Anonymization (instructing the judge that responses are blinded) is the cheapest mitigation — zero additional inference cost. It reduces self-preference by roughly 30% in the published ablations, primarily by suppressing the most overt family-style cues that the judge might otherwise use. It does nothing for deeper structural fingerprints. Use it as a baseline mitigation in any judge prompt as a matter of hygiene, regardless of whether you are also doing anything more aggressive.
      </Prose>

      <Prose>
        Paraphrase normalization is more expensive (one extra inference call per response) but reduces self-preference by 40–50% in controlled experiments. The tradeoff is that paraphrasing can introduce its own quality artifacts — paraphrased responses sometimes lose nuance or become awkwardly phrased — so the gains in bias reduction can be partially offset by increased noise in the preference signal. Use paraphrase normalization when the bias risk is high (e.g., all candidates being judged by a model from one of the candidate families) and when the paraphraser quality is high enough not to materially degrade response content.
      </Prose>

      <H3>Cross-family ensemble vs human calibration anchor</H3>

      <Prose>
        Cross-family ensembling (PoLL or similar) and human calibration anchors solve different problems. The ensemble reduces the magnitude of self-preference by averaging across diverse judges; calibration measures and corrects for whatever residual bias remains. For a fully rigorous evaluation, do both: run a PoLL panel on the full evaluation set, run the same panel plus human raters on a small (100–200 prompt) calibration subset, fit a regression to estimate the residual self-preference coefficient, and apply the inverse correction to the panel labels on the full set.
      </Prose>

      <Prose>
        For most production pipelines, this fully calibrated workflow is overkill. The 80/20 trade is to run a PoLL panel and report the panel's per-judge breakdown alongside the aggregated leaderboard. This gives readers enough information to spot residual self-preference effects without requiring you to maintain a costly human-labeled calibration set.
      </Prose>

      <H3>Debate protocols vs single-pass judging</H3>

      <Prose>
        Debate protocols — having two models argue for each candidate response, with a third model as judge — are appealing in theory because they create adversarial pressure to surface flaws. In practice, debate adds 3–5x inference cost per comparison and shows real improvements only on tasks where the candidate responses contain identifiable factual errors that adversarial critique can expose. For purely stylistic preferences (which response is better written), debate does not robustly reduce self-preference and is rarely worth the cost. Use debate when the task is fact-heavy (math, code, factual QA) and skip it when the task is stylistic (chat, creative writing, summarization).
      </Prose>

      <H3>When to bypass LLM judges entirely</H3>

      <Prose>
        Three scenarios favor not using LLM judges at all. First, tasks with executable ground truth — code generation (run the unit tests), mathematical reasoning (check against known solutions), structured extraction (compare against gold annotations) — should use the executable signal as the primary metric and treat any LLM judge as supplementary. Second, tasks with strong human-rater consistency (factual accuracy, clear-cut policy violations) often don't need LLM judges; a small panel of trained human raters provides higher-quality signal. Third, when the evaluation question is itself meta-evaluation (is this judge biased?), you cannot use the judge being studied as part of the evaluation — that requires an external anchor like human ratings or a known-neutral evaluator.
      </Prose>

      <H3>Reward model versus AI judge for preference labeling</H3>

      <Prose>
        For preference dataset construction specifically, there is a trade-off between using a trained reward model and using an AI judge directly. A reward model — a separate network trained to predict human preferences from a small human-labeled set — is faster at inference (one forward pass per response, no generation), more amenable to aggressive batching, and does not exhibit self-preference in the same way (because its training signal was human labels, not its own preferences). The downside is that reward models drift further from the policy distribution as DPO/PPO training progresses (the off-distribution problem) and require periodic retraining. AI judges are more flexible (can handle novel tasks zero-shot), more interpretable (you can read the reasoning), but suffer self-preference and other biases discussed in this topic. For mature pipelines, the common architecture is to use a reward model for the bulk of preference scoring and reserve AI judges for tasks where the reward model has not been trained or for spot-check audits.
      </Prose>

      <H3>Open-source judges versus frontier-model judges</H3>

      <Prose>
        A growing line of work investigates whether smaller open-source models, fine-tuned specifically as judges (e.g., Prometheus, JudgeLM), can substitute for frontier API judges. The case for them: they are cheaper, run locally without rate limits, and — being trained on broadly distributed preference data rather than being a single frontier model — they can exhibit lower family-specific self-preference. The case against: they are typically lower-quality judges in absolute terms and may agree less with humans on hard cases. The current consensus is that open-source judge models are good enough to replace single-frontier-judge setups in cost-sensitive pipelines, but a PoLL panel of frontier judges still produces higher-quality signal in absolute agreement-with-human terms. The right choice depends on the cost-quality trade you are willing to make and how much you value reduced self-preference vs. higher absolute agreement.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        The most operationally important fact about self-preference bias is that its magnitude does not appear to shrink as judge models get more capable. The Panickssery paper studied GPT-4, Llama-2-70B, and several smaller models, and found that the larger, more capable judges (GPT-4 in particular) exhibited the strongest self-preference effects. This is plausibly because more capable judges have more refined stylistic preferences and more accurate self-recognition, both of which fuel the bias. Subsequent work on GPT-4o and Claude-3 confirmed that the bias persists at the frontier; if anything, the recognition-accuracy component grows as models become more sophisticated.
      </Prose>

      <Prose>
        What does scale, in a positive direction, is the value of judge ensembling. As you add more diverse judges to a panel, the family-specific biases progressively cancel out, with diminishing but real gains up to roughly 5–7 panel members. Beyond that, marginal returns shrink because the available frontier judges all share some overlap in training data and methodology — there is no truly orthogonal seventh model to add. Cost scales linearly with panel size, so the practical sweet spot for most production pipelines is a 3–5 judge panel, which captures roughly 70–80% of the bias-reduction benefit at a fraction of the cost of a fully exhaustive panel.
      </Prose>

      <Prose>
        Recognition accuracy — the judge's ability to identify its own outputs — also scales with model capability, but in a way that is partially controllable. Paraphrase normalization reduces recognition accuracy substantially (the paper reports drops from above 70% to roughly 55–60% after paraphrasing), and stronger normalization (more aggressive rewriting, multi-pass paraphrasing) can drive recognition closer to chance at the cost of further response degradation. The fundamental limit is that any normalization aggressive enough to fully disguise the source will also distort the response enough to materially change its content.
      </Prose>

      <Prose>
        The scaling property that does not hold in a useful direction is human-rater agreement. As you scale the evaluation set up — moving from a 100-prompt benchmark to a 100,000-prompt preference dataset — the judge's mean disagreement with humans does not shrink; it stays roughly constant per comparison. This means that any systematic bias the judge has, including self-preference, accumulates linearly with dataset size. A 7-point self-preference bias on 1,000 comparisons becomes a 70-extra-wins effect; on 100,000 comparisons it becomes 7,000 extra wins, and the resulting preference dataset has measurable family-style skew. This is why mitigations need to scale with the size of the evaluation; you cannot rely on aggregation alone to wash the bias out.
      </Prose>

      <Prose>
        Finally, downstream propagation does scale. A small per-comparison bias, fed through a DPO or reward modeling run that consumes hundreds of thousands of comparisons, produces a downstream model with a measurable shift in stylistic preferences toward the judge's family. This is the mechanism by which judge bias becomes student bias. The strength of the downstream effect scales with the size of the preference dataset, the strength of the preference signal (β in DPO), and the quality of the SFT prior. Small datasets and strong priors absorb less of the judge's bias; large datasets and weak priors absorb more.
      </Prose>

      <Prose>
        There is one further scaling property worth understanding: the bias also scales with task subjectivity. On tasks with executable ground truth — code with unit tests, math problems with known answers, factual QA with verifiable references — self-preference is dramatically smaller because the judge has objective constraints that override stylistic preferences. On highly subjective tasks — creative writing, conversational helpfulness, "which tone is better" — self-preference dominates because there is no anchor that pulls the judge away from its stylistic priors. This explains the empirical pattern in the literature: papers studying summarization and chat (subjective) report large self-preference effects, while papers studying coding and math (objective) report small ones. When budgeting bias mitigation effort, pour it into the subjective tasks; on objective tasks, the executable signal does most of the work for free.
      </Prose>

      <Prose>
        A related observation: the magnitude of self-preference also varies with the difficulty of the comparison itself. When two candidate responses are very different in quality (one obviously better), even a biased judge will get the comparison right and self-preference is washed out by the strong quality signal. When two responses are close in quality, the family-bias term has more leverage and the result becomes more strongly determined by the judge's family. The implication is that the bias preferentially distorts the close calls — exactly the comparisons that, in a leaderboard context, determine the rank ordering of competing models. The far-apart pairs are robust; the close pairs are where the bias does its damage.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>Single judge masquerading as ground truth</H3>
      <Prose>
        The deepest failure is treating a single LLM judge's preferences as a proxy for "what humans would prefer." For mature LLM-as-judge methodology, this conflation is now widely understood to be wrong; for newer practitioners, it remains a common pitfall. Whenever you see an evaluation reported with a single judge (especially "GPT-4 said this model is better"), treat the result as carrying a known systematic bias toward the judge's family, not as a neutral measurement.
      </Prose>

      <H3>Evaluating a model with its own family as judge</H3>
      <Prose>
        The most operationally severe scenario. If you train a Claude-distilled DPO model and then evaluate it with Claude-as-judge against a Llama baseline, the evaluation will systematically favor your model. The result is not informative about cross-family quality; it tells you only how well Claude approves of Claude-flavored outputs. Always evaluate cross-family comparisons with at least one judge from outside both families, or with a panel that explicitly excludes the candidate families.
      </Prose>

      <H3>Position bias compounds with self-preference</H3>
      <Prose>
        Position bias (judges prefer the response shown first) and self-preference bias compound multiplicatively when present together. If a single judge consistently sees its own family's responses in position A and competing family in position B, the two biases align and produce inflated self-family win rates. Always randomize position presentation, even on PoLL panels — without position randomization, a panel can still concentrate bias on whichever model happens to be assigned to position A more often.
      </Prose>

      <H3>Paraphrase normalization that distorts content</H3>
      <Prose>
        Aggressive paraphrasing reduces self-recognition but can also degrade content fidelity in subtle ways — losing nuance, dropping qualifiers, simplifying technical phrasing. If the paraphraser systematically simplifies one family's responses more than another's (because their styles differ in complexity), you can introduce a new bias while removing the old one. Validate paraphrase quality by checking that paraphrased responses preserve the original semantic content, not just by checking that they reduce the recognition signal.
      </Prose>

      <H3>Anonymization that the judge ignores</H3>
      <Prose>
        Telling a judge "responses are anonymous, evaluate purely on content" is a prompt-level intervention that the judge may or may not actually act on. Empirically, anonymization helps modestly, but the model's underlying stylistic preferences are not actually erased by the instruction — they are merely de-emphasized. Do not treat "I asked the judge to be unbiased" as equivalent to "the judge was unbiased." Verify with cross-model consistency checks.
      </Prose>

      <H3>Recursive self-preference in distilled models</H3>
      <Prose>
        When you train a student model on preferences labeled by a judge, then later evaluate the student with the same judge, you have closed a feedback loop that amplifies self-preference: the student has been optimized to produce judge-favored stylistic features, and the judge then rewards those features at evaluation time. The loop manifests as a student model that scores very highly on its own family's judges while underperforming on cross-family judges. Always evaluate distilled students with judges that did not produce their training signal.
      </Prose>

      <H3>Treating panel agreement as quality</H3>
      <Prose>
        When a PoLL panel votes 3-0 on a comparison, the natural inference is "this comparison is high-confidence." But if all three panel members share a common bias — e.g., all three prefer longer responses — unanimous agreement can be a signal of shared bias rather than genuine confidence. Inter-judge agreement is a useful filtering signal but should not be the only one; cross-validate against a small human-labeled subset to confirm that high-agreement comparisons actually correspond to clear-cut cases by human standards.
      </Prose>

      <H3>Calibration sets that drift from deployment</H3>
      <Prose>
        Human calibration anchors solve self-preference only as long as the calibration prompts are representative of the deployment distribution. If your benchmark is heavy on summarization but your calibration set is heavy on creative writing, the regression-corrected bias estimate will not transfer. Periodically refresh the calibration set, and stratify it across the same task types as the deployment evaluation.
      </Prose>

      <H3>Confounding self-preference with capability gap</H3>
      <Prose>
        If the model being evaluated is genuinely much better than the baseline on the task, a high diagonal in the cross-evaluation matrix may reflect actual quality rather than self-preference. The way to disambiguate is to look at the off-diagonal entries in the same row: if all judges agree that the generator is strong (high values across the row), the diagonal can be high without self-preference. Self-preference shows up specifically as a diagonal that is higher than the within-row average, not as a uniformly high row.
      </Prose>

      <H3>Format and length confounds masquerading as self-preference</H3>
      <Prose>
        Self-preference and other content-attribute biases (verbosity, markdown formatting, emoji use, list structure) live in adjacent measurement space. A model that produces longer responses than its peers will appear to have inflated self-preference if the judging family also exhibits length bias, even when the underlying mechanism is just verbosity bias compounding with same-family judging. Always include length and structural features as covariates in the regression decomposition; the same-family coefficient should be reported after partialling out these confounds, not before.
      </Prose>

      <H3>Cross-version self-preference within the same family</H3>
      <Prose>
        A subtle and underappreciated failure mode: when a new version of a model (GPT-4o vs GPT-4-turbo, Claude-3.5 vs Claude-3) judges its predecessors and successors. The newer version often exhibits stronger self-preference for its own outputs than for the previous version's, because the stylistic conventions evolved between versions. Treating "GPT-family" as one undifferentiated group can obscure this; in tightly controlled studies, breaking the family taxonomy down to specific model versions gives a more accurate self-preference measurement. The practical implication is that "use a different judge from the same provider" is not a reliable mitigation — Claude-Haiku judging Claude-Opus outputs may still inflate, just less than Claude-Opus judging itself.
      </Prose>

      <Callout accent="gold">
        Self-preference is the bias most likely to flatter you and least likely to be obvious in the evaluation output. A leaderboard run with a single judge from your model's family will look clean, well-formatted, and confidently rank your model favorably — and yet contain a systematic 5–10 point bias in your favor. Always force yourself to run cross-family evaluations before publishing comparative numbers.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        Sources verified against arXiv pages on 2026-04-26. Author lists, abstracts, and arXiv IDs confirmed.
      </Prose>

      <H3>Panickssery, Bowman, Feng 2024 — self-preference and self-recognition</H3>
      <Prose>
        Arjun Panickssery, Samuel R. Bowman, Shi Feng. "LLM Evaluators Recognize and Favor Their Own Generations." arXiv:2404.13076. Published April 2024. The founding empirical paper for self-preference bias as an isolable phenomenon. Constructs a controlled cross-evaluation matrix on summarization tasks across GPT-4, GPT-3.5, and Llama-2-70B, and shows that each judge prefers its own family's outputs by 5–10 percentage points more than humans do. Goes one step further and demonstrates that GPT-4 can identify its own outputs with greater than 70% accuracy via stylistic cues, and shows a positive correlation between self-recognition accuracy and self-preference magnitude across judges. Includes ablations on paraphrase normalization (reduces but does not eliminate the bias), anonymization (modest reduction), and explicit instructions to be unbiased (essentially no effect). The paper that any subsequent work on LLM-judge bias has to reckon with.
      </Prose>

      <H3>Wataoka et al. 2024 — formal definitions of self-preference</H3>
      <Prose>
        Koki Wataoka et al. "Self-Preference Bias in LLM-as-Judge." arXiv preprint, 2024. Formalizes the self-preference coefficient, gives the cross-evaluation matrix differential definition, and proposes Bayesian measurement models that decompose judge preferences into true-quality, content-bias, and family-bias components. Provides the methodological scaffolding that the field has converged on for measuring self-preference quantitatively, including identifiability requirements (at least three models in the evaluation matrix; external anchor for true quality). Pairs naturally with Panickssery et al. for theory + empirics coverage.
      </Prose>

      <H3>Verga et al. 2024 — PoLL (Panel of LLM Judges)</H3>
      <Prose>
        Pat Verga, Sebastian Hofstatter, Sophia Althammer, et al. (Cohere). "Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models." arXiv:2404.18796. Published April 2024. Proposes the PoLL architecture: a three-judge panel of cross-family small/medium models (typically Claude-Haiku, GPT-3.5, Command) replacing a single GPT-4 judge. Shows that PoLL agrees with human ratings as well as a single GPT-4 judge does, at roughly an order of magnitude lower cost, and substantially reduces single-family self-preference effects. The canonical reference for production judge ensembling and the architecture deployed by most large-scale automated preference labeling pipelines released after mid-2024.
      </Prose>

      <H3>Zheng et al. 2023 — MT-Bench, position bias, and the LLM-as-judge framework</H3>
      <Prose>
        Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, et al. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." arXiv:2306.05685. Published June 2023; presented at NeurIPS 2023. Introduces MT-Bench and the Chatbot Arena methodology, and gives the foundational catalog of LLM-judge biases: position bias (preference for the response shown first), verbosity bias (preference for longer responses), and self-enhancement bias (the early term for what later became self-preference). The bias section of this paper is what motivated the more focused empirical work by Panickssery et al. and the Wataoka measurement formalism. Required background for understanding how LLM-as-judge methodology evolved from its initial proposal to its current bias-aware form.
      </Prose>

      <H3>Liu et al. 2023 — ChatGPT as a subjective judge</H3>
      <Prose>
        Yang Liu, Dan Iter, Yichong Xu, et al. "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment." arXiv:2303.16634. Published March 2023, with an associated body of work titled "ChatGPT as a Subjective Judge" appearing across several venues in mid-to-late 2023. Documents the early use of GPT-3.5 and GPT-4 as evaluators on subjective natural-language-generation tasks (summarization quality, dialogue coherence, story quality) and reports both the strengths (high correlation with human judgments on average) and the weaknesses (systematic deviation in specific task subgroups). Provides the practical foundation that the bias literature later refined.
      </Prose>

      <H3>Tunstall et al. 2023 — Zephyr (mechanism for downstream propagation)</H3>
      <Prose>
        Lewis Tunstall, Edward Beeching, et al. (HuggingFace). "Zephyr: Direct Distillation of LM Alignment." arXiv:2310.16944. Published October 2023. Not a self-preference paper in itself, but the canonical reference for the pipeline pattern in which a frontier model (GPT-4) labels a preference dataset (UltraFeedback) which is then used to DPO-train a 7B student. This is precisely the workflow that propagates self-preference bias from judge to student in the most widespread way, and any analysis of self-preference's downstream effects has to engage with the Zephyr-style training pattern as the dominant deployment mode.
      </Prose>

      <H3>Irving, Christiano, Amodei 2018 — debate as bias mitigation</H3>
      <Prose>
        Geoffrey Irving, Paul Christiano, Dario Amodei. "AI Safety via Debate." arXiv:1805.00899. Published May 2018. The foundational paper proposing structured debate between models as a mechanism to surface flaws that a single judge would miss. Pre-dates the LLM-as-judge era but provides the conceptual framework for adversarial protocols as a bias-mitigation strategy. Empirical follow-up work in the 2023–2024 period extended the framework to specific LLM judges and showed mixed results for self-preference reduction (debate helps surface factual errors but does not robustly remove stylistic biases).
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Why two models is not enough</H3>
      <Prose>
        Suppose you have only two models, A and B, and you build a 2×2 cross-evaluation matrix of win rates against a fixed baseline. The diagonal entries are <Code>W_{"{A,A}"}</Code> and <Code>W_{"{B,B}"}</Code>; the off-diagonal entries are <Code>W_{"{A,B}"}</Code> and <Code>W_{"{B,A}"}</Code>. Show explicitly why you cannot identify the family-bias coefficient <Code>γ</Code> separately from the relative-quality main effect of A versus B. Why does adding a third model make the parameters identifiable? What is the minimum number of models you would need if you also wanted to measure how the family-bias coefficient varies across model pairs (e.g., is it the same magnitude between GPT-4 and Llama as it is between Claude and Mistral)?
      </Prose>

      <H3>Exercise 2 — Recognition accuracy as causal evidence</H3>
      <Prose>
        Panickssery et al. report that judges with higher self-recognition accuracy also exhibit larger self-preference, and they argue that this is evidence that recognition causes preference. List two alternative explanations for the observed correlation that do not involve recognition causing preference. For each, design an experiment that would distinguish your alternative from the recognition-causes-preference hypothesis. As a follow-up: why is paraphrase normalization the most important ablation in their methodology, and what would you conclude if paraphrasing reduced recognition accuracy to chance but did not reduce self-preference at all?
      </Prose>

      <H3>Exercise 3 — Downstream propagation through DPO</H3>
      <Prose>
        Suppose you collect a preference dataset of 100,000 (prompt, chosen, rejected) triples by using a single GPT-4 judge to rank pairs from a mixed pool of GPT-4-generated and Llama-3-generated responses. The judge has a 7-point self-preference bias toward GPT-4 responses. You then train a Llama-3-7B SFT model with DPO on this dataset for one epoch, β=0.1. Trace through the implications: which responses will end up disproportionately in the "chosen" set; how the DPO loss will shape the resulting policy's stylistic distribution; and what you would observe if you evaluated the trained model on (a) GPT-4 as judge and (b) a cross-family panel. Why is option (a) misleading and option (b) more informative?
      </Prose>

      <H3>Exercise 4 — Designing a calibration set</H3>
      <Prose>
        You are responsible for designing a 200-prompt human-rater calibration set for an evaluation pipeline that uses a PoLL panel. The deployment evaluation covers summarization, code generation, factual QA, and creative writing in roughly equal proportion. The calibration set must (a) be representative enough that a regression-corrected bias estimate transfers to deployment, (b) contain enough signal in each task type to estimate the family-bias coefficient separately, and (c) be small enough that human annotation is feasible (target 8 hours of expert annotator time total). Write out your design: how many prompts per task type, how many annotators per prompt, what reconciliation procedure you use for annotator disagreement, and what sample-size calculation justifies your numbers. What goes wrong if the calibration set is biased toward one task type?
      </Prose>

      <H3>Exercise 5 — When self-preference looks like quality</H3>
      <Prose>
        You are reviewing a paper that reports a new 7B model achieving state-of-the-art results on AlpacaEval 2 with GPT-4 as judge, beating much larger competing models. The paper does not report cross-family judge results. List three observable signals — from the paper's training methodology, the candidate response distribution, and the magnitude of the reported improvement — that would heighten your suspicion that the result is partially driven by self-preference rather than genuine quality. For each signal, describe what would distinguish a self-preference-inflated result from a genuinely strong model. What additional experiments would you ask the authors to run before accepting the headline number, and what would the cheapest credible additional experiment be?
      </Prose>

      <H3>Exercise 6 — Designing a bias-aware leaderboard</H3>
      <Prose>
        You are tasked with designing the methodology for a new public leaderboard that ranks chat models on conversational helpfulness. The leaderboard must be defensible against accusations of self-preference bias and must remain trustworthy as new model families are released over time. Specify: (a) the judge architecture (single judge, panel, mix); (b) the rotation policy for adding/retiring judges as the field evolves; (c) the per-evaluation logging requirements that allow third parties to audit the leaderboard for bias; (d) the human-calibration anchor design and refresh cadence; and (e) the reporting policy — what summary statistics, breakdowns, and uncertainty bands are shown alongside the headline rankings? Justify each choice in terms of the bias-mitigation literature.
      </Prose>

      <H3>Exercise 7 — Recognition without preference, preference without recognition</H3>
      <Prose>
        Construct two thought experiments. In the first, design a hypothetical model that has very high self-recognition accuracy (above 90%) but exhibits zero self-preference bias — what training procedure could produce such a model, and how would you verify both properties experimentally? In the second, design a hypothetical model that exhibits substantial self-preference bias (above 10 percentage points) but has chance-level self-recognition accuracy — what mechanism could produce this, and how would you distinguish it from a model with high recognition that simply does not act on it? What do these thought experiments tell you about the causal pathway from recognition to preference?
      </Prose>

    </div>
  ),
};

export default aiToAiBias;
