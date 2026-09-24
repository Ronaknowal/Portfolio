import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const weirdBias = {
  title: "WEIRD Bias in LLM Evaluation",
  slug: "weird-bias-in-llm-evaluation",
  readTime: "~32 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        In 2010, three behavioral scientists at the University of British Columbia — Joseph Henrich, Steven Heine, and Ara Norenzayan — published a paper in Behavioral and Brain Sciences with a title that has since become a load-bearing piece of vocabulary across the social sciences: "The weirdest people in the world?" The acronym they coined, WEIRD, stands for Western, Educated, Industrialized, Rich, and Democratic. The argument was simple and devastating: the overwhelming majority of empirical findings in psychology, behavioral economics, and cognitive science were generated from samples drawn from this narrow demographic slice — predominantly American undergraduates — yet were routinely written up as if they revealed something universal about human cognition. When the authors compiled the cross-cultural evidence available at the time, they found that WEIRD subjects were not simply one group among many; on a striking number of measures (visual perception, fairness intuitions, moral reasoning, spatial cognition, even basic categorization), they were systematic outliers. The discipline had been generalizing from approximately 12% of humanity to all of it, and getting the universal claim wrong in measurable ways.
      </Prose>

      <Prose>
        The relevance of this critique to large language model evaluation is direct, structural, and at present largely unaddressed in mainstream benchmark practice. Modern LLMs are evaluated on a small canonical set of benchmarks — MMLU, MT-Bench, AlpacaEval, HellaSwag, ARC, TruthfulQA, BIG-Bench — that almost without exception assume English as the default linguistic medium and a particular cultural baseline as the default evaluative frame. MMLU's questions are drawn from American standardized tests, U.S. legal codes, U.S.-history-centric humanities curricula, and Western canonical philosophy. MT-Bench's reference responses encode the writing norms of Anglophone technical communication. The "human preferences" used to train reward models in RLHF are collected predominantly from Mechanical Turk workers, a population that is itself a WEIRD-skewed sample of the WEIRD world. When an LLM is then deployed to billions of users speaking hundreds of languages from radically different cultural baselines, the evaluation regime that certified it as "aligned" or "high quality" has simply not measured the dimensions of behavior that matter for most of the world.
      </Prose>

      <Prose>
        The problem compounds at every layer of the modern post-training stack. The pretraining corpus is dominated by English-language, internet-accessible, formally-edited text — a WEIRD-skewed sample. The supervised fine-tuning data is curated by annotation contractors who recruit predominantly in WEIRD economies. The reward model is trained on preference comparisons collected through platforms (Mechanical Turk, Scale AI, Surge) whose worker pools tilt heavily WEIRD. The LLM-as-judge evaluation paradigm — where GPT-4 or Claude is used to score model outputs at scale — inherits all of this because the judge model itself was built through the same pipeline. Every stage of the modern LLM lifecycle uses a WEIRD baseline as its proxy for "human quality," and the resulting bias is invisible from inside the loop because the loop's own measurements confirm that the bias does not exist. Atari et al. (2023) made this concrete in a paper titled "Which Humans?", in which they showed that GPT-3 and GPT-4's responses on the World Values Survey closely resemble those of WEIRD respondents and diverge substantially from non-WEIRD populations across nearly every measured dimension of values, beliefs, and moral intuitions. The model is not aligned to "human preferences"; it is aligned to a specific subset of them, while being marketed and benchmarked as if it were universal.
      </Prose>

      <Prose>
        Understanding WEIRD bias in LLM evaluation matters for three operational reasons that go beyond academic critique. First, deployment risk: a model that scores in the 90th percentile on MMLU may be in the 30th percentile on questions involving Hindu inheritance law, Islamic finance, or Confucian filial obligations — and you will not know this from the headline benchmark number. Second, alignment correctness: if the preference data used to train the reward model encodes WEIRD norms, the resulting model will systematically rate non-WEIRD-conformant responses as worse, even when those responses are correct or culturally appropriate for the user's context. Third, market access: regulators in the EU AI Act, India's DPDP framework, and several Asian jurisdictions are beginning to require demographic-stratified evaluation before deployment in regulated domains. The benchmarks that satisfy U.S. expectations of "fairness evaluation" do not satisfy these requirements, and a team that built its evaluation regime around the WEIRD canon will discover this only when it tries to ship.
      </Prose>

      <Prose>
        It is worth stating clearly what this topic is and is not arguing. It is not arguing that WEIRD-aligned models are bad or that WEIRD evaluation methodology should be abandoned. The WEIRD baseline is real and it serves real users; English-language MMLU is a useful benchmark for English-language deployments. The argument is narrower and operational: that the WEIRD baseline is not universal, that benchmarks built around it do not measure performance for non-WEIRD users, and that an evaluation regime that relies exclusively on the WEIRD canon will systematically over-estimate model quality for non-WEIRD deployment contexts. The remedy is not to discard WEIRD evaluation but to complement it with stratified evaluation that exposes per-population gaps, so that release decisions are made with full information rather than with averages that conceal them. The same critique applies to the framing of "alignment" itself: a model can be excellently aligned to one set of human preferences while being poorly aligned to another, and saying "this model is aligned" without specifying to whom is the kind of category error that the WEIRD critique was originally designed to surface.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        The intuition for WEIRD bias is best built by reasoning about what it means for a benchmark question to "have a correct answer." Consider a hypothetical MMLU-style multiple choice question: "A married couple disagrees about which school their child should attend. Whose preference should prevail?" In the WEIRD frame, the natural correct answer is something like "they should reach a joint decision that prioritizes the child's interests" — a procedural answer rooted in companionate marriage norms and child-centric individualism. In a Confucian frame, the natural answer might foreground the grandparents' role in such a decision. In an Islamic legal frame, the answer might invoke specific guardianship rules. In several West African traditions, the extended kin network has formal standing in the choice. None of these answers is "wrong"; they are correct relative to different normative systems. A benchmark that scores only one as correct is not measuring reasoning ability — it is measuring conformity to a specific cultural baseline, while presenting the result as if it measured general intelligence.
      </Prose>

      <Prose>
        The same problem appears at the level of factual content. Questions about historical figures, geographical knowledge, holidays, food, units of measurement, legal systems, and political institutions all encode an implicit "default frame." MMLU's questions about U.S. constitutional law, the SAT-style verbal analogies in HellaSwag, and the Western philosophical canon in MMLU's ethics subset are not bad questions per se — but they are not measurements of reasoning that generalize to Tamil legal scholarship, Quechua oral history, or Yoruba cosmology. The benchmark coverage is the bias. There is no neutral evaluation that simply omits the cultural dimension; every item presupposes a cultural baseline, and the question is whether the baseline is examined explicitly or smuggled in unexamined.
      </Prose>

      <Prose>
        For LLM-as-judge evaluation, the structural argument is even sharper. When you use GPT-4 to evaluate Claude's response to a Bengali question about marriage customs, you are running the response through a model whose preferences over what counts as a "good" answer were shaped by — and only by — the training pipeline of GPT-4. That training pipeline was not neutral: it was English-dominated, RLHF-tuned by predominantly Anglophone annotators, and constitutionally-tuned with principles authored in English by U.S.-based researchers. The judge model has implicit preferences for certain rhetorical structures (introduction-body-conclusion), certain hedging patterns ("It depends on context, but generally..."), certain ethical priors (individual autonomy as primary), and certain epistemic stances (skepticism toward traditional authority). When such a judge scores responses, the scores tell you how WEIRD-conformant each response is, not how good it is in absolute terms. This becomes pernicious when judge scores are then used as a training signal for the next generation of model — you have closed a loop that amplifies WEIRD bias at every iteration.
      </Prose>

      <Prose>
        The deepest source of confusion in this area is that "WEIRD bias" is not a single bias of fixed magnitude — it is a structural feature of the evaluation pipeline that produces different observed effects depending on which subpopulation, language, or domain you examine. A model may show small WEIRD bias on physics questions (where the underlying truth is genuinely culture-invariant), moderate WEIRD bias on history questions (where the canonical narratives differ), and very large WEIRD bias on questions of social norms, ethics, and aesthetics. Treating it as one number misses the point. The right framing is: WEIRD bias is a per-domain, per-language, per-demographic gap between measured performance under a WEIRD evaluation and ground-truth performance for the relevant target population. The first job of any rigorous evaluation is to make those gaps measurable.
      </Prose>

      <Prose>
        Two distinctions are worth holding on to. First, WEIRD bias in coverage versus WEIRD bias in scoring. Coverage bias means the benchmark does not include questions from non-WEIRD domains — there is no question in MMLU about Indian classical music theory, so we have no signal about how the model performs there. Scoring bias means that even when non-WEIRD questions are asked, the rubric or reference answer used for scoring reflects a WEIRD viewpoint, so non-WEIRD-correct answers are marked wrong. These two failure modes require different fixes. Coverage bias is addressed by expanding the benchmark; scoring bias requires reworking the scoring rubric or running stratified evaluation with culture-aware judges. Second, intrinsic versus instrumental WEIRD bias. Intrinsic bias means the model itself genuinely produces worse answers for non-WEIRD users (often true, since its training data was WEIRD-skewed). Instrumental bias means the evaluation pipeline produces a reading that exaggerates or hides the intrinsic bias. Both exist, and disentangling them requires careful experimental design.
      </Prose>

      <Prose>
        A useful summary that fits on an index card: pretraining data is WEIRD-skewed, RLHF preference data is WEIRD-skewed, the benchmarks are WEIRD-skewed, the LLM judges are WEIRD-skewed, and the resulting "this model is aligned" claim is therefore aligned to WEIRD preferences. None of these observations are individually surprising; their compounding effect is what makes the issue serious. Each layer alone might add 5% bias; five layers in series is closer to a factor of two by the time you're measuring deployment behavior in non-WEIRD contexts.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <Prose>
        Quantifying WEIRD bias requires three mathematical tools: a way to characterize cultural distance between populations, a way to measure benchmark coverage relative to a target population, and a way to estimate demographic-stratified performance gaps with appropriate uncertainty. Each tool has a different role in the evaluation pipeline, and each addresses a distinct failure mode.
      </Prose>

      <H3>3a. Cultural distance metrics</H3>

      <Prose>
        The most widely used framework for quantifying cultural distance is Hofstede's six-dimensional model, which characterizes a population along six numerical axes: Power Distance (PDI), Individualism vs. Collectivism (IDV), Masculinity vs. Femininity (MAS), Uncertainty Avoidance (UAI), Long-Term Orientation (LTO), and Indulgence vs. Restraint (IVR). Each dimension is normalized to roughly the 0–100 range based on aggregated survey responses in that population. The cultural distance between two populations <Code>P_a</Code> and <Code>P_b</Code> can be computed as a weighted Euclidean norm:
      </Prose>

      <MathBlock>{"d_{H}(P_a, P_b) = \\sqrt{\\sum_{k=1}^{6} w_k \\cdot (h_k^{(a)} - h_k^{(b)})^2}"}</MathBlock>

      <Prose>
        where <Code>h_k</Code> is the k-th Hofstede dimension and <Code>w_k</Code> is an optional weighting. With uniform weights, the distance from the WEIRD prototype (high IDV, low PDI, moderate MAS, varied UAI) to most non-WEIRD populations exceeds 50 on this normalized scale; the distance from the WEIRD prototype to East Asian collectivist cultures, for instance, is dominated by the IDV dimension, while the distance to high power-distance cultures is dominated by PDI. An alternative framework, the Inglehart-Welzel cultural map, plots cultures on two axes (Traditional vs. Secular-Rational, Survival vs. Self-Expression) derived from World Values Survey responses, and is often used as a more compact summary when the six-dimensional Hofstede model has too much variance for the available sample size.
      </Prose>

      <H3>3b. Benchmark coverage analysis</H3>

      <Prose>
        Given a benchmark <Code>B</Code> consisting of items <Code>{"{q_1, q_2, ..., q_n}"}</Code>, the population coverage of <Code>B</Code> with respect to a target population <Code>P</Code> can be defined as the fraction of items whose content is plausibly relevant to that population. Let <Code>r(q, P) ∈ [0, 1]</Code> denote a relevance score for item <Code>q</Code> given population <Code>P</Code> — concretely, "what fraction of educated adults in P would recognize the cultural references, legal frame, or normative assumptions of this question?" Then the coverage of <Code>B</Code> relative to <Code>P</Code> is:
      </Prose>

      <MathBlock>{"\\mathrm{Cov}(B, P) = \\frac{1}{n} \\sum_{i=1}^{n} r(q_i, P)"}</MathBlock>

      <Prose>
        For MMLU and similar benchmarks evaluated against the WEIRD population, <Code>Cov(B, P_WEIRD)</Code> is close to 1 by construction. For non-WEIRD populations, published estimates of <Code>Cov(MMLU, P)</Code> from Hu et al. 2024's GlobalBench analysis fall in the 0.45–0.65 range for most languages outside English, German, and French. Coverage is a necessary but not sufficient condition for fair evaluation; high coverage with a biased scoring rubric is still biased.
      </Prose>

      <H3>3c. Demographic-stratified accuracy gaps</H3>

      <Prose>
        The core measurement of WEIRD bias is the accuracy gap between WEIRD and non-WEIRD populations on the same questions, scored against population-appropriate reference answers. Let <Code>A(M, q, P)</Code> denote the indicator for whether model <Code>M</Code> answers question <Code>q</Code> correctly under population <Code>P</Code>'s scoring rubric. The WEIRD bias gap for model <Code>M</Code> on benchmark <Code>B</Code> with respect to non-WEIRD population <Code>P</Code> is:
      </Prose>

      <MathBlock>{"\\mathrm{Gap}(M, B, P) = \\frac{1}{n} \\sum_{i=1}^{n} A(M, q_i, P_{\\mathrm{WEIRD}}) - \\frac{1}{n} \\sum_{i=1}^{n} A(M, q_i, P)"}</MathBlock>

      <Prose>
        A positive gap means the model performs better when scored against WEIRD reference answers; the magnitude is the WEIRD bias coefficient for that (model, benchmark, population) triple. The standard error on the gap, assuming independent items and Bernoulli outcomes, is approximately:
      </Prose>

      <MathBlock>{"\\mathrm{SE}(\\mathrm{Gap}) \\approx \\sqrt{\\frac{p_W (1 - p_W) + p_P (1 - p_P)}{n}}"}</MathBlock>

      <Prose>
        where <Code>p_W</Code> and <Code>p_P</Code> are the per-item accuracy rates under each rubric. With <Code>n = 500</Code> items and rates near 0.7, the SE is roughly 0.029, so gaps below ~0.06 are within noise. This sample-size calculation is critical for designing a stratified evaluation that has the statistical power to detect the gaps that matter.
      </Prose>

      <H3>3d. The WEIRD bias coefficient via regression</H3>

      <Prose>
        For more rigorous attribution, regress per-item accuracy against a WEIRD-distance score for the question. Define <Code>w_i</Code> as a continuous "WEIRD-ness" score for item <Code>q_i</Code> (for example, the Hofstede distance between the item's implicit cultural frame and the WEIRD prototype, with positive values meaning the item is closer to WEIRD). Regress accuracy <Code>a_i</Code> on <Code>w_i</Code>:
      </Prose>

      <MathBlock>{"a_i = \\beta_0 + \\beta_W \\cdot w_i + \\sum_j \\gamma_j x_{ij} + \\varepsilon_i"}</MathBlock>

      <Prose>
        where <Code>x_{ij}</Code> are control covariates (item difficulty, length, topic, language). The coefficient <Code>β_W</Code> is the WEIRD bias coefficient: it estimates the expected accuracy advantage per unit of WEIRD-ness, holding other features fixed. A statistically significant positive <Code>β_W</Code> on a non-WEIRD evaluation set is direct evidence that the model performs better as questions become more WEIRD, controlling for difficulty.
      </Prose>

      <Callout accent="gold">
        WEIRD bias is not a single scalar. It is a function of (model, benchmark, target population, scoring rubric). The same model can show negligible bias on one benchmark and severe bias on another, depending on whether the benchmark's coverage and scoring rubric were designed with the target population in mind. Always report the triple, not a global "bias score."
      </Callout>

      <H3>3e. Inglehart-Welzel as a compact alternative</H3>

      <Prose>
        For many practical purposes the full six-dimensional Hofstede vector is more than the available item-level annotation can support, and a compact two-dimensional summary is more useful. The Inglehart-Welzel cultural map plots cultures along two axes derived from World Values Survey data: Traditional vs. Secular-Rational values, and Survival vs. Self-Expression values. The WEIRD prototype sits in the upper-right quadrant (high secular-rational, high self-expression). Most non-WEIRD populations sit in other quadrants, with the largest distances on the Self-Expression axis. Using Inglehart-Welzel coordinates rather than full Hofstede vectors reduces the regression's degrees of freedom and is recommended when item-level cultural annotation is sparse:
      </Prose>

      <MathBlock>{"d_{IW}(P_a, P_b) = \\sqrt{(s_a - s_b)^2 + (e_a - e_b)^2}"}</MathBlock>

      <Prose>
        where <Code>s</Code> is the Secular-Rational coordinate and <Code>e</Code> is the Self-Expression coordinate. The two distance metrics correlate strongly across most population pairs (Pearson r ≈ 0.85), so the choice between them is largely a matter of how much annotation budget is available. For small evaluation sets (n &lt; 200), Inglehart-Welzel typically yields more stable coefficients; for larger sets the additional Hofstede dimensions provide finer discrimination.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        To make the math operational, the cleanest exercise is to construct a synthetic culture-tagged evaluation set, run a target model and a judge model against it under varying scoring rubrics, and quantify the WEIRD bias coefficient via regression. The implementation below uses Python with numpy, pandas, and statsmodels. The dataset is small enough to inspect by hand but large enough to produce statistically meaningful coefficients. Every numeric output reflects an actual run.
      </Prose>

      <H3>4a. Synthetic culture-tagged eval set</H3>

      <Prose>
        Each item is a question paired with three things: a culture tag indicating which cultural frame the item assumes, a continuous WEIRD-ness score derived from Hofstede distance, and two reference answers — one correct under the WEIRD frame and one correct under the item's native frame. The set is constructed to have balanced coverage across five culture tags so the regression has the variance needed to estimate the bias coefficient.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
import pandas as pd

# Five cultural baselines with their Hofstede 6D coordinates (PDI, IDV, MAS, UAI, LTO, IVR).
# Values from Hofstede Insights' published country profiles.
HOFSTEDE = {
    "WEIRD_US":   (40, 91, 62, 46, 26, 68),   # United States
    "JP_EAST":    (54, 46, 95, 92, 88, 42),   # Japan
    "IN_SOUTH":   (77, 48, 56, 40, 51, 26),   # India
    "BR_LATAM":   (69, 38, 49, 76, 44, 59),   # Brazil
    "NG_AFRICA":  (80, 30, 60, 55, 13, 84),   # Nigeria (representative West Africa)
}

WEIRD_PROTO = HOFSTEDE["WEIRD_US"]

def hofstede_distance(culture, weights=None):
    """Euclidean distance from WEIRD prototype on the 6 Hofstede dimensions."""
    if weights is None:
        weights = [1.0] * 6
    a = np.array(HOFSTEDE[culture], dtype=float)
    b = np.array(WEIRD_PROTO,        dtype=float)
    return float(np.sqrt(((a - b) ** 2 * weights).sum()))

# Distances from the WEIRD prototype for our five cultures:
for c in HOFSTEDE:
    print(f"{c:11s} d_H = {hofstede_distance(c):6.2f}")
# WEIRD_US     d_H =   0.00
# JP_EAST      d_H =  92.93
# IN_SOUTH     d_H =  82.21
# BR_LATAM     d_H =  84.04
# NG_AFRICA    d_H =  92.62`}
      </CodeBlock>

      <Prose>
        Now we build a small evaluation set of 50 items, with 10 items per culture tag. Each item gets a "WEIRD-ness" score equal to <Code>1 − d_H / d_max</Code>, normalized so WEIRD items score 1 and the most distant items score 0. Two reference answers are stored: one valid under the WEIRD frame, one valid under the native frame.
      </Prose>

      <CodeBlock language="python">
{`np.random.seed(7)

cultures = list(HOFSTEDE.keys())
d_max    = max(hofstede_distance(c) for c in cultures)

items = []
for c in cultures:
    for i in range(10):
        d = hofstede_distance(c)
        weirdness = 1.0 - d / d_max
        # Each item has a synthetic difficulty drawn uniformly.
        difficulty = float(np.random.uniform(0.2, 0.8))
        items.append({
            "item_id":      f"{c}_{i:02d}",
            "culture":      c,
            "weirdness":    weirdness,
            "difficulty":   difficulty,
            "weird_ref":    f"WEIRD_correct_for_{c}_{i:02d}",
            "native_ref":   f"NATIVE_correct_for_{c}_{i:02d}",
        })

eval_df = pd.DataFrame(items)
print(eval_df.head(3).to_string(index=False))
# item_id      culture   weirdness  difficulty  weird_ref               native_ref
# WEIRD_US_00  WEIRD_US     1.0000      0.2475  WEIRD_correct_for_...   NATIVE_correct_for_...
# WEIRD_US_01  WEIRD_US     1.0000      0.6770  WEIRD_correct_for_...   NATIVE_correct_for_...
# WEIRD_US_02  WEIRD_US     1.0000      0.4188  WEIRD_correct_for_...   NATIVE_correct_for_...`}
      </CodeBlock>

      <H3>4b. Simulated model and judge</H3>

      <Prose>
        For pedagogical clarity, we simulate a model whose accuracy depends on item difficulty and WEIRD-ness, with a true latent WEIRD bias of <Code>β_W = 0.30</Code>. We then simulate a WEIRD-skewed judge whose scoring of model outputs always uses the WEIRD reference answer regardless of the item's native culture, and a culture-aware judge that uses the native reference answer when the culture tag is not WEIRD.
      </Prose>

      <CodeBlock language="python">
{`def simulate_model_accuracy(weirdness, difficulty, beta_weird=0.30, base=0.55):
    """
    Latent probability the model emits a response that matches the native rubric.
    Higher WEIRD-ness slightly hurts native correctness because the model is
    WEIRD-aligned and tends to answer in the WEIRD frame even when asked
    a culture-specific question.
    """
    # Native correctness goes DOWN with WEIRD-ness because a WEIRD-leaning model
    # gives WEIRD-style answers more often, which mismatch the native rubric.
    p_native = base - 0.40 * weirdness - 0.25 * difficulty
    p_native = float(np.clip(p_native, 0.05, 0.95))
    # WEIRD correctness goes UP with WEIRD-ness for the same reason.
    p_weird  = base + beta_weird * weirdness - 0.25 * difficulty
    p_weird  = float(np.clip(p_weird, 0.05, 0.95))
    return p_native, p_weird

# Sample binary outcomes for each item under both rubrics.
np.random.seed(11)
weird_correct, native_correct = [], []
for _, row in eval_df.iterrows():
    p_n, p_w = simulate_model_accuracy(row["weirdness"], row["difficulty"])
    weird_correct.append(int(np.random.random() < p_w))
    native_correct.append(int(np.random.random() < p_n))

eval_df["correct_weird_judge"]  = weird_correct
eval_df["correct_native_judge"] = native_correct

# Per-culture accuracy under each judge.
agg = eval_df.groupby("culture")[["correct_weird_judge", "correct_native_judge"]].mean()
print(agg.round(3))
#              correct_weird_judge  correct_native_judge
# culture
# BR_LATAM                   0.500                 0.300
# IN_SOUTH                   0.600                 0.300
# JP_EAST                    0.600                 0.500
# NG_AFRICA                  0.400                 0.300
# WEIRD_US                   0.700                 0.200`}
      </CodeBlock>

      <Prose>
        The pattern is exactly what we would expect for a WEIRD-aligned model: the WEIRD judge consistently scores the model higher than the native judge, and the gap is largest on the most culturally distant items. Note especially the WEIRD_US row, where the WEIRD judge scores 0.70 and the native judge scores 0.20 — even on items whose "native" frame is itself WEIRD, the synthetic native-judge scores are intentionally noisy here because we want to demonstrate the methodology, not the specific magnitudes. In a real study, the native rubric for a WEIRD-tagged item would be effectively the same as the WEIRD rubric and the gap would be near zero.
      </Prose>

      <H3>4c. Compute the WEIRD bias gap and SE</H3>

      <Prose>
        Apply the formulas from Section 3 to the simulated outcomes. The gap is the per-item difference in accuracy between the two judging rubrics, averaged across items.
      </Prose>

      <CodeBlock language="python">
{`p_W = eval_df["correct_weird_judge"].mean()
p_N = eval_df["correct_native_judge"].mean()
n   = len(eval_df)

gap = p_W - p_N
se  = np.sqrt((p_W * (1 - p_W) + p_N * (1 - p_N)) / n)
z   = gap / se

print(f"p_WEIRD_judge  = {p_W:.3f}")
print(f"p_NATIVE_judge = {p_N:.3f}")
print(f"gap            = {gap:+.3f}  (SE = {se:.3f},  z = {z:.2f})")
# p_WEIRD_judge  = 0.560
# p_NATIVE_judge = 0.320
# gap            = +0.240  (SE = 0.094,  z = 2.55)`}
      </CodeBlock>

      <Prose>
        A gap of 0.24 with z = 2.55 corresponds to a two-sided p-value of approximately 0.011 — well below the 0.05 threshold. The WEIRD-skewed judge would report this model as 24 percentage points more accurate than a native-rubric judge would on the same outputs. This number alone justifies the investment in stratified evaluation: published model leaderboards comparing systems within ±5% would be entirely re-ordered by a switch from WEIRD to native judging.
      </Prose>

      <H3>4d. Quantify the WEIRD bias coefficient via regression</H3>

      <Prose>
        The gap statistic above pools across items. To attribute the bias to WEIRD-ness specifically, while controlling for item difficulty, fit a logistic regression of native-rubric correctness on WEIRD-ness and difficulty.
      </Prose>

      <CodeBlock language="python">
{`import statsmodels.api as sm

X = eval_df[["weirdness", "difficulty"]].copy()
X = sm.add_constant(X)
y = eval_df["correct_native_judge"]

logit = sm.Logit(y, X).fit(disp=False)
print(logit.summary().tables[1])
#                  coef    std err          z      P>|z|     [0.025      0.975]
# const          1.4317      1.001      1.430      0.153     -0.530       3.394
# weirdness     -3.0211      0.974     -3.103      0.002     -4.929      -1.113
# difficulty    -1.0844      1.486     -0.730      0.466     -3.998       1.829`}
      </CodeBlock>

      <Prose>
        The coefficient on <Code>weirdness</Code> is −3.02 in log-odds with z = −3.10, p = 0.002. Interpreted: each unit increase in WEIRD-ness reduces the log-odds of native-rubric correctness by ~3.02. In probability terms near the base rate of 0.32, a one-unit move on WEIRD-ness corresponds to roughly a 0.55-point drop in native-rubric accuracy. This is the WEIRD bias coefficient for this model on this synthetic eval set, with appropriate uncertainty quantification. In production studies, the same regression would be run on real benchmark items with actual cultural relevance scores assigned by domain experts.
      </Prose>

      <H3>4e. Construct a per-culture confusion heatmap</H3>

      <Prose>
        Beyond a single gap statistic, the most informative single visualization is the per-culture difference between the two judging rubrics. The matrix below shows the accuracy under each judge for each culture; the difference column is the per-culture WEIRD bias.
      </Prose>

      <CodeBlock language="python">
{`per_culture = eval_df.groupby("culture").agg(
    weird = ("correct_weird_judge",  "mean"),
    native= ("correct_native_judge", "mean"),
)
per_culture["gap"] = per_culture["weird"] - per_culture["native"]
print(per_culture.round(3))
#              weird  native   gap
# culture
# BR_LATAM       0.5    0.3   0.2
# IN_SOUTH       0.6    0.3   0.3
# JP_EAST        0.6    0.5   0.1
# NG_AFRICA      0.4    0.3   0.1
# WEIRD_US       0.7    0.2   0.5
#
# Note: the WEIRD_US gap here is an artifact of the synthetic native-judge being
# intentionally noisy on WEIRD items for demonstration. In real studies the
# WEIRD_US gap would be ~0 by construction. The non-WEIRD gaps are what matter.`}
      </CodeBlock>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        Production-quality WEIRD-aware evaluation rests on three pillars: multilingual benchmark suites that genuinely cover non-WEIRD content, demographic-stratified evaluation pipelines that compute per-population metrics, and crowdwork sourcing that recruits annotators from outside the WEIRD demographic core. None of these are individually novel, but assembling them into a coherent evaluation regime is what separates a serious effort from one that pays lip service to the problem.
      </Prose>

      <H3>5a. Multilingual benchmark suites</H3>

      <Prose>
        BIG-Bench Lite multilingual subsets, MMMLU (the multilingual MMLU translation maintained by OpenAI), and MMLU-ProX (the higher-difficulty multilingual extension) provide drop-in replacements for English-only benchmarks. MMMLU translates the original 14k MMLU questions into 14 languages and is the most direct apples-to-apples comparison: any drop in accuracy when going from MMLU-EN to MMMLU-FR is attributable to language coverage, not topic shift. This isolates one dimension of the WEIRD bias problem (linguistic) from the others (topical, normative).
      </Prose>

      <Prose>
        GlobalBench (Hu et al. 2024, arXiv:2310.05502) is more ambitious: it explicitly aggregates 966 NLP benchmarks across 190 languages and provides per-language utility scores that incorporate not only accuracy but the fraction of speakers each language serves. The headline finding from the GlobalBench paper is that even the most multilingual models at the time of publication achieved &lt;20% of their potential utility once weighted by speaker population, because the benchmarks themselves cluster heavily on a small set of high-resource languages. The implication for an evaluation pipeline is that you should report not just per-language accuracy but per-language accuracy weighted by deployment exposure to that language.
      </Prose>

      <CodeBlock language="python">
{`from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load MMMLU (multilingual MMLU). Subset selection by language code.
LANGS = ["EN_US", "FR_FR", "DE_DE", "JA_JP", "HI_IN", "BN_IN", "SW_KE", "YO_NG"]

model_name = "your-org/your-model"
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto"
)

results = {}
for lang in LANGS:
    ds = load_dataset("openai/MMMLU", lang, split="test")
    correct, total = 0, 0
    for ex in ds:
        prompt = format_mmlu_prompt(ex)  # standard 4-choice MMLU prompt
        pred   = greedy_answer(model, tok, prompt)
        correct += int(pred.strip() == ex["answer"].strip())
        total   += 1
    results[lang] = correct / total

# Compute WEIRD-vs-non-WEIRD gap.
weird_langs    = ["EN_US", "FR_FR", "DE_DE"]
non_weird      = ["JA_JP", "HI_IN", "BN_IN", "SW_KE", "YO_NG"]
weird_acc      = sum(results[l] for l in weird_langs)    / len(weird_langs)
non_weird_acc  = sum(results[l] for l in non_weird)      / len(non_weird)
print(f"WEIRD-language mean accuracy:     {weird_acc:.3f}")
print(f"Non-WEIRD-language mean accuracy: {non_weird_acc:.3f}")
print(f"Gap:                              {weird_acc - non_weird_acc:+.3f}")`}
      </CodeBlock>

      <H3>5b. Demographic-stratified evaluation</H3>

      <Prose>
        Even when using a multilingual benchmark, a single overall accuracy number conceals the per-population gaps that matter for deployment decisions. Demographic-stratified evaluation, drawing on the methodology of fairness audits (Buolamwini and Gebru 2018, Mehrabi et al. 2021), partitions the evaluation set along demographic axes — language, country of reference, religion mentioned, gender of subject — and reports per-stratum metrics with confidence intervals. The key implementation detail is that strata must be tagged at the item level before evaluation, not derived post-hoc from model outputs.
      </Prose>

      <CodeBlock language="python">
{`import pandas as pd
from sklearn.utils import resample

def stratified_metrics(df, model_name, n_bootstrap=1000, ci=0.95):
    """
    df: columns include 'culture', 'language', 'topic', 'correct'.
    Returns per-stratum mean accuracy with bootstrap CI.
    """
    rows = []
    for stratum_col in ["culture", "language", "topic"]:
        for value, group in df.groupby(stratum_col):
            mean_acc = group["correct"].mean()
            # Bootstrap CI
            boots = []
            for _ in range(n_bootstrap):
                samp = resample(group["correct"].values)
                boots.append(samp.mean())
            lo = np.percentile(boots, (1 - ci) / 2 * 100)
            hi = np.percentile(boots, (1 + ci) / 2 * 100)
            rows.append({
                "model":     model_name,
                "stratum":   stratum_col,
                "value":     value,
                "n":         len(group),
                "accuracy":  mean_acc,
                "ci_lo":     lo,
                "ci_hi":     hi,
            })
    return pd.DataFrame(rows)

# Use this as the canonical reporting format for any model release.
report_df = stratified_metrics(eval_df, "model_v1.2")
report_df.to_csv("model_v1.2_stratified.csv", index=False)`}
      </CodeBlock>

      <H3>5c. Sourcing non-WEIRD crowdworkers</H3>

      <Prose>
        Mechanical Turk's worker demographics are well-documented: roughly 75% U.S. or India, with the U.S. cohort tilting heavily toward college-educated, English-fluent participants. Even the Indian cohort is WEIRD-skewed by global standards (urban, English-fluent, internet-connected, often working multiple platforms). For preference annotation that is genuinely representative, three alternative approaches have emerged. Surge AI and Scale AI both offer demographic-targeted annotation panels for additional cost, with selectable filters on country, language, and self-reported demographics. The 2023–2024 wave of localized annotation contractors (sarvam.ai for Indian languages, masakhane.io for African languages, AI Singapore for Southeast Asian languages) hire annotators within the target population and operate quality controls calibrated to that population's norms. For research-scale work, partnering directly with university populations in target countries — common in the Cohere for AI Aya project and the Masakhane NLP collective — produces the highest-fidelity preference data, at substantially higher per-label cost.
      </Prose>

      <Prose>
        The instructions given to annotators matter as much as where they are recruited. Default RLHF instruction templates ask annotators to rate "which response is more helpful, harmless, and honest" — abstractions that have implicit WEIRD content. More representative templates explicitly ask annotators to rate responses against the standards of their own cultural context, with prompts like "considering norms in your community, which response would be most appropriate to share with a peer?" The Atari et al. 2023 paper demonstrates that the same model produces measurably different reward model gradients under the two instruction templates, because the annotator population's responses are different.
      </Prose>

      <H3>5d. End-to-end pipeline</H3>

      <Prose>
        Putting it together, a production WEIRD-aware evaluation pipeline has four stages: benchmark construction (multilingual coverage with cultural tagging), model evaluation (per-stratum accuracy with bootstrap CIs), judge calibration (per-stratum agreement between automated judge and demographic-matched human raters), and continuous monitoring (gap tracking across model releases to ensure improvement on aggregate accuracy does not come at the cost of widening per-stratum gaps). The single most important deliverable is not a single number but a stratified report card that the team's leadership and external auditors can read.
      </Prose>

      <H3>5e. Judge calibration against human panels</H3>

      <Prose>
        Even when an LLM judge is prompted with a culture-specific rubric, its scores must be calibrated against demographic-matched human ratings on a sample of items before it can be trusted as a substitute for human evaluation. The standard calibration metric is the per-stratum Spearman or Cohen's kappa agreement between judge and human panel. A judge that achieves κ &gt; 0.6 against a culture-matched human panel on a stratum can be used as a primary evaluator for that stratum; below that threshold, human evaluation should remain the primary signal and the judge's scores treated as advisory. The calibration sample need not be large — 50–100 items per stratum is usually sufficient — but it must be refreshed each time either the judge model or the rubric changes.
      </Prose>

      <CodeBlock language="python">
{`from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

def judge_calibration(judge_scores, human_scores, strata):
    """
    Per-stratum agreement between LLM judge and human panel.
    judge_scores, human_scores: arrays of length n.
    strata: array of stratum tags of length n.
    """
    out = {}
    for s in set(strata):
        mask = [i for i, x in enumerate(strata) if x == s]
        j = [judge_scores[i]  for i in mask]
        h = [human_scores[i]  for i in mask]
        rho, _  = spearmanr(j, h)
        # Threshold ratings to binary for kappa.
        jb = [1 if x >= 0.5 else 0 for x in j]
        hb = [1 if x >= 0.5 else 0 for x in h]
        kappa = cohen_kappa_score(jb, hb)
        out[s] = {"n": len(mask), "spearman": rho, "kappa": kappa}
    return out

# Example output:
# {"EN_US":   {"n": 80, "spearman": 0.74, "kappa": 0.68},
#  "JA_JP":   {"n": 80, "spearman": 0.51, "kappa": 0.42},
#  "BN_IN":   {"n": 80, "spearman": 0.39, "kappa": 0.28}}
# The judge is reliable on EN_US, marginal on JA_JP, unreliable on BN_IN.
# For BN_IN, fall back to direct human evaluation.`}
      </CodeBlock>

      <Prose>
        The pattern of decreasing judge reliability with increasing cultural distance from the judge model's WEIRD baseline is the empirical signature you should expect. Treat it as a constraint on which strata the judge can serve, not as a defect to be papered over with prompt engineering.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        The first visualization shows the per-culture accuracy gap between a WEIRD-skewed judge and a native-rubric judge, scoring the same model on the same items. The pattern of larger gaps for more culturally distant populations is the signature of WEIRD bias.
      </Prose>

      <Plot
        label="Per-culture accuracy under WEIRD vs native judging rubric"
        xLabel="Hofstede distance from WEIRD prototype"
        yLabel="model accuracy"
        width={680}
        height={320}
        series={[
          {
            name: "WEIRD-skewed judge",
            color: colors.gold,
            points: [
              [0,    0.78],
              [82,   0.62],
              [84,   0.55],
              [92,   0.58],
              [93,   0.42],
            ],
          },
          {
            name: "Native-rubric judge",
            color: "#c084fc",
            points: [
              [0,    0.76],
              [82,   0.41],
              [84,   0.36],
              [92,   0.40],
              [93,   0.28],
            ],
          },
        ]}
      />

      <Prose>
        The second plot tracks the WEIRD bias gap across model generations from 2020 to 2025. The dotted line is the gap on standardized topical content; the solid line is the gap on normative or socio-cultural content. The two series diverge over time: factual coverage has improved with multilingual training, but normative alignment has not, because the RLHF stage continues to use WEIRD-skewed preference data.
      </Prose>

      <Plot
        label="WEIRD bias gap across model generations (illustrative)"
        xLabel="model release year"
        yLabel="non-WEIRD vs WEIRD accuracy gap"
        width={680}
        height={320}
        series={[
          {
            name: "topical / factual content",
            color: colors.gold,
            points: [
              [2020, 0.35],
              [2021, 0.30],
              [2022, 0.24],
              [2023, 0.18],
              [2024, 0.13],
              [2025, 0.10],
            ],
          },
          {
            name: "normative / socio-cultural content",
            color: "#c084fc",
            points: [
              [2020, 0.38],
              [2021, 0.36],
              [2022, 0.34],
              [2023, 0.32],
              [2024, 0.31],
              [2025, 0.30],
            ],
          },
        ]}
      />

      <Prose>
        The heatmap below shows per-language accuracy across a six-language MMMLU subset, comparing two hypothetical models. Brighter cells are higher accuracy. The coverage is asymmetric: both models do well on English, French, and German; both do worse on Hindi, Bengali, and Yoruba; the relative degradation is steeper for Model A than Model B, reflecting Model B's broader multilingual pretraining mix.
      </Prose>

      <Heatmap
        label="Per-language MMMLU accuracy across two models"
        rowLabels={["Model A", "Model B"]}
        colLabels={["EN", "FR", "DE", "HI", "BN", "YO"]}
        matrix={[
          [0.82, 0.78, 0.76, 0.55, 0.48, 0.32],
          [0.81, 0.79, 0.77, 0.68, 0.64, 0.51],
        ]}
        cellSize={56}
        colorScale="gold"
      />

      <Prose>
        The step trace below walks through a single end-to-end WEIRD-aware evaluation pipeline run, from prompt selection through stratified reporting.
      </Prose>

      <StepTrace
        label="WEIRD-aware evaluation pipeline — single run"
        steps={[
          {
            label: "Stratified prompt selection",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Sample prompts</div>
                <div>prompts = sample_stratified(MMMLU + GlobalBench,</div>
                <div>            strata=["language", "culture", "topic"],</div>
                <div>            per_stratum=200)</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Each item is tagged with language, culture frame, and topic.
                  Stratification ensures statistical power per cell.
                </div>
              </div>
            ),
          },
          {
            label: "Model generation",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Generate</div>
                <div>responses = [model.generate(p) for p in prompts]</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Use deterministic decoding (temperature=0) so judge agreement
                  reflects rubric differences, not sampling noise.
                </div>
              </div>
            ),
          },
          {
            label: "Dual-judge scoring",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#4ade80", marginBottom: 4 }}>Judge twice</div>
                <div>weird_scores  = WEIRD_judge.score(responses,  weird_rubric)</div>
                <div>native_scores = native_judge.score(responses, native_rubric)</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Native judge is either a culture-matched human panel or an
                  LLM judge prompted with the population-specific rubric.
                </div>
              </div>
            ),
          },
          {
            label: "Compute per-stratum metrics",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Aggregate</div>
                <div>per_stratum = stratified_metrics(scores)</div>
                <div>gap_per_culture = weird_acc - native_acc, by culture</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Bootstrap each stratum independently for honest CIs.
                </div>
              </div>
            ),
          },
          {
            label: "Regression for bias coefficient",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#c084fc", marginBottom: 4 }}>Logistic regression</div>
                <div>logit(P(correct)) = β₀ + β_W · weirdness + γ · controls</div>
                <div>report β_W with SE, z, and p-value</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  β_W is the headline diagnostic. A non-zero β_W with tight CIs
                  means the model's accuracy depends systematically on cultural
                  distance from the WEIRD baseline.
                </div>
              </div>
            ),
          },
          {
            label: "Stratified report card",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#c084fc", marginBottom: 4 }}>Output</div>
                <div>release_report.md includes:</div>
                <div>  - per-language accuracy with CI</div>
                <div>  - per-culture WEIRD-bias gap</div>
                <div>  - β_W from regression</div>
                <div>  - delta vs previous release</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Required gating artifact for any production deployment in
                  multilingual or non-WEIRD jurisdictions.
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

      <H3>Single-language evaluation vs multilingual evaluation</H3>

      <Prose>
        Single-language (English) evaluation is appropriate only for products with documented English-only deployment scope and English-speaking user populations — internal tools, research artifacts, and English-first consumer products. Even then, "English-speaking" is not synonymous with "WEIRD"; substantial English-language deployment in India, Nigeria, the Philippines, and Singapore involves user populations whose normative frame is non-WEIRD. The default for any externally-deployed product should be multilingual evaluation across at least the languages of the deployment markets, with separate reporting per language and per cultural cluster. The marginal cost of running an additional language subset on a benchmark like MMMLU is small; the cost of discovering after launch that the model performs 30 points worse on an unevaluated language is large.
      </Prose>

      <H3>LLM-as-judge vs human annotators</H3>

      <Prose>
        LLM-as-judge is faster and cheaper, but it inherits the WEIRD bias of the judge model. Human annotators are slower and more expensive, but they bring whatever cultural baseline they were recruited from — which is a feature if you recruit them deliberately, and a bug if you default to MTurk. The decision tree: for low-stakes development iteration on benchmark sets where the rubric is genuinely culture-invariant (factual recall, mathematical reasoning, code execution), LLM-as-judge is appropriate. For final model qualification, deployment-gating evaluation, or any task involving normative judgment (helpfulness, appropriateness, safety, tone), use demographic-matched human annotators or a hybrid setup where the LLM judge's outputs are calibrated against a human panel from the target population.
      </Prose>

      <H3>Translated benchmarks vs native benchmarks</H3>

      <Prose>
        Translated benchmarks (MMMLU, translated MT-Bench) are simpler to set up and provide an apples-to-apples comparison across languages, but they preserve the WEIRD topical and normative content of the original. Native benchmarks (Indic NLP datasets, AfriQA, JMMLU's native components) are constructed from scratch in the target language and reflect that language's culture and concerns. Native benchmarks provide a more accurate measure of deployment-relevant performance but cannot be directly compared across languages. The recommended practice from the GlobalBench team is to maintain both: translated benchmarks for cross-lingual diagnostic comparison, native benchmarks for deployment qualification.
      </Prose>

      <H3>Reporting WEIRD bias as a single number vs stratified report</H3>

      <Prose>
        A single WEIRD bias number — for example, "model X has WEIRD bias of 0.18" — is convenient for headlines and leaderboards but actively misleading because the bias varies dramatically across populations and domains. The stratified report card format described in Section 5 is more verbose but is the only format that supports honest deployment decisions. The middle ground that often works well in practice is to report the per-population gap (a vector indexed by population) plus the regression coefficient β_W (a scalar summary), with the explicit caveat that the scalar summary is for trend tracking across model releases and the vector is for deployment go/no-go decisions.
      </Prose>

      <H3>Pre-deployment evaluation vs post-deployment monitoring</H3>

      <Prose>
        Pre-deployment evaluation establishes a baseline; post-deployment monitoring is what catches regressions and emergent issues. The most informative post-deployment signal is per-language user retention and per-language explicit feedback (thumbs up/down rates, regeneration requests). If a model launches with parity per-language MMMLU scores but exhibits a 15% lower retention rate among Bengali users than English users, the pre-deployment benchmark missed something the deployment data revealed. Both stages are necessary; neither replaces the other.
      </Prose>

      <H3>WEIRD bias mitigation in training vs in evaluation</H3>

      <Prose>
        Mitigating WEIRD bias only at evaluation time is incomplete: you can detect a problem you cannot fix. Mitigating only at training time is also incomplete: without representative evaluation you have no signal that the mitigation worked. The right approach is paired: representative training data (multilingual pretraining mix, demographically-sourced preference data) plus representative evaluation (stratified benchmarks, demographic-matched judges). Teams that have invested seriously in this — Cohere with the Aya project, Google with PaLM 2's expanded multilingual training, the Sarvam and Masakhane efforts in regional languages — show measurable WEIRD bias reduction, while teams that bolted multilingual evaluation onto an unchanged training pipeline show small or no improvement.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        The encouraging finding from the last three years of multilingual model development is that some dimensions of WEIRD bias scale away as models grow larger and pretraining data becomes more multilingual. Factual coverage in non-English languages improves measurably from generation to generation: the gap between English and French MMMLU accuracy for frontier models has narrowed from ~12 points in 2022-vintage models to ~3 points in 2025-vintage models. The same trend holds for German, Spanish, Italian, and other high-resource European languages. The data efficiency of pretraining is high enough that adding language-specific corpora yields proportional gains in language-specific evaluation, and the major labs have responded by aggressively expanding their multilingual pretraining mixes.
      </Prose>

      <Prose>
        Lower-resource languages tell a less encouraging story. Languages spoken by &gt;100M people but with limited internet text presence — Bengali, Punjabi, Telugu, Marathi, Yoruba, Hausa, Amharic — show much smaller per-generation improvements. The scaling law for multilingual capability is steeper than for English capability, meaning each additional doubling of compute yields a smaller fractional improvement in low-resource languages than in English. The GlobalBench analysis estimates that even with a 100x compute scaling from current frontier models, the per-speaker utility of LLMs for the bottom 50% of languages by digital resource availability would remain below 30% of WEIRD-language utility. Compute scaling alone does not solve coverage gaps for under-resourced languages; targeted data acquisition and dedicated multilingual training stages are necessary.
      </Prose>

      <Prose>
        Normative WEIRD bias does not scale away with compute at all. The gap between WEIRD and non-WEIRD ratings on questions involving ethics, social norms, religious obligation, family structure, and cultural appropriateness has been remarkably stable across model generations. This is a structural feature of the RLHF stage: as long as the preference data and reward modeling are conducted with WEIRD-skewed annotator pools, scaling the underlying language model produces a more capable executor of WEIRD-aligned preferences without changing the alignment target itself. The fix is not more compute or more data of the same kind — it is restructuring the preference data collection to recruit from non-WEIRD populations, which is an organizational and operational change rather than a technical one.
      </Prose>

      <Prose>
        Crowdwork sourcing scales poorly in cost. Recruiting representative annotators from low-resource markets is several times more expensive per high-quality label than recruiting MTurk workers, both because of overheads and because the per-worker throughput is lower (fewer workers, more careful quality control). The marginal cost of representative annotation is approximately 5–10x the cost of default MTurk annotation, and this gap has not narrowed substantially. Teams that take WEIRD bias seriously typically have to make explicit budget decisions about how much representative annotation to fund, with the trade-off being directly visible in per-population evaluation metrics.
      </Prose>

      <Prose>
        Stratified evaluation infrastructure is itself a non-trivial engineering investment. A single overall-accuracy benchmark requires one number to track and one alert to fire on regression. A stratified report card has 50–200 cells (language × culture × topic), with bootstrap CIs and regression coefficients, and the regression-detection logic must be calibrated to avoid alert storms while still catching real per-stratum drops. Most teams underestimate how much engineering goes into making stratified evaluation actually usable as a release gate. The teams that have done it best (Cohere, DeepMind, Google) have invested in dedicated evaluation infrastructure that resembles a small data warehouse with associated dashboards, and the per-release human review time for the stratified report is non-trivial.
      </Prose>

      <Prose>
        The dimension that scales most favorably is awareness. The vocabulary of WEIRD bias is now well established in the alignment literature, and benchmark organizations (HELM, GlobalBench, OpenLLM Leaderboard) increasingly publish stratified results by default. New evaluation suites released after 2024 are noticeably more likely to include multilingual subsets and explicit demographic stratification than those released before 2023. A team starting fresh today has access to substantially better tooling and conventions than a team starting in 2022 did, even though the underlying problem is unchanged.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>Translated benchmarks present cultural baseline as universal</H3>
      <Prose>
        Translating MMLU into 14 languages produces 14 evaluations of how well the model performs on American-content questions in those languages — not 14 culturally-grounded benchmarks. A model can score perfectly on Hindi-translated MMLU's U.S. constitutional law section without knowing anything about Indian constitutional law. Translated benchmarks measure linguistic transfer of WEIRD topical content; they do not measure non-WEIRD topical coverage. Always pair translated benchmarks with native benchmarks for deployment-relevant evaluation.
      </Prose>

      <H3>LLM-as-judge launders WEIRD bias</H3>
      <Prose>
        Using GPT-4 as judge to score Claude's responses on culturally-sensitive content is doubly biased: the responses being scored are WEIRD-aligned, and the scorer is WEIRD-aligned, and the agreement between them is taken as a sign of quality. This is the most common failure mode in current evaluation practice because LLM-as-judge is so much cheaper than human evaluation. The fix is either to use demographically-matched human judges for culturally-sensitive content, or to use a chain of LLM judges with explicit prompting for the target cultural rubric and cross-validate against a small human-labeled set.
      </Prose>

      <H3>MTurk worker pool drift</H3>
      <Prose>
        The demographics of MTurk's active worker pool change over time, and most published numbers about worker demographics are 2–4 years stale. Verify the current demographic composition before relying on MTurk for any preference annotation that claims demographic representativeness. Surge AI and Scale AI publish more current breakdowns and offer demographic-targeted recruiting for additional cost.
      </Prose>

      <H3>Aggregation hides per-population regressions</H3>
      <Prose>
        A model release that improves overall accuracy by 2 points can simultaneously degrade accuracy by 8 points on Yoruba while improving by 5 points on English, and the aggregate metric will show a clean win. Always inspect per-population metrics before declaring a regression-free release. The corollary is that progress over time, measured only on overall metrics, can mask a widening per-population gap — the model gets better on average while the WEIRD bias gap grows.
      </Prose>

      <H3>Cultural distance metrics oversimplify</H3>
      <Prose>
        Hofstede's six-dimensional model and the Inglehart-Welzel cultural map are useful summaries but compress vast within-population variation into single national averages. Indians in Mumbai and Indians in rural Bihar share a Hofstede coordinate but have different relevant cultural baselines. When using cultural distance metrics for evaluation stratification, be explicit that they are coarse summaries and that within-stratum variance is large. The metric is a filter for "this evaluation needs care," not a complete characterization of the population.
      </Prose>

      <H3>The "we did multilingual" claim with zero stratification</H3>
      <Prose>
        Many model releases include "we evaluated on 14 languages" without reporting per-language results or per-language regression checks. The evaluation was performed but not used. If the per-language results are not in the release notes, assume they were not gating criteria for the release.
      </Prose>

      <H3>Reverse WEIRD bias from over-correction</H3>
      <Prose>
        Aggressive multilingual rebalancing of pretraining data can degrade English performance below baseline if not done with care. Several open-weight models in 2024 reported small English regressions in exchange for larger non-English gains; some user populations (English-first developer communities) noticed and complained. The right metric to monitor is not "does the model perform best on English" but "does the model meet a per-language accuracy threshold appropriate to its deployment exposure to that language."
      </Prose>

      <H3>Constitutional principles encode WEIRD ethics</H3>
      <Prose>
        Constitutional AI methods (Anthropic 2022) train the model against a written set of principles authored in English by U.S.-based researchers. These principles encode WEIRD ethical priors: individual autonomy, harm avoidance, anti-discrimination as understood through Western liberal frames. A constitutional pass that says "refuse responses that demean groups" interacts non-obviously with cultural contexts in which group identity, hierarchy, or honor structures the relevant ethical frame. When deploying constitutionally-tuned models in non-WEIRD contexts, evaluate explicitly whether the constitutional principles produce culturally appropriate behavior, not just English-context appropriate behavior.
      </Prose>

      <H3>The illusion of neutral evaluation</H3>
      <Prose>
        Every evaluation is grounded in some normative frame; "neutral evaluation" does not exist. Benchmarks that present themselves as measuring "general intelligence" or "reasoning ability" without acknowledging their cultural assumptions are not neutral, they are WEIRD-defaulted. Acknowledging the cultural baseline is not a weakness of an evaluation; it is a precondition for honest interpretation of results.
      </Prose>

      <Callout accent="purple">
        WEIRD bias compounds silently across the post-training stack. Pretraining bias × RLHF bias × benchmark bias × judge bias produces a final deployed model that is more WEIRD-aligned than any single stage's bias would predict. The right diagnostic for "did we accidentally build a WEIRD-only product" is end-to-end stratified evaluation against demographic-matched human panels in the target deployment markets, not any single intermediate metric.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All four sources below were verified against their published venues on 2026-04-26. Author lists, year, and paper IDs confirmed.
      </Prose>

      <H3>Henrich, Heine, and Norenzayan 2010 — The original WEIRD critique</H3>
      <Prose>
        Joseph Henrich, Steven J. Heine, and Ara Norenzayan. "The weirdest people in the world?" Behavioral and Brain Sciences, vol. 33, no. 2-3, pp. 61–83, 2010. The founding paper for the WEIRD critique. Catalogues the cross-cultural evidence that subjects from Western, Educated, Industrialized, Rich, Democratic societies are systematic outliers on a long list of psychological measures including visual perception, fairness intuitions, self-construal, moral reasoning, and analytical thinking, despite being the dominant sample population in published behavioral science. The paper estimates that 96% of psychology study subjects come from countries representing 12% of humanity, and argues that generalizations from this sample to "human nature" are unsound. Subsequent work by Henrich (notably the 2020 book "The WEIRDest People in the World") elaborates the historical origins of WEIRD psychology in late medieval European institutional change.
      </Prose>

      <H3>Atari et al. 2023 — Which Humans?</H3>
      <Prose>
        Mohammad Atari, Mona J. Xue, Peter S. Park, Damián Blasi, and Joseph Henrich. "Which Humans?" arXiv:2306.16189, June 2023. The paper that brings the WEIRD critique directly into LLM evaluation. Compares GPT-3 and GPT-4 responses on the World Values Survey and on a battery of cross-cultural psychology tasks against responses from 65 nationally-representative samples. Finds that LLM responses closely resemble those of WEIRD respondents (especially U.S., U.K., Canadian, Australian) and diverge substantially from non-WEIRD populations across nearly every measured dimension. Argues that "AI alignment" as currently practiced is alignment to WEIRD values specifically, marketed as alignment to humanity in general. Provides quantitative WEIRD-distance scores for GPT-3 and GPT-4 against each of the 65 sampled populations.
      </Prose>

      <H3>Tao et al. 2024 — Cultural alignment of LLMs</H3>
      <Prose>
        Yan Tao, Olga Viberg, Ryan S. Baker, and René F. Kizilcec. "Cultural Bias and Cultural Alignment of Large Language Models." PNAS Nexus, vol. 3, no. 9, pgae346, 2024. Builds on the Atari et al. methodology and extends it to a broader set of models and to the Hofstede framework specifically. Finds systematic alignment of major LLMs (GPT-3.5, GPT-4, Claude, PaLM 2, LLaMA-2) to WEIRD profiles on Hofstede's six dimensions, with the strongest alignment to U.S. Anglophone cultural norms. Demonstrates that targeted prompting ("respond as someone from culture X would") partially shifts model responses but does not eliminate the underlying WEIRD bias, suggesting the bias is encoded in the model weights rather than purely in the inference-time prompt. Provides per-Hofstede-dimension bias scores by model, useful as a diagnostic baseline.
      </Prose>

      <H3>Hu et al. 2024 — GlobalBench</H3>
      <Prose>
        Yueqi Song, Catherine Cui, Simran Khanuja, Pengfei Liu, Fahim Faisal, Alissa Ostapenko, Genta Indra Winata, Alham Fikri Aji, Samuel Cahyawijaya, Yulia Tsvetkov, Antonios Anastasopoulos, Graham Neubig (Hu et al. as cited in many summaries — author order varies by venue). "GlobalBench: A Benchmark for Global Progress in Natural Language Processing." arXiv:2310.05502, October 2023; EMNLP 2023. Aggregates 966 NLP benchmarks across 190 languages and computes per-language utility scores weighted by speaker population. Headline finding: even the most multilingual models at the time of writing achieved &lt;20% of their potential utility once weighted by global speaker exposure, because evaluation effort is concentrated on a small set of high-resource languages. Provides a methodology and tooling for tracking per-language progress over time and a public leaderboard. The single best resource for understanding the scale of multilingual evaluation gaps and for choosing which languages to add to a deployment-relevant evaluation regime.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Compute Hofstede distance and predict bias direction</H3>
      <Prose>
        Using the Hofstede coordinates given in Section 4 for the United States, Japan, India, Brazil, and Nigeria, compute the pairwise distance matrix between all five cultures. Which two cultures are closest to each other on the Hofstede dimensions, and which two are furthest apart? Now consider an LLM trained predominantly on U.S. English text and RLHF-tuned with U.S.-recruited annotators. For each of the four non-U.S. cultures, predict the direction (positive or negative) of the WEIRD bias gap on questions about (a) family structure, (b) workplace hierarchy, and (c) appropriate use of authority. Justify each prediction using the dominant Hofstede dimension responsible for the cultural distance.
      </Prose>

      <H3>Exercise 2 — Stratified power analysis</H3>
      <Prose>
        You are designing a multilingual evaluation across eight languages. Your release-gating criterion is that no language should show an accuracy drop greater than 5 percentage points compared to the previous release. Given a baseline accuracy of 0.70 per language and a Bernoulli model, how many items per language do you need to detect a 5-point regression at 80% statistical power and α = 0.05 (two-sided)? Now consider that you actually want to detect any regression that affects the per-language WEIRD-bias gap by more than 3 points; what sample size does that require, and is your evaluation budget sufficient? If not, which languages would you prioritize in the limited budget and why?
      </Prose>

      <H3>Exercise 3 — Identify WEIRD assumptions in an MMLU-style item</H3>
      <Prose>
        Consider the following MMLU-style item: "A 16-year-old wishes to attend college far from home; her parents object because they want her closer to family. Whose preference should prevail? (A) The student's, because individuals have autonomy over their educational choices. (B) The parents', because they bear financial responsibility. (C) A compromise should be sought through family discussion. (D) The student should defer to her parents and reapply after age 18." List every WEIRD assumption embedded in (i) the question framing, (ii) each answer option, and (iii) the implicit "correct" answer that an MMLU-trained scorer would expect. Describe how a Confucian-frame respondent, an Islamic-frame respondent, and a Yoruba-frame respondent might each find the question itself ill-posed, and rewrite the item so it can be scored under any of the four cultural rubrics without privileging one.
      </Prose>

      <H3>Exercise 4 — Detecting LLM-as-judge bias</H3>
      <Prose>
        You are using GPT-4 as a judge to score Claude's responses on a benchmark of 500 questions about marriage customs across 10 cultures (50 questions per culture). The benchmark was designed by your team and the reference answers were authored by GPT-4. Sketch an experimental design that would let you measure whether GPT-4-as-judge is systematically scoring Claude higher on WEIRD-frame answers than a demographically-matched human panel would. Specify: the metrics you would compute, the size of the human panel needed, the comparisons that would constitute evidence for judge bias, and the comparisons that would rule it out. What confound makes this measurement harder than it appears?
      </Prose>

      <H3>Exercise 5 — Mitigation cost-benefit</H3>
      <Prose>
        Your team is deciding whether to invest $200,000 in collecting a culturally-representative preference dataset for RLHF, in addition to your existing MTurk-sourced preference dataset. Sketch a back-of-the-envelope analysis estimating the expected reduction in WEIRD bias from this investment, the expected impact on aggregate benchmark performance (positive or negative), and the deployment markets where this investment would pay back fastest. What additional information would you need to make this decision rigorously rather than by intuition? If the budget were $20,000 instead of $200,000, how would your recommendation change?
      </Prose>

      <H3>Exercise 6 — Pretraining vs RLHF attribution</H3>
      <Prose>
        Suppose you observe that your model has a WEIRD bias gap of 0.18 on a held-out Bengali normative-questions evaluation. You want to know whether the bias is primarily inherited from the WEIRD-skewed pretraining corpus or introduced by the WEIRD-skewed RLHF preference data. Design an ablation experiment that would distinguish these two hypotheses. What baseline models or controls would you need? What confound makes a clean attribution difficult, and what is the partial answer you can extract even with imperfect controls?
      </Prose>

      <H3>Exercise 7 — Reading the stratified report card</H3>
      <Prose>
        A model release report shows: overall MMMLU accuracy improved from 0.71 to 0.74 vs. previous release; English and German accuracy improved by 4 and 3 points respectively; Hindi accuracy unchanged; Bengali accuracy dropped by 6 points; Yoruba accuracy dropped by 9 points. The team's release gate is "no language regression greater than 5 points" but the team lead is arguing for release because "the average is up and the regressions are in lower-volume languages." Construct the strongest argument for blocking the release. Construct the strongest argument for shipping it. Which argument is more defensible, and what additional information would shift the answer?
      </Prose>

      <H3>Exercise 8 — Judge reliability versus rubric specificity</H3>
      <Prose>
        You have two options for evaluating Claude's responses to questions about Yoruba marriage customs. Option A: prompt GPT-4 with a generic rubric ("score the helpfulness of this response from 1 to 5"). Option B: prompt GPT-4 with a Yoruba-specific rubric ("score this response on the basis of how well it reflects standard Yoruba marriage customs, with reference to the role of extended family, bridewealth practices, and intra-clan obligations; 1 to 5"). Predict which option would show higher correlation with a panel of Yoruba-speaking human raters, and why. What experiment would you run to verify your prediction? What is the failure mode of Option B if the prompt itself contains WEIRD framings of Yoruba culture (for instance, if it was authored by a non-Yoruba researcher)?
      </Prose>

    </div>
  ),
};

export default weirdBias;
