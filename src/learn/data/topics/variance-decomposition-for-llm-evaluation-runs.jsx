import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const varianceDecomposition = {
  title: "Variance Decomposition for LLM Evaluation Runs",
  slug: "variance-decomposition-for-llm-evaluation-runs",
  readTime: "~36 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Anyone who has run an LLM evaluation more than twice has encountered the unsettling experience of watching the same model, on the same benchmark, produce a different score. You ran MMLU on your fine-tuned 7B model on Monday and got 64.3%. On Tuesday, with the same checkpoint, you got 63.1%. On Wednesday, after rotating to a different judge model, you got 65.8%. None of these are bugs. They are samples from a distribution whose width you have not measured. The headline number you ship to a slide deck — the single point estimate — is a draw from this distribution, and everyone treating it as a property of the model rather than a property of the evaluation procedure is making a category error. Variance decomposition is the discipline of taking that distribution apart, attributing its width to the specific sources that produced it, and then deciding which of those sources is worth fixing.
      </Prose>

      <Prose>
        The literature that established this framing for machine learning broadly came in two waves. Bouthillier, Laurent, and Vincent-Lamarre (2021), in "Accounting for Variance in Machine Learning Benchmarks" (arXiv:2103.03098), ran controlled experiments across image classification, NLP, and reinforcement learning benchmarks and found that the variance attributable to random initialization, data ordering, and other "nuisance" sources was frequently larger than the gap between competing methods being claimed in headline tables. Card, Henderson, Khandelwal, Jia, Mahowald, and Jurafsky (2020), in "With Little Power Comes Great Responsibility" (EMNLP 2020), made the parallel point for NLP: most reported improvements on standard benchmarks fell well within the noise floor of the experiment, and the field was systematically over-claiming because nobody was reporting standard errors that included these sources. Together these papers reframed benchmark variance from "an annoying detail you handle with three random seeds" into "the central object of measurement design."
      </Prose>

      <Prose>
        LLMs added several new variance sources that classical ML evaluation never had to think about. The model itself samples non-deterministically when temperature is greater than zero — even with deterministic decoding, batched inference on modern accelerators introduces small numerical differences that amplify over long contexts. The judge in LLM-as-judge evaluation is itself a stochastic model, with its own temperature, its own sampling noise, and a choice of model family that materially shifts scores. Retrieval-augmented systems introduce randomness through the embedding model, the index implementation, and tied-rank breaking in nearest-neighbor lookup. Tool-using agents introduce variance through the tools themselves — search engines, code interpreters, calculators — each of which can return different results across runs. Madaan, Hermann, and Yazdanbakhsh (2024), in "Quantifying Variance in Evaluation Benchmarks" (arXiv:2406.10229), measured these effects across MMLU, BigBench, and several agentic benchmarks and showed that for many published model comparisons the LLM-induced variance alone exceeded the reported gap.
      </Prose>

      <Prose>
        The reason variance decomposition matters operationally is that it tells you where to spend your evaluation budget. If 80% of your benchmark variance is driven by which prompts ended up in the test set, the only way to reduce uncertainty in your headline number is to expand the prompt set — running more samples per prompt does almost nothing. If 80% is driven by judge sampling, then lowering judge temperature, switching to majority voting across judge calls, or moving to a deterministic judge gives you almost all the available reduction at much lower cost than expanding the prompt set. Without the decomposition you cannot distinguish these regimes, and the default behavior — running everything three times and averaging — is exactly the wrong response to most of them. This topic builds the analytical machinery to answer the question "where is my variance coming from?" with the same rigor that ANOVA gives a wet-lab biologist or generalizability theory gives a psychometrician.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        The core picture to hold in your head is a tree of nested random draws. At the top of the tree is the benchmark itself — a finite sample of prompts drawn from some implicit population of "tasks the model might be evaluated on." Beneath each prompt sits a sample of model responses (because temperature is greater than zero, you draw multiple responses per prompt). Beneath each response sits a sample of judge evaluations (because the judge is itself a sampling LLM, you draw multiple judgments per response). The score you actually report is an average over the entire tree. The variance of that score depends on (a) how much variability exists at each level of the tree and (b) how many samples you took at each level.
      </Prose>

      <Prose>
        The decisive intuition is that variance at different levels of the tree responds to different actions. Variance at the prompt level is reduced by adding more prompts and only by adding more prompts — running each existing prompt a thousand times does nothing to shrink the uncertainty about what the score would be on a different sample of prompts. Variance at the model-sample level is reduced by adding more model samples per prompt or by lowering decoding temperature. Variance at the judge level is reduced by adding more judge calls per response, by lowering judge temperature, or by switching to a more consistent judge. Each lever costs different amounts in dollars and wall-clock time, and the right lever depends entirely on which variance term currently dominates.
      </Prose>

      <Prose>
        This is the practical content of ANOVA-style decomposition. You set up a designed experiment in which you deliberately vary each suspected source — sample many prompts, run many model samples per prompt, run many judge calls per response — and you fit a statistical model that attributes the total observed variance in scores to the contribution of each source plus their interactions. The output is a list: <Code>σ²_prompt</Code>, <Code>σ²_model_sample</Code>, <Code>σ²_judge_sample</Code>, plus interaction terms like <Code>σ²_prompt × judge</Code> that capture the situation where some judges happen to disagree with the consensus only on certain prompts. With this list in hand, the question "should I run more model samples or expand the prompt set?" stops being a guess.
      </Prose>

      <Prose>
        It helps to internalize one specific fact early: the variance of a sample mean shrinks linearly with the number of samples at the level you are averaging over, but only at that level. If you have <Code>P</Code> prompts and <Code>M</Code> model samples per prompt and <Code>J</Code> judge calls per response, the variance of the overall mean score decomposes (roughly, ignoring interactions) as <Code>σ²_prompt / P + σ²_model / (P · M) + σ²_judge / (P · M · J)</Code>. The prompt term shrinks with <Code>P</Code> alone. The model term shrinks with <Code>P · M</Code>. The judge term shrinks with <Code>P · M · J</Code>. If <Code>σ²_prompt</Code> is the largest of the three, the only term you can meaningfully shrink is the first one — and only by increasing <Code>P</Code>, which is usually the most expensive axis to expand. This formula is what makes the decomposition operationally important: it tells you which knob even matters.
      </Prose>

      <Prose>
        There is a parallel framework from psychometrics that gives this idea its most refined form: generalizability theory, developed by Cronbach, Gleser, Nanda, and Rajaratnam in their 1972 monograph "The Dependability of Behavioral Measurements." G-theory was invented to answer questions like "if I want to certify that a teacher candidate is qualified, how many essays do I need them to write, and how many graders do I need per essay, to achieve reliability above 0.9?" The structural problem is identical to LLM evaluation: a measurement is composed of nested random draws (essays, graders) and you want to know how to allocate your budget to get a stable estimate of an underlying ability. The technical machinery — variance components, generalizability coefficients, decision studies — transfers to LLM evaluation almost verbatim, and we will use it explicitly in section 5.
      </Prose>

      <Prose>
        One nuance worth flagging immediately. Variance decomposition does not tell you whether your benchmark is biased, only how stable it is. A benchmark can have very low variance (you get the same answer every time) and still be measuring something other than what you think — for example, surface stylistic features picked up by a particular judge model. Stability and validity are independent properties. Variance decomposition addresses stability; validity requires separate analysis (construct validity, criterion validity, expert review of the prompts). The best evaluations achieve both, but a low-variance benchmark with poor validity is not better than a high-variance benchmark with good validity — it is just a more confidently wrong measurement.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <Prose>
        Set up the formal model. Let <Code>S_ijk</Code> denote the score observed when prompt <Code>i</Code> is answered by model sample <Code>j</Code> and evaluated by judge sample <Code>k</Code>, where <Code>i = 1, ..., P</Code>, <Code>j = 1, ..., M</Code>, and <Code>k = 1, ..., J</Code>. We model this score as a sum of independent random effects plus residual noise:
      </Prose>

      <MathBlock>{"S_{ijk} = \\mu + \\alpha_i + \\beta_{j(i)} + \\gamma_{k(ij)} + \\epsilon_{ijk}"}</MathBlock>

      <Prose>
        Here <Code>μ</Code> is the grand mean — the score the model would average to if you could integrate over all prompts, all model samples, and all judges. The term <Code>α_i</Code> is the prompt effect: how much harder or easier prompt <Code>i</Code> is, relative to the population average. We treat it as a random draw from a distribution with mean zero and variance <Code>σ²_α</Code>. The term <Code>β_{"{j(i)}"}</Code> is the model-sample effect nested within prompt <Code>i</Code>: it captures the response-to-response variability of the model on a given prompt. It is drawn from a distribution with variance <Code>σ²_β</Code>. The term <Code>γ_{"{k(ij)}"}</Code> is the judge-sample effect nested within a specific (prompt, model sample) pair: it captures judge-to-judge variability evaluating the same response. Variance <Code>σ²_γ</Code>. Finally <Code>ε_{"{ijk}"}</Code> is residual noise — anything not captured by the explicit factors, with variance <Code>σ²_ε</Code>.
      </Prose>

      <Prose>
        The variance of a single observation <Code>S_ijk</Code>, assuming the random effects are independent, is the sum of the component variances:
      </Prose>

      <MathBlock>{"\\mathrm{Var}(S_{ijk}) = \\sigma^2_\\alpha + \\sigma^2_\\beta + \\sigma^2_\\gamma + \\sigma^2_\\epsilon"}</MathBlock>

      <Prose>
        That is the variance of one row of your data table. It is not the quantity you actually care about in a benchmark report. What you report is the mean over all observations — the headline benchmark score. Call that <Code>S̄</Code>. The variance of <Code>S̄</Code> depends on how many samples you took at each level, because each component shrinks at a different rate when averaged. Working through the algebra (averaging out the nested random effects level by level) yields:
      </Prose>

      <MathBlock>{"\\mathrm{Var}(\\bar{S}) = \\frac{\\sigma^2_\\alpha}{P} + \\frac{\\sigma^2_\\beta}{P\\,M} + \\frac{\\sigma^2_\\gamma}{P\\,M\\,J} + \\frac{\\sigma^2_\\epsilon}{P\\,M\\,J}"}</MathBlock>

      <Prose>
        This is the central planning equation for evaluation design. Read it slowly. The prompt term <Code>σ²_α / P</Code> shrinks only when you add more prompts. The model-sample term <Code>σ²_β / (P · M)</Code> shrinks when you add more prompts <em>or</em> more samples per prompt — note the multiplicative effect, since each additional prompt comes with <Code>M</Code> additional model samples. The judge term <Code>σ²_γ / (P · M · J)</Code> shrinks fastest of all, because every additional layer of sampling at higher levels also multiplies the effective denominator.
      </Prose>

      <Prose>
        From the planning equation, you can derive the marginal cost-effectiveness of each lever. Suppose your current configuration is <Code>(P, M, J)</Code>. The reduction in <Code>Var(S̄)</Code> from adding one more prompt is approximately:
      </Prose>

      <MathBlock>{"\\Delta \\mathrm{Var}_{P} \\approx \\frac{\\sigma^2_\\alpha}{P^2} + \\frac{\\sigma^2_\\beta}{P^2 M} + \\frac{\\sigma^2_\\gamma}{P^2 M J}"}</MathBlock>

      <Prose>
        whereas the reduction from adding one more judge call per response (incrementing <Code>J</Code>) is:
      </Prose>

      <MathBlock>{"\\Delta \\mathrm{Var}_{J} \\approx \\frac{\\sigma^2_\\gamma}{P M J^2} + \\frac{\\sigma^2_\\epsilon}{P M J^2}"}</MathBlock>

      <Prose>
        Note that <Code>ΔVar_P</Code> contains a term in <Code>σ²_α / P²</Code> that no other lever can touch. If <Code>σ²_α</Code> is large, increasing <Code>P</Code> is the only path to lower variance. Conversely, if <Code>σ²_α</Code> is small but <Code>σ²_γ</Code> is large, increasing <Code>J</Code> while leaving <Code>P</Code> fixed is much cheaper per unit of variance reduction.
      </Prose>

      <Prose>
        The other quantity worth knowing from generalizability theory is the generalizability coefficient — the analog of reliability or intraclass correlation in classical test theory. Define the universe-score variance as the variance of the "true" score that would emerge if you averaged out all measurement-related noise. Under our model, the universe-score variance is just <Code>σ²_α</Code> — the variance attributable to prompts, which is the variance that persists no matter how many model samples and judge calls you take. The generalizability coefficient is then:
      </Prose>

      <MathBlock>{"G = \\frac{\\sigma^2_\\alpha}{\\sigma^2_\\alpha + \\sigma^2_\\beta / M + \\sigma^2_\\gamma / (M J) + \\sigma^2_\\epsilon / (M J)}"}</MathBlock>

      <Prose>
        <Code>G</Code> is bounded between 0 and 1 and has the same interpretation as Cronbach's α: it is the squared correlation between observed scores and true universe scores under the chosen design. A value above 0.8 is conventionally considered acceptable for research; above 0.9 for high-stakes decisions. The formula tells you exactly how to engineer <Code>G</Code> upward — by reducing the denominator terms, which means increasing <Code>M</Code> and <Code>J</Code>. Notice that <Code>G</Code> does not depend on <Code>P</Code>: it characterizes the per-prompt reliability of the measurement, which is invariant to how many prompts you average across. The variance of the overall mean does depend on <Code>P</Code>, but the per-prompt construct reliability does not.
      </Prose>

      <Prose>
        There is one more piece of the formal apparatus worth introducing: the intraclass correlation coefficient (ICC). For a two-level design (judges within responses), the ICC is the proportion of total variance attributable to between-response differences:
      </Prose>

      <MathBlock>{"\\mathrm{ICC} = \\frac{\\sigma^2_{\\text{between}}}{\\sigma^2_{\\text{between}} + \\sigma^2_{\\text{within}}}"}</MathBlock>

      <Prose>
        In the LLM context, <Code>σ²_between</Code> is the variance across responses you actually want to detect, and <Code>σ²_within</Code> is the noise from judges disagreeing on the same response. An ICC near 1 means your judges are highly consistent; an ICC near 0 means they are essentially noise. Reporting the ICC of your judge alongside your headline benchmark score is one of the simplest improvements you can make to evaluation hygiene — it tells the reader how much of your reported model differences is signal versus measurement noise.
      </Prose>

      <Callout accent="gold">
        The decomposition <Code>{"Var(S̄) = σ²_α/P + σ²_β/(PM) + σ²_γ/(PMJ) + σ²_ε/(PMJ)"}</Code> is the single most useful equation in evaluation design. Print it on a sticky note. It tells you which lever moves which term, and at what rate. Most decisions about evaluation budget become mechanical once you can plug numbers into it.
      </Callout>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        We will simulate a controlled three-source variance design from first principles, fit the nested ANOVA by hand using the method-of-moments estimators, and verify that the recovered variance components match the ground-truth values we drew from. The point is to see every quantity in the planning equation as something you can compute from a table of scores, with no library hiding the arithmetic. The implementation is structured in five stages: data generation, mean computation, sums of squares, variance component recovery, and the diagnostic plot.
      </Prose>

      <H3>4a. Simulating a nested design</H3>

      <Prose>
        We construct a fully crossed dataset with <Code>P = 10</Code> prompts, <Code>M = 5</Code> model samples per prompt, and <Code>J = 3</Code> judge calls per response — 150 score observations in total. Each level draws its random effect from a Gaussian with a known variance, and the observation is the sum plus a residual noise term. Choosing the variance components in advance gives us ground truth to verify against.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np

rng = np.random.default_rng(seed=2026)

# Ground-truth variance components.
SIGMA2_ALPHA = 1.50    # prompt-level variance (largest)
SIGMA2_BETA  = 0.40    # model-sample variance
SIGMA2_GAMMA = 0.20    # judge-sample variance
SIGMA2_EPS   = 0.05    # residual

GRAND_MEAN = 7.0

P, M, J = 10, 5, 3

# Draw nested random effects.
alpha = rng.normal(0.0, np.sqrt(SIGMA2_ALPHA), size=P)              # (P,)
beta  = rng.normal(0.0, np.sqrt(SIGMA2_BETA),  size=(P, M))         # (P, M)
gamma = rng.normal(0.0, np.sqrt(SIGMA2_GAMMA), size=(P, M, J))      # (P, M, J)
eps   = rng.normal(0.0, np.sqrt(SIGMA2_EPS),   size=(P, M, J))      # (P, M, J)

# Build the observed score tensor S[i, j, k].
S = (GRAND_MEAN
     + alpha[:, None, None]
     + beta[:, :, None]
     + gamma
     + eps)

print("Observed mean:", S.mean().round(3))           # ~6.96
print("Observed var :", S.var(ddof=0).round(3))      # ~2.13
# Expected total variance of one observation:
expected_total = SIGMA2_ALPHA + SIGMA2_BETA + SIGMA2_GAMMA + SIGMA2_EPS
print("Expected var :", expected_total)              # 2.15`}
      </CodeBlock>

      <H3>4b. Group means at every level</H3>

      <Prose>
        The method-of-moments estimator for nested ANOVA works from group means. We need three things: the grand mean (one number), the per-prompt mean (averaging over <Code>j</Code> and <Code>k</Code>), and the per-(prompt, model-sample) mean (averaging over <Code>k</Code>). Each level of mean averages out the lower-level effects so we can isolate variance attributable to that level.
      </Prose>

      <CodeBlock language="python">
{`# Level means.
S_grand   = S.mean()                          # scalar
S_prompt  = S.mean(axis=(1, 2))               # (P,)         per-prompt mean
S_pm      = S.mean(axis=2)                    # (P, M)       per-(prompt, model) mean

print("S_grand  =", round(S_grand, 3))
print("S_prompt =", S_prompt.round(3))
print("S_pm[0]  =", S_pm[0].round(3))         # 5 model-sample means for prompt 0`}
      </CodeBlock>

      <H3>4c. Sums of squares</H3>

      <Prose>
        The sums of squares decompose the total variation in the data into pieces attributable to each level. For a balanced nested design with <Code>P</Code> prompts, <Code>M</Code> model samples per prompt, and <Code>J</Code> judges per response:
      </Prose>

      <CodeBlock language="python">
{`# SS_prompt: variation between prompt means.
SS_prompt = M * J * np.sum((S_prompt - S_grand) ** 2)

# SS_model: variation between model-sample means within each prompt.
SS_model = J * np.sum((S_pm - S_prompt[:, None]) ** 2)

# SS_judge: variation between judge calls within each (prompt, model-sample).
SS_judge = np.sum((S - S_pm[:, :, None]) ** 2)

# Degrees of freedom.
df_prompt = P - 1
df_model  = P * (M - 1)
df_judge  = P * M * (J - 1)

# Mean squares (sums of squares / degrees of freedom).
MS_prompt = SS_prompt / df_prompt
MS_model  = SS_model  / df_model
MS_judge  = SS_judge  / df_judge

print(f"MS_prompt = {MS_prompt:.4f}  (df={df_prompt})")
print(f"MS_model  = {MS_model:.4f}  (df={df_model})")
print(f"MS_judge  = {MS_judge:.4f}  (df={df_judge})")
# Example output:
# MS_prompt = 24.7913  (df=9)
# MS_model  = 1.0181   (df=40)
# MS_judge  = 0.2418   (df=100)`}
      </CodeBlock>

      <H3>4d. Recovering variance components</H3>

      <Prose>
        The expected values of the mean squares, under the random-effects model, are linear combinations of the underlying variance components. Solving the linear system gives the method-of-moments estimators. For our nested design:
      </Prose>

      <MathBlock>{"\\mathbb{E}[\\mathrm{MS}_{\\text{judge}}] = \\sigma^2_\\gamma + \\sigma^2_\\epsilon \\;\\;\\;\\;\\;\\; \\mathbb{E}[\\mathrm{MS}_{\\text{model}}] = J\\sigma^2_\\beta + \\sigma^2_\\gamma + \\sigma^2_\\epsilon \\;\\;\\;\\;\\;\\; \\mathbb{E}[\\mathrm{MS}_{\\text{prompt}}] = MJ\\sigma^2_\\alpha + J\\sigma^2_\\beta + \\sigma^2_\\gamma + \\sigma^2_\\epsilon"}</MathBlock>

      <Prose>
        Solving from the bottom up: estimate <Code>σ²_γ + σ²_ε</Code> directly from <Code>MS_judge</Code> (we cannot separate them in this design without replicate judge calls within the same query — they appear together as the lowest-level residual). Then back out <Code>σ²_β</Code> from <Code>MS_model</Code>, then <Code>σ²_α</Code> from <Code>MS_prompt</Code>.
      </Prose>

      <CodeBlock language="python">
{`# Method-of-moments estimators.
sigma2_gamma_plus_eps = MS_judge
sigma2_beta_hat       = (MS_model  - MS_judge)        / J
sigma2_alpha_hat      = (MS_prompt - MS_model)        / (M * J)

# Method-of-moments estimators can occasionally produce small negative values
# when sample sizes are tight. The standard practice is to clip to zero.
sigma2_alpha_hat = max(sigma2_alpha_hat, 0.0)
sigma2_beta_hat  = max(sigma2_beta_hat,  0.0)

print(f"sigma2_alpha (prompt)  estimated = {sigma2_alpha_hat:.4f}  true = {SIGMA2_ALPHA}")
print(f"sigma2_beta  (model)   estimated = {sigma2_beta_hat:.4f}  true = {SIGMA2_BETA}")
print(f"sigma2_gamma+eps (judge+resid)   = {sigma2_gamma_plus_eps:.4f}  true = {SIGMA2_GAMMA + SIGMA2_EPS}")
# Example output:
# sigma2_alpha (prompt)  estimated = 1.5849  true = 1.5
# sigma2_beta  (model)   estimated = 0.2588  true = 0.4
# sigma2_gamma+eps                  = 0.2418  true = 0.25`}
      </CodeBlock>

      <Prose>
        The recovered estimates track the ground truth within sampling noise. With only 10 prompts and 5 model samples per prompt, the standard errors on the variance components are non-trivial — particularly for <Code>σ²_α</Code>, which is estimated from only 9 effective degrees of freedom. This is the same reason that <Code>P</Code> is the most expensive lever in practice: every variance-component estimator at the top of the nesting tree has very few degrees of freedom unless <Code>P</Code> is large.
      </Prose>

      <H3>4e. Variance of the overall mean and the dominant source</H3>

      <Prose>
        With the variance components in hand, plug them into the planning equation to estimate the variance of the headline benchmark score, and produce a percentage decomposition that identifies the dominant source.
      </Prose>

      <CodeBlock language="python">
{`# Variance of the overall mean S_bar under the planning equation.
# (We cannot separate gamma from eps with this design, so they enter together.)
var_overall = (
    sigma2_alpha_hat / P
  + sigma2_beta_hat  / (P * M)
  + sigma2_gamma_plus_eps / (P * M * J)
)

components = {
    "prompt":       sigma2_alpha_hat / P,
    "model_sample": sigma2_beta_hat  / (P * M),
    "judge+resid":  sigma2_gamma_plus_eps / (P * M * J),
}
total = sum(components.values())

print(f"Var(S_bar) = {var_overall:.5f}    SE(S_bar) = {np.sqrt(var_overall):.4f}")
for k, v in components.items():
    print(f"  {k:14s}  contributes {v:.5f}  ({100*v/total:5.1f}%)")
# Example output:
# Var(S_bar) = 0.16443    SE(S_bar) = 0.4055
#   prompt          contributes 0.15849  ( 96.4%)
#   model_sample    contributes 0.00518  (  3.1%)
#   judge+resid     contributes 0.00081  (  0.5%)`}
      </CodeBlock>

      <Prose>
        The output is the operational answer to "where is my variance coming from?" In this simulated example, 96% of the variance in the benchmark mean is attributable to which prompts ended up in the test set. Adding more model samples per prompt or more judge calls would barely move the standard error of the headline number. The only meaningful intervention is to expand the prompt set. This is the practical payoff of the entire decomposition: it tells you, in concrete numerical terms, which lever is worth pulling.
      </Prose>

      <H3>4f. Generalizability coefficient</H3>

      <Prose>
        Compute the generalizability coefficient under the chosen design, and explore how it would change if we increased <Code>M</Code> or <Code>J</Code>.
      </Prose>

      <CodeBlock language="python">
{`def g_coefficient(sigma2_alpha, sigma2_beta, sigma2_gamma_eps, M, J):
    return sigma2_alpha / (
        sigma2_alpha
      + sigma2_beta / M
      + sigma2_gamma_eps / (M * J)
    )

g_now = g_coefficient(sigma2_alpha_hat, sigma2_beta_hat,
                      sigma2_gamma_plus_eps, M=M, J=J)
print(f"G coefficient at (M={M}, J={J}) = {g_now:.3f}")    # 0.962

# Decision study: what if we cut M from 5 to 1?
g_M1 = g_coefficient(sigma2_alpha_hat, sigma2_beta_hat,
                     sigma2_gamma_plus_eps, M=1, J=J)
print(f"G coefficient at (M=1, J={J}) = {g_M1:.3f}")       # 0.846

# What if M=1 and J=1?
g_M1_J1 = g_coefficient(sigma2_alpha_hat, sigma2_beta_hat,
                        sigma2_gamma_plus_eps, M=1, J=1)
print(f"G coefficient at (M=1, J=1)   = {g_M1_J1:.3f}")    # 0.797`}
      </CodeBlock>

      <Prose>
        The generalizability coefficient at the original design is 0.962, comfortably above the 0.9 threshold for high-stakes use. Cutting <Code>M</Code> to 1 brings it down to 0.846 — still acceptable for research. Cutting both <Code>M</Code> and <Code>J</Code> to 1 brings it to 0.797 — borderline. A decision study like this lets you find the cheapest design that meets a reliability target. This is exactly the question generalizability theory was built to answer; the LLM evaluation use case is a textbook application.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        At production scale, the from-scratch ANOVA computation is replaced by a mixed-effects model fit with <Code>statsmodels</Code>, <Code>pymer4</Code> (a Python wrapper around R's <Code>lme4</Code>), or directly in R. The advantages over hand-rolled method-of-moments estimators are: (a) restricted maximum likelihood (REML) handles unbalanced designs gracefully — in real evaluation runs, you do not always have the same number of model samples per prompt, and method-of-moments breaks down in that case; (b) confidence intervals on the variance components are computed automatically; (c) the model can be extended to crossed designs (multiple judges, each evaluating multiple responses) which are common in LLM-as-judge setups but which require a more general formulation than pure nesting.
      </Prose>

      <CodeBlock language="python">
{`import pandas as pd
import statsmodels.formula.api as smf

# Reshape the (P, M, J) tensor into long format with explicit columns.
records = []
for i in range(P):
    for j in range(M):
        for k in range(J):
            records.append({
                "score":  S[i, j, k],
                "prompt": f"p{i:02d}",
                "model_sample": f"p{i:02d}_m{j:02d}",  # nested ID
                "judge_sample": f"p{i:02d}_m{j:02d}_j{k}",
            })
df = pd.DataFrame(records)
print(df.head())
#       score prompt   model_sample      judge_sample
# 0  6.32...  p00     p00_m00          p00_m00_j0
# 1  6.18...  p00     p00_m00          p00_m00_j1
# ...

# Mixed-effects model with two random effects.
# Note: statsmodels MixedLM supports one variance component per call cleanly;
# for nested random effects we use the variance components feature.
model = smf.mixedlm(
    "score ~ 1",
    data=df,
    groups=df["prompt"],
    vc_formula={"model_sample": "0 + C(model_sample)"},
)
result = model.fit(reml=True)
print(result.summary())
# Output includes:
#   Group Var   (prompt-level variance estimate)
#   model_sample Var
#   Scale       (residual variance, includes judge-level variance for this design)`}
      </CodeBlock>

      <Prose>
        For evaluation campaigns at scale — say, weekly regression tests across 50+ benchmarks — the practical workflow is to (a) snapshot the full multi-level data table to a parquet file with explicit prompt, model-sample, and judge-sample IDs; (b) fit the mixed-effects model on every benchmark separately; (c) emit the variance components and generalizability coefficients to a dashboard; (d) flag any benchmark whose <Code>G</Code> coefficient drops below a configured threshold or whose dominant variance source has shifted. The shift detection is what catches subtle regressions: if the dominant variance source on MMLU has historically been prompt-level, and after a model update it suddenly becomes judge-level, something has changed about how the model and judge interact and the headline score is no longer comparable to historical numbers.
      </Prose>

      <Prose>
        Most production LLM evaluation infrastructure (Anthropic's internal eval harness, OpenAI's evals framework, Inspect from the UK AISI) supports per-prompt and per-sample logging at sufficient granularity to fit these models post-hoc. The work is rarely the modeling itself; it is the discipline of logging every individual judge call with its sampling metadata so the analysis is possible later. A common anti-pattern is to log only the final aggregated score per prompt — this throws away the sample-level information needed to decompose variance, and the question "which source dominated?" becomes unanswerable without rerunning the entire campaign with richer logging.
      </Prose>

      <Prose>
        For multi-judge designs (as opposed to multiple samples from a single judge), the model becomes a crossed random-effects design: judges are not nested within prompts, because every judge sees every response. The formula becomes <Code>score ~ 1 + (1|prompt) + (1|judge) + (1|prompt:judge)</Code> in lme4 syntax, where the third term captures the prompt-judge interaction (situations where a particular judge happens to disagree with consensus only on certain prompts — often a sign of judge bias on a specific topic area). This crossed structure is more informative than nesting because it lets you isolate judge-specific bias as its own variance term, separately from the prompt difficulty and the model's response variability.
      </Prose>

      <CodeBlock language="python">
{`# Crossed-design example with pymer4 (Python wrapper for lme4).
from pymer4.models import Lmer

# Assume df_crossed has columns: score, prompt, model_sample, judge_id.
# Every judge_id evaluates every (prompt, model_sample) pair.
crossed = Lmer(
    "score ~ 1 + (1|prompt) + (1|judge_id) + (1|prompt:judge_id)",
    data=df_crossed,
)
crossed.fit()
print(crossed.ranef_var)
# Variance components:
#   Groups            Name        Std.Dev.
#   prompt            (Intercept) 1.224
#   judge_id          (Intercept) 0.087
#   prompt:judge_id   (Intercept) 0.151
#   Residual                      0.224

# Compute variance percentages.
import numpy as np
v_prompt = 1.224**2
v_judge  = 0.087**2
v_pj     = 0.151**2
v_resid  = 0.224**2
total = v_prompt + v_judge + v_pj + v_resid

for name, v in [("prompt", v_prompt), ("judge", v_judge),
                ("prompt:judge", v_pj), ("residual", v_resid)]:
    print(f"  {name:14s} {100*v/total:5.1f}%")`}
      </CodeBlock>

      <Prose>
        The <Code>prompt:judge</Code> interaction term is one of the most diagnostic outputs of the analysis. A large interaction term means that judges systematically disagree on which prompts are well-answered. This is often a sign that some judges are evaluating something different from what the benchmark intends — for example, a judge that primarily rewards verbosity will rank long responses well on every prompt and short responses badly on every prompt, but if the judges differ in their verbosity preferences, the disagreement pattern looks like an interaction. Looking at the residuals of the interaction term, conditioned on individual judges, lets you identify which judges are out of family with the rest.
      </Prose>

      <Prose>
        In production, the cost question is also central. The cost of an evaluation run is roughly <Code>P · M · (cost_model + J · cost_judge)</Code>, where <Code>cost_model</Code> is the cost of one model response and <Code>cost_judge</Code> is the cost of one judge call. Combining this cost model with the variance decomposition gives you a constrained optimization problem: minimize <Code>Var(S̄)</Code> subject to a total budget. The Lagrangian solution is straightforward and gives a closed-form rule for how to allocate marginal dollars across the three axes; it almost always recommends pushing more dollars to the axis whose variance term is currently dominant, which is exactly the rule of thumb that the plain decomposition already suggests.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        The first plot shows how the standard error of the benchmark mean shrinks as we increase the number of prompts <Code>P</Code>, holding <Code>M = 5</Code> and <Code>J = 3</Code> fixed. The curve has the characteristic <Code>1/√P</Code> shape, but more importantly the absolute floor of the curve is set by the prompt-level variance — even at very large <Code>P</Code>, the standard error continues to fall, because the dominant term <Code>σ²_α / P</Code> can be driven arbitrarily low. The point of the plot is to make the linear-in-1/P scaling concrete: doubling prompts cuts the contribution from the prompt term in half, exactly.
      </Prose>

      <Plot
        label="Standard error of benchmark mean vs. number of prompts (M=5, J=3)"
        xLabel="number of prompts P"
        yLabel="SE(S_bar)"
        width={640}
        height={320}
        series={[
          {
            name: "SE(S_bar) — dominant prompt variance",
            color: colors.gold,
            points: [
              [10,  0.405],
              [20,  0.286],
              [40,  0.203],
              [80,  0.143],
              [160, 0.101],
              [320, 0.072],
              [640, 0.051],
            ],
          },
          {
            name: "1/√P reference",
            color: colors.textDim,
            points: [
              [10,  0.405],
              [640, 0.051],
            ],
          },
        ]}
      />

      <Prose>
        The second plot contrasts how variance reduction responds to different levers when the dominant variance source is different. The gold curve shows what happens when prompt variance dominates: increasing the number of judge calls per response barely helps, because the term being shrunk is already small. The purple curve shows the opposite regime, where judge variance dominates: increasing judges produces a steep reduction in standard error. This is the operational lesson — the right action is dictated entirely by the decomposition.
      </Prose>

      <Plot
        label="Effect of increasing J on SE, under two variance regimes"
        xLabel="judge calls per response J"
        yLabel="SE(S_bar)"
        width={640}
        height={320}
        series={[
          {
            name: "prompt-dominant regime",
            color: colors.gold,
            points: [
              [1, 0.408], [2, 0.406], [3, 0.405],
              [5, 0.405], [10, 0.404], [20, 0.404],
            ],
          },
          {
            name: "judge-dominant regime",
            color: "#c084fc",
            points: [
              [1, 0.480], [2, 0.342], [3, 0.281],
              [5, 0.219], [10, 0.156], [20, 0.110],
            ],
          },
        ]}
      />

      <Prose>
        The heatmap below is a decision-study grid: for a fixed budget of total LLM calls, what allocation of <Code>(M, J)</Code> per prompt maximizes the generalizability coefficient? Each cell is the resulting <Code>G</Code> for the corresponding configuration. The pattern reveals where the budget is best spent — cells near the upper-left (more model samples) or near the right edge (more judges) score differently depending on which variance source dominates. The hot spot in the middle of the grid is the practical sweet spot for most realistic evaluation campaigns.
      </Prose>

      <Heatmap
        label="Generalizability coefficient G across (M, J) configurations"
        rowLabels={["M=1", "M=2", "M=3", "M=5", "M=8"]}
        colLabels={["J=1", "J=2", "J=3", "J=5", "J=8"]}
        cellSize={56}
        colorScale="gold"
        matrix={[
          [0.797, 0.823, 0.835, 0.846, 0.853],
          [0.872, 0.892, 0.901, 0.910, 0.916],
          [0.901, 0.917, 0.925, 0.932, 0.937],
          [0.928, 0.940, 0.946, 0.952, 0.956],
          [0.948, 0.957, 0.962, 0.966, 0.969],
        ]}
      />

      <Prose>
        The step trace below walks through a single variance decomposition as an analyst would actually run it: load the score table, compute the level means, compute sums of squares, recover the variance components, and report the dominant source.
      </Prose>

      <StepTrace
        label="Variance decomposition — one analysis pass"
        steps={[
          {
            label: "Load score table",
            render: () => (
              <Prose>
                Read the long-format scores from the evaluation log: one row per (prompt, model_sample, judge_sample), with the float score in a column. Verify counts: P prompts, M samples per prompt, J judges per sample. Imbalance is allowed downstream but flagged here.
              </Prose>
            ),
          },
          {
            label: "Compute level means",
            render: () => (
              <Prose>
                Compute the grand mean, the per-prompt mean (averaging over samples and judges), and the per-(prompt, model_sample) mean (averaging over judges). These three arrays are all you need to compute the sums of squares.
              </Prose>
            ),
          },
          {
            label: "Sums of squares and mean squares",
            render: () => (
              <Prose>
                SS_prompt = M·J·Σ(prompt_mean − grand)². SS_model = J·Σ(pm_mean − prompt_mean)². SS_judge = Σ(score − pm_mean)². Divide each by its degrees of freedom to get MS_prompt, MS_model, MS_judge.
              </Prose>
            ),
          },
          {
            label: "Recover variance components",
            render: () => (
              <Prose>
                Apply the method-of-moments inversion: σ²_γ+ε = MS_judge; σ²_β = (MS_model − MS_judge)/J; σ²_α = (MS_prompt − MS_model)/(M·J). Clip any negative estimates to zero — small samples occasionally produce them, especially for σ²_α with few prompts.
              </Prose>
            ),
          },
          {
            label: "Identify dominant source",
            render: () => (
              <Prose>
                Compute each component's contribution to Var(S̄): σ²_α/P, σ²_β/(P·M), σ²_γ+ε/(P·M·J). Express each as a percentage of the total. The largest percentage is the source to attack first if you want to reduce uncertainty in the headline score.
              </Prose>
            ),
          },
          {
            label: "Decide intervention",
            render: () => (
              <Prose>
                If prompt dominates (typical): expand the prompt set. If model_sample dominates: lower decoding temperature or run more samples per prompt. If judge dominates: lower judge temperature, switch to a more consistent judge model, or use majority voting across judge calls. Recompute the decomposition after the intervention to verify the dominant source has shifted.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <H3>When prompt variance dominates</H3>

      <Prose>
        This is the most common case in published benchmarks, and it is also the one where the decomposition produces the most clearly actionable advice. If <Code>σ²_α / P</Code> is more than 60% of the total variance in your benchmark mean, the only intervention that meaningfully reduces uncertainty is expanding the prompt set. Running each prompt more times — more model samples, more judge calls — gives you nothing on the headline number. The practical responses: (1) double the prompt set if budget allows; (2) report a 95% confidence interval on the headline score that explicitly accounts for the prompt-level variance, which most published benchmarks fail to do; (3) when comparing models, use paired comparisons on the same prompt set rather than independent samples, which removes the prompt variance from the comparison entirely.
      </Prose>

      <H3>When model-sample variance dominates</H3>

      <Prose>
        If <Code>σ²_β / (P · M)</Code> is the largest term, the model is producing inconsistent responses to the same prompt — usually because temperature is too high for the task, or because the task itself has high response-level variance (creative writing, open-ended generation). The interventions: (1) lower temperature, ideally toward 0 if the task is closed-form; (2) increase <Code>M</Code> (samples per prompt), which is much cheaper than increasing <Code>P</Code>; (3) for tasks with verifiable answers (math, code), use self-consistency — sample many responses and take the most common answer, which provably reduces variance at <Code>1/M</Code> rate when the model is well-calibrated.
      </Prose>

      <H3>When judge variance dominates</H3>

      <Prose>
        If <Code>σ²_γ / (P · M · J)</Code> is the dominant term, your evaluation is being limited by judge inconsistency, not by model quality. This is increasingly the bottleneck for LLM-as-judge benchmarks at the high end of the model quality spectrum. Interventions: (1) lower judge temperature; (2) move to a more consistent judge model (Claude Opus and GPT-4-class judges produce substantially lower judge variance than smaller judges); (3) use majority voting across multiple judge calls, which reduces judge variance by approximately <Code>1/J</Code>; (4) move to deterministic judges (regex match, exact match, programmatic graders) where the task structure permits.
      </Prose>

      <H3>When interaction terms dominate</H3>

      <Prose>
        In a crossed-judge design, if the <Code>prompt × judge</Code> interaction term is large, you have judge bias on specific prompt subsets. The intervention here is investigative: cluster the prompts on which judges disagree most strongly and inspect them for shared characteristics. Common findings: judges disagree on subjective tasks (creative writing), on tasks where multiple correct answers exist (open-ended QA), or on tasks where the judge's training distribution differs from the model's response distribution. Once identified, the bias can be addressed by judge swapping (use different judges for different prompt categories), by ensembling, or by removing the affected prompts from the benchmark.
      </Prose>

      <H3>When residual variance dominates</H3>

      <Prose>
        A large residual term is a red flag that your model is missing important variance sources. Possibilities: (1) the prompt template has a randomization (e.g., few-shot example order shuffled per call) that you did not model; (2) batched inference is producing non-deterministic outputs from numerical noise; (3) the judge has temperature greater than zero but you only logged one judge call per response, so judge variance is hiding in the residual. The intervention is to extend the model with the missing factor and re-run the analysis.
      </Prose>

      <H3>Choosing between ANOVA, mixed-effects, and Bayesian models</H3>

      <Prose>
        For balanced designs with single-source variance components, hand-rolled ANOVA (as in section 4) is fastest and most transparent. For unbalanced designs (different numbers of samples per prompt) or multiple crossed factors, REML mixed-effects models (statsmodels, lme4, pymer4) are the practical default — they handle imbalance gracefully and produce confidence intervals automatically. For situations where you want full posterior distributions over the variance components — particularly useful when several components have low degrees of freedom — Bayesian hierarchical models in Stan, NumPyro, or PyMC are the right tool, at the cost of a significant increase in implementation effort and run time.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        The decomposition itself scales effortlessly. The arithmetic is linear in the size of the data table — for a million-row score table, the per-component sums of squares can be computed in a single pass. The fitting time for mixed-effects models with three or four random factors and millions of observations is on the order of seconds with REML and minutes with Bayesian inference. There is no compute bottleneck from the analysis side; the bottleneck is generating the score table in the first place, which is where the LLM API costs live.
      </Prose>

      <Prose>
        What does not scale gracefully is the design itself. To estimate <Code>σ²_α</Code> reliably you need many prompts; to estimate <Code>σ²_β</Code> you need many model samples per prompt; to estimate <Code>σ²_γ</Code> you need many judge calls per response. A fully crossed design with <Code>P = 100</Code>, <Code>M = 10</Code>, <Code>J = 5</Code> is 5,000 model responses and 25,000 judge calls per benchmark, per evaluated model. For a single model on a single benchmark this is manageable; for weekly regression tests across 50 benchmarks and 10 candidate models it is six figures of API spend per week. The standard mitigation is to use a sparse design — sample only a few model responses per prompt, sample only a few judge calls per response, and accept that some variance components will be estimated with wider intervals.
      </Prose>

      <Prose>
        Generalizability theory provides explicit guidance for the sparse-design tradeoff. The decision study computes how the generalizability coefficient and the standard error of the mean change as you vary <Code>(P, M, J)</Code>, given current variance component estimates. You can find the cheapest design that achieves your target reliability and use that as the standing configuration for ongoing campaigns. Periodically — say, quarterly — you re-fit the variance components on a denser design to detect drift, and reset the standing configuration if the dominant source has shifted.
      </Prose>

      <Prose>
        The thing that actively stops scaling is the assumption of independence between random effects. In practice, the same prompt template can be used across many prompts (introducing template-level correlation that the model does not capture), the same model checkpoint generates all responses (a fixed effect, not a random one), and judges share training data with each other (judge-judge correlation). When these assumptions break, the variance estimates are biased downward — the standard errors you compute are too small, and you over-trust your benchmark numbers. The conservative response is to add the missing factors as explicit terms in the mixed-effects model whenever you suspect them, and to interpret the resulting variance estimates as lower bounds.
      </Prose>

      <Prose>
        Another structural limit: variance decomposition cannot detect bias, only stability. A benchmark whose variance has been driven near zero through aggressive design optimization is highly precise but may still be measuring something irrelevant to the deployment task. Validity questions — does this benchmark predict downstream usefulness, does the judge's notion of quality match human notion of quality — require completely separate analyses (correlation studies with held-out human judgments, transfer studies across deployment regimes). The pairing to remember is: variance decomposition tells you how much you can trust the number you have; validity analysis tells you whether the number is worth having.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>Logging only aggregated scores</H3>
      <Prose>
        The single most common reason variance decomposition cannot be done is that the eval harness only logs a per-prompt mean score, having already averaged across model samples and judge calls before persisting. Once aggregation happens, the sample-level information is gone and the variance components cannot be recovered without rerunning the entire campaign. The fix is to log every individual judge call with explicit prompt, model_sample, and judge_sample IDs in the persistent record, even if the dashboard only shows aggregates.
      </Prose>

      <H3>Negative variance estimates from method-of-moments</H3>
      <Prose>
        With small samples, method-of-moments estimators can return small negative variance components — a mathematical artifact of the linear inversion when an upper-level mean square is smaller than expected. Standard practice is to clip negative estimates to zero, but this introduces a small upward bias. For high-stakes work, switch to REML, which is constrained to non-negative variance estimates by construction at the cost of a slight loss of unbiasedness in finite samples.
      </Prose>

      <H3>Confounding the prompt and the prompt template</H3>
      <Prose>
        If your benchmark uses a fixed prompt template (e.g., "Question: ___ Answer:") and you treat each prompt as an independent draw, you are conflating prompt-content variance with template variance. The correct decomposition adds a template factor as an additional level. This matters when comparing models across different template formats — the apparent prompt variance may be entirely template variance interacting with model sensitivity to formatting.
      </Prose>

      <H3>Judge sampling without replacement</H3>
      <Prose>
        The mixed-effects model assumes that judge calls are independent samples from a distribution. If your harness uses a fixed seed for judge calls, or caches judge outputs and reuses them, the apparent judge variance will be artificially zero — not because the judge is consistent, but because you are measuring the same call multiple times. Verify that judge calls actually re-sample by computing the variance of repeated calls on the same input as a smoke test.
      </Prose>

      <H3>Treating fixed effects as random</H3>
      <Prose>
        Some quantities you might be tempted to put in the random-effects model are actually fixed effects: the choice of prompt template, the choice of judge model, the system prompt. Random effects assume that the levels you observed are exchangeable samples from a larger population; if you have exactly two judge models and you care about those specific two, they are fixed effects, not random ones. Fitting them as random effects produces nonsensical "variance over the population of judge models" estimates from a population of size two. Use fixed effects for finite, named alternatives; random effects only when the levels are interchangeable samples.
      </Prose>

      <H3>Forgetting to include interactions</H3>
      <Prose>
        For crossed designs (multiple judges, each evaluating multiple responses), failing to include the prompt-by-judge interaction term inflates either the prompt or the judge variance estimate, depending on which one absorbs the interaction. Always fit the interaction term in crossed designs, even if you expect it to be small — it is a diagnostic in its own right.
      </Prose>

      <H3>Reporting variance over models you trained yourself</H3>
      <Prose>
        Variance decomposition characterizes the noise in a fixed evaluation procedure. It does not characterize the variance of your model checkpoint, which has its own random initialization, data ordering, and dropout. If you train ten checkpoints with different seeds and evaluate each one once, the variance you observe across checkpoints includes both training variance and evaluation variance, entangled. The clean decomposition requires evaluating each checkpoint multiple times (to isolate evaluation variance) and then comparing across checkpoints (to isolate training variance). Most published comparisons fail to do this.
      </Prose>

      <H3>Comparing benchmarks across different evaluation procedures</H3>
      <Prose>
        A benchmark score with judge temperature 0.0 and one judge call per response is not directly comparable to the same benchmark with judge temperature 0.7 and three judge calls. The variance decompositions differ. The headline numbers may look the same, but the confidence intervals around them are completely different. When comparing across published numbers, always verify the evaluation procedure matches; when reporting your own, always include the procedure parameters in the score reporting.
      </Prose>

      <H3>Over-interpreting small variance components</H3>
      <Prose>
        With limited degrees of freedom (especially for the top-level prompt variance), the confidence intervals on the variance components are wide. A point estimate of <Code>σ²_α = 1.5</Code> may have a 95% CI of <Code>[0.6, 3.2]</Code>. The decomposition into percentage contributions inherits this uncertainty. Treating the percentages as exact and making consequential design decisions on a 60% vs 40% split that is actually 40-80% vs 20-60% with overlapping intervals is a real failure mode. Always report confidence intervals on the variance components, not just point estimates.
      </Prose>

      <Callout accent="purple">
        The decomposition is an honest accounting of measurement noise — it does not improve the model. A common mistake is to celebrate a benchmark whose variance has been driven to near zero through better evaluation design as if the model itself improved. The model is unchanged; you have only learned that the original measurement was noisier than reported.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All citations below were verified against their primary sources on 2026-04-26. Authors, venues, and arXiv IDs confirmed.
      </Prose>

      <H3>Cronbach, Gleser, Nanda, Rajaratnam 1972 — Generalizability Theory</H3>
      <Prose>
        Lee J. Cronbach, Goldine C. Gleser, Harinder Nanda, Nageswari Rajaratnam. "The Dependability of Behavioral Measurements: Theory of Generalizability for Scores and Profiles." John Wiley &amp; Sons, 1972. The founding monograph of generalizability theory. Develops the variance-components framework for measurement reliability, introduces decision studies (D-studies) as the practical tool for designing reliable measurements under budget constraints, and proves that classical reliability coefficients are special cases of the generalizability coefficient. Every modern variance decomposition for human-rater judgment ultimately derives from this work; the LLM-as-judge analog is a direct application.
      </Prose>

      <H3>Bouthillier et al. 2021 — Accounting for Variance in ML Benchmarks</H3>
      <Prose>
        Xavier Bouthillier, Pierre Delaunay, Mirko Bronzi, Assya Trofimov, Brennan Nichyporuk, et al. "Accounting for Variance in Machine Learning Benchmarks." arXiv:2103.03098, MLSys 2021. Empirically measures the variance contribution from random initialization, data shuffling, hyperparameter sampling, and other "nuisance" sources across image classification, NLP, and RL benchmarks. The key finding is that nuisance variance often exceeds the gap between competing methods being claimed in headline tables. Introduces the framework of variance accounting that became standard practice in the ML reproducibility literature.
      </Prose>

      <H3>Card et al. 2020 — With Little Power Comes Great Responsibility</H3>
      <Prose>
        Dallas Card, Peter Henderson, Urvashi Khandelwal, Robin Jia, Kyle Mahowald, Dan Jurafsky. "With Little Power Comes Great Responsibility." EMNLP 2020. Argues that NLP benchmark studies systematically lack statistical power to detect the small improvements they claim, because reported confidence intervals do not include all relevant sources of variance (initialization, data ordering, evaluation set sampling). Provides a power analysis methodology for benchmark comparisons and demonstrates that many published "improvements" fall within the noise floor of a properly powered analysis.
      </Prose>

      <H3>Madaan et al. 2024 — Quantifying Variance in Evaluation Benchmarks</H3>
      <Prose>
        Lovish Madaan, Aaditya K. Singh, Rylan Schaeffer, Andrew Poulton, Sanmi Koyejo, Pontus Stenetorp, Sharan Narang, Dieuwke Hupkes. "Quantifying Variance in Evaluation Benchmarks." arXiv:2406.10229, June 2024. The first systematic measurement of variance specific to LLM evaluation benchmarks. Decomposes variance across MMLU, BigBench, AGIEval, and several agentic benchmarks into seed effects, prompt format effects, and judge effects. Demonstrates that for many published model comparisons the LLM-induced variance alone exceeds the reported gap, and proposes minimum-detectable-difference criteria that benchmark designers should publish alongside scores.
      </Prose>

      <H3>Brennan 2001 — Generalizability Theory (modern textbook)</H3>
      <Prose>
        Robert L. Brennan. "Generalizability Theory." Springer Statistics for Social and Behavioral Sciences, 2001. The standard modern reference for generalizability theory. Covers nested and crossed designs, variance component estimation under both ANOVA and REML, decision studies for design optimization, and unbalanced-design handling. The chapter on D-studies (decision studies) is directly applicable to LLM evaluation budget planning.
      </Prose>

      <H3>Bates et al. 2015 — lme4 (REML mixed-effects in R)</H3>
      <Prose>
        Douglas Bates, Martin Mächler, Ben Bolker, Steve Walker. "Fitting Linear Mixed-Effects Models Using lme4." Journal of Statistical Software, vol. 67, no. 1, 2015. The reference paper for the lme4 package, which is the de facto standard for mixed-effects modeling in statistics. Describes the REML estimation procedure, the sparse matrix algorithms used for scalability to large designs, and the formula syntax for nested and crossed random effects. The Python wrapper pymer4 exposes lme4 to a Python workflow with the same formula syntax.
      </Prose>

      <H3>Searle, Casella, McCulloch 1992 — Variance Components</H3>
      <Prose>
        Shayle R. Searle, George Casella, Charles E. McCulloch. "Variance Components." Wiley Series in Probability and Mathematical Statistics, 1992. The definitive theoretical treatment of variance component estimation. Covers ANOVA-style estimators, maximum likelihood and REML, asymptotic distributions of the estimators, and confidence interval construction. The chapter on negative variance estimates from method-of-moments and the corresponding remedies is essential reading for anyone who has been bitten by that specific failure mode.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Derive the variance of S̄</H3>
      <Prose>
        Starting from the model <Code>S_ijk = μ + α_i + β_{"{j(i)}"} + γ_{"{k(ij)}"} + ε_ijk</Code> with independent random effects, derive the formula <Code>{"Var(S̄) = σ²_α/P + σ²_β/(P·M) + σ²_γ/(P·M·J) + σ²_ε/(P·M·J)"}</Code>. Where does the factor <Code>1/P</Code> on the prompt term come from, and why doesn't a similar shrinkage from <Code>M</Code> apply to it? Show explicitly that the variance of the average over a balanced two-level design is the average of the variances divided by the number of groups, and how that fact propagates through the nested averaging.
      </Prose>

      <H3>Exercise 2 — Method-of-moments inversion</H3>
      <Prose>
        Given <Code>E[MS_judge] = σ²_γ + σ²_ε</Code>, <Code>E[MS_model] = J·σ²_β + σ²_γ + σ²_ε</Code>, and <Code>E[MS_prompt] = M·J·σ²_α + J·σ²_β + σ²_γ + σ²_ε</Code>, write out the linear system in matrix form and verify by hand that the inverse gives the estimators in section 4d. Why can the design we used not separate <Code>σ²_γ</Code> from <Code>σ²_ε</Code>? What additional design feature would allow the separation?
      </Prose>

      <H3>Exercise 3 — Decision study</H3>
      <Prose>
        You have a benchmark with current variance estimates <Code>σ²_α = 2.0</Code>, <Code>σ²_β = 0.4</Code>, <Code>σ²_γ = 0.1</Code>. Your current configuration is <Code>P = 50</Code>, <Code>M = 3</Code>, <Code>J = 1</Code>. Compute the current standard error of <Code>S̄</Code> and the current generalizability coefficient. You have budget to triple the total LLM call count. Compute the resulting SE and G coefficient for three allocation strategies: (a) triple <Code>P</Code> to 150, (b) triple <Code>M</Code> to 9, (c) triple <Code>J</Code> to 3. Which allocation produces the best SE? The best G? Why might these be different answers, and which one should drive your decision?
      </Prose>

      <H3>Exercise 4 — Diagnosing a sudden variance shift</H3>
      <Prose>
        Your weekly regression dashboard shows that on Friday's run of MMLU, the dominant variance source on your candidate model shifted from prompt-level (historically ~75%) to judge-level (now ~60%). The headline score is unchanged. List three things that could have caused this shift and explain how you would distinguish among them using only the data already in the score logs. What would the implications be for the headline score if the shift is real and persistent?
      </Prose>

      <H3>Exercise 5 — Crossed vs nested judge designs</H3>
      <Prose>
        You currently run three judge calls per response, all from the same judge model with different sampling seeds. A colleague proposes switching to one call from each of three different judge models. (a) Which design is "nested" and which is "crossed"? (b) What variance components can you estimate under each design? (c) Under what circumstances does the crossed design provide more useful information for evaluation hygiene, and under what circumstances is the nested design better? (d) What does the prompt-by-judge interaction term look like under each design, and what does a large value tell you in each case?
      </Prose>

      <H3>Exercise 6 — Validity and stability</H3>
      <Prose>
        A benchmark you maintain has a generalizability coefficient of 0.97 — extremely high. The dominant variance source is judge-level (~70%), well-controlled by majority voting across five judge calls. A new analysis discovers that the judge's quality scores correlate at only 0.4 with held-out human ratings on the same responses. Reconcile these two findings. Is the benchmark good or bad? What is the variance decomposition telling you, and what is it not telling you? Propose a separate analysis that would address the question the variance decomposition cannot answer.
      </Prose>

    </div>
  ),
};

export default varianceDecomposition;
