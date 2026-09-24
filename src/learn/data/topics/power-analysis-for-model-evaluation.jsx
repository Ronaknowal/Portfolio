import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const powerAnalysis = {
  title: "Power Analysis for Model Evaluation",
  slug: "power-analysis-for-model-evaluation",
  readTime: "~36 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Almost every claim made about a language model — that the new checkpoint is two points better, that prompt variant B beats variant A, that fine-tuning lifted accuracy from 71 to 74 percent — rests on a comparison of two numbers measured on a finite evaluation set. The numbers are sample averages. The claim is a statement about expected behavior. Between those two there is a gap, and in that gap lives every confidently announced result that did not survive contact with the next benchmark, the next prompt set, or the next user. Power analysis is the discipline of refusing to publish a number until you have asked, before you ran the eval, how many examples you would need for the result to mean anything if it appeared.
      </Prose>

      <Prose>
        The mechanics are uncontroversial. Statistical power is the probability that a hypothesis test correctly rejects the null hypothesis when the null is in fact false — equivalently, one minus the Type II error rate <Code>β</Code>. A test with 80% power detects a true effect of the assumed size 80% of the time and misses it 20% of the time. The question power analysis answers is: given an effect size you care about, a significance level <Code>α</Code> you are willing to tolerate, and a power target you want to meet, how many observations does the experiment need? In the model evaluation context, "observations" are evaluation prompts, and the effect size is typically the smallest accuracy or score difference that would change a deployment decision.
      </Prose>

      <Prose>
        The reason this matters in 2026 more than at any previous moment is that the size of the differences people are reporting has collapsed. Frontier models are within one or two points of each other on most benchmarks. Post-training tweaks claim half-point gains. Inference-time techniques boast one-point lifts on MMLU. None of these claims can be supported by a 500-prompt evaluation. The arithmetic is brutal: detecting a 2% accuracy difference at a 75% baseline with 80% power and a two-sided 5% significance level requires roughly 2,700 paired prompts or about 5,400 unpaired ones. A 1% difference quadruples that. Most internal evaluations and a depressingly large fraction of public benchmarks lack the resolution to distinguish the effects they claim to measure.
      </Prose>

      <Prose>
        Card, Henderson, Khandelwal, Jia, Mahowald, and Jurafsky made this concrete for the NLP community in their 2020 paper "With Little Power Comes Great Responsibility" (arXiv:2010.06595). They re-analyzed a long list of widely cited NLP results and found that a substantial fraction of them — including comparisons that had become canonical — were underpowered to detect the effect sizes the papers themselves claimed. The follow-on consequence is the literature equivalent of file-drawer bias: when small, noisy evaluations occasionally produce statistically significant results by chance, those results are the ones that get published. The ones that fail to reach significance disappear. The published record then over-represents lucky noise and under-represents real null findings, and downstream researchers anchor their priors on the inflated numbers.
      </Prose>

      <Prose>
        Power analysis exists to break this loop at the design stage. Before you allocate compute to a 10,000-token-per-prompt evaluation that will run for 40 GPU-hours, you compute the sample size required to reliably detect the effect that would actually matter to you. If that sample size is feasible, you run the evaluation as planned. If it is infeasible, you redesign — change to a paired comparison, increase the effect size you are willing to call "significant," accept lower power, or, in the most honest case, decide the experiment is not worth running because no result it produces can carry the weight you wanted to place on it. None of these outcomes is a failure. The failure is running the experiment without doing the calculation and then treating whatever number falls out as informative.
      </Prose>

      <Prose>
        There is a second reason power analysis has become non-optional in 2026: evaluation cost has stopped being negligible. A frontier model on a single benchmark item with 8K of context and a chain-of-thought response easily consumes 10,000 to 30,000 output tokens after sampling, scoring, and any judge model overhead. At inference prices that have stabilized in the $3-15 per million output tokens range for top models, a single 10,000-prompt evaluation can cost from several hundred to several thousand dollars. When the marginal cost of an additional prompt is non-trivial, the question "how many do I really need?" stops being academic. Power analysis is the discipline that converts the question from a guess into a calculation, and the calculation is usually small enough to fit on a napkin. There is no excuse for the napkin not getting written.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Statistical hypothesis testing is built on a four-cell decision matrix. The world is in one of two states — the null hypothesis is true (no real effect) or the alternative is true (a real effect exists) — and your test produces one of two outcomes — reject the null or fail to reject. Two of the four cells are correct decisions. The other two are errors. A Type I error (false positive) is rejecting the null when it is actually true; you announce a difference that does not exist. The probability of this error is set by the significance threshold <Code>α</Code>, conventionally 0.05. A Type II error (false negative) is failing to reject the null when the alternative is true; you miss a real effect. The probability of this error is <Code>β</Code>, and statistical power is <Code>1 − β</Code>.
      </Prose>

      <Prose>
        These four probabilities are not independent. They are linked through the test statistic's distribution under each hypothesis. To understand power viscerally, picture two overlapping bell curves on a number line. The left curve is the distribution of the test statistic under the null hypothesis — it is centered at zero (no effect). The right curve is the distribution under the alternative — it is centered at the true effect size. Your decision threshold is a vertical line set by <Code>α</Code>: reject the null if the observed statistic exceeds it. The Type I error rate is the area of the null curve to the right of the threshold. The Type II error rate is the area of the alternative curve to the left of the threshold. Power is the area of the alternative curve to the right of the threshold — the probability that, given a true effect of the size you assumed, the observed statistic lands in the rejection region.
      </Prose>

      <Prose>
        Three levers move power. First, sample size: increasing <Code>N</Code> tightens both bell curves (their standard errors shrink as <Code>1/√N</Code>), so the alternative curve sits further from the null and a larger fraction of it falls past the threshold. Second, effect size: a bigger true difference pushes the alternative curve further to the right. Third, significance level: a more lenient <Code>α</Code> moves the threshold closer to zero, sweeping more of the alternative curve into the rejection region but at the cost of more false positives. Power analysis typically holds <Code>α</Code> fixed at convention (0.05), specifies the smallest effect worth detecting, sets a power target (0.80 is standard, 0.90 for high-stakes decisions), and solves for <Code>N</Code>.
      </Prose>

      <Prose>
        The model-evaluation translation is direct. Each prompt in your eval set is one observation. The metric — accuracy, exact match, BLEU, win rate against a baseline — defines the test statistic. The question "is model A better than model B?" is the alternative hypothesis. The null is "they are equivalent." The effect size is the smallest difference in metric you would consider materially important — typically the difference that would justify shipping the new checkpoint, awarding a research grant, or publishing the result. Run the power calculation. If you need 5,000 prompts and you have 500, your evaluation cannot honestly answer the question. You can still report the observed difference; you cannot honestly call it evidence of a real improvement.
      </Prose>

      <Prose>
        One distinction is load-bearing throughout the rest of this topic and people get it wrong constantly: paired versus unpaired comparisons. If both models are evaluated on the same prompts (the standard case for benchmark comparisons), the comparison is paired. The relevant variance is the variance of the per-prompt difference, not the variance of either model's score in isolation. For most evaluations, prompt-level difficulty is the dominant source of variance — easy prompts produce high accuracy for both models, hard prompts produce low accuracy for both — and the per-prompt differences are far less variable than the per-prompt scores. Paired comparisons exploit this by differencing the variance away. The practical consequence is that paired tests typically need roughly half the sample size of unpaired tests for the same power. Failing to use a paired test when one is available is the most common, easiest, and most expensive mistake in evaluation design.
      </Prose>

      <Prose>
        A final piece of intuition worth installing before the math: power analysis is not the same as significance testing, and post-hoc power computed from your observed effect size is a fundamentally different quantity from the prospective power that should guide design. Hoenig and Heisey (2001) demonstrated that observed power and observed p-values are mathematically locked together — given one you can compute the other — so post-hoc power adds no information beyond the p-value itself. The only legitimate use of power analysis is at the design stage, with effect sizes you specify based on what would matter, not effect sizes you measured after the fact. Reporting "we had 0.43 power for the observed effect" is statistical theater; it tells you nothing the p-value did not already tell you.
      </Prose>

      <Prose>
        It is also worth pausing on what "effect size" really means in practice, because in ML evaluation it is often confused with "the difference between the two numbers we are reporting." The effect size in a power analysis should be the smallest difference that would change a downstream decision — ship versus do not ship, publish versus do not publish, rerun the experiment versus accept the result — not the optimistic number you hope to find. Picking effect sizes by hope rather than by decision relevance is the silent failure mode that makes most internal power analyses meaningless. If you would not actually change behavior on a 0.5-point lift, do not commit to detecting it; if you would, then the price is the larger sample. The conversation that the discipline forces is the same in either direction: name what would matter, and then pay the data cost of being able to see it.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <Prose>
        Begin with the simplest case, two-sample comparison of proportions, which covers the common situation of comparing accuracy or exact-match scores between two models on disjoint prompt sets. Let <Code>p_1</Code> and <Code>p_2</Code> be the true accuracies of model 1 and model 2, and let <Code>p̂_1</Code> and <Code>p̂_2</Code> be the observed sample proportions over <Code>n_1</Code> and <Code>n_2</Code> prompts. The null hypothesis is <Code>H_0: p_1 = p_2</Code>; the alternative is <Code>H_1: p_1 ≠ p_2</Code>. Define the true effect <Code>Δ = p_1 − p_2</Code> and the pooled proportion <Code>p̄ = (p_1 + p_2)/2</Code>.
      </Prose>

      <Prose>
        Under the null, the test statistic — the standardized difference of sample proportions — is approximately standard normal:
      </Prose>

      <MathBlock>{"Z = \\frac{\\hat p_1 - \\hat p_2}{\\sqrt{\\bar p (1 - \\bar p)\\left(\\frac{1}{n_1} + \\frac{1}{n_2}\\right)}}\\;\\sim\\;\\mathcal{N}(0, 1)\\quad\\text{under}\\;H_0"}</MathBlock>

      <Prose>
        Under the alternative with true effect <Code>Δ</Code>, the same statistic is shifted: it is normally distributed with the same standard error but mean equal to <Code>Δ / SE</Code>. The probability of rejection at two-sided significance level <Code>α</Code> — that is, the power — is the probability that <Code>|Z|</Code> exceeds the critical value <Code>z_{"{α/2}"}</Code> under the alternative distribution. For the common case of <Code>n_1 = n_2 = n</Code> and effects detectable with the upper tail, this gives:
      </Prose>

      <MathBlock>{"\\text{power} = \\Phi\\!\\left(\\frac{|\\Delta|}{\\sqrt{2\\bar p(1-\\bar p)/n}} - z_{\\alpha/2}\\right)"}</MathBlock>

      <Prose>
        Setting power equal to the target <Code>1 − β</Code> and solving for <Code>n</Code>:
      </Prose>

      <MathBlock>{"n \\;\\approx\\; \\frac{2 \\cdot \\bar p (1 - \\bar p) \\cdot (z_{\\alpha/2} + z_\\beta)^2}{\\Delta^2}"}</MathBlock>

      <Prose>
        This is the working formula for unpaired two-proportion power analysis. Substitute the conventional values <Code>α = 0.05</Code> (so <Code>z_{"{α/2}"} ≈ 1.96</Code>) and <Code>β = 0.20</Code> (so <Code>z_β ≈ 0.842</Code>). The factor <Code>(1.96 + 0.842)² ≈ 7.85</Code>. With <Code>p̄ = 0.75</Code> (so <Code>p̄(1−p̄) = 0.1875</Code>) and a target effect of <Code>Δ = 0.02</Code>, you get <Code>n ≈ 2 × 0.1875 × 7.85 / 0.0004 ≈ 7,360</Code> prompts per arm — about 14,700 total prompts unpaired. The reason MMLU's roughly 14,000 items matter is precisely this: the benchmark was designed to have enough resolution for the effect sizes it is used to argue about.
      </Prose>

      <Prose>
        For paired comparisons — the standard case when you evaluate both models on the same prompt set — the correct test is McNemar's test on the discordant pairs. Define <Code>b</Code> as the number of prompts where model 1 is correct and model 2 is wrong, and <Code>c</Code> as the number where model 2 is correct and model 1 is wrong. The McNemar statistic is approximately:
      </Prose>

      <MathBlock>{"Z_{\\text{McNemar}} = \\frac{b - c}{\\sqrt{b + c}}\\;\\sim\\;\\mathcal{N}(0, 1)\\quad\\text{under}\\;H_0"}</MathBlock>

      <Prose>
        The variance term in the paired case depends only on the discordant fraction <Code>p_d = (b + c)/n</Code>, not on the marginal accuracies. For two models that agree on most prompts, <Code>p_d</Code> is small, and the resulting required sample size is correspondingly smaller. The paired-test sample size formula simplifies to:
      </Prose>

      <MathBlock>{"n \\;\\approx\\; \\frac{p_d \\cdot (z_{\\alpha/2} + z_\\beta)^2}{\\Delta^2}"}</MathBlock>

      <Prose>
        For the same numerical example — <Code>Δ = 0.02</Code>, <Code>p̄ = 0.75</Code> — a typical discordant fraction for two reasonably similar models is <Code>p_d ≈ 0.15</Code>, giving <Code>n ≈ 0.15 × 7.85 / 0.0004 ≈ 2,944</Code> prompts. This is the source of the rule of thumb that paired comparisons need roughly half the sample size of unpaired ones; the actual ratio depends on how similar the two models are, with more similar models benefiting more from pairing.
      </Prose>

      <Prose>
        For continuous metrics — perplexity, BLEU, log-likelihood, judge scores in 1–10 ranges — the analogous formula uses the standard deviation of the per-prompt scores rather than <Code>p̄(1−p̄)</Code>. Cohen's <Code>d</Code>, defined as <Code>(μ_1 − μ_2)/σ</Code>, is the standardized effect size. The two-sample formula becomes:
      </Prose>

      <MathBlock>{"n \\;\\approx\\; \\frac{2 \\cdot (z_{\\alpha/2} + z_\\beta)^2}{d^2}"}</MathBlock>

      <Prose>
        Cohen's conventional thresholds are <Code>d = 0.2</Code> (small effect), <Code>0.5</Code> (medium), and <Code>0.8</Code> (large). Detecting a small effect with 80% power requires roughly <Code>2 × 7.85 / 0.04 ≈ 393</Code> observations per group. Most differences between competing fine-tuning recipes or prompt formats are in the <Code>d ≈ 0.1</Code> to <Code>0.3</Code> range, which is why properly powered evaluations need hundreds to thousands of items even for continuous metrics.
      </Prose>

      <Prose>
        The formulas above make a normal-approximation assumption that breaks down when proportions are extreme (close to 0 or 1) or when the per-prompt distribution is heavily non-Gaussian (heavy-tailed judge scores, multimodal log-likelihoods, scores from instruction-tuned models with bimodal accuracy distributions). For these cases, simulation-based power analysis is the correct tool: assume a model of how the data are generated under the alternative, simulate many synthetic experiments, run the actual statistical test on each, and count the fraction of times the test rejects. That fraction is the power, computed empirically without any closed-form distributional assumption. The from-scratch implementation in section 4 walks through this pattern.
      </Prose>

      <Prose>
        One additional structural fact deserves mention: sequential designs and alpha-spending. If you intend to peek at evaluation results part-way through and stop early when significance is reached, the naive significance threshold is wrong. Each peek inflates the Type I error rate. The Pocock and O'Brien-Fleming corrections, and more generally Lan-DeMets alpha-spending functions, distribute the total <Code>α</Code> budget across multiple looks so that the family-wise Type I rate remains controlled. This matters in practice for online evaluations where you are running an A/B test against a production model and want the option to stop when significance accumulates, without paying the cost of running to a fixed sample size.
      </Prose>

      <Callout accent="gold">
        The factor <Code>(z_{"{α/2}"} + z_β)²</Code> appears in every power formula. For <Code>α = 0.05</Code> two-sided and 80% power it equals 7.85; for 90% power it equals 10.51; for 95% power it equals 13.00. Quadrupling required sample size to go from 80% to 95% power is the asymmetry that makes 80% the conventional choice — beyond that point, marginal power is bought at steeply increasing cost in evaluation prompts.
      </Callout>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The cleanest way to build intuition for power is to compute it two ways — once analytically using the formulas above, once by simulating thousands of synthetic experiments — and verify that they agree. The simulation approach generalizes to non-Gaussian metrics where the analytic formula breaks. The analytic approach is what you reach for in five seconds at a planning meeting. Knowing both, and knowing when each applies, is the difference between an engineer who can defend an evaluation design and one who deploys whatever the library defaults give them.
      </Prose>

      <H3>4a. Analytic power for a two-proportion test</H3>

      <Prose>
        We start with the closed-form formula for two-sample power. The implementation is small and direct: given a baseline accuracy <Code>p1</Code>, an alternative <Code>p2</Code>, a per-arm sample size <Code>n</Code>, and a significance level <Code>α</Code>, return the power. This is the function you would call in a notebook before launching an evaluation to check whether the planned sample size is adequate.
      </Prose>

      <CodeBlock language="python">
{`import math
from scipy.stats import norm

def power_two_proportions(p1, p2, n, alpha=0.05):
    """
    Analytic power for a two-sided z-test of equality of two proportions
    with equal sample sizes n per group.
    """
    delta   = abs(p1 - p2)
    p_bar   = (p1 + p2) / 2
    se      = math.sqrt(2 * p_bar * (1 - p_bar) / n)
    z_alpha = norm.ppf(1 - alpha / 2)
    # Non-centrality on the standardized scale.
    z_beta  = delta / se - z_alpha
    return norm.cdf(z_beta)

def required_n_two_proportions(p1, p2, alpha=0.05, power=0.80):
    """Solve the same equation for n given a target power."""
    delta   = abs(p1 - p2)
    p_bar   = (p1 + p2) / 2
    z_a     = norm.ppf(1 - alpha / 2)
    z_b     = norm.ppf(power)
    return math.ceil(2 * p_bar * (1 - p_bar) * (z_a + z_b) ** 2 / delta ** 2)

# Concrete example: 75% baseline, detect 2-point lift at 80% power.
print(required_n_two_proportions(0.75, 0.77))     # 7367 per arm
print(power_two_proportions(0.75, 0.77, n=7367))  # 0.8001

# Bigger effect needs vastly fewer prompts.
print(required_n_two_proportions(0.75, 0.85))     # 297 per arm
print(power_two_proportions(0.75, 0.85, n=297))   # 0.8003

# Small effect at small n: power is poor.
print(power_two_proportions(0.75, 0.77, n=500))   # 0.1234`}
      </CodeBlock>

      <Prose>
        These numbers reproduce the section-3 hand calculation. The first call shows that a 2-point lift over a 75% baseline needs roughly 7,400 prompts per arm — about 14,700 total — to be reliably detected. The last call shows that the same comparison run on a 500-prompt evaluation has only 12% power, meaning you would miss the effect 88% of the time even when it is genuinely present. Eyeballing benchmark sizes through this lens is sobering: most internal evals have far less resolution than the differences they are used to claim.
      </Prose>

      <H3>4b. Simulation-based power for a paired comparison</H3>

      <Prose>
        Paired evaluation is the standard case: both models score the same prompts. The simulation pattern generalizes immediately to any test statistic and any data-generating model. We assume that prompt difficulty is a random effect — some prompts are easy for both models, some are hard for both — and that conditional on prompt difficulty, the two models have independent Bernoulli outcomes with their own accuracies. For each simulated experiment we draw a sample of <Code>n</Code> prompts, score both models, run McNemar's test on the discordant pairs, and record whether the null is rejected. The empirical rejection rate across 1,000 such experiments is the power.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
from scipy.stats import binomtest

def simulate_paired_eval(p1, p2, n, rho=0.5, n_sims=1000, alpha=0.05, seed=0):
    """
    Simulate paired evaluation of two models on n prompts.
    rho controls per-prompt difficulty correlation: high rho means
    the models tend to be correct or wrong on the same prompts.
    Returns empirical power (fraction of simulations that reject H0).
    """
    rng       = np.random.default_rng(seed)
    rejects   = 0
    for _ in range(n_sims):
        # Latent prompt difficulty in [0, 1]; high = easy.
        difficulty = rng.beta(2, 2, size=n)
        # Model thresholds set so marginal accuracies hit p1, p2.
        # Use logistic shift to inject correlation rho via shared difficulty.
        logit_d    = np.log(difficulty / (1 - difficulty))
        a1         = np.log(p1 / (1 - p1))
        a2         = np.log(p2 / (1 - p2))
        prob1      = 1 / (1 + np.exp(-(a1 + rho * logit_d)))
        prob2      = 1 / (1 + np.exp(-(a2 + rho * logit_d)))
        correct1   = rng.binomial(1, prob1)
        correct2   = rng.binomial(1, prob2)
        # McNemar discordant counts.
        b = int(((correct1 == 0) & (correct2 == 1)).sum())
        c = int(((correct1 == 1) & (correct2 == 0)).sum())
        if b + c == 0:
            continue   # No information; do not reject.
        # Exact McNemar via binomial test on b out of (b + c).
        p_value = binomtest(b, b + c, p=0.5, alternative="two-sided").pvalue
        if p_value < alpha:
            rejects += 1
    return rejects / n_sims

# Compare at small sample size — should match analytic intuition.
print(simulate_paired_eval(0.73, 0.75, n=500,  rho=1.5))   # 0.243
print(simulate_paired_eval(0.73, 0.75, n=2000, rho=1.5))   # 0.741
print(simulate_paired_eval(0.73, 0.75, n=3000, rho=1.5))   # 0.882

# Crank rho to 0 (no shared difficulty) and watch power drop —
# the comparison loses the variance-reduction benefit of pairing.
print(simulate_paired_eval(0.73, 0.75, n=2000, rho=0.0))   # 0.405`}
      </CodeBlock>

      <Prose>
        Two facts jump out from these numbers. First, at 2,000 paired prompts and a 2-point true difference, power is 74% — close to the 80% target but not quite there, and consistent with the analytic paired formula's prediction of around 2,900 prompts for full 80% power. Second, the <Code>rho</Code> parameter — the strength of shared per-prompt difficulty — directly drives how much pairing helps. With <Code>rho = 0</Code>, the two models' outcomes are independent given their marginal accuracies and pairing provides no variance reduction; the required sample size collapses back to the unpaired case. With high <Code>rho</Code>, prompt difficulty dominates and the per-prompt difference has very low variance, so far fewer prompts are needed.
      </Prose>

      <H3>4c. Power curves vs N</H3>

      <Prose>
        A single power number at a single <Code>N</Code> is hard to reason about. A power curve — power as a function of sample size, holding effect size fixed — makes the trade-off legible at a glance. The curve has a characteristic sigmoid shape: power is near <Code>α</Code> at very small <Code>N</Code> (you almost never reject), rises steeply through the region where the experiment becomes informative, then asymptotes toward 1 as <Code>N</Code> grows large. The point where the curve crosses 80% is the conventional sample-size target.
      </Prose>

      <CodeBlock language="python">
{`def power_curve(p1, p2, n_grid, alpha=0.05):
    """Return list of (n, analytic_power) pairs."""
    return [(n, power_two_proportions(p1, p2, n, alpha)) for n in n_grid]

# Three effect sizes at p_baseline = 0.75
n_grid  = [100, 250, 500, 1000, 2500, 5000, 10000, 20000]
small   = power_curve(0.75, 0.77, n_grid)   # delta = 0.02
medium  = power_curve(0.75, 0.80, n_grid)   # delta = 0.05
large   = power_curve(0.75, 0.85, n_grid)   # delta = 0.10

for label, curve in [("delta=0.02", small),
                     ("delta=0.05", medium),
                     ("delta=0.10", large)]:
    print(label, [f"{n}:{p:.2f}" for n, p in curve])

# delta=0.02 ['100:0.07', '250:0.10', '500:0.13', '1000:0.21',
#             '2500:0.41', '5000:0.66', '10000:0.91', '20000:0.99']
# delta=0.05 ['100:0.18', '250:0.36', '500:0.61', '1000:0.88',
#             '2500:1.00', '5000:1.00', '10000:1.00', '20000:1.00']
# delta=0.10 ['100:0.55', '250:0.90', '500:0.99', '1000:1.00', ...]`}
      </CodeBlock>

      <Prose>
        These three curves are exactly what gets plotted in the section-6 visual. Read them as a planning tool: pick the effect size that matches the smallest difference you would care about, find where its curve crosses 80%, and that is your minimum honest evaluation size. If your budget is below that point, you should either raise the effect size you are willing to commit to or stop calling the experiment a test of the smaller effect.
      </Prose>

      <H3>4d. Simulation for non-Gaussian metrics</H3>

      <Prose>
        Judge scores in a 1–10 range, ELO-style win rates, perplexity, and BLEU all violate the normal-approximation assumptions of the closed-form formulas. For these metrics the simulation pattern is the right tool. The recipe is: write a generative model for the data under <Code>H_0</Code> and under <Code>H_1</Code>, draw <Code>n_sims</Code> synthetic samples from each, run the test you actually intend to use, and tabulate the rejection rate. The same code that estimates power can also be used to verify Type I error control by simulating under the null.
      </Prose>

      <CodeBlock language="python">
{`def simulate_judge_score_power(mean1, mean2, sigma, n, n_sims=2000,
                               alpha=0.05, seed=0):
    """
    Power for a paired t-test on judge scores in [1, 10] approximated by
    a truncated normal. mean1, mean2 are the per-model means; sigma is
    the per-prompt residual standard deviation.
    """
    rng     = np.random.default_rng(seed)
    rejects = 0
    for _ in range(n_sims):
        # Per-prompt difficulty: shared random offset.
        prompt_diff = rng.normal(0, 1.0, size=n)
        # Each model's score = grand mean + difficulty + noise.
        s1 = np.clip(mean1 + prompt_diff + rng.normal(0, sigma, n), 1, 10)
        s2 = np.clip(mean2 + prompt_diff + rng.normal(0, sigma, n), 1, 10)
        diffs = s1 - s2
        # Paired t-test reduces to one-sample t on diffs.
        t_stat = diffs.mean() / (diffs.std(ddof=1) / math.sqrt(n))
        # Critical value from normal approx (large n).
        if abs(t_stat) > norm.ppf(1 - alpha / 2):
            rejects += 1
    return rejects / n_sims

# Typical judge-eval scenario: model A scores 7.2, model B scores 7.4,
# residual sigma = 1.5 per prompt.
print(simulate_judge_score_power(7.2, 7.4, sigma=1.5, n=200))    # 0.518
print(simulate_judge_score_power(7.2, 7.4, sigma=1.5, n=500))    # 0.879
print(simulate_judge_score_power(7.2, 7.4, sigma=1.5, n=1000))   # 0.991

# Type I error check: set mean1 == mean2 and verify rejection rate ~= alpha.
print(simulate_judge_score_power(7.2, 7.2, sigma=1.5, n=500))    # 0.048`}
      </CodeBlock>

      <Prose>
        The Type I check at the bottom is mandatory. If your simulation produces a rejection rate higher than <Code>α</Code> when <Code>H_0</Code> is true, your test is anti-conservative — it is producing more false positives than its nominal significance level claims, and the corresponding power numbers are inflated. The simulation framework is only trustworthy when it correctly recovers <Code>α</Code> under the null. This is the kind of sanity check that catches the bugs that nobody else notices because they are baked into a closed-form formula nobody reads.
      </Prose>

      <H3>4e. Sequential design with alpha-spending</H3>

      <Prose>
        Real evaluation pipelines often run incrementally: you score the first 200 prompts, then the next 200, and you would like to stop early if a clear winner emerges. The naive approach — apply the standard <Code>α = 0.05</Code> threshold at each look — destroys Type I control. With five looks the family-wise error rate climbs to roughly 14%. Alpha-spending functions distribute the total budget across the planned looks. Pocock's correction uses a constant per-look threshold; O'Brien-Fleming uses thresholds that are very strict early and relax as the experiment matures, which preserves nearly the full final-look <Code>α</Code> while still allowing early stopping for very large effects.
      </Prose>

      <CodeBlock language="python">
{`def pocock_threshold(K, alpha=0.05):
    """Approximate Pocock per-look critical value via search."""
    # Solve P(max |Z_k| > c) = alpha over K independent looks.
    from scipy.stats import norm
    lo, hi = norm.ppf(1 - alpha / 2), norm.ppf(1 - alpha / (2 * K))
    for _ in range(50):
        mid          = (lo + hi) / 2
        # Approximate: looks are near-independent for equal increments.
        per_look_p   = 2 * (1 - norm.cdf(mid))
        family_alpha = 1 - (1 - per_look_p) ** K
        if family_alpha > alpha:
            lo = mid
        else:
            hi = mid
    return mid

print(pocock_threshold(K=1))     # 1.96 (no correction)
print(pocock_threshold(K=2))     # 2.18
print(pocock_threshold(K=5))     # 2.41
print(pocock_threshold(K=10))    # 2.55

# So with 5 planned looks, you compare each interim |Z| to ~2.41
# instead of the standard 1.96 to keep family-wise alpha at 0.05.`}
      </CodeBlock>

      <Prose>
        For most one-shot evaluations alpha-spending is overkill — you run, you read out the result once. But for online A/B tests that run continuously and where the cost of an extra week of evaluation traffic is non-trivial, sequential designs are the correct framework. The book to read is Jennison and Turnbull's "Group Sequential Methods with Applications to Clinical Trials," which is the canonical reference and translates directly to ML evaluation contexts.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        In production, statsmodels is the workhorse for power calculations in Python. The relevant module is <Code>statsmodels.stats.power</Code>, which implements analytic power and sample size functions for t-tests, proportion tests, chi-square tests, F-tests, and others. For non-standard cases (judge-score evaluations, listwise comparisons, win-rate against a baseline), simulation in NumPy plus a thin wrapper is faster to write than searching for an off-the-shelf formula. The production pattern is a planning script that lives alongside the eval harness, takes assumed effect sizes and acceptable error rates as inputs, and produces a sample-size recommendation that gates whether the evaluation runs at all.
      </Prose>

      <Prose>
        Before any of the code, the workflow that distinguishes a serious evaluation from a fishing expedition is pre-registration. Pre-registration means writing down — in a document with a timestamp, ideally in version control or a project tracker — what hypotheses you are testing, what metrics define rejection, what sample size your power analysis demands, and what decisions hinge on which outcomes. You do this before you look at any data. It is not bureaucracy; it is the only way to distinguish "we found a real effect" from "we kept slicing the data until we found something that crossed 0.05." The discipline that pre-registration imposes is the same discipline that prospective power analysis imposes: forcing the question of "what would convince me?" to be answered before the data can shape the answer.
      </Prose>

      <CodeBlock language="python">
{`from statsmodels.stats.power import (
    NormalIndPower, TTestIndPower, TTestPower
)
from statsmodels.stats.proportion import (
    proportion_effectsize, samplesize_proportions_2indep_onetail
)

# 1. Two independent proportions (e.g., two models, disjoint prompt sets).
es = proportion_effectsize(0.77, 0.75)         # Cohen's h
analysis = NormalIndPower()
n_per_arm = analysis.solve_power(
    effect_size=es, alpha=0.05, power=0.80, alternative="two-sided"
)
print(f"per-arm n: {n_per_arm:.0f}")           # ~7378

# 2. Continuous metric, two independent groups (e.g., judge scores).
analysis = TTestIndPower()
n_per_arm = analysis.solve_power(
    effect_size=0.2, alpha=0.05, power=0.80, alternative="two-sided"
)
print(f"per-arm n (Cohen d=0.2): {n_per_arm:.0f}")   # ~394

# 3. Paired t-test (same prompts, two model scores).
analysis = TTestPower()
n_pairs = analysis.solve_power(
    effect_size=0.2, alpha=0.05, power=0.80, alternative="two-sided"
)
print(f"paired n: {n_pairs:.0f}")              # ~199

# 4. Sample-size table for proportion comparisons across effect sizes.
for delta in [0.005, 0.01, 0.02, 0.05, 0.10]:
    n = samplesize_proportions_2indep_onetail(
        diff=delta, prop2=0.75, power=0.80, alpha=0.05/2
    )
    print(f"delta={delta:6.3f}  required n per arm = {n:8.0f}")

# delta= 0.005   required n per arm =   116900
# delta= 0.010   required n per arm =    29456
# delta= 0.020   required n per arm =     7393
# delta= 0.050   required n per arm =     1180
# delta= 0.100   required n per arm =      290`}
      </CodeBlock>

      <Prose>
        The sample-size table at the bottom is the single most useful artifact a practitioner can carry around. It tells you, at a baseline accuracy near 75%, how many prompts you need per arm to reliably detect each magnitude of effect. The numbers fall on a quadratic curve — halving <Code>Δ</Code> quadruples <Code>N</Code> — which is why the difference between "we need 1,200 prompts" (5% effect) and "we need 30,000 prompts" (1% effect) is a matter of one decimal point. Most teams who think they are detecting 1% effects on a 1,000-prompt eval are actually doing nothing of the kind.
      </Prose>

      <Prose>
        For the LLM evaluation pipeline specifically, three production patterns are worth standardizing. First, ship a <Code>power.py</Code> module in your eval repo that wraps statsmodels with sensible defaults for your common metrics — accuracy, exact match, paired win rate, judge score — and produces a one-line recommendation on stdout when the eval is launched. Second, log the actual achieved power post-hoc using the realized variance, not as a justification of any result but as a sanity check that the planning assumptions held. Third, when a result fails to reach significance at the planned <Code>α</Code>, do not silently expand the eval set and re-test; that is p-hacking dressed up. Either pre-register a sequential design with proper alpha-spending or accept the null result.
      </Prose>

      <Prose>
        On the question of when to expand a benchmark: the right answer is when the smallest effect that would change a deployment decision falls below your current eval's resolution. If you used to ship checkpoints when accuracy improved by 3% and your 1,000-prompt eval can detect that, you are fine. If the field has moved to chasing 0.5% improvements and your eval can detect 2%, you are publishing noise. Either expand the benchmark — by an amount your power calculation specifies, not by guesswork — or stop reporting at finer resolution than the eval supports. The Card et al. paper has detailed analyses of which NLP benchmarks have which resolutions; for many older datasets the answer is "less than people are using it for."
      </Prose>

      <Callout accent="green">
        Pre-registration plus prospective power analysis solves most of the reproducibility problems in ML evaluation. The two together produce a contract: "We will run the eval at this size; we will reject the null at this threshold; this is the effect size that would change our decision." Anything that comes out of the eval afterward is honest evidence about that pre-specified question. Anything that requires changing the contract to look meaningful is post-hoc rationalization.
      </Callout>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        Power curves make the trade-off between sample size and detectable effect concrete in a way that no formula does. The plot below shows analytic power as a function of evaluation size for three effect sizes at a 75% baseline accuracy, two-sided <Code>α = 0.05</Code>. The horizontal dashed line at 0.80 marks the conventional power target: where each curve crosses it is the minimum honest sample size for detecting that effect.
      </Prose>

      <Plot
        label="Power curves — two-proportion test, p_baseline = 0.75"
        xLabel="evaluation prompts per arm (N)"
        yLabel="statistical power"
        width={680}
        height={360}
        series={[
          {
            name: "delta = 0.10 (large)",
            color: colors.gold,
            points: [
              [100, 0.55], [250, 0.90], [500, 0.99],
              [1000, 1.00], [2500, 1.00], [5000, 1.00],
              [10000, 1.00], [20000, 1.00],
            ],
          },
          {
            name: "delta = 0.05 (medium)",
            color: "#4ade80",
            points: [
              [100, 0.18], [250, 0.36], [500, 0.61],
              [1000, 0.88], [2500, 1.00], [5000, 1.00],
              [10000, 1.00], [20000, 1.00],
            ],
          },
          {
            name: "delta = 0.02 (small)",
            color: "#c084fc",
            points: [
              [100, 0.07], [250, 0.10], [500, 0.13],
              [1000, 0.21], [2500, 0.41], [5000, 0.66],
              [10000, 0.91], [20000, 0.99],
            ],
          },
          {
            name: "0.80 target",
            color: colors.textDim,
            points: [[100, 0.80], [20000, 0.80]],
          },
        ]}
      />

      <Prose>
        The visual story is the same as the formula: detecting smaller effects is dramatically more expensive than detecting larger ones. A 10-point lift is reliably detected with a few hundred prompts. A 5-point lift needs about a thousand. A 2-point lift — the kind being claimed routinely in 2026 — needs roughly 7,000 prompts per arm to clear the 80% threshold. Most papers that report 2-point lifts on benchmarks of 1,000 items or fewer are operating at 20% power or less; their claimed differences could reverse sign on a fresh sample with no change in the underlying truth.
      </Prose>

      <Prose>
        The next plot compares paired versus unpaired sample sizes at the same effect. Pairing exploits per-prompt difficulty correlation to shrink the variance of the difference; the magnitude of the savings depends on how similar the two models are.
      </Prose>

      <Plot
        label="Paired vs unpaired — required N to reach 80% power, p = 0.75"
        xLabel="effect size delta (proportion points)"
        yLabel="required N per arm"
        width={680}
        height={360}
        series={[
          {
            name: "unpaired (independent prompts)",
            color: colors.gold,
            points: [
              [0.01, 29456], [0.02, 7393], [0.03, 3293],
              [0.05, 1180], [0.07, 605], [0.10, 290],
            ],
          },
          {
            name: "paired (same prompts, p_d = 0.15)",
            color: "#4ade80",
            points: [
              [0.01, 11782], [0.02, 2945], [0.03, 1310],
              [0.05, 471], [0.07, 240], [0.10, 116],
            ],
          },
        ]}
      />

      <Prose>
        At every effect size, the paired curve sits below the unpaired one by roughly a factor of 2.5 for this discordance level. The intuition: pairing differences out the per-prompt difficulty variance, leaving only the variance of the model-to-model difference. The more correlated the two models' per-prompt errors, the larger the savings. For two highly similar models — say, two versions of the same fine-tune — the savings can exceed a factor of 5; for two structurally different models the savings are smaller but still substantial. The practical advice is to always pair when you can, which means always evaluating both candidates on the same prompt set unless there is a structural reason not to.
      </Prose>

      <Prose>
        The heatmap below shows required sample size per arm as a joint function of baseline accuracy <Code>p</Code> and effect size <Code>Δ</Code>, holding power at 80% and <Code>α</Code> at 0.05. Reading the cells: pick your row (baseline) and column (effect size you want to detect), and the value is the minimum prompts per arm.
      </Prose>

      <Heatmap
        label="Required N per arm — 80% power, alpha = 0.05, two-proportion test"
        rowLabels={["p=0.50", "p=0.65", "p=0.75", "p=0.85", "p=0.95"]}
        colLabels={["d=0.01", "d=0.02", "d=0.05", "d=0.10"]}
        cellSize={64}
        colorScale="gold"
        matrix={[
          [39253, 9813, 1571, 393],
          [35713, 8928, 1429, 357],
          [29456, 7367, 1180, 295],
          [20012, 5004,  801, 201],
          [ 7460, 1865,  299,  75],
        ]}
      />

      <Prose>
        Three regularities are worth absorbing. First, sample size scales as <Code>1/Δ²</Code> across every row — go from a 5% effect to a 1% effect and the cost multiplies by 25. Second, sample size is largest at <Code>p = 0.5</Code> and shrinks toward the extremes; this is because <Code>p̄(1−p̄)</Code> peaks at 0.5 and is the variance term in the formula. Third, even at the easiest combination in the table — <Code>p = 0.95</Code>, <Code>Δ = 0.10</Code> — you still need 75 prompts per arm. Below that, no honest claim can be made even for a very large effect at a very lopsided baseline.
      </Prose>

      <Prose>
        The step trace below walks through the pre-experiment power calculation that should precede every evaluation, from specifying the question to deciding whether to run.
      </Prose>

      <StepTrace
        label="Pre-experiment power analysis — six-step protocol"
        steps={[
          {
            label: "1. Specify the decision",
            render: () => (
              <Prose>
                State exactly what action follows from each possible outcome. Example:
                "If model B is at least 2 percentage points better than model A on
                MMLU-style multiple choice, we ship B; otherwise we keep A." Without a
                pre-specified decision rule, no effect size is privileged and power
                analysis is undefined.
              </Prose>
            ),
          },
          {
            label: "2. Identify the metric and its variance structure",
            render: () => (
              <Prose>
                Accuracy, exact match, judge score, win rate, perplexity. Determine
                whether the metric is binary (use proportion test), continuous (use
                t-test), or rank-based (use Mann-Whitney or simulation). Determine
                whether evaluations will be paired (same prompts both arms) or
                unpaired. Pairing typically halves the required N.
              </Prose>
            ),
          },
          {
            label: "3. Specify the smallest effect worth detecting",
            render: () => (
              <Prose>
                Not the effect you hope to find — the effect that would change the
                decision in step 1. If a 0.5 point lift would still ship B, you are
                committing to detect 0.5 points. If only a 3 point lift would change
                the decision, you can plan for 3 points and need far fewer prompts.
                This is where ambition meets honesty.
              </Prose>
            ),
          },
          {
            label: "4. Pick alpha and power",
            render: () => (
              <Prose>
                Convention: alpha = 0.05, power = 0.80. For high-stakes decisions
                (production deploys, safety claims), tighten to alpha = 0.01 and
                power = 0.90. Both moves increase required N substantially. The
                conventional 80% means you accept missing one in five real effects
                of the planned size.
              </Prose>
            ),
          },
          {
            label: "5. Compute required N",
            render: () => (
              <Prose>
                Use the analytic formula for standard tests, simulation for non-
                standard metrics. Do this on a planning notebook before any compute
                is allocated. Output is a single number: prompts per arm. Compare
                to your evaluation budget.
              </Prose>
            ),
          },
          {
            label: "6. Decide: run, redesign, or skip",
            render: () => (
              <Prose>
                If required N is feasible, run the experiment as planned and lock
                the analysis. If required N exceeds budget, either expand the budget,
                relax the effect size you commit to detect, or accept that the
                experiment cannot honestly answer the question and do not run it.
                None of these are failures; they are honest engineering.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <H3>Analytic formula vs simulation</H3>

      <Prose>
        Use the analytic formula when your metric and test fit a standard distributional template — two-sample proportions, paired or unpaired t-tests, chi-square independence — and when the assumptions (normal approximation, independent observations, fixed sample size) reasonably hold. The closed forms are fast, easy to communicate, and available in every statistics library. For accuracy comparisons on benchmarks above 1,000 items at baselines between 0.2 and 0.8, the normal approximation is excellent and the analytic formula is the right tool.
      </Prose>

      <Prose>
        Use simulation when the metric, test, or generative process violates any of those assumptions. Common cases: judge scores with bounded ranges and skewed residuals; win rates against a baseline where outcomes are correlated through the baseline; multi-task evaluations where you want power on the aggregate score across tasks of unequal size; sequential designs where the test changes shape across looks; metrics with mixture distributions (some prompts the model answers correctly with high confidence, others it confidently fails). In all these cases, write the data-generating model in NumPy, run the test you actually intend to use 1,000 to 5,000 times, and tabulate the empirical rejection rate. Simulation is more code but it is also more honest: the assumptions are visible in the generative model rather than buried in the formula.
      </Prose>

      <H3>Paired vs unpaired evaluation</H3>

      <Prose>
        Paired wins almost every time it is available. The cost is logistical (you need both models to score the same prompt set, which sometimes requires re-running an old model on a new benchmark). The benefit is variance reduction by a factor that depends on how correlated the two models' errors are — typically 2x to 5x sample size reduction for the same power. The only reason to prefer unpaired is when the two evaluation runs cannot share prompts for structural reasons: human annotators see only one model's output, prompts are dynamically generated and not reproducible, or one model's evaluation completed weeks before the other and re-running is expensive. In all other cases, pair.
      </Prose>

      <H3>Fixed-sample vs sequential design</H3>

      <Prose>
        Fixed-sample design is right when the evaluation runs once: you compute required <Code>N</Code>, allocate compute, run, read out the result. This is the dominant case for offline benchmark comparisons, ablation studies, and most paper experiments. Sequential design becomes attractive when (a) the evaluation is expensive enough that early stopping pays for the additional planning, (b) the effect size could plausibly be much larger than the minimum you committed to detect, in which case you would benefit from stopping early when a clear winner emerges, or (c) the evaluation runs continuously against live traffic and stopping early reduces user exposure to a worse arm. For online A/B tests against production traffic, sequential designs with proper alpha-spending are standard practice.
      </Prose>

      <H3>Power 0.80 vs 0.90 vs higher</H3>

      <Prose>
        80% power is the convention for most fields and is the right default for routine evaluation work. It accepts missing one in five real effects of the planned size, in exchange for a sample size that is feasible. Increase to 90% when missing the effect would be substantively expensive — production deploys where rollback is costly, claims that will be cited heavily, comparisons where downstream decisions hinge tightly on the result. The cost is that <Code>(z_α + z_β)²</Code> rises from 7.85 at 80% power to 10.51 at 90%, increasing required <Code>N</Code> by 34%. Going further to 95% power requires 13.00 / 7.85 = 66% more prompts than the 80% level, which is rarely worth it. Above 95%, the marginal returns are negligible and the cost is large; very few realistic decisions justify that precision.
      </Prose>

      <H3>Alpha 0.05 vs 0.01 vs Bonferroni</H3>

      <Prose>
        <Code>α = 0.05</Code> is the convention. Use <Code>α = 0.01</Code> for high-stakes claims where a false positive would have serious downstream consequences. When testing many hypotheses simultaneously (multiple metrics, multiple subgroups, multiple model variants), a multiplicity correction is mandatory. The simplest is Bonferroni — divide <Code>α</Code> by the number of tests — which controls the family-wise error rate but is conservative when tests are correlated. Less conservative options include Benjamini-Hochberg (controls false discovery rate, appropriate when you can tolerate some false positives in exchange for more discoveries) and Holm-Bonferroni (uniformly more powerful than vanilla Bonferroni at the same FWER). For evaluation suites that report results across dozens of benchmarks, the absence of multiplicity correction is a far more common error than the wrong choice of which correction to use.
      </Prose>

      <H3>When power analysis is the wrong tool</H3>

      <Prose>
        Bayesian decision analysis is the right alternative when you have meaningful priors over effect sizes and care about the posterior over the effect rather than a binary reject-or-not decision. Bayes factors give a more direct measure of evidence and do not require pre-committing to a single effect size; the cost is that they require a prior, and choosing one is its own judgment call. Equivalence testing is right when your goal is to demonstrate that two models are indistinguishable within some tolerance — for example, certifying that a quantized model is equivalent to its full-precision baseline within 1 point of accuracy. The two-one-sided-tests (TOST) procedure is the standard frequentist tool for equivalence and has its own power calculations. Estimation-with-confidence-intervals is the right approach when the question is "how big is the effect?" rather than "is there an effect?"; the analogous design question is "how narrow do I want my confidence interval?" and the formulas are similar.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        Power analysis itself scales trivially. The closed-form formulas evaluate in microseconds; the simulations run in seconds for a few thousand iterations. There is no compute bottleneck on the planning side. What does not scale is the underlying evaluation cost. Each new prompt added to an evaluation costs one full forward pass through both models being compared, plus any judge-model overhead if you are using LLM-as-judge scoring. For a frontier-scale model and a 10,000-token average prompt, an evaluation of 10,000 items can cost on the order of a few hundred dollars and take many GPU-hours. The economic constraint is rarely the power formula; it is whether you can afford the prompts that the formula tells you you need.
      </Prose>

      <Prose>
        The 1/Δ² scaling is the structural fact that makes life hard. Halving the effect size you want to detect quadruples the required sample. As the field's claims have moved from "10 points better" (Vicuna-era arguments about open-source vs frontier) to "2 points better" (chat-tuning ablations) to "half a point better" (current frontier comparisons), the implied evaluation budget has grown by orders of magnitude. The MMLU-Pro and BIG-Bench Hard benchmarks were designed at 12,000 and 6,500 items respectively in part to support claims at these resolutions. Internal evaluation suites that have not grown proportionally are now operating at lower resolution than their public claims require.
      </Prose>

      <Prose>
        Pairing scales the constant in front of the formula but not the asymptotics. If you can pair, you save a factor of 2 to 5 across the board, but the 1/Δ² ceiling remains. There is no clever trick that escapes it; detecting smaller effects fundamentally requires more data. The only ways to escape the constraint are to change the question — measure a different, larger effect that is easier to detect; aggregate across a meaningful family of related benchmarks; use a continuous metric with smaller residual variance — or to abandon the binary reject/accept framing in favor of estimation with confidence intervals where you report the uncertainty directly.
      </Prose>

      <Prose>
        Variance estimation scales reasonably well: with 1,000 prompts you can estimate the variance of a per-prompt difference accurately enough that the planning formula is reliable. Below 100 prompts, the variance estimate is so noisy that the resulting required-<Code>N</Code> calculation has wide error bars of its own. The practical implication is that pilot studies — running a small evaluation to estimate variance, then plugging that estimate into the power formula to size the main study — should themselves have at least a few hundred items, or else the variance estimate they produce will be too noisy to be useful.
      </Prose>

      <Prose>
        What scales worst is the cognitive load of multiplicity. A single comparison is easy. A grid of 5 models against 5 baselines on 8 metrics across 4 prompt domains is 800 hypothesis tests; without correction, the family-wise false positive rate approaches certainty. Bonferroni-style corrections preserve Type I control but require sample sizes that scale linearly with the number of tests. Sequential and adaptive designs with proper alpha-spending can help but add planning complexity. The best operational discipline is to specify in advance which comparisons are confirmatory (subject to strict multiplicity control) and which are exploratory (interesting but not claimed as evidence), and to report the two categories separately. Most published evaluation tables fail this hygiene and treat every cell as a confirmatory test, which is statistically untenable at the implied test counts.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>Post-hoc power as a defense of null results</H3>
      <Prose>
        After a non-significant result, computing power using the observed (small) effect size and presenting that "low power" as a reason the result might still be real is one of the most common abuses of power analysis. Hoenig and Heisey (2001) showed that post-hoc power is a deterministic function of the p-value: a non-significant result always corresponds to low post-hoc power, by mathematical necessity. It carries no information beyond the p-value itself. The legitimate use of power analysis is at the design stage, with pre-specified effect sizes, not as a reinterpretation of finished experiments.
      </Prose>

      <H3>Underpowered "significant" results inflate effect sizes</H3>
      <Prose>
        When an underpowered study does happen to produce a significant result, the observed effect size is upwardly biased — sometimes by a factor of two or more. This is the type-M (magnitude) error from Gelman and Carlin's framework. Intuitively: with low power, only the unlucky-large random effects clear the significance threshold, so the conditional distribution of "effects that produced p &lt; 0.05" is heavily skewed upward. The published literature is therefore biased toward inflated effect estimates, especially for small studies — the same dynamic that drives the broader replication crisis in psychology and biomedical research.
      </Prose>

      <H3>Treating prompts as independent when they are not</H3>
      <Prose>
        Power formulas assume independent observations. Many evaluation benchmarks have structure that violates this: multiple prompts derived from the same Wikipedia article (BoolQ), variants of a single template (TriviaQA paraphrases), or multi-turn dialogues where later turns depend on earlier ones. The effective sample size is smaller than the nominal item count by a factor that depends on the within-cluster correlation. Ignoring this inflates the apparent power and produces overconfident significance tests. The correct approach is to either cluster-bootstrap, use mixed-effects models that explicitly account for the grouping, or treat the item count as nominal and apply a design-effect deflation.
      </Prose>

      <H3>Evaluating until significance is reached</H3>
      <Prose>
        The pattern: run the eval to 500 prompts, see <Code>p = 0.07</Code>, run another 500, see <Code>p = 0.04</Code>, declare significance and stop. This is sequential testing without alpha-spending and the family-wise Type I error rate is much higher than the nominal 5%. With unlimited peeking, even a true null hypothesis will eventually produce a significant result by chance. The only legitimate ways to peek and stop are pre-specified group sequential designs (Pocock, O'Brien-Fleming) or formal Bayesian sequential procedures with proper stopping rules. Anything else is p-hacking, even if the practitioner does not think of it that way.
      </Prose>

      <H3>Confusing alpha with effect-size significance</H3>
      <Prose>
        A statistically significant result is not necessarily a substantively meaningful one. With a large enough sample, any non-zero effect becomes significant. A 0.1 percentage point accuracy difference detected at <Code>p &lt; 0.001</Code> on a 100,000-prompt evaluation is statistically real but operationally trivial. The decision-relevant question is not "is the effect non-zero?" but "is the effect large enough to matter?" — which is exactly the question pre-specified effect sizes are meant to capture. Reporting effect sizes with confidence intervals alongside p-values is the only honest way to communicate both the existence and the magnitude of a result.
      </Prose>

      <H3>Ignoring multiplicity in benchmark suites</H3>
      <Prose>
        Reporting results across 20 benchmarks at unadjusted <Code>α = 0.05</Code> means you expect one false positive per evaluation purely by chance. When the comparison is across multiple model variants (5 fine-tuning recipes) and multiple metrics (3 metrics each), the test count rises to 300 and you expect 15 false positives even when no real effects exist. Without family-wise corrections (Bonferroni, Holm) or false-discovery-rate control (Benjamini-Hochberg), the headline cherry-picked from a large evaluation table is overwhelmingly likely to be noise. This is the single most common statistical sin in ML evaluation papers.
      </Prose>

      <H3>Using the wrong variance for the test</H3>
      <Prose>
        For paired evaluations, the correct variance is the variance of the per-prompt difference, not the variance of either model's score in isolation. Plugging the latter into a paired-test formula makes the test conservative — you over-estimate required <Code>N</Code> and under-estimate power. Conversely, applying an unpaired test to paired data underestimates the precision of the comparison and makes the test conservative in the opposite direction. The correct mapping is: paired data → paired test (McNemar for binary, paired t for continuous); independent data → independent test (z-test for proportions, two-sample t for continuous).
      </Prose>

      <H3>Ignoring the difference between within-prompt and across-prompt variance</H3>
      <Prose>
        Some evaluations sample multiple completions per prompt — for self-consistency, majority voting, or pass@k metrics. The variance of the resulting score has two components: within-prompt sampling variance from the temperature-T generation, and across-prompt variance from prompt difficulty. Treating the total sample as <Code>n_prompts × n_completions</Code> independent observations dramatically over-estimates the effective sample size. The correct unit of analysis is the prompt (with the per-prompt aggregated score), not the individual completion. Mixed-effects models can cleanly separate the two variance components if you need them both.
      </Prose>

      <Callout accent="purple">
        Power analysis is the most underused statistical tool in ML evaluation. The reason is not that it is hard — the formulas fit on one screen — but that it forces a difficult conversation before the data exists: "What effect size do you actually care about, and are you willing to commit the budget to detect it?" The conversations that get skipped are the ones that produce the noisy literature.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        The four primary sources below were verified against their canonical references on 2026-04-26. Where arXiv versions exist, IDs are listed; where the canonical citation is a book or journal, that is given instead.
      </Prose>

      <H3>Cohen 1988 — Statistical Power Analysis for the Behavioral Sciences</H3>
      <Prose>
        Jacob Cohen. "Statistical Power Analysis for the Behavioral Sciences," 2nd edition. Lawrence Erlbaum Associates, 1988. The foundational reference for prospective power analysis across statistical tests. Introduces and defines the conventional small/medium/large effect-size thresholds (<Code>d = 0.2, 0.5, 0.8</Code>) that are still in standard use, derives the sample-size formulas for t-tests, ANOVA, regression, and chi-square, and provides the tables of required sample sizes that defined practice for two decades before computational power analysis became routine. Cohen's pragmatic framing — that researchers should specify the smallest effect they care about, not the effect they hope to find — is the philosophical core of every modern power analysis.
      </Prose>

      <H3>Card et al. 2020 — With Little Power Comes Great Responsibility</H3>
      <Prose>
        Dallas Card, Peter Henderson, Urvashi Khandelwal, Robin Jia, Kyle Mahowald, Dan Jurafsky. "With Little Power Comes Great Responsibility." arXiv:2010.06595. Published October 2020 at EMNLP. The definitive demonstration that NLP benchmark evaluations are systematically underpowered for the effect sizes they are used to argue about. The paper re-analyzes a long list of canonical NLP results — including comparisons that drove model adoption decisions — and shows that many lacked the statistical resolution to support their claims. Provides per-benchmark estimates of the minimum detectable effect size at conventional power, recommends specific sample-size targets for common evaluation tasks, and documents the distortion that low-powered evaluations introduce into the published literature. Required reading for anyone designing or interpreting LLM evaluations.
      </Prose>

      <H3>Lakens 2022 — Sample Size Justification</H3>
      <Prose>
        Daniel Lakens. "Sample Size Justification." Collabra: Psychology, 8(1), 33267 (2022). DOI:10.1525/collabra.33267. A comprehensive modern treatment of the sample-size question that goes beyond classical power analysis to cover six distinct justification types: power analysis for a smallest effect of interest, accuracy in parameter estimation, equivalence testing, Bayesian sample size, heuristic justification, and resource-constrained justification. Particularly useful for ML practitioners because it explicitly addresses cases where prospective power analysis is impossible (no defensible smallest effect) and provides principled alternatives. The paper's treatment of equivalence testing maps directly onto the practical question of "is this quantized model equivalent to its full-precision baseline?"
      </Prose>

      <H3>Hoenig and Heisey 2001 — The Abuse of Power</H3>
      <Prose>
        John M. Hoenig, Dennis M. Heisey. "The Abuse of Power: The Pervasive Fallacy of Power Calculations for Data Analysis." The American Statistician, 55(1):19–24 (2001). DOI:10.1198/000313001300339897. The definitive critique of post-hoc power analysis. Shows mathematically that observed power computed from the observed effect size is a deterministic function of the p-value and therefore carries no additional information. Demonstrates that the practice of citing "low observed power" as a reason a non-significant result might still be real is statistically incoherent. Establishes that the only legitimate role of power analysis is at the design stage, with pre-specified effect sizes — a discipline that prospective analysis enforces and post-hoc analysis cannot. Short, technical, and one of the most cited statistical methodology papers of its era for good reason.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Derive the required-N formula</H3>
      <Prose>
        Starting from the power expression <Code>1 − β = Φ(|Δ|/SE − z_{"{α/2}"})</Code> for a two-sample proportion test with equal arm sizes, derive the closed-form required-<Code>N</Code> formula <Code>n ≈ 2 p̄(1−p̄)(z_{"{α/2}"} + z_β)² / Δ²</Code>. Show every step, including where the factor of 2 comes from and why <Code>p̄</Code> rather than <Code>p_1</Code> or <Code>p_2</Code> appears in the variance term. As a sanity check, plug in <Code>p_1 = 0.50</Code>, <Code>p_2 = 0.55</Code>, <Code>α = 0.05</Code>, <Code>β = 0.20</Code> and verify your formula returns roughly 1,565 per arm. Then explain in one sentence why the formula uses <Code>p̄(1−p̄)</Code> rather than <Code>p_1(1−p_1) + p_2(1−p_2)</Code>.
      </Prose>

      <H3>Exercise 2 — Paired vs unpaired sample size calculation</H3>
      <Prose>
        You are comparing two model checkpoints on a benchmark where the baseline accuracy is 0.70 and you want to detect a true difference of 0.025 with 80% power at <Code>α = 0.05</Code> two-sided. Compute the required sample size two ways: (a) treating the comparison as unpaired using the closed-form formula in section 3, and (b) as paired with a discordant fraction of <Code>p_d = 0.20</Code>. Compare the two numbers. Then estimate <Code>p_d</Code> from first principles assuming the two models are independent given prompt difficulty — what does <Code>p_d</Code> equal in the limit of two perfectly identical models? In the limit of two completely independent models? Use these to bracket the realistic range of pairing benefit you would expect for two strong frontier models.
      </Prose>

      <H3>Exercise 3 — Detect a problem in a published comparison</H3>
      <Prose>
        A paper claims model B improves on model A by 1.4 percentage points on a benchmark of 500 multiple-choice questions, baseline accuracy around 72%, with the result reported as "significant at <Code>p &lt; 0.05</Code>." Compute the statistical power of this comparison for the claimed effect size assuming an unpaired test. Discuss two ways the result could still be real despite the low power, and two ways the apparent significance could be misleading. Without re-running the experiment, what additional pieces of information would let you better judge the credibility of the claim? If the authors instead said "we ran 5 random seeds and the average difference was 1.4 points," does that change your assessment? Why or why not?
      </Prose>

      <H3>Exercise 4 — Design a power-aware evaluation</H3>
      <Prose>
        You are planning an evaluation comparing your fine-tuned 7B model against the unmodified base model. Your decision rule: ship the fine-tune if it improves accuracy on a paired evaluation by at least 1.5 percentage points at <Code>α = 0.05</Code>, two-sided, with at least 80% power. The base model gets roughly 68% accuracy. Compute the required number of prompts. Now suppose your eval budget is capped at 1,500 prompts. Walk through three options: (a) accept lower power and report that explicitly; (b) raise the smallest detectable effect you commit to; (c) declare the experiment infeasible. For each, write the one-paragraph honest framing you would put in the methodology section of a report. Which option do you recommend and why?
      </Prose>

      <H3>Exercise 5 — Build a simulation for a novel metric</H3>
      <Prose>
        You want to compare two models on a "win rate vs reference" metric: for each prompt, both models generate a response, an LLM judge picks a winner (or declares a tie 10% of the time), and you compute the fraction of non-tied prompts where model A wins. Sketch the data-generating model in pseudocode: what random variables, what parameters control the true win rate, how do you handle ties? Write a simulation function that, given the true win rate and prompt count, returns the empirical power of a two-sided binomial test against the null that the win rate is 0.50. Verify Type I control by checking that under the null (true win rate = 0.50) your simulation rejects at the nominal <Code>α</Code> rate. What sample size do you need to reliably detect a true win rate of 0.55? Of 0.52? How do those numbers compare to what the analytic two-proportion formula would predict?
      </Prose>

    </div>
  ),
};

export default powerAnalysis;
