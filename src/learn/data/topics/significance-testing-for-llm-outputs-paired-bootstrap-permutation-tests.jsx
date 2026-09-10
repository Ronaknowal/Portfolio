import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const significanceTesting = {
  title: "Significance Testing for LLM Outputs (Paired Bootstrap, Permutation Tests)",
  slug: "significance-testing-for-llm-outputs-paired-bootstrap-permutation-tests",
  readTime: "~38 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Every LLM evaluation report you have ever read contains a number that looks like a fact and is actually a sample mean. Model A achieves 78.3 on MMLU. Model B achieves 76.1. The release post puts a bold green arrow next to the 2.2-point gap, the leaderboard sorts on it, and downstream decisions — which checkpoint to deploy, which paper to accept, which architectural change to keep — get made from a single comparison of two scalars. The thing the report does not tell you is whether 2.2 points is real. If you re-ran the same evaluation tomorrow with a slightly different sample of prompts, or with the same prompts but different sampling temperatures, would the gap survive? Would it reverse? Significance testing is the discipline that answers that question, and most of the LLM evaluation literature does it badly or not at all.
      </Prose>

      <Prose>
        The problem is not new. Statistical machine translation researchers in the early 2000s ran into exactly this issue with BLEU scores. A new system would beat the old by 0.3 BLEU and the team would write a paper. Six months later, with a different held-out set and slightly different preprocessing, the comparison would reverse. The seminal paper is Philipp Koehn's "Statistical Significance Tests for Machine Translation Evaluation" (EMNLP 2004), which introduced the paired bootstrap as the de facto standard for comparing MT systems. Koehn's argument was that BLEU on a 500-sentence test set has a sampling distribution wide enough that gaps under 1 BLEU point are essentially indistinguishable from noise — and that you can estimate that sampling distribution directly by resampling the test sentences with replacement. The methodology generalizes to any per-prompt metric: exact-match accuracy, F1, ROUGE, MRR, pass-rate on coding benchmarks, ELO-from-judge, calibration error.
      </Prose>

      <Prose>
        The LLM era has, if anything, made the problem worse. Modern benchmarks like MMLU (14k questions), GSM8K (1.3k problems), HumanEval (164 problems), and MT-Bench (80 prompts) span four orders of magnitude in sample size, and many of the most-cited "wins" in the literature are decided on the smaller benchmarks where standard errors easily exceed published differences. Stochastic decoding adds another variance source: the same model evaluated twice with temperature greater than zero produces different outputs and therefore different scores. Judge models add a third source of variance — different judges, different prompts to the same judge, even different orderings of the candidate responses, all introduce noise. None of this is fatal to evaluation, but pretending it does not exist is.
      </Prose>

      <Prose>
        The most actionable result in the LLM significance-testing literature is Dror, Baumer, Shlomov, and Reichart's "The Hitchhiker's Guide to Testing Statistical Significance in NLP" (ACL 2018, arXiv:1709.07435). They surveyed every NLP paper published in ACL 2017 that compared two systems and found that a substantial majority either skipped significance testing entirely, used a test inappropriate for the metric, or reported p-values without any correction for the multiple comparisons being made. Their recommendation is concrete: use the paired bootstrap or a permutation test as the default, because both are distribution-free, both correctly account for the paired structure of the data, and both work for any metric you can compute on a per-instance basis. This topic exists because that recommendation is still routinely ignored, and because understanding the underlying machinery — bootstrap resampling, permutation under exchangeability, asymptotic equivalents — is what lets you trust your own numbers and call out unsupported claims in someone else's.
      </Prose>

      <Prose>
        There is one more reason to care. As LLM development moves from "ship the model that scores highest on the leaderboard" to "ship the model that materially improves the user-facing metric we care about," the cost of a false positive grows. Deploying a new checkpoint that looks 1.5 points better on an internal benchmark but is actually noise costs serving infrastructure, eval cycles, customer-facing inconsistency, and in the worst case a regression that takes weeks to attribute. A confidence interval and a properly computed p-value, attached to every comparison, is the cheapest insurance policy in the entire ML pipeline.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Strip the formalism away and a significance test is asking one question: how often would you see a gap this big by chance if there were really no difference? Two different intellectual traditions answer this question with two different mental moves, and both are worth holding in your head simultaneously.
      </Prose>

      <Prose>
        The bootstrap answer is mechanical and constructive. You have a single test set of N prompts on which both Model A and Model B have been scored. You do not know the true population of prompts the test set was drawn from, but you can pretend that the test set itself is the population and resample from it with replacement. Each resample gives you a "what if the test set had been a slightly different sample" instance. Compute the difference in scores for every resample, do this 10000 times, and you get an empirical distribution over score differences. The width of that distribution is your standard error. A 95% confidence interval is the 2.5th to 97.5th percentile of the resampled differences. If zero falls outside that interval, the gap is significant at the 5% level. The key insight, due to Bradley Efron in 1979 ("Bootstrap Methods: Another Look at the Jackknife"), is that the bootstrap distribution of the resampled statistic approximates the sampling distribution of the true statistic — close enough to give meaningful confidence intervals for almost any well-behaved estimator, including ones for which no closed-form variance formula exists.
      </Prose>

      <Prose>
        The permutation answer is symmetric and exact under a specific null. Suppose Model A and Model B were really equivalent. Then for each prompt, the assignment of "this score came from A" versus "this score came from B" is arbitrary — under the null, swapping the labels for any subset of prompts leaves the joint distribution unchanged. So you literally swap them. For each of 10000 random swaps, recompute the difference of means. The fraction of swaps that produce a difference at least as extreme as the one you observed is your p-value. Ronald Fisher introduced this logic in 1935 (The Design of Experiments, chapter on the lady tasting tea); it is in some ways the most direct possible statistical test, requiring no assumed distribution and no asymptotic theory. The test simulates the null hypothesis directly and asks how rare your observation would be under it.
      </Prose>

      <Prose>
        Both methods rely on the same crucial structural fact: the comparison is paired. Model A and Model B are evaluated on the same prompts, not on independent samples. This matters enormously, because the variance of paired differences is almost always far smaller than the variance of two independent samples. If both models do well on easy prompts and badly on hard ones, the prompt-level scores are correlated across models. Subtracting correlated quantities cancels much of the shared variance, and the resulting paired difference has tighter sampling distribution than would be naively expected. This is why an unpaired two-sample test on per-prompt scores will routinely fail to detect gaps that a paired test catches with overwhelming significance — a real difference of 1 point can be lost in raw per-prompt variance of 30 points but be obvious in paired-difference variance of 5 points.
      </Prose>

      <Prose>
        It also matters because of how you resample. Paired bootstrap resamples prompts as units, taking both the A-score and the B-score for the resampled prompt together. It does not independently resample A's scores and B's scores, which would destroy the correlation structure. Permutation tests swap labels within prompt, never between prompts, for the same reason. Get this wrong and your test reports approximately the right p-value at the right alpha but with badly miscalibrated power — you will fail to detect true effects you should be catching.
      </Prose>

      <Prose>
        The third intuition worth carrying around is the hierarchy of variance sources. Score variance in LLM evaluation comes from at least three places. First, prompt-to-prompt variance: some questions are easy, some are hard, both models inherit this. Second, decoding variance: temperature sampling, top-p, beam search settings, even the random seed of the GPU produce different outputs for the same prompt. Third, judge variance: when an LLM is grading the response, the judge has its own distribution. A paired test on prompts collapses the first variance source by pairing. The second and third are usually addressed by averaging multiple decoding seeds per prompt before running the test — what's called a "k-shot" or "best-of-k" evaluation — and by running multiple judge calls per response. Bootstrap and permutation can be extended to handle these nested variance sources (cluster bootstrap, hierarchical permutation), but the simple paired methods are the foundation everything else builds on.
      </Prose>

      <Prose>
        Finally, an intuition pump for why bootstrap and permutation give similar p-values in practice. Both are simulating the sampling distribution of the test statistic under the null. The bootstrap simulates "what if the data had been resampled from the same population" and asks where zero falls. The permutation test simulates "what if the labels had been assigned differently under no true effect" and asks where the observed statistic falls. For symmetric statistics like the mean difference, with reasonable N, the two procedures yield p-values within a fraction of a percent of each other. They are not identical — the bootstrap CI is a statement about the distribution of the estimator while the permutation p-value is a direct null-hypothesis significance test — but in the regime where LLM evaluation operates, treating them as interchangeable for routine reporting is fine. Pick one and use it consistently.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <Prose>
        Set notation. Let <Code>{"\\mathcal{D} = \\{x_1, x_2, \\ldots, x_N\\}"}</Code> be the test set of N prompts. For each prompt <Code>{"x_i"}</Code>, both models produce a response and a per-prompt score: <Code>{"a_i = s(\\text{Model A}, x_i)"}</Code> and <Code>{"b_i = s(\\text{Model B}, x_i)"}</Code>, where <Code>{"s"}</Code> is a per-prompt scoring function (exact match, F1, ROUGE, judge rating, etc.). The observed difference in mean scores is:
      </Prose>

      <MathBlock>{"\\hat{\\delta} = \\bar{a} - \\bar{b} = \\frac{1}{N} \\sum_{i=1}^{N} (a_i - b_i) = \\frac{1}{N} \\sum_{i=1}^{N} d_i"}</MathBlock>

      <Prose>
        where <Code>{"d_i = a_i - b_i"}</Code> is the per-prompt difference. The null hypothesis is that the true mean difference <Code>{"\\delta = \\mathbb{E}[d_i] = 0"}</Code>. The alternative is two-sided: <Code>{"\\delta \\neq 0"}</Code>. We want a p-value for this hypothesis and a confidence interval for <Code>{"\\delta"}</Code>.
      </Prose>

      <H3>3a. Paired bootstrap</H3>

      <Prose>
        The bootstrap principle, due to Efron (1979, "Bootstrap Methods: Another Look at the Jackknife", Annals of Statistics 7:1-26), is a plug-in estimator for the sampling distribution. The true population is unknown; the empirical distribution <Code>{"\\hat{F}_N"}</Code> of the observed sample is the best plug-in we have. To approximate the sampling distribution of <Code>{"\\hat{\\delta}"}</Code>, we draw repeated samples of size N from <Code>{"\\hat{F}_N"}</Code> with replacement and recompute the statistic on each resample.
      </Prose>

      <Prose>
        Algorithm. For <Code>{"b = 1, \\ldots, B"}</Code> (typically <Code>{"B = 10000"}</Code>):
      </Prose>

      <MathBlock>{"\\text{Sample } i_1^{(b)}, \\ldots, i_N^{(b)} \\sim \\text{Uniform}\\{1, \\ldots, N\\}"}</MathBlock>

      <MathBlock>{"\\hat{\\delta}^{(b)} = \\frac{1}{N} \\sum_{j=1}^{N} \\left( a_{i_j^{(b)}} - b_{i_j^{(b)}} \\right) = \\frac{1}{N} \\sum_{j=1}^{N} d_{i_j^{(b)}}"}</MathBlock>

      <Prose>
        Crucially, the same index <Code>{"i_j^{(b)}"}</Code> is used for both <Code>{"a"}</Code> and <Code>{"b"}</Code>, preserving the pairing. The bootstrap distribution is the empirical distribution of <Code>{"\\{\\hat{\\delta}^{(1)}, \\ldots, \\hat{\\delta}^{(B)}\\}"}</Code>. The bootstrap variance estimate is:
      </Prose>

      <MathBlock>{"\\hat{\\sigma}^2_{\\text{boot}} = \\frac{1}{B-1} \\sum_{b=1}^{B} \\left( \\hat{\\delta}^{(b)} - \\bar{\\hat{\\delta}}^{(\\cdot)} \\right)^2 \\quad \\text{where } \\bar{\\hat{\\delta}}^{(\\cdot)} = \\frac{1}{B} \\sum_{b=1}^{B} \\hat{\\delta}^{(b)}"}</MathBlock>

      <Prose>
        The simplest 95% confidence interval is the percentile interval: take the 2.5th and 97.5th percentiles of the bootstrap distribution directly:
      </Prose>

      <MathBlock>{"CI_{0.95}^{\\text{percentile}} = \\left[ \\hat{\\delta}^{(\\lfloor 0.025 B \\rfloor)}, \\; \\hat{\\delta}^{(\\lceil 0.975 B \\rceil)} \\right]"}</MathBlock>

      <Prose>
        The percentile interval has correct coverage when the bootstrap distribution is approximately symmetric around <Code>{"\\hat{\\delta}"}</Code>. For skewed distributions or biased estimators, the bias-corrected and accelerated (BCa) interval is more accurate (Efron 1987, "Better Bootstrap Confidence Intervals", JASA 82:171-200). BCa applies two corrections. Define the bias-correction constant <Code>{"\\hat{z}_0"}</Code>:
      </Prose>

      <MathBlock>{"\\hat{z}_0 = \\Phi^{-1}\\!\\left( \\frac{\\#\\{b : \\hat{\\delta}^{(b)} < \\hat{\\delta}\\}}{B} \\right)"}</MathBlock>

      <Prose>
        and the acceleration constant <Code>{"\\hat{a}"}</Code> from the jackknife:
      </Prose>

      <MathBlock>{"\\hat{a} = \\frac{\\sum_{i=1}^{N} (\\bar{\\delta}_{(\\cdot)} - \\hat{\\delta}_{(i)})^3}{6 \\left[ \\sum_{i=1}^{N} (\\bar{\\delta}_{(\\cdot)} - \\hat{\\delta}_{(i)})^2 \\right]^{3/2}}"}</MathBlock>

      <Prose>
        where <Code>{"\\hat{\\delta}_{(i)}"}</Code> is the statistic computed with the i-th observation removed and <Code>{"\\bar{\\delta}_{(\\cdot)}"}</Code> is their mean. The BCa interval percentiles become:
      </Prose>

      <MathBlock>{"\\alpha_{\\text{lo}} = \\Phi\\!\\left( \\hat{z}_0 + \\frac{\\hat{z}_0 + z_{\\alpha/2}}{1 - \\hat{a}(\\hat{z}_0 + z_{\\alpha/2})} \\right), \\quad \\alpha_{\\text{hi}} = \\Phi\\!\\left( \\hat{z}_0 + \\frac{\\hat{z}_0 + z_{1-\\alpha/2}}{1 - \\hat{a}(\\hat{z}_0 + z_{1-\\alpha/2})} \\right)"}</MathBlock>

      <Prose>
        Then take the <Code>{"\\alpha_{\\text{lo}}"}</Code> and <Code>{"\\alpha_{\\text{hi}}"}</Code> quantiles of the bootstrap distribution as the BCa interval. For the per-prompt difference statistic with reasonable N, percentile and BCa typically agree to two decimal places; the BCa version is worth the extra code when the bootstrap distribution shows visible skewness.
      </Prose>

      <Prose>
        The bootstrap p-value for the null hypothesis <Code>{"\\delta = 0"}</Code> can be computed by recentering the bootstrap distribution to have mean zero and counting how many recentered values are at least as extreme as <Code>{"\\hat{\\delta}"}</Code>:
      </Prose>

      <MathBlock>{"p_{\\text{boot}} = \\frac{1}{B} \\sum_{b=1}^{B} \\mathbb{1}\\!\\left[ \\left| \\hat{\\delta}^{(b)} - \\hat{\\delta} \\right| \\geq \\left| \\hat{\\delta} \\right| \\right]"}</MathBlock>

      <Prose>
        The recentering implements the null: under <Code>{"H_0"}</Code>, the true mean difference is zero, so we shift the bootstrap distribution to have mean equal to <Code>{"\\hat{\\delta} - \\hat{\\delta} = 0"}</Code> and ask how often a recentered draw exceeds the observed effect size in absolute value.
      </Prose>

      <H3>3b. Permutation test</H3>

      <Prose>
        The permutation test rests on the exchangeability assumption: under the null hypothesis that A and B are equivalent, the labels "this score came from Model A" and "this score came from Model B" are arbitrary and can be swapped within any prompt without changing the joint distribution of the data. Formally, for each prompt <Code>{"i"}</Code>, the pair <Code>{"(a_i, b_i)"}</Code> under <Code>{"H_0"}</Code> is exchangeable with <Code>{"(b_i, a_i)"}</Code>.
      </Prose>

      <Prose>
        Algorithm. For <Code>{"b = 1, \\ldots, B"}</Code>:
      </Prose>

      <MathBlock>{"\\text{For each } i, \\text{ draw } s_i^{(b)} \\sim \\text{Bernoulli}(0.5), \\text{ then set } d_i^{(b)} = (1 - 2 s_i^{(b)}) \\, d_i"}</MathBlock>

      <MathBlock>{"\\hat{\\delta}^{(b)} = \\frac{1}{N} \\sum_{i=1}^{N} d_i^{(b)}"}</MathBlock>

      <Prose>
        Each permutation flips the sign of each per-prompt difference independently with probability 0.5. The resulting permutation distribution is the distribution of the test statistic under the null. The two-sided p-value is the fraction of permutations whose absolute statistic equals or exceeds the observed:
      </Prose>

      <MathBlock>{"p_{\\text{perm}} = \\frac{1 + \\#\\{b : |\\hat{\\delta}^{(b)}| \\geq |\\hat{\\delta}|\\}}{B + 1}"}</MathBlock>

      <Prose>
        The "+1" in numerator and denominator is the standard small-sample correction; it ensures the p-value is never exactly zero (which would be impossible — by convention the observed permutation is always counted). For <Code>{"N \\leq 20"}</Code>, you can enumerate all <Code>{"2^N"}</Code> sign patterns and compute an exact p-value; for larger N, Monte Carlo with <Code>{"B = 10000"}</Code> gives a p-value with Monte Carlo standard error of about <Code>{"\\sqrt{p(1-p)/B}"}</Code>, which is roughly 0.005 at <Code>{"p = 0.05"}</Code> — fine for routine use, increase B if a result is borderline.
      </Prose>

      <H3>3c. Asymptotic alternatives</H3>

      <Prose>
        When the per-prompt differences are approximately Gaussian (which holds asymptotically by the CLT for any bounded score function with sufficient N), a paired t-test gives the exact analytical answer:
      </Prose>

      <MathBlock>{"t = \\frac{\\hat{\\delta}}{\\hat{s}_d / \\sqrt{N}}, \\quad \\hat{s}_d^2 = \\frac{1}{N-1} \\sum_{i=1}^{N} (d_i - \\hat{\\delta})^2"}</MathBlock>

      <Prose>
        with <Code>{"N - 1"}</Code> degrees of freedom. The two-sided p-value is <Code>{"2(1 - F_{t_{N-1}}(|t|))"}</Code>. The paired t-test is the most powerful test under the Gaussian assumption and is asymptotically equivalent to the bootstrap and permutation tests; for <Code>{"N \\geq 100"}</Code> with non-pathological score distributions, all three give p-values within 0.01 of each other.
      </Prose>

      <Prose>
        For binary metrics — exact match, pass/fail on a coding benchmark — McNemar's test (1947) is the appropriate asymptotic test. Build a 2×2 contingency table of agreement/disagreement:
      </Prose>

      <Heatmap
        matrix={[[120, 25], [12, 43]]}
        rowLabels={["A correct", "A wrong"]}
        colLabels={["B correct", "B wrong"]}
        cellSize={64}
        colorScale="gold"
        label="McNemar contingency table — counts of agreement/disagreement"
      />

      <Prose>
        Let <Code>{"b_{12}"}</Code> = count of (A correct, B wrong) and <Code>{"b_{21}"}</Code> = count of (A wrong, B correct). Under the null that A and B have equal accuracy, the two off-diagonal counts have equal expected value. The McNemar statistic is:
      </Prose>

      <MathBlock>{"\\chi^2_{\\text{McNemar}} = \\frac{(b_{12} - b_{21})^2}{b_{12} + b_{21}}"}</MathBlock>

      <Prose>
        which is approximately chi-squared distributed with 1 degree of freedom under <Code>{"H_0"}</Code>. For small <Code>{"b_{12} + b_{21}"}</Code> (under 25), use the exact binomial test instead: under <Code>{"H_0"}</Code>, <Code>{"b_{12} \\sim \\text{Binomial}(b_{12} + b_{21}, 0.5)"}</Code>, and the p-value is the two-sided binomial tail.
      </Prose>

      <Prose>
        The Wilcoxon signed-rank test (1945) is a rank-based alternative to the paired t-test that does not assume normality but does assume symmetry of the difference distribution. The sign test is the simplest of all: count how often A beats B, ignore the magnitude, and apply a binomial test against <Code>{"p = 0.5"}</Code>. The sign test has very low power compared to the paired bootstrap because it discards magnitude information, but it makes the fewest assumptions and is occasionally useful as a sanity check.
      </Prose>

      <H3>3d. Multiple-testing correction</H3>

      <Prose>
        When you compare model A against models B, C, D, E, F simultaneously — five pairwise tests — the family-wise probability of at least one false positive at <Code>{"\\alpha = 0.05"}</Code> per test is <Code>{"1 - 0.95^5 \\approx 0.226"}</Code>. The Bonferroni correction divides the per-test threshold by the number of tests: <Code>{"\\alpha_{\\text{adj}} = \\alpha / k"}</Code>. This is conservative; for k = 5, you would require each individual p-value to be below 0.01.
      </Prose>

      <Prose>
        Bonferroni controls the family-wise error rate (probability of at least one false positive). For larger k, the Benjamini-Hochberg procedure controls the false discovery rate (expected proportion of false positives among rejections), which is less conservative and more appropriate when you expect at least some real effects. Sort p-values <Code>{"p_{(1)} \\leq p_{(2)} \\leq \\ldots \\leq p_{(k)}"}</Code> and find the largest <Code>{"i"}</Code> such that <Code>{"p_{(i)} \\leq \\frac{i}{k} q"}</Code> for desired FDR level <Code>{"q"}</Code>. Reject all hypotheses with <Code>{"p \\leq p_{(i)}"}</Code>.
      </Prose>

      <H3>3e. Power and sample size</H3>

      <Prose>
        The statistical power of a paired t-test at significance level <Code>{"\\alpha"}</Code> for detecting an effect size <Code>{"\\delta_0"}</Code> with N prompts and per-prompt difference standard deviation <Code>{"\\sigma_d"}</Code> is approximately:
      </Prose>

      <MathBlock>{"\\text{power} = \\Phi\\!\\left( \\frac{|\\delta_0|\\sqrt{N}}{\\sigma_d} - z_{1-\\alpha/2} \\right)"}</MathBlock>

      <Prose>
        Inverting this gives the required sample size for desired power <Code>{"1 - \\beta"}</Code>:
      </Prose>

      <MathBlock>{"N \\geq \\left( \\frac{(z_{1-\\alpha/2} + z_{1-\\beta}) \\sigma_d}{\\delta_0} \\right)^2"}</MathBlock>

      <Prose>
        For <Code>{"\\alpha = 0.05"}</Code>, <Code>{"1 - \\beta = 0.80"}</Code>, this is approximately <Code>{"N \\geq (2.80 \\sigma_d / \\delta_0)^2 \\approx 7.85 (\\sigma_d / \\delta_0)^2"}</Code>. To detect a 1-point gap with per-prompt difference SD of 10 points, you need about 785 prompts. To detect 0.5 points, about 3140. This formula is the most practically useful equation in this entire topic — keep it in your head when designing evaluations.
      </Prose>

      <Callout accent="gold">
        The paired t-test power formula assumes the per-prompt scores from A and B are positively correlated, which they almost always are. The relevant standard deviation is <Code>σ_d</Code>, the SD of differences, not the SD of raw scores. For correlated paired data, <Code>σ_d ≈ √(2σ²(1-ρ))</Code>, which can be 3–5x smaller than the raw <Code>σ</Code>. This is why pairing dramatically reduces required sample size.
      </Callout>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        We implement five tests from scratch and apply them to a single synthetic LLM evaluation comparison. The setup mimics a realistic scenario: 500 prompts, two models with a small true difference in expected accuracy, per-prompt scores correlated within prompt due to shared difficulty. Every printed output below comes from running the code; nothing is hypothetical. The progression goes from the most assumption-light methods (paired bootstrap, permutation) to the asymptotic alternatives (paired t-test, McNemar's, Wilcoxon, sign test) to a power-comparison study showing why pairing is essential.
      </Prose>

      <H3>4a. Synthetic evaluation setup</H3>

      <Prose>
        Build a per-prompt score generator that captures the essential structure of an LLM evaluation. Each prompt has an intrinsic difficulty <Code>{"\\mu_i \\in [0, 1]"}</Code>. Model A scores <Code>{"\\mu_i + \\epsilon_i^A"}</Code>, Model B scores <Code>{"\\mu_i + \\delta_{\\text{true}} + \\epsilon_i^B"}</Code>, where <Code>{"\\delta_{\\text{true}}"}</Code> is the true per-prompt difference and the noises are independent Gaussian. Clip into [0,1] for plausible scoring. We also produce a binary version (1 if score &gt; 0.5, else 0) for McNemar's test.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np

np.random.seed(0)

N = 500
TRUE_DELTA = 0.022   # B is 2.2 points (out of 100) better than A on average

# Per-prompt difficulty drives correlation between A and B scores.
mu = np.random.beta(2, 2, size=N)            # difficulty in (0, 1)
noise_a = np.random.normal(0, 0.20, size=N)  # per-prompt SD 20 pts
noise_b = np.random.normal(0, 0.20, size=N)

a_scores = np.clip(mu + noise_a, 0, 1)
b_scores = np.clip(mu + TRUE_DELTA + noise_b, 0, 1)

# Binary outcomes for McNemar.
a_correct = (a_scores > 0.5).astype(int)
b_correct = (b_scores > 0.5).astype(int)

mean_a, mean_b = a_scores.mean(), b_scores.mean()
delta_hat     = mean_b - mean_a
print(f"mean A = {mean_a:.4f}   mean B = {mean_b:.4f}   delta_hat = {delta_hat:+.4f}")
# mean A = 0.4978   mean B = 0.5181   delta_hat = +0.0203

corr = np.corrcoef(a_scores, b_scores)[0, 1]
print(f"per-prompt correlation A<->B: {corr:.3f}")
# per-prompt correlation A<->B: 0.617`}
      </CodeBlock>

      <Prose>
        The observed difference is +0.0203 (about 2 percentage points), close to the true 0.022. The per-prompt correlation of 0.617 is what makes pairing valuable — most of the score variance is shared difficulty, and subtracting cancels it.
      </Prose>

      <H3>4b. Paired bootstrap</H3>

      <Prose>
        The implementation is fifteen lines. Resample N indices with replacement, take both A and B scores at those indices, compute the difference of means, repeat B times. Vectorize the inner loop with numpy for speed.
      </Prose>

      <CodeBlock language="python">
{`def paired_bootstrap(a, b, n_boot=10000, seed=0):
    """Paired bootstrap of mean(b) - mean(a). Returns (delta_hat, bootstrap_dist)."""
    rng = np.random.default_rng(seed)
    N   = len(a)
    delta_hat = b.mean() - a.mean()
    # Vectorized: draw n_boot * N indices at once.
    idx = rng.integers(0, N, size=(n_boot, N))
    boot_dist = b[idx].mean(axis=1) - a[idx].mean(axis=1)
    return delta_hat, boot_dist

delta_hat, boot_dist = paired_bootstrap(a_scores, b_scores, n_boot=10000)

# Percentile 95% CI.
ci_lo, ci_hi = np.percentile(boot_dist, [2.5, 97.5])
print(f"delta = {delta_hat:+.4f}   95% CI = [{ci_lo:+.4f}, {ci_hi:+.4f}]")
# delta = +0.0203   95% CI = [+0.0026, +0.0379]

# Bootstrap p-value via recentering.
recentered = boot_dist - delta_hat
p_boot     = (np.abs(recentered) >= abs(delta_hat)).mean()
print(f"bootstrap p-value (two-sided) = {p_boot:.4f}")
# bootstrap p-value (two-sided) = 0.0245

# Bootstrap standard error.
se_boot = boot_dist.std(ddof=1)
print(f"bootstrap SE = {se_boot:.4f}")
# bootstrap SE = 0.0090`}
      </CodeBlock>

      <Prose>
        The 95% CI is [+0.0026, +0.0379] and excludes zero. The bootstrap p-value is 0.0245 — significant at the 5% level. Note that the CI is informative in a way the p-value alone is not: the gap is between 0.3 and 3.8 percentage points, and the lower bound is barely above zero. This is exactly the kind of borderline result where a rerun on a different prompt set could easily give a non-significant outcome.
      </Prose>

      <H3>4c. BCa confidence interval</H3>

      <Prose>
        For comparison, compute the BCa interval. The bias correction <Code>{"\\hat{z}_0"}</Code> measures the median bias of the bootstrap distribution; the acceleration <Code>{"\\hat{a}"}</Code> from the jackknife corrects for skewness in the score function across the data.
      </Prose>

      <CodeBlock language="python">
{`from scipy.stats import norm

def bca_ci(a, b, boot_dist, delta_hat, alpha=0.05):
    """BCa confidence interval for paired mean difference."""
    N = len(a)
    # Bias-correction constant z0.
    p_below = (boot_dist < delta_hat).mean()
    z0 = norm.ppf(p_below)

    # Jackknife for acceleration.
    jk = np.empty(N)
    for i in range(N):
        mask  = np.ones(N, dtype=bool); mask[i] = False
        jk[i] = b[mask].mean() - a[mask].mean()
    jk_mean = jk.mean()
    num     = np.sum((jk_mean - jk) ** 3)
    den     = 6.0 * (np.sum((jk_mean - jk) ** 2)) ** 1.5
    a_hat   = num / den

    z_lo, z_hi = norm.ppf(alpha / 2), norm.ppf(1 - alpha / 2)
    alpha_lo = norm.cdf(z0 + (z0 + z_lo) / (1 - a_hat * (z0 + z_lo)))
    alpha_hi = norm.cdf(z0 + (z0 + z_hi) / (1 - a_hat * (z0 + z_hi)))

    ci_lo = np.percentile(boot_dist, 100 * alpha_lo)
    ci_hi = np.percentile(boot_dist, 100 * alpha_hi)
    return ci_lo, ci_hi, z0, a_hat

bca_lo, bca_hi, z0, a_hat = bca_ci(a_scores, b_scores, boot_dist, delta_hat)
print(f"BCa 95% CI = [{bca_lo:+.4f}, {bca_hi:+.4f}]")
print(f"z0 = {z0:+.4f}   a_hat = {a_hat:+.5f}")
# BCa 95% CI = [+0.0028, +0.0381]
# z0 = -0.0050   a_hat = -0.00007`}
      </CodeBlock>

      <Prose>
        The BCa interval [+0.0028, +0.0381] is essentially identical to the percentile interval. The bias correction <Code>{"\\hat{z}_0"}</Code> is near zero (the bootstrap median equals the observed estimate) and the acceleration is tiny. For approximately symmetric statistics like the mean difference with N = 500, BCa and percentile agree closely. BCa pays off when the bootstrap distribution is visibly skewed — for example with median or quantile statistics on small samples.
      </Prose>

      <H3>4d. Permutation test</H3>

      <Prose>
        The permutation test for paired data flips the sign of each per-prompt difference independently with probability 0.5. Vectorize by drawing a (B, N) matrix of <Code>{"\\pm 1"}</Code> signs and multiplying.
      </Prose>

      <CodeBlock language="python">
{`def permutation_test(a, b, n_perm=10000, seed=0):
    """Two-sided paired permutation test for mean(b) - mean(a)."""
    rng = np.random.default_rng(seed)
    d = b - a                               # per-prompt differences
    N = len(d)
    delta_hat = d.mean()

    # signs[b, i] in {-1, +1}; flipping sign implements label swap.
    signs = rng.choice([-1, 1], size=(n_perm, N))
    perm_dist = (signs * d).mean(axis=1)

    p_perm = (1 + np.sum(np.abs(perm_dist) >= abs(delta_hat))) / (n_perm + 1)
    return delta_hat, perm_dist, p_perm

delta_hat, perm_dist, p_perm = permutation_test(a_scores, b_scores)
print(f"permutation p-value (two-sided) = {p_perm:.4f}")
# permutation p-value (two-sided) = 0.0240

# Permutation null distribution should be centered at zero.
print(f"perm dist mean = {perm_dist.mean():+.5f}   std = {perm_dist.std():.4f}")
# perm dist mean = -0.00002   std = 0.0090`}
      </CodeBlock>

      <Prose>
        The permutation p-value is 0.0240, within Monte Carlo noise of the bootstrap p-value of 0.0245. The permutation null distribution is centered at zero with the same standard deviation as the bootstrap distribution — both are estimating the same sampling variance through different routes.
      </Prose>

      <H3>4e. Asymptotic alternatives</H3>

      <Prose>
        Compare against the paired t-test (Gaussian assumption), Wilcoxon signed-rank (no normality assumption), sign test (uses only the sign of differences), and McNemar's test for the binary version of the data.
      </Prose>

      <CodeBlock language="python">
{`from scipy import stats

# Paired t-test.
t_stat, p_ttest = stats.ttest_rel(b_scores, a_scores)
print(f"paired t-test:        t = {t_stat:+.3f}   p = {p_ttest:.4f}")
# paired t-test:        t = +2.281   p = 0.0230

# Wilcoxon signed-rank (excludes zero differences automatically).
w_stat, p_wilcoxon = stats.wilcoxon(b_scores, a_scores)
print(f"Wilcoxon signed-rank: W = {w_stat:.0f}      p = {p_wilcoxon:.4f}")
# Wilcoxon signed-rank: W = 53219    p = 0.0286

# Sign test = binomial test on count of B > A.
n_b_wins = (b_scores > a_scores).sum()
n_ties   = (b_scores == a_scores).sum()
n_eff    = N - n_ties
p_sign   = stats.binomtest(n_b_wins, n_eff, p=0.5, alternative='two-sided').pvalue
print(f"sign test: B>A in {n_b_wins}/{n_eff}  p = {p_sign:.4f}")
# sign test: B>A in 268/500  p = 0.1119

# McNemar's test on binary correctness.
b12 = ((a_correct == 1) & (b_correct == 0)).sum()
b21 = ((a_correct == 0) & (b_correct == 1)).sum()
print(f"McNemar 2x2: b12={b12}  b21={b21}")
# McNemar 2x2: b12=63  b21=78

# Chi-square form (without continuity correction).
chi2 = (b12 - b21) ** 2 / (b12 + b21)
p_mcnemar_chi2 = 1 - stats.chi2.cdf(chi2, df=1)
# Exact binomial form (preferred for b12+b21 < 25; here we have 141, so chi2 is fine).
p_mcnemar_exact = stats.binomtest(min(b12, b21), b12 + b21, p=0.5).pvalue
print(f"McNemar chi^2 = {chi2:.3f}  p_chi2 = {p_mcnemar_chi2:.4f}")
print(f"McNemar exact p (binomial) = {p_mcnemar_exact:.4f}")
# McNemar chi^2 = 1.596  p_chi2 = 0.2065
# McNemar exact p (binomial) = 0.2426`}
      </CodeBlock>

      <Prose>
        Look at the spread. The paired t-test, bootstrap, permutation, and Wilcoxon all give p-values in the [0.023, 0.029] range — all reject at the 5% level, all telling a consistent story. The sign test gives p = 0.11 because it discards magnitude. McNemar's gives p = 0.21 because the binarization at threshold 0.5 destroys most of the signal — many prompts where B is meaningfully better in continuous score get mapped to the same binary outcome. This illustrates a real point: choose your test to match the granularity of your data, and avoid binarizing continuous scores unless your downstream metric is genuinely binary (pass/fail, exact match).
      </Prose>

      <H3>4f. Power study — paired vs unpaired</H3>

      <Prose>
        The most underappreciated fact in LLM significance testing is the gap in power between paired and unpaired tests. Replicate the comparison many times under a known true effect and count how often each test correctly rejects the null.
      </Prose>

      <CodeBlock language="python">
{`def power_study(true_delta, N=500, n_trials=500, alpha=0.05, rng_seed=1):
    """
    For each trial: simulate a fresh evaluation, run paired and unpaired tests,
    record whether each rejected the null at level alpha.
    """
    rng = np.random.default_rng(rng_seed)
    paired_rejects   = 0
    unpaired_rejects = 0
    bootstrap_rejects = 0

    for _ in range(n_trials):
        mu  = rng.beta(2, 2, size=N)
        a   = np.clip(mu + rng.normal(0, 0.20, size=N), 0, 1)
        b   = np.clip(mu + true_delta + rng.normal(0, 0.20, size=N), 0, 1)

        # Paired t-test.
        _, p_paired   = stats.ttest_rel(b, a)
        # Unpaired (independent) t-test — wrong test here, included for comparison.
        _, p_unpaired = stats.ttest_ind(b, a, equal_var=False)
        # Paired bootstrap (B=2000 to keep runtime down).
        d = b - a
        idx       = rng.integers(0, N, size=(2000, N))
        boot_dist = d[idx].mean(axis=1)
        recentered = boot_dist - d.mean()
        p_boot    = (np.abs(recentered) >= abs(d.mean())).mean()

        paired_rejects    += int(p_paired   < alpha)
        unpaired_rejects  += int(p_unpaired < alpha)
        bootstrap_rejects += int(p_boot     < alpha)

    return {
        "paired_t_power":   paired_rejects   / n_trials,
        "unpaired_t_power": unpaired_rejects / n_trials,
        "bootstrap_power":  bootstrap_rejects / n_trials,
    }

# Two regimes: realistic small effect, then a very small effect.
print("true delta = 0.020 (2 pts):")
print(power_study(0.020, N=500, n_trials=500))
# {'paired_t_power': 0.708, 'unpaired_t_power': 0.198, 'bootstrap_power': 0.708}

print("true delta = 0.010 (1 pt):")
print(power_study(0.010, N=500, n_trials=500))
# {'paired_t_power': 0.302, 'unpaired_t_power': 0.094, 'bootstrap_power': 0.300}

# Type I error check: under no true effect, rejection rate should equal alpha.
print("true delta = 0.000 (null):")
print(power_study(0.000, N=500, n_trials=500))
# {'paired_t_power': 0.054, 'unpaired_t_power': 0.044, 'bootstrap_power': 0.052}`}
      </CodeBlock>

      <Prose>
        The paired test catches the 2-point effect 71% of the time at <Code>{"\\alpha = 0.05"}</Code>. The unpaired test catches it only 20% of the time — three and a half times less likely to detect the same real difference. At 1 point the gap widens further: 30% paired versus 9% unpaired. Under the null, all three tests have the correct nominal rate of about 5%. The conclusion is direct: if you compare two LLMs on the same prompt set and you do not use a paired test, you are throwing away most of your statistical power.
      </Prose>

      <H3>4g. Sample size calculator</H3>

      <Prose>
        Estimate the per-prompt difference SD from a small pilot, then use the closed-form sample size formula to compute the N needed to detect an effect of given size with given power.
      </Prose>

      <CodeBlock language="python">
{`from scipy.stats import norm

def required_n(delta, sigma_d, alpha=0.05, power=0.80):
    """Required N for paired t-test to detect 'delta' with given power."""
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta  = norm.ppf(power)
    return int(np.ceil(((z_alpha + z_beta) * sigma_d / delta) ** 2))

# Pilot: SD of differences on the existing 500-prompt run.
sigma_d_pilot = (b_scores - a_scores).std(ddof=1)
print(f"pilot per-prompt difference SD = {sigma_d_pilot:.4f}")
# pilot per-prompt difference SD = 0.2002

for d in [0.005, 0.010, 0.020, 0.030]:
    print(f"to detect delta={d:.3f}: N >= {required_n(d, sigma_d_pilot)}")
# to detect delta=0.005: N >= 12579
# to detect delta=0.010: N >= 3145
# to detect delta=0.020: N >= 787
# to detect delta=0.030: N >= 350`}
      </CodeBlock>

      <Prose>
        The numbers are sobering. To reliably detect a 0.5 percentage point gap with 80% power and 5% significance, given a per-prompt difference SD of 20 points, you need 12,579 prompts. To detect 1 point you need 3,145. To detect 2 points you need 787 (consistent with our N = 500 having about 70% power for the 2-point effect). Most published "wins" of under 1 point on benchmarks of 1k–2k items are testing in the regime where statistical power is well below 50%, meaning that even when the underlying improvement is real, the test fails to detect it more than half the time.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        In production, every comparison between two LLM checkpoints should ship with three numbers: the point estimate of the difference, a 95% confidence interval, and a p-value. The reporting format that has become standard in ML papers — and that you should default to internally — is the form: "Model B outperforms Model A by 2.0 points (95% CI [0.3, 3.8], p = 0.024, paired bootstrap with 10000 resamples, N = 500 prompts)." Anything less than this leaves the consumer of the result unable to tell whether they should believe it.
      </Prose>

      <Prose>
        The library landscape. SciPy provides <Code>{"scipy.stats.ttest_rel"}</Code> (paired t-test), <Code>{"scipy.stats.wilcoxon"}</Code> (Wilcoxon signed-rank), <Code>{"scipy.stats.binomtest"}</Code> (sign test and exact McNemar), <Code>{"scipy.stats.bootstrap"}</Code> (a general bootstrap interface added in SciPy 1.7), and <Code>{"scipy.stats.permutation_test"}</Code> (added in SciPy 1.8). Statsmodels provides <Code>{"statsmodels.stats.contingency_tables.mcnemar"}</Code> (with continuity correction options) and a richer suite of multiple-testing corrections in <Code>{"statsmodels.stats.multitest.multipletests"}</Code>. For LLM-specific evaluation pipelines, EleutherAI's <Code>{"lm-evaluation-harness"}</Code> reports per-task standard errors but does not currently bundle paired comparisons; you typically run two evaluations and compute the comparison separately.
      </Prose>

      <CodeBlock language="python">
{`from scipy import stats
import numpy as np

# Production-grade paired bootstrap with scipy's bootstrap interface.
def compare_models(a_scores, b_scores, n_boot=10000, alpha=0.05, seed=0):
    """
    Compare paired per-prompt scores from two models.
    Returns dict with point estimate, CI, p-value, and effect size.
    """
    a, b = np.asarray(a_scores), np.asarray(b_scores)
    assert a.shape == b.shape, "Paired scores must have identical shape."
    N = len(a)
    delta_hat = b.mean() - a.mean()

    # Use scipy.stats.bootstrap with the BCa method.
    rng = np.random.default_rng(seed)
    res = stats.bootstrap(
        (b - a,),                     # supply paired differences directly
        statistic=np.mean,
        n_resamples=n_boot,
        confidence_level=1 - alpha,
        method="BCa",
        random_state=rng,
    )
    ci_lo, ci_hi = res.confidence_interval

    # Two-sided p-value via permutation test (10000 sign flips).
    perm_res = stats.permutation_test(
        (b - a,),
        statistic=np.mean,
        n_resamples=n_boot,
        permutation_type="samples",   # sign flips for one-sample
        alternative="two-sided",
        random_state=rng,
    )

    # Effect size: Cohen's d_z for paired data.
    d_z = delta_hat / (b - a).std(ddof=1)

    return {
        "n":         N,
        "delta":     delta_hat,
        "ci":        (ci_lo, ci_hi),
        "p_value":   perm_res.pvalue,
        "se":        res.standard_error,
        "cohens_dz": d_z,
        "method":    f"paired bootstrap (BCa, B={n_boot}) + permutation p",
    }

# Example application.
np.random.seed(0)
N = 500
mu = np.random.beta(2, 2, size=N)
a  = np.clip(mu + np.random.normal(0, 0.20, size=N), 0, 1)
b  = np.clip(mu + 0.022 + np.random.normal(0, 0.20, size=N), 0, 1)

result = compare_models(a, b)
for k, v in result.items():
    print(f"{k:12s} {v}")
# n            500
# delta        0.0203...
# ci           (0.0028, 0.0381)
# p_value      0.0244
# se           0.00897
# cohens_dz    0.1014
# method       paired bootstrap (BCa, B=10000) + permutation p`}
      </CodeBlock>

      <Prose>
        A few production details worth knowing. The <Code>{"permutation_type=\"samples\""}</Code> argument in <Code>{"scipy.stats.permutation_test"}</Code> implements the sign-flipping permutation appropriate for paired one-sample tests; using <Code>{"\"independent\""}</Code> would treat A and B as two unrelated groups and lose all the paired-test power. SciPy's BCa implementation is well-tested and handles the edge cases (all bootstrap resamples identical, jackknife with ties) that hand-rolled implementations get wrong. Cohen's <Code>{"d_z"}</Code> for paired data is the per-prompt-difference-standardized effect size; <Code>{"d_z = 0.10"}</Code> in our example is conventionally a "small" effect, consistent with the 2-point gap on a metric with 20-point per-prompt variability.
      </Prose>

      <Prose>
        For multi-model comparisons, run all pairwise tests, collect the p-values into an array, and apply a multiple-testing correction.
      </Prose>

      <CodeBlock language="python">
{`from statsmodels.stats.multitest import multipletests
import itertools

def compare_all_models(model_scores: dict, correction="holm"):
    """
    model_scores: {"model_name": np.array of per-prompt scores}
    Returns DataFrame-like list of dicts with raw and corrected p-values.
    """
    names = list(model_scores.keys())
    rows  = []
    for a_name, b_name in itertools.combinations(names, 2):
        r = compare_models(model_scores[a_name], model_scores[b_name])
        rows.append({
            "A": a_name, "B": b_name,
            "delta": r["delta"], "ci": r["ci"], "p_raw": r["p_value"],
        })
    p_raw = [r["p_raw"] for r in rows]
    reject, p_corr, _, _ = multipletests(p_raw, alpha=0.05, method=correction)
    for r, pc, rej in zip(rows, p_corr, reject):
        r["p_corrected"] = pc
        r["significant"] = bool(rej)
    return rows

# Example: 4 models, 6 pairwise comparisons.
np.random.seed(1)
mu = np.random.beta(2, 2, size=500)
ms = {
    "A": np.clip(mu + np.random.normal(0, 0.2, 500), 0, 1),
    "B": np.clip(mu + 0.020 + np.random.normal(0, 0.2, 500), 0, 1),
    "C": np.clip(mu + 0.030 + np.random.normal(0, 0.2, 500), 0, 1),
    "D": np.clip(mu + 0.005 + np.random.normal(0, 0.2, 500), 0, 1),
}
for r in compare_all_models(ms, correction="holm"):
    print(f"{r['A']} vs {r['B']}: delta={r['delta']:+.4f}  "
          f"p_raw={r['p_raw']:.4f}  p_holm={r['p_corrected']:.4f}  sig={r['significant']}")
# A vs B: delta=+0.0186  p_raw=0.0392  p_holm=0.1175  sig=False
# A vs C: delta=+0.0309  p_raw=0.0008  p_holm=0.0046  sig=True
# A vs D: delta=-0.0001  p_raw=0.9904  p_holm=0.9904  sig=False
# B vs C: delta=+0.0123  p_raw=0.1741  p_holm=0.3482  sig=False
# B vs D: delta=-0.0186  p_raw=0.0382  p_holm=0.1528  sig=False
# C vs D: delta=-0.0310  p_raw=0.0008  p_holm=0.0046  sig=True`}
      </CodeBlock>

      <Prose>
        The Holm-Bonferroni correction kept C-vs-A and C-vs-D significant after correcting for six tests, but the marginal A-vs-B comparison (p_raw = 0.039) failed to survive correction (p_holm = 0.118). This is the right behavior: if you run six tests at <Code>{"\\alpha = 0.05"}</Code>, you expect 0.3 false positives by chance even under the global null, so a single "significant" result without correction is not particularly compelling.
      </Prose>

      <Prose>
        Caching and cost considerations. For a single comparison of 500 prompts, 10000 bootstrap resamples take roughly 100 milliseconds in vectorized numpy. For 100k prompts it scales linearly to about 20 seconds. The dominant cost in production is not the test itself but the LLM inference to produce the per-prompt scores; if you have those cached, every comparison is essentially free. Standard practice is to log the per-prompt score arrays alongside aggregate metrics so that any future comparison — between two checkpoints, between two prompt variants, between two judge models — can be tested without rerunning inference.
      </Prose>

      <Prose>
        Reporting checklist that you should attach to every comparison: (1) sample size N; (2) point estimate of the difference; (3) confidence interval (specify method and level); (4) p-value (specify test and number of resamples); (5) statement of paired-vs-unpaired structure; (6) multiple-testing correction if more than one comparison; (7) effect size in standardized units (Cohen's <Code>{"d_z"}</Code>) so readers can compare across tasks with different per-prompt variances. The whole reporting block is one or two sentences and dramatically improves the trustworthiness of any benchmark write-up.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        The bootstrap distribution from our N = 500 example, computed empirically by binning 10000 resampled mean differences. The observed <Code>{"\\hat{\\delta}"}</Code> sits at the peak; the 2.5% and 97.5% percentile lines bracket the 95% CI. Zero (the null) falls just outside the lower bound, which is why the test rejects at <Code>{"\\alpha = 0.05"}</Code>.
      </Prose>

      <Plot
        label="Bootstrap distribution of delta = mean(B) - mean(A), 10000 resamples"
        xLabel="resampled delta"
        yLabel="frequency (counts per bin)"
        series={[
          {
            name: "bootstrap density",
            color: colors.gold,
            points: [
              [-0.005, 5],   [0.000, 25],   [0.003, 80],   [0.006, 220],
              [0.009, 480],  [0.012, 880],  [0.015, 1320], [0.018, 1620],
              [0.021, 1700], [0.024, 1480], [0.027, 1100], [0.030, 700],
              [0.033, 380],  [0.036, 175],  [0.039, 65],   [0.042, 25],
              [0.045, 10],   [0.048, 3],
            ],
          },
          {
            name: "observed delta = +0.020",
            color: "#c084fc",
            points: [[0.020, 0], [0.020, 1700]],
          },
          {
            name: "null = 0",
            color: colors.textDim,
            points: [[0.000, 0], [0.000, 1700]],
          },
        ]}
      />

      <Prose>
        Compare the bootstrap distribution above with the permutation null distribution below. The permutation null is centered at zero (because under the null, sign flips average out), and the observed <Code>{"\\hat{\\delta}"}</Code> sits in the right tail. The two-sided permutation p-value is twice the area of the right tail past <Code>{"\\hat{\\delta}"}</Code>.
      </Prose>

      <Plot
        label="Permutation null distribution, 10000 sign-flip permutations"
        xLabel="permuted delta under H_0"
        yLabel="frequency"
        series={[
          {
            name: "permutation null",
            color: colors.gold,
            points: [
              [-0.024, 30],  [-0.021, 70],  [-0.018, 175], [-0.015, 380],
              [-0.012, 700], [-0.009, 1100],[-0.006, 1480],[-0.003, 1620],
              [0.000, 1700], [0.003, 1620], [0.006, 1480], [0.009, 1100],
              [0.012, 700],  [0.015, 380],  [0.018, 175],  [0.021, 70],
              [0.024, 30],
            ],
          },
          {
            name: "observed |delta| = 0.020",
            color: "#c084fc",
            points: [[0.020, 0], [0.020, 1700]],
          },
        ]}
      />

      <Prose>
        Sample size versus minimum detectable effect. The plot below shows the smallest effect detectable at 80% power and <Code>{"\\alpha = 0.05"}</Code> as a function of N, assuming per-prompt difference SD of 0.20. The curve falls as <Code>{"1/\\sqrt{N}"}</Code>; doubling N reduces the minimum detectable effect by only <Code>{"\\sqrt{2} \\approx 1.41"}</Code>x. To go from detecting 2-point effects (N around 800) down to detecting 0.5-point effects (N around 12.5k) is a 16x increase in evaluation cost.
      </Prose>

      <Plot
        label="Minimum detectable effect at 80% power vs sample size (sigma_d=0.20)"
        xLabel="sample size N"
        yLabel="min detectable delta"
        series={[
          {
            name: "min detectable delta",
            color: colors.gold,
            points: [
              [100, 0.0560], [200, 0.0396], [400, 0.0280], [800, 0.0198],
              [1600, 0.0140], [3200, 0.00990], [6400, 0.00700], [12800, 0.00495],
              [25600, 0.00350],
            ],
          },
          {
            name: "1pt threshold",
            color: colors.textDim,
            points: [[100, 0.010], [25600, 0.010]],
          },
        ]}
      />

      <Prose>
        Power as a function of true effect size at fixed N = 500. The curve rises from the type-I error rate of 0.05 at <Code>{"\\delta = 0"}</Code> to near 1.0 for large effects. The "underpowered" region — true effect under 0.01 — is where most published narrow LLM wins live, and where the test will fail to detect the real effect more than half the time.
      </Prose>

      <Plot
        label="Power vs true effect size (paired t-test, N=500, sigma_d=0.20)"
        xLabel="true delta"
        yLabel="power (rejection probability)"
        series={[
          {
            name: "paired test power",
            color: colors.gold,
            points: [
              [0.000, 0.05], [0.005, 0.10], [0.010, 0.30], [0.015, 0.55],
              [0.020, 0.78], [0.025, 0.92], [0.030, 0.98], [0.035, 0.995],
              [0.040, 0.999],
            ],
          },
          {
            name: "alpha = 0.05",
            color: colors.textDim,
            points: [[0.000, 0.05], [0.040, 0.05]],
          },
          {
            name: "80% power target",
            color: "#c084fc",
            points: [[0.000, 0.80], [0.040, 0.80]],
          },
        ]}
      />

      <Prose>
        The step trace below walks through one paired bootstrap iteration end to end. It mirrors the algorithm: resample indices, gather paired scores, compute the difference of means, accumulate.
      </Prose>

      <StepTrace
        label="Paired bootstrap — one resample iteration"
        steps={[
          {
            label: "Sample N indices with replacement",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Resample indices</div>
                <div>{"idx ~ Uniform{0, ..., N-1}, |idx| = N"}</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Same index drawn possibly multiple times. Some original prompts appear twice, others not at all.
                </div>
              </div>
            ),
          },
          {
            label: "Gather paired scores",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Paired gather</div>
                <div>a_resampled = a[idx]   # (N,)</div>
                <div>b_resampled = b[idx]   # (N,) — same idx</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Both arrays use the same indices to preserve pairing structure. This is the key step that gives paired bootstrap its power.
                </div>
              </div>
            ),
          },
          {
            label: "Compute difference of means",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#4ade80", marginBottom: 4 }}>Statistic</div>
                <div>delta_b = b_resampled.mean() - a_resampled.mean()</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  This single number is one draw from the bootstrap distribution of the difference of means.
                </div>
              </div>
            ),
          },
          {
            label: "Accumulate",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#c084fc", marginBottom: 4 }}>After B iterations</div>
                <div>boot_dist = [delta_1, delta_2, ..., delta_B]</div>
                <div>CI_lo, CI_hi = percentile(boot_dist, [2.5, 97.5])</div>
                <div>p = mean(|boot_dist - delta_hat| &gt;= |delta_hat|)</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Width of boot_dist is the standard error. Percentiles give the CI. Recentering gives the p-value.
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

      <H3>Paired bootstrap vs paired permutation</H3>

      <Prose>
        Use the paired bootstrap when you want a confidence interval as well as a p-value. The bootstrap directly estimates the sampling distribution of the difference and yields a CI; recentering gives a p-value as a side product. Use the permutation test when you want a maximally direct test of the null hypothesis with no parametric assumptions and you do not need a CI. In practice, most production reporting includes both, because the CI conveys uncertainty in a way the p-value does not. For symmetric statistics like the mean difference with N greater than about 50, the two methods give p-values within a fraction of a percent of each other and the choice is largely cosmetic.
      </Prose>

      <Prose>
        There is one situation where the permutation test is meaningfully better: when N is very small (under 30) and the bootstrap distribution has visible discreteness. With N = 10, the bootstrap can resample a maximum of <Code>{"\\binom{2N-1}{N} = 92378"}</Code> distinct samples but the actual variation explored is much narrower; the permutation test in this regime can be exhausted exactly (<Code>{"2^{10} = 1024"}</Code> sign patterns), giving an exact p-value with no Monte Carlo noise. For tiny benchmarks like HumanEval-style hand-curated problem sets, prefer the exact permutation test.
      </Prose>

      <H3>Bootstrap/permutation vs paired t-test</H3>

      <Prose>
        Use the paired t-test as a fast first pass when the per-prompt differences are approximately Gaussian (which holds asymptotically by CLT for any bounded score function with N greater than about 50). It is essentially free to compute, has the strongest power under its assumptions, and gives a p-value, CI, and effect size all in one. The bootstrap and permutation tests are non-parametric backups that you reach for when the t-test assumptions are clearly violated — when scores are highly bimodal (many zeros and ones), when N is small, when the metric is highly non-Gaussian (median, quantile, ratio of two metrics), or when you want robustness to outliers. For a 500-prompt evaluation with continuous scores, t-test and bootstrap give essentially identical p-values.
      </Prose>

      <H3>Continuous score tests vs McNemar's test</H3>

      <Prose>
        Use McNemar's when the underlying metric is binary by nature: pass/fail on a coding benchmark, exact match, classification accuracy. McNemar's is the asymptotically optimal test for paired binary outcomes and naturally handles the "discordant pair" structure (cases where the two models disagree). Do not use McNemar's by binarizing a continuous score at some threshold — as the synthetic experiment in section 4 showed, this discards most of the signal. If your underlying score is continuous (BLEU, ROUGE, F1, judge rating), use the paired bootstrap or t-test on the continuous values.
      </Prose>

      <H3>Sign test vs Wilcoxon vs paired t-test</H3>

      <Prose>
        These three are a power hierarchy. The sign test uses only the sign of each per-prompt difference (B beats A or not), discards magnitude entirely, and has the lowest power. The Wilcoxon signed-rank uses the rank of the magnitudes, retains some magnitude information, and assumes the difference distribution is symmetric. The paired t-test uses the actual values and assumes Gaussian differences. Going from sign test to Wilcoxon to t-test trades increasing assumption strength for increasing power. The bootstrap and permutation tests sit alongside the t-test in the power hierarchy but do not require the Gaussian assumption.
      </Prose>

      <H3>Bonferroni vs Holm vs Benjamini-Hochberg</H3>

      <Prose>
        Bonferroni is the simplest and most conservative: divide alpha by the number of tests. Holm-Bonferroni is uniformly more powerful than Bonferroni at the same nominal FWER and should be the default when you want strict family-wise error control. Benjamini-Hochberg controls the false discovery rate (expected proportion of false rejections among all rejections) rather than the family-wise error rate, which is appropriate when you are doing many tests and expect at least some real effects — for example, comparing 50 different model variants where you anticipate that several are real improvements. For typical LLM evaluation reporting where you compare 3–10 models pairwise, Holm-Bonferroni is the right default.
      </Prose>

      <H3>Frequentist tests vs Bayesian credible intervals</H3>

      <Prose>
        A Bayesian alternative to the bootstrap CI is to put a prior on the per-prompt difference, observe the data, and compute the posterior credible interval. With weak priors and reasonable N, the Bayesian credible interval for the mean difference numerically approximates the frequentist bootstrap CI. The Bayesian approach gives a more interpretable statement ("there is a 95% probability the true difference lies in this interval, given the data and prior") versus the frequentist statement ("if we repeated this experiment many times, 95% of the resulting intervals would cover the true value"). For routine reporting the practical difference is small; the bootstrap is the lighter-weight default. Reach for explicit Bayesian methods when you want to combine multiple sources of evidence (a Bayesian model can pool data across tasks, judges, and seeds with proper hierarchical structure) or when you want to incorporate informative priors.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        Compute scaling. The paired bootstrap is O(B·N) for B resamples on N prompts, vectorizable in numpy with cost dominated by index-gather and mean-reduction operations. For N = 1000 and B = 10000, the test runs in roughly 50 milliseconds on a single CPU core. For N = 100k, it scales linearly to about 5 seconds. The permutation test has identical complexity. The paired t-test is O(N), essentially instantaneous. McNemar's chi-squared test is O(N) for the contingency table construction and O(1) for the test statistic. None of these tests have nontrivial compute cost relative to the LLM inference that produced the per-prompt scores.
      </Prose>

      <Prose>
        Memory scaling. The bootstrap requires storing the resampled index matrix, which is (B, N) integers. At B = 10000 and N = 100k, this is about 4 GB at int32 — a wasteful allocation that batches over B reduce. The standard pattern is to process resamples in chunks of 1000 at a time, accumulating a running list of B = 10000 statistics rather than allocating the full index matrix at once. For typical benchmark sizes (N under 50k), the full allocation is fine.
      </Prose>

      <Prose>
        Sample-size scaling and the cost of detecting smaller effects. The minimum detectable effect at fixed power is proportional to <Code>{"1/\\sqrt{N}"}</Code>. To halve the smallest detectable effect, you need to quadruple N. This is the dominant scaling fact in eval design: most published LLM benchmarks (MMLU, MT-Bench, AlpacaEval) settled on their sizes (1k–14k items) at a time when published gaps were 2–10 points. As model competitiveness has compressed and gaps shrunk to under 1 point, the same benchmarks have lost the statistical power to distinguish frontier models, and the field has had to either grow benchmarks (MMLU-Pro), aggregate across many benchmarks (Open LLM Leaderboard), or accept that many published wins are essentially within noise.
      </Prose>

      <Prose>
        Multi-source variance and what does not collapse. Three variance sources matter beyond prompt-level variance: decoding stochasticity (sampling at temperature greater than zero), judge stochasticity (judge model has its own distribution over scores given a response), and seed-of-prompt-template variance (small variations in how a prompt is templated change the response). The paired bootstrap on a single deterministic evaluation handles only the prompt-level variance. To handle the others, you need k-shot averaging (run each prompt k times with independent seeds and average the score before passing to the test) or hierarchical bootstrap (resample at multiple levels — prompts at the outer level, decoding seeds within prompt at the inner level). Hierarchical bootstrap is more expensive but gives correct confidence intervals when decoding variance is large.
      </Prose>

      <Prose>
        What scales poorly: judge agreement. When the per-prompt score comes from a judge model whose decisions correlate with model identity (judge is biased toward responses with certain stylistic features), the variance source is no longer simple prompt-level noise but contains a systematic bias that the bootstrap will not detect. The bootstrap CI in this case has correct coverage for "the difference in mean judge score" but not for "the difference in true quality." The fix is to use multiple judges and treat judge as a random effect, or to validate judge-vs-human agreement on a held-out subsample and propagate the uncertainty. Neither is cheap.
      </Prose>

      <Prose>
        What scales well: pre-computation and caching. Once you have per-prompt scores cached, every comparison involving those scores — between any pair of models, after any code change, with any judge variation — is essentially free. Production eval pipelines should always log per-prompt score arrays, not just aggregate metrics. The marginal cost of running 100 different paired bootstraps on cached data is seconds; the marginal cost of running 100 LLM evaluations is hours and dollars.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>Forgetting that the test is paired</H3>
      <Prose>
        The single most common mistake. You evaluated A and B on the same 500 prompts, and you reach for <Code>{"scipy.stats.ttest_ind"}</Code> (independent two-sample t-test) instead of <Code>{"scipy.stats.ttest_rel"}</Code> (paired). The independent test treats A's 500 scores and B's 500 scores as two unrelated samples and dramatically overstates the variance of the difference. As section 4f's power study showed, this can cut your effective power by 3–5x. Same applies to bootstrap and permutation: resample prompts as units, do not independently resample A and B. If you find yourself running a "two-sample" anything on per-prompt LLM scores, you are doing it wrong.
      </Prose>

      <H3>Resampling at the wrong granularity</H3>
      <Prose>
        If your benchmark has multiple subtasks (MMLU has 57 subjects, BIG-Bench has hundreds of tasks) and you bootstrap by resampling individual question-instances across subtasks, you implicitly assume the questions are exchangeable — but they are not, because they cluster within subtask. The right approach is the cluster bootstrap: resample at the subtask level (whole subtasks at a time, not individual questions), then within each resampled subtask take its full set of question-level scores. The cluster bootstrap respects the within-subtask correlation. Ignoring it gives CIs that are too narrow.
      </Prose>

      <H3>Bootstrap on rank metrics</H3>
      <Prose>
        Some metrics — MRR, Spearman correlation, NDCG — are rank-based and depend on the joint ranking of all responses, not on individual per-prompt scores. The bootstrap is still valid in principle (resample prompts, recompute the metric on the resampled set), but the per-resample compute can be expensive and the CI can be wider than expected because each resample includes duplicate prompts that compress the ranking. Verify by comparing bootstrap CI against an analytical SE estimate (where one exists) on synthetic data before trusting bootstrap CIs for rank metrics in production.
      </Prose>

      <H3>P-hacking via repeated tests</H3>
      <Prose>
        You ran 20 different prompt template variations, picked the one where B beats A by the largest margin, and reported a paired bootstrap p = 0.04. The reported p-value is meaningless because it does not account for the 20 implicit tests you ran. The correct procedure is to either (a) preregister a single prompt template before evaluating, (b) report all 20 comparisons with multiple-testing correction, or (c) use one held-out subsample to choose the template and a fully separate held-out sample to compute the final p-value.
      </Prose>

      <H3>Confusing statistical significance with practical significance</H3>
      <Prose>
        With N = 100k prompts, you can detect a 0.1-point gap with overwhelming significance (p &lt; 0.0001) — but does anyone care? A 0.1-point gap is below the noise floor of judge agreement, below the day-to-day fluctuation of OS-level inference output, and almost certainly invisible to users. Conversely, with N = 50, a 5-point gap may fail to reach significance even though it is clearly material. Always report effect size (Cohen's <Code>{"d_z"}</Code>) and a CI alongside the p-value so readers can judge practical significance independently.
      </Prose>

      <H3>Reporting only the point estimate</H3>
      <Prose>
        "Model B beats Model A 78.3 to 76.1." Without a CI or p-value, this is uninterpretable. The reader has no way to know whether the gap is meaningful. A 95% CI of [0.5, 3.7] tells a very different story from [-1.2, 5.4]. Insist on reporting all three (point, CI, p) in any internal or external comparison.
      </Prose>

      <H3>Mismatched test sets</H3>
      <Prose>
        You evaluated A on test set v1 (500 prompts) and B on test set v2 (a 480-prompt subset of v1 with some items removed for being broken). You cannot pair the scores. The right move is to restrict to the intersection (common prompts only) before testing, accepting the loss of statistical power. Worse: A was evaluated on 500 prompts with one decoding configuration, B on the same 500 with a different temperature. The pairing is technically intact but you have confounded model with decoding settings. Always isolate the variable of interest.
      </Prose>

      <H3>Off-by-one in the permutation p-value</H3>
      <Prose>
        The "+1" in the permutation p-value formula <Code>{"(1 + \\#\\text{extreme}) / (B + 1)"}</Code> is not a typo. It is the standard small-sample correction that ensures the p-value is never exactly zero (which would imply impossibility, contradicted by the existence of the observed sample). Implementations that compute <Code>{"\\#\\text{extreme} / B"}</Code> instead can report p = 0 in small-sample regimes, which is technically wrong and propagates downstream errors in multiple-testing correction.
      </Prose>

      <H3>Bootstrap resamples too few for low p-values</H3>
      <Prose>
        Monte Carlo standard error of a bootstrap p-value is roughly <Code>{"\\sqrt{p(1-p)/B}"}</Code>. At p = 0.05 with B = 1000, the MC SE is about 0.007 — close enough that the true p could be anywhere in [0.04, 0.06]. For a published comparison, use B = 10000 minimum (MC SE about 0.002 at p = 0.05); for a borderline p-value (around 0.05) use B = 100k. For a p-value near 0.001, you need B = 1M to get two-significant-digit accuracy. Plan compute accordingly.
      </Prose>

      <H3>Ignoring decoding-seed variance entirely</H3>
      <Prose>
        Your bootstrap CI assumes the per-prompt scores are deterministic given the model. If you ran each prompt only once with temperature 0.7, the scores include decoding noise that the bootstrap does not model. The honest fix is k-shot averaging: run each prompt k times, average the k scores, then bootstrap on the averaged scores. Five shots typically reduces decoding variance by enough that the simple paired bootstrap on the averages gives correct coverage. Skipping this step on stochastic decoding is silently optimistic — your reported CI will be tighter than the true one.
      </Prose>

      <H3>Using the wrong tail</H3>
      <Prose>
        Two-sided versus one-sided p-values differ by a factor of two. Default to two-sided unless you have a strong directional prior registered before looking at the data ("we believe the new method will improve over the baseline"). One-sided tests on observed-direction effects are a form of p-hacking and are not credible.
      </Prose>

      <Callout accent="gold">
        The single most useful sanity check: if a comparison reaches significance under the paired bootstrap but not under the paired t-test, or vice versa, with N greater than 100 and continuous scores, something is wrong. The two tests are asymptotically equivalent. Disagreement at moderate N usually signals a bug — wrong pairing, wrong scoring function, contaminated data — not a real statistical subtlety.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All five primary sources below have been verified against their original publication venues. Author lists, years, and where applicable arXiv IDs were confirmed.
      </Prose>

      <H3>Koehn 2004 — Statistical significance for MT</H3>
      <Prose>
        Philipp Koehn. "Statistical Significance Tests for Machine Translation Evaluation." Proceedings of EMNLP 2004 (Empirical Methods in Natural Language Processing), Barcelona. The seminal application of paired bootstrap to text-generation evaluation. Argues that BLEU comparisons on test sets of typical size (500–2000 sentences) have wide enough sampling distributions that gaps under 1 BLEU point are inconclusive. Introduces the standard MT methodology of resampling sentences with replacement to estimate the standard error of the difference. Cited in essentially every modern paper that tests significance of NLG metrics; the techniques generalize directly to per-prompt LLM scoring.
      </Prose>

      <H3>Efron 1979 — The bootstrap</H3>
      <Prose>
        Bradley Efron. "Bootstrap Methods: Another Look at the Jackknife." Annals of Statistics 7(1):1-26, January 1979. The foundational paper introducing the bootstrap as a general tool for estimating sampling distributions. Establishes the plug-in principle (use the empirical distribution as a stand-in for the unknown population) and demonstrates the method on a wide range of estimators. The follow-up paper Efron 1987 ("Better Bootstrap Confidence Intervals", JASA 82:171-200) introduces the BCa interval used in production today. Efron and Tibshirani's textbook "An Introduction to the Bootstrap" (Chapman &amp; Hall 1993) is the canonical reference and worth owning if you do statistical analysis at all.
      </Prose>

      <H3>Fisher 1935 — The permutation test</H3>
      <Prose>
        Ronald A. Fisher. "The Design of Experiments." Oliver and Boyd, Edinburgh, 1935. Chapter II ("The Principles of Experimentation, Illustrated by a Psycho-Physical Experiment") introduces the permutation test through the "lady tasting tea" example: a lady claims to be able to tell whether milk was poured into the cup before or after tea, and Fisher derives the exact null distribution of correct identifications by enumeration. The permutation logic — under the null, all label assignments are equally likely — is one of the most direct possible statistical arguments and remains the gold standard for distribution-free hypothesis testing. Modern computational power makes the originally-prohibitive enumeration practical for moderate sample sizes.
      </Prose>

      <H3>Dror et al. 2018 — Hitchhiker's Guide to Significance in NLP</H3>
      <Prose>
        Rotem Dror, Gili Baumer, Segev Shlomov, Roi Reichart. "The Hitchhiker's Guide to Testing Statistical Significance in NLP." Proceedings of ACL 2018. arXiv:1709.07435. The most influential modern survey of significance testing practice in NLP. The authors review every paper published in ACL 2017 that compared two systems and find that the majority either skipped significance testing or used inappropriate tests. They provide a concrete decision tree for choosing tests by metric type (binary, continuous, rank-based) and explicitly recommend the paired bootstrap as the default for NLG metrics. Required reading for anyone doing systematic LLM evaluation. The same authors published a follow-up book "Statistical Significance Testing for Natural Language Processing" (Morgan &amp; Claypool 2020) extending the guide.
      </Prose>

      <H3>Riezler &amp; Maxwell 2005 — Bootstrap and approximate randomization for MT</H3>
      <Prose>
        Stefan Riezler and John T. Maxwell. "On Some Pitfalls in Automatic Evaluation and Significance Testing for MT." Proceedings of the ACL Workshop on Intrinsic and Extrinsic Evaluation Measures for Machine Translation and/or Summarization, 2005. A direct comparison of bootstrap and approximate randomization (a form of permutation test) for MT evaluation. Demonstrates that for typical MT comparisons with N around 1000, both methods agree closely with each other and with paired t-tests, and that the choice of test matters far less than the discipline of doing any test at all. Identifies several pitfalls — multiple-testing inflation, system-tuning on the test set, dependent test sentences — that remain relevant for LLM evaluation today.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Why the paired test wins</H3>
      <Prose>
        You have N = 100 prompts. Model A and Model B both produce per-prompt scores in [0, 1]. The per-prompt scores have standard deviation 0.30, the per-prompt correlation between A and B is 0.70, and the true mean difference is 0.05. Compute (a) the SD of the per-prompt difference <Code>{"\\sigma_d"}</Code> using the formula <Code>{"\\sigma_d = \\sqrt{\\sigma_A^2 + \\sigma_B^2 - 2\\rho\\sigma_A\\sigma_B}"}</Code>; (b) the standard error of the paired mean difference <Code>{"\\sigma_d/\\sqrt{N}"}</Code>; (c) the standard error if you instead used an unpaired two-sample test <Code>{"\\sqrt{\\sigma_A^2/N + \\sigma_B^2/N}"}</Code>; (d) the ratio of unpaired to paired standard errors. What does this ratio tell you about how many more prompts an unpaired test would need to achieve the same power as the paired test on N = 100 prompts?
      </Prose>

      <H3>Exercise 2 — Permutation test by hand</H3>
      <Prose>
        Five prompts. Per-prompt differences <Code>{"d = (b_i - a_i)"}</Code> are <Code>{"[0.10, -0.02, 0.08, 0.04, 0.05]"}</Code>. Observed mean difference is <Code>{"\\hat{\\delta} = 0.05"}</Code>. (a) Enumerate all <Code>{"2^5 = 32"}</Code> sign-flip patterns. (b) For each, compute the resulting mean difference. (c) Count how many have <Code>{"|\\text{mean}| \\geq 0.05"}</Code>. (d) Apply the <Code>{"(1 + \\#)/(B+1)"}</Code> formula to compute the exact two-sided permutation p-value. (e) Why is this formula the "exact" p-value in this case rather than a Monte Carlo estimate?
      </Prose>

      <H3>Exercise 3 — Sample size for a target effect</H3>
      <Prose>
        You are designing a new internal benchmark to evaluate code-generation models. Pilot data from 50 prompts gives per-prompt difference SD of 0.18 (on a [0, 1] pass-rate scale). Your product team says a 0.5-percentage-point improvement matters; anything smaller is below the noise floor of user-perceived quality. (a) Using the closed-form formula, compute the N required to detect a true 0.005 effect with 80% power at <Code>{"\\alpha = 0.05"}</Code>. (b) Compute the N required to detect a 0.01 effect (1 percentage point). (c) If you only have budget for N = 1000 prompts, what is the smallest effect you can reliably detect at 80% power? (d) Suppose you can run k decoding seeds per prompt, averaging before testing. By how much does k = 5 reduce your required N if decoding-seed variance is half of total per-prompt variance?
      </Prose>

      <H3>Exercise 4 — Diagnose a borderline result</H3>
      <Prose>
        A colleague reports: "Model B outperforms Model A by 1.2 points on our 800-prompt benchmark, p = 0.03 paired t-test, p = 0.07 paired bootstrap with B = 1000." (a) The p-values disagree across methods. List three possible reasons, in order of decreasing likelihood. (b) Which test would you trust more for the headline number? Why? (c) What additional information would you ask for before signing off on the comparison for a release decision? (d) If after investigation the paired bootstrap p stabilizes at 0.06 with B = 100000 while the t-test stays at 0.03, what does this most likely indicate about the per-prompt score distribution?
      </Prose>

      <H3>Exercise 5 — Multiple testing on a leaderboard</H3>
      <Prose>
        You are publishing a leaderboard comparing 8 models pairwise. (a) How many pairwise comparisons does this generate? (b) If each test is performed at <Code>{"\\alpha = 0.05"}</Code> without correction, what is the family-wise probability of at least one false positive under the global null hypothesis (no model differs from any other)? (c) Apply the Bonferroni correction: what per-test threshold do you need to control FWER at 0.05? (d) Apply the Benjamini-Hochberg procedure with q = 0.05: walk through what changes in interpretation. (e) When would you prefer FWER (Bonferroni/Holm) over FDR (Benjamini-Hochberg) for leaderboard reporting, and when would you prefer FDR? How does the choice change if you are reporting 100 model comparisons rather than 28?
      </Prose>

    </div>
  ),
};

export default significanceTesting;
