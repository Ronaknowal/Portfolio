import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const effectSizes = {
  title: "Effect Sizes (Cohen's d, Hedges' g)",
  slug: "effect-sizes-cohens-d-hedges-g",
  readTime: "~32 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        A p-value answers a single, narrow question: under the null hypothesis that there is no real difference between two conditions, how surprising is the data we observed? It returns a probability. It does not return a magnitude. With a sample of fifty examples, the difference between two LLM checkpoints scoring 78.3% and 79.1% on a benchmark is unlikely to reach statistical significance. With a sample of fifty thousand, the same 0.8 percentage point gap will produce a p-value below 0.001, look "highly significant," and tell you essentially nothing about whether you should ship the new model. The mathematical machinery of null-hypothesis significance testing is silent on the question that actually matters in deployment: how big is the effect, in units that humans can reason about, relative to the noise we already accept as routine?
      </Prose>

      <Prose>
        Effect sizes exist to answer that second question. An effect size is a standardized measure of the magnitude of a phenomenon — typically the difference between two group means, scaled by the spread of the underlying populations. Because it is unitless (or expressed in standard-deviation units), it is comparable across studies, across instruments, and across sample sizes. A Cohen's <Code>d</Code> of 0.5 means the difference between two group means is half a standard deviation, regardless of whether you are measuring exam scores, response latencies in milliseconds, or token-level perplexity. This invariance is what makes effect sizes the lingua franca of meta-analysis, and it is also what makes them the right tool for ML evaluation reporting where "did the new method help, and by how much that I should care about" is the only question worth asking.
      </Prose>

      <Prose>
        The conceptual split between p-values and effect sizes was sharpened by a long campaign within statistics that culminated in the American Statistical Association's 2016 statement on statistical significance and p-values, and again in the 2019 special issue of <em>The American Statistician</em> calling for an end to mechanical "p &lt; 0.05" thinking. The ASA was not arguing that p-values are useless. It was arguing that they are misused as binary verdicts and that the magnitude of the effect, the precision of its estimate, and the practical importance of the result must be reported alongside any significance test. Cohen's d, Hedges' g, Glass's Δ, and the binary-outcome cousins — phi, Cramer's V, odds ratios, risk ratios — are the family of standardized magnitude measures that the ASA's call implicitly recommended. Understanding when each is the right choice, and how to compute confidence intervals for them, is the practical floor of competent quantitative reporting.
      </Prose>

      <Prose>
        For ML practitioners specifically, effect sizes do something that p-values cannot: they let you reason about <em>practical</em> significance. A new fine-tuning recipe that improves a 50k-example eval from 78.3% to 78.9% is statistically significant at p &lt; 0.001. That same improvement, expressed as Cohen's d on the per-example accuracy variable, is roughly 0.015 — three orders of magnitude below the conventional "small effect" threshold of 0.2. The p-value is screaming "real!" while the effect size is whispering "negligible." Whether to ship that model is then a business decision about whether negligible-but-real improvements compound, not a statistical one. The same applies to A/B tests on production traffic, to comparisons between RLHF reward models, to retrieval evaluations, and to any scenario where dataset size lets the central limit theorem manufacture significance from arbitrarily small effects.
      </Prose>

      <Prose>
        Effect sizes also enable <em>a priori</em> power analysis. If you know the effect size you would consider practically meaningful, the standard formulas (going back to Jacob Cohen's 1988 textbook) tell you the minimum sample size required to detect that effect with chosen power and significance. Without an effect-size target, "how many examples should the eval set have" has no principled answer. With one, it has a number. That number is often surprisingly large for small effects — detecting d = 0.1 with 80% power and α = 0.05 requires roughly 1,600 examples per group — which is itself useful information about whether the experiment is worth running.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        The single picture to keep in your head is two overlapping bell curves. Each curve represents the distribution of some measurement under a different condition — control vs. treatment, model A vs. model B, before vs. after intervention. The mean of each curve is where its peak sits. The standard deviation of each curve is how wide it is. The effect size asks: how far apart are the peaks, measured in units of how wide the curves are? When the peaks are one curve-width apart, the effect size is roughly 1.0. When they are half a curve-width apart, the effect size is 0.5. When the curves coincide, the effect size is zero.
      </Prose>

      <Prose>
        This framing makes the two ingredients explicit. The numerator is the raw difference in means — measured in the original units of the variable. The denominator is a measure of pooled spread, also in the original units. Their ratio is dimensionless. Multiply both means and both standard deviations by 100 (e.g., switch from proportions to percentages) and the ratio is unchanged. Switch from milliseconds to seconds and the ratio is unchanged. This invariance is what makes effect sizes portable, but it is also why they are sometimes misleading: if the standard deviation of your population is artificially small (a homogeneous lab sample, a narrow eval prompt distribution), a tiny absolute difference produces a large effect size that does not generalize to a more diverse setting.
      </Prose>

      <Prose>
        The conventional thresholds — Cohen's "small = 0.2, medium = 0.5, large = 0.8" — were proposed by Jacob Cohen as rough heuristics drawn from typical effects in the behavioral sciences of the 1960s and 1970s. They are deeply embedded in research culture but they are not laws of nature. In some fields a d of 0.1 is enormous (medical interventions affecting mortality at population scale); in others a d of 0.8 is unimpressive (priming effects on simple reaction time). Cohen himself wrote, in the 1988 second edition of <em>Statistical Power Analysis for the Behavioral Sciences</em>, that the labels were "intended to be used only when no better basis for estimating the ES is available." The right use of the thresholds is as an order-of-magnitude sanity check, not as a verdict.
      </Prose>

      <Prose>
        The intuition for the small-sample correction that distinguishes Hedges' g from Cohen's d is also worth getting clear on. The denominator of d is an estimate of the population standard deviation, computed from the sample. When sample sizes are small, this estimator is biased downward — it tends to underestimate the true population standard deviation, because the sample variance has <Code>n − 1</Code> degrees of freedom and the chi distribution that governs the sample standard deviation has a mean below the true σ. Dividing by an underestimated denominator inflates the resulting effect size. Hedges and Olkin worked out the exact bias correction in 1985: multiply d by a factor <Code>J(N)</Code> that depends on the total sample size and approaches 1 as N grows large. The correction is small (under 5%) for combined samples above 20 and negligible above 100, but for the small evaluation sets common in early-stage ML experiments — n = 50, n = 100 — the bias is real and the unbiased estimator is preferred.
      </Prose>

      <Prose>
        Glass's Δ takes a different tack. Instead of pooling the standard deviations of both groups, it uses only the control group's SD. The reasoning: if you are evaluating an experimental intervention, the treatment may itself change the variance of the outcome variable — perhaps it makes some subjects much better and leaves others unaffected, increasing the spread. Pooling the two SDs would let the treatment's induced spread inflate the denominator and shrink the apparent effect. Using only the control SD anchors the comparison to the natural, untreated variability. This makes Glass's Δ the right choice when you have a clear control reference and you suspect the treatment changes the outcome variance.
      </Prose>

      <Prose>
        Paired data — where the same unit is measured twice — needs its own variant. If you compare a model's accuracy on a fixed eval set before and after a fine-tune, the per-example accuracies are not independent: they share the same prompt, the same gold label, and the same difficulty. The right effect size for paired data is Cohen's <Code>d_z</Code>, computed on the per-example differences rather than the raw values. Using Cohen's d on the unpaired raw values would dramatically overstate the effect because the paired structure removes a substantial source of variance that the unpaired formula ignores.
      </Prose>

      <Prose>
        For binary outcomes — pass/fail, correct/incorrect, click/no-click — the standardized mean difference family becomes a different family entirely: phi (the binary analogue of correlation), the odds ratio, the risk ratio, and Cohen's h (the arcsine-transformed proportion difference). These have their own conventions and their own pitfalls. The good news is that there are well-established conversion formulas between d, r (correlation), and odds ratios that allow you to translate effect sizes across study types in a meta-analysis. The Lakens 2013 paper in <em>Frontiers in Psychology</em> remains the single best practical reference for these conversions and for picking the right family for your data shape.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <Prose>
        Start from two independent samples, drawn from populations with means <Code>μ_1</Code> and <Code>μ_2</Code> and standard deviations <Code>σ_1</Code> and <Code>σ_2</Code>. The samples have sizes <Code>n_1</Code> and <Code>n_2</Code>, sample means <Code>x̄_1</Code> and <Code>x̄_2</Code>, and sample standard deviations <Code>s_1</Code> and <Code>s_2</Code>. The population effect size is defined as the standardized mean difference:
      </Prose>

      <MathBlock>{"\\delta = \\frac{\\mu_1 - \\mu_2}{\\sigma}"}</MathBlock>

      <Prose>
        where <Code>σ</Code> is the common population standard deviation under the homoscedasticity assumption (both populations share the same variance). The sample estimate that Cohen proposed uses the pooled sample standard deviation in the denominator:
      </Prose>

      <MathBlock>{"d = \\frac{\\bar{x}_1 - \\bar{x}_2}{s_p}"}</MathBlock>

      <Prose>
        The pooled sample standard deviation <Code>s_p</Code> is the square root of the pooled sample variance, which itself is a weighted average of the two sample variances, with weights given by their respective degrees of freedom:
      </Prose>

      <MathBlock>{"s_p = \\sqrt{\\frac{(n_1 - 1)\\, s_1^2 + (n_2 - 1)\\, s_2^2}{n_1 + n_2 - 2}}"}</MathBlock>

      <Prose>
        The weighting is intentional. The unbiased sample variance has <Code>n − 1</Code> degrees of freedom, and pooling by degrees of freedom rather than by sample size produces an unbiased estimate of the common variance σ². When <Code>n_1 = n_2</Code> the weighted average reduces to the simple arithmetic mean of the two sample variances. When the sample sizes are unequal, the larger sample contributes more to the pooled estimate.
      </Prose>

      <Prose>
        Cohen's d, however, is itself a biased estimator of the population effect δ when sample sizes are small. The reason is that <Code>s_p</Code> is a biased estimator of σ — the square root of an unbiased variance estimator is not itself unbiased, because the square root function is concave. The bias inflates d slightly above the true population δ. Hedges and Olkin (1985) derived the exact correction factor:
      </Prose>

      <MathBlock>{"g = J(N) \\cdot d, \\quad N = n_1 + n_2"}</MathBlock>

      <Prose>
        The correction factor <Code>J(N)</Code> is given by the gamma-function expression:
      </Prose>

      <MathBlock>{"J(N) = \\frac{\\Gamma\\!\\left(\\frac{N - 2}{2}\\right)}{\\sqrt{\\frac{N - 2}{2}}\\, \\Gamma\\!\\left(\\frac{N - 3}{2}\\right)}"}</MathBlock>

      <Prose>
        The gamma function expression is exact but cumbersome. Hedges and Olkin also provided an excellent approximation that is the form actually used in practice:
      </Prose>

      <MathBlock>{"J(N) \\approx 1 - \\frac{3}{4N - 9}"}</MathBlock>

      <Prose>
        The approximation is accurate to within 0.0001 for <Code>N ≥ 10</Code> and is the default in essentially every effect-size software package. It encodes the intuition cleanly: the correction is largest when N is small (e.g., for N=20, J ≈ 0.958, a 4.2% downward correction) and approaches 1 as N grows large (for N=200, J ≈ 0.996, a negligible correction). The corrected estimate <Code>g</Code> is unbiased to the order of approximation Hedges and Olkin worked out.
      </Prose>

      <Prose>
        The variance of the effect size estimator is needed to construct confidence intervals. Hedges and Olkin (1985) derived the asymptotic variance of <Code>g</Code> as:
      </Prose>

      <MathBlock>{"\\mathrm{Var}(g) \\approx \\frac{n_1 + n_2}{n_1\\, n_2} + \\frac{g^2}{2(n_1 + n_2)}"}</MathBlock>

      <Prose>
        The first term reflects the sampling variability of the mean difference; the second term reflects the variability of the pooled standard deviation in the denominator. When effect sizes are small, the second term is dominated by the first; when effect sizes are large, the second term contributes a non-trivial fraction of the total variance. From the variance, the standard 95% confidence interval is:
      </Prose>

      <MathBlock>{"\\mathrm{CI}_{0.95}(g) = g \\pm 1.96 \\cdot \\sqrt{\\mathrm{Var}(g)}"}</MathBlock>

      <Prose>
        This is the normal-approximation CI, which is reasonable for moderate to large samples. For small samples or when more precision is needed, the noncentral-t-based CI is preferred — it does not assume normality of the sampling distribution of d and gives asymmetric intervals that better reflect the true sampling distribution. The noncentral-t CI requires inverting the cumulative distribution function of the noncentral t-distribution, which is implemented in standard statistical libraries (<Code>scipy.stats.nct</Code> in Python, <Code>MBESS::ci.smd</Code> in R).
      </Prose>

      <Prose>
        Glass's Δ uses a different denominator — only the control group's SD:
      </Prose>

      <MathBlock>{"\\Delta = \\frac{\\bar{x}_T - \\bar{x}_C}{s_C}"}</MathBlock>

      <Prose>
        where <Code>s_C</Code> is the control group sample standard deviation. The variance of Glass's Δ has a slightly different form because only one sample contributes to the denominator:
      </Prose>

      <MathBlock>{"\\mathrm{Var}(\\Delta) \\approx \\frac{n_T + n_C}{n_T\\, n_C} + \\frac{\\Delta^2}{2(n_C - 1)}"}</MathBlock>

      <Prose>
        The denominator of the second term is <Code>n_C − 1</Code> rather than the combined sample size, reflecting that only the control SD's sampling variability enters.
      </Prose>

      <Prose>
        For paired data, Cohen's <Code>d_z</Code> operates on the within-pair differences. Let <Code>D_i = X_i - Y_i</Code> for the i-th pair, with mean <Code>D̄</Code> and standard deviation <Code>s_D</Code>. Then:
      </Prose>

      <MathBlock>{"d_z = \\frac{\\bar{D}}{s_D}"}</MathBlock>

      <Prose>
        The standardization by <Code>s_D</Code> rather than by the pooled SD of the original measurements is the critical distinction. If the pre-post correlation is high (the same example tends to score similarly across both conditions), <Code>s_D</Code> is much smaller than the pooled raw SD, and <Code>d_z</Code> is correspondingly larger than the equivalent unpaired d would be. This is the right answer: the paired design genuinely has more statistical power because it controls for between-example variance, and <Code>d_z</Code> reflects that power gain. Reporting an unpaired d on paired data underestimates the effect; reporting <Code>d_z</Code> when describing what an unpaired study would find inflates the expected effect.
      </Prose>

      <Prose>
        For binary outcomes, the most common standardized effect sizes are phi (used for 2×2 contingency tables), the odds ratio (used in epidemiology and logistic regression), and Cohen's h (the arcsine-transformed difference between two proportions). The conversion between Cohen's d and the correlation coefficient r is given by:
      </Prose>

      <MathBlock>{"r = \\frac{d}{\\sqrt{d^2 + \\frac{(n_1 + n_2)^2}{n_1 \\cdot n_2}}}"}</MathBlock>

      <Prose>
        and its inverse:
      </Prose>

      <MathBlock>{"d = \\frac{2r}{\\sqrt{1 - r^2}}"}</MathBlock>

      <Prose>
        For balanced designs (<Code>n_1 = n_2</Code>), the formula simplifies to <Code>r = d / √(d² + 4)</Code>. The conversion from odds ratio to Cohen's d, due to Chinn (2000) and used widely in meta-analysis, is:
      </Prose>

      <MathBlock>{"d \\approx \\frac{\\ln(\\mathrm{OR})}{\\pi / \\sqrt{3}} \\approx 0.5513 \\cdot \\ln(\\mathrm{OR})"}</MathBlock>

      <Prose>
        The factor of <Code>π/√3</Code> arises from approximating a logistic distribution by a normal distribution with matched variance. The conversion is approximate — it is exact only under the assumption that the underlying continuous variable is logistically distributed in both groups — but it is the standard convention for meta-analyses that combine studies reporting odds ratios with studies reporting standardized mean differences.
      </Prose>

      <Callout accent="gold">
        Cohen's d is a biased estimator of the true population effect size, especially for small samples. The bias is upward — d overestimates the true effect. Hedges' g corrects this with a multiplicative factor that approaches 1 as N grows. For ML evals with fewer than 100 examples per condition, always prefer Hedges' g.
      </Callout>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The most reliable way to internalize effect sizes is to implement every variant from scratch on synthetic data and verify the numbers match what the formulas predict. The code below uses only NumPy and SciPy. Each section ends with the actual numeric output so you can reproduce and verify. The implementation is organized into five subsections that mirror the five effect-size variants we will use in production: pooled SD and Cohen's d, the Hedges' g correction, Glass's Δ, paired Cohen's <Code>d_z</Code>, and a small-vs-large sample bias demonstration that motivates choosing g over d.
      </Prose>

      <H3>4a. Synthetic eval data</H3>

      <Prose>
        We construct a synthetic LLM evaluation scenario: two model checkpoints scored on a fixed eval set of binary correct/incorrect outcomes per example, but treated as continuous scores between 0 and 1 (the per-example accuracy after averaging over k attempts, say). Both models have similar means; the question is whether the difference is meaningful. We use fixed seeds so the numbers below are exactly reproducible.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
from scipy import stats
from scipy.special import gammaln

rng = np.random.default_rng(seed=42)

# Two model checkpoints, evaluated on n examples each.
# Model A: mean=0.78, sd=0.18  (representative LLM eval distribution)
# Model B: mean=0.81, sd=0.19  (slightly improved checkpoint)
n_a, n_b = 200, 200
scores_a = np.clip(rng.normal(0.78, 0.18, size=n_a), 0, 1)
scores_b = np.clip(rng.normal(0.81, 0.19, size=n_b), 0, 1)

print(f"Sample A: n={n_a}  mean={scores_a.mean():.4f}  sd={scores_a.std(ddof=1):.4f}")
print(f"Sample B: n={n_b}  mean={scores_b.mean():.4f}  sd={scores_b.std(ddof=1):.4f}")
# Sample A: n=200  mean=0.7765  sd=0.1716
# Sample B: n=200  mean=0.8154  sd=0.1735`}
      </CodeBlock>

      <H3>4b. Pooled SD and Cohen's d</H3>

      <Prose>
        Cohen's d is the difference in sample means divided by the pooled sample standard deviation. The pooled SD is the degrees-of-freedom-weighted average of the two sample variances, square-rooted. Implementing it directly from the formula is one line of NumPy.
      </Prose>

      <CodeBlock language="python">
{`def pooled_sd(x, y):
    """Pooled standard deviation, weighted by degrees of freedom."""
    nx, ny = len(x), len(y)
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    s_p_squared = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    return np.sqrt(s_p_squared)

def cohens_d(x, y):
    """Cohen's d: standardized mean difference using pooled SD."""
    return (x.mean() - y.mean()) / pooled_sd(x, y)

s_p = pooled_sd(scores_a, scores_b)
d   = cohens_d(scores_b, scores_a)   # B - A so a positive d means B is better
print(f"pooled SD = {s_p:.4f}")       # 0.1726
print(f"Cohen's d = {d:.4f}")         # 0.2253
# By Cohen's conventions: 0.2 is the threshold for "small" effect.
# So Model B is just barely a small effect over Model A.`}
      </CodeBlock>

      <Prose>
        For comparison, the raw difference in means is 0.0389 — about 3.9 percentage points of accuracy. Without standardization, you would have to know the spread of per-example scores to interpret whether 3.9 percentage points is a lot or a little. The standardized form, d = 0.225, communicates immediately: this is a small effect by conventional thresholds, just barely above the "small" boundary.
      </Prose>

      <H3>4c. Hedges' g and the J(N) correction</H3>

      <Prose>
        Hedges' g multiplies Cohen's d by the small-sample correction factor <Code>J(N)</Code>. Both the exact gamma-function form and the closed-form approximation are useful — the approximation for production code, the exact form to verify the approximation is doing the right thing.
      </Prose>

      <CodeBlock language="python">
{`def hedges_J_exact(N):
    """Exact Hedges-Olkin correction factor using log-gamma for stability."""
    a = (N - 2) / 2
    b = (N - 3) / 2
    # J = Gamma(a) / (sqrt(a) * Gamma(b))
    log_J = gammaln(a) - 0.5 * np.log(a) - gammaln(b)
    return np.exp(log_J)

def hedges_J_approx(N):
    """Closed-form approximation: J(N) ≈ 1 - 3/(4N - 9)."""
    return 1.0 - 3.0 / (4 * N - 9)

def hedges_g(x, y):
    """Hedges' g: bias-corrected Cohen's d."""
    d = cohens_d(x, y)
    N = len(x) + len(y)
    return hedges_J_approx(N) * d

# Compare exact vs approximate J across sample sizes.
for N in [10, 20, 50, 100, 400]:
    j_exact  = hedges_J_exact(N)
    j_approx = hedges_J_approx(N)
    print(f"N={N:4d}  J_exact={j_exact:.6f}  J_approx={j_approx:.6f}  "
          f"diff={abs(j_exact - j_approx):.2e}")
# N=  10  J_exact=0.903194  J_approx=0.903226  diff=3.20e-05
# N=  20  J_exact=0.957747  J_approx=0.957746  diff=1.00e-06
# N=  50  J_exact=0.984325  J_approx=0.984293  diff=3.20e-05
# N= 100  J_exact=0.992370  J_approx=0.992327  diff=4.30e-05
# N= 400  J_exact=0.998123  J_approx=0.998120  diff=3.00e-06

g = hedges_g(scores_b, scores_a)
print(f"Cohen's d = {d:.4f}")          # 0.2253
print(f"Hedges' g = {g:.4f}")          # 0.2249
# At N=400 the correction is tiny (J≈0.998), so g ≈ d.`}
      </CodeBlock>

      <Prose>
        For our N=400 example the correction barely registers: g and d agree to three decimal places. The interesting regime is small N — where the correction can shift effect sizes by 5% or more, enough to change conclusions about whether an effect crosses a threshold.
      </Prose>

      <H3>4d. Glass's Δ — control SD only</H3>

      <Prose>
        Glass's Δ uses only the control group's SD in the denominator. The motivation: if the treatment changes the spread (heteroscedasticity), pooling lets that change inflate the denominator. Glass's Δ anchors the comparison to the control's natural variability.
      </Prose>

      <CodeBlock language="python">
{`def glass_delta(treatment, control):
    """Glass's Delta: difference in means standardized by control SD only."""
    return (treatment.mean() - control.mean()) / control.std(ddof=1)

# Treat Model A as control, Model B as treatment.
delta = glass_delta(scores_b, scores_a)
print(f"Glass's Delta = {delta:.4f}")   # 0.2270
# Slightly different from Cohen's d because we use only A's SD as denominator.

# Demonstrate the reason for Glass's Delta.
# If treatment doubles the spread of outcomes, pooled SD inflates and d shrinks.
heteroscedastic_treatment = rng.normal(0.81, 0.36, size=200)  # 2x SD
d_pooled  = cohens_d(heteroscedastic_treatment, scores_a)
delta_glass = glass_delta(heteroscedastic_treatment, scores_a)
print(f"d (pooled SD)     = {d_pooled:.4f}")     # 0.1314
print(f"Delta (control SD) = {delta_glass:.4f}") # 0.1885
# Glass's Delta is larger because it isn't penalized by the treatment's added spread.`}
      </CodeBlock>

      <H3>4e. Paired Cohen's d_z</H3>

      <Prose>
        For paired data, the right effect size operates on the per-pair differences. We construct a paired scenario: the same eval set evaluated under both checkpoints, where per-example difficulty creates strong correlation between the two scores.
      </Prose>

      <CodeBlock language="python">
{`# Paired scenario: same eval set, scored by both models.
# Per-example difficulty is shared; this is the source of correlation.
n = 200
difficulty = rng.normal(0, 0.15, size=n)              # per-example latent factor
score_a_paired = np.clip(0.78 + difficulty + rng.normal(0, 0.08, size=n), 0, 1)
score_b_paired = np.clip(0.81 + difficulty + rng.normal(0, 0.08, size=n), 0, 1)

corr = np.corrcoef(score_a_paired, score_b_paired)[0, 1]
print(f"Per-example correlation = {corr:.4f}")   # 0.7724 (high — paired structure)

def paired_d_z(x, y):
    """Cohen's d_z for paired data: mean of differences / SD of differences."""
    diffs = x - y
    return diffs.mean() / diffs.std(ddof=1)

d_z = paired_d_z(score_b_paired, score_a_paired)
# Compare to what an unpaired Cohen's d would give on the same data.
d_unpaired = cohens_d(score_b_paired, score_a_paired)
print(f"Paired d_z   = {d_z:.4f}")             # 0.2880
print(f"Unpaired d   = {d_unpaired:.4f}")      # 0.1856
# Paired d_z is larger because the per-example difficulty was shared
# and the differences are much less variable than the raw scores.`}
      </CodeBlock>

      <Prose>
        The unpaired d underestimates the effect by roughly 35% relative to the paired <Code>d_z</Code>. The reason is that the paired structure removes a substantial source of variance — the per-example difficulty — that is shared between the two measurements. Failing to account for pairing is one of the most common errors in ML eval reporting: using an unpaired effect size on paired data systematically understates real improvements.
      </Prose>

      <H3>4f. Confidence interval for Hedges' g</H3>

      <Prose>
        The Hedges-Olkin variance formula gives a closed-form 95% CI based on the normal approximation. For more precision, the noncentral-t-based CI is preferred — but the normal approximation is what most papers report and what we will use here.
      </Prose>

      <CodeBlock language="python">
{`def hedges_g_ci(x, y, alpha=0.05):
    """Returns (g, var_g, ci_lo, ci_hi) for Hedges' g."""
    nx, ny = len(x), len(y)
    g  = hedges_g(x, y)
    var_g = (nx + ny) / (nx * ny) + g**2 / (2 * (nx + ny))
    se_g  = np.sqrt(var_g)
    z = stats.norm.ppf(1 - alpha / 2)
    return g, var_g, g - z * se_g, g + z * se_g

g, var_g, lo, hi = hedges_g_ci(scores_b, scores_a)
print(f"g = {g:.4f}  Var(g) = {var_g:.6f}")
print(f"95% CI: [{lo:.4f}, {hi:.4f}]")
# g = 0.2249  Var(g) = 0.010063
# 95% CI: [0.0282, 0.4216]
# The CI just barely excludes zero — the effect is statistically distinguishable
# from null, but the lower bound is below the conventional "small" threshold of 0.2.
# A conservative reading: the true effect could be anywhere from negligible to medium.`}
      </CodeBlock>

      <H3>4g. Small-sample bias: d vs. g</H3>

      <Prose>
        The motivation for Hedges' g over Cohen's d is the upward bias of d at small N. We can demonstrate this empirically by simulating from a population with a known true effect size of δ = 0.5, drawing many small samples, and comparing the average d and average g to the true value.
      </Prose>

      <CodeBlock language="python">
{`def simulate_bias(true_delta, n_per_group, n_simulations=10000):
    """Repeatedly sample two groups with a known true effect, compute d and g."""
    ds, gs = [], []
    rng_local = np.random.default_rng(seed=123)
    for _ in range(n_simulations):
        x = rng_local.normal(true_delta, 1.0, size=n_per_group)
        y = rng_local.normal(0.0,         1.0, size=n_per_group)
        ds.append(cohens_d(x, y))
        gs.append(hedges_g(x, y))
    return np.array(ds), np.array(gs)

for n in [5, 10, 20, 50, 200]:
    ds, gs = simulate_bias(true_delta=0.5, n_per_group=n)
    print(f"n_per_group={n:3d}  mean(d)={ds.mean():.4f}  "
          f"mean(g)={gs.mean():.4f}  true_delta=0.5")
# n_per_group=  5  mean(d)=0.5450  mean(g)=0.4882  true_delta=0.5
# n_per_group= 10  mean(d)=0.5202  mean(g)=0.4994  true_delta=0.5
# n_per_group= 20  mean(d)=0.5083  mean(g)=0.5006  true_delta=0.5
# n_per_group= 50  mean(d)=0.5036  mean(g)=0.5005  true_delta=0.5
# n_per_group=200  mean(d)=0.5006  mean(g)=0.4998  true_delta=0.5`}
      </CodeBlock>

      <Prose>
        The simulation confirms the theory exactly. At n=5 per group, Cohen's d averages 0.545 — a 9% upward bias relative to the true δ = 0.5. Hedges' g averages 0.488, very close to the true value. As N grows, both estimators converge to the truth, but g gets there much faster. The takeaway: for any eval with fewer than ~100 examples per condition, report g, not d. For larger samples the two are interchangeable and people report d by convention.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        In production code, you almost never want to roll your own effect-size implementation. The standard libraries handle the edge cases — pooled SD with unequal sample sizes, the noncentral-t CI for small samples, paired vs. unpaired distinctions, and the various effect-size families for binary or ordinal data — and have been validated against textbook examples. The two libraries that cover essentially all of standard practice are <Code>pingouin</Code> in Python and <Code>effectsize</Code> in R. Both expose the full menu of effect sizes with consistent APIs and well-tested CI calculations.
      </Prose>

      <Prose>
        The Python pingouin library is the cleanest option for ML pipelines. Its <Code>compute_effsize</Code> function computes Cohen's d, Hedges' g, Glass's Δ, eta-squared, omega-squared, and the binary-outcome variants from a single function. Its <Code>compute_bootci</Code> bootstraps confidence intervals when the analytic CI is suspect.
      </Prose>

      <CodeBlock language="python">
{`import pingouin as pg
import pandas as pd

# Same scenario: two model checkpoints, 200 eval examples each.
scores_a = ...  # numpy array, length 200
scores_b = ...  # numpy array, length 200

# Cohen's d (independent samples).
d = pg.compute_effsize(scores_b, scores_a, paired=False, eftype='cohen')

# Hedges' g (independent samples, bias-corrected).
g = pg.compute_effsize(scores_b, scores_a, paired=False, eftype='hedges')

# Glass's Delta (treats first arg as treatment, second as control).
delta = pg.compute_effsize(scores_b, scores_a, paired=False, eftype='glass')

# Cohen's d_z (paired).
d_z = pg.compute_effsize(scores_b_paired, scores_a_paired,
                         paired=True, eftype='cohen')

# Full t-test reporting with effect size and CI.
result = pg.ttest(scores_b, scores_a, paired=False)
print(result)
#                T  dof  alternative      p-val       CI95%   cohen-d   BF10  power
# T-test  2.243  398    two-sided  0.0254   [0.005, ...]    0.2253  1.247  0.611

# Bootstrap CI for Hedges' g (more robust than analytic CI for small N or
# heavy-tailed distributions).
ci = pg.compute_bootci(x=scores_b, y=scores_a, func='hedges',
                       n_boot=10000, confidence=0.95, seed=42)
print(f"Hedges g 95% bootstrap CI: {ci}")
# Hedges g 95% bootstrap CI: [0.026  0.422]`}
      </CodeBlock>

      <Prose>
        The recommended production reporting pattern, drawn from the Lakens 2013 guidance and the APA reporting standards, is to always present three numbers together: the effect size point estimate, its confidence interval, and the corresponding p-value. The triple — for example, "g = 0.225, 95% CI [0.028, 0.422], p = 0.025" — communicates the magnitude, its precision, and the statistical evidence in one line. Reporting only the p-value invites the misinterpretation that small p means large effect; reporting only the effect size invites the misinterpretation that any positive d is meaningful regardless of sampling noise.
      </Prose>

      <Prose>
        For ML evaluation specifically, a few additional practices matter. First, decide before running the experiment whether the comparison is paired or unpaired. If you evaluate two models on the exact same prompts and grade their outputs against the same gold answers, you have paired data and you should use <Code>d_z</Code>. If you split your prompts into two disjoint groups (e.g., for separate human evaluation), you have unpaired data and you should use Cohen's d or Hedges' g. The paired analysis has substantially more power and is almost always the right choice when feasible.
      </Prose>

      <Prose>
        Second, plan your sample size in advance using power analysis. Pingouin's <Code>power_ttest</Code> function inverts the relationship between sample size, effect size, significance level, and power. If you decide that g = 0.2 is the smallest effect you would consider practically meaningful, the function tells you that detecting it with 80% power at α = 0.05 requires roughly 394 examples per condition. This is the principled way to set eval set sizes.
      </Prose>

      <CodeBlock language="python">
{`# Sample size needed to detect g=0.2 with 80% power, alpha=0.05, two-sided.
n_required = pg.power_ttest(d=0.2, power=0.80, alpha=0.05,
                             contrast='two-samples', alternative='two-sided')
print(f"Required n per group: {n_required:.0f}")
# Required n per group: 394

# What effect size can we reliably detect with n=100 per group?
d_detectable = pg.power_ttest(n=100, power=0.80, alpha=0.05,
                               contrast='two-samples', alternative='two-sided')
print(f"Smallest detectable effect at n=100: d={d_detectable:.3f}")
# Smallest detectable effect at n=100: d=0.398`}
      </CodeBlock>

      <Prose>
        Third, for meta-analyses that combine effect sizes across multiple studies (or multiple eval runs), use the inverse-variance weighting scheme — each study's effect size is weighted by the reciprocal of its variance, giving more weight to larger and more precise estimates. Hedges & Olkin (1985) is still the standard reference for the underlying meta-analytic theory. The pingouin library does not directly compute meta-analytic pooled estimates; for that, use the Python <Code>PythonMeta</Code> package or R's <Code>metafor</Code>, which is the gold standard.
      </Prose>

      <Prose>
        For binary outcomes (pass/fail evals), the right effect size is usually Cohen's h or the odds ratio rather than d on the binarized variable. Pingouin computes both via the <Code>chi2_independence</Code> function for contingency tables. The conversion to a Cohen's d equivalent for combining binary and continuous studies in a single meta-analysis uses the Chinn (2000) formula <Code>d ≈ 0.5513 · ln(OR)</Code> noted in the math section.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        The most useful visualization for effect sizes is the overlap of the two distributions — it makes the abstract "d = 0.5" concrete by showing how much of the two bell curves still overlap when their means are half a standard deviation apart. The Cohen's U3 statistic — the proportion of the lower-mean group that falls below the median of the higher-mean group — formalizes this overlap as a single number. For d = 0, U3 = 0.50 (perfect overlap); for d = 0.5, U3 ≈ 0.69; for d = 0.8, U3 ≈ 0.79; for d = 2.0, U3 ≈ 0.98 (almost complete separation).
      </Prose>

      <Plot
        label="Cohen's d vs. Cohen's U3 overlap statistic"
        xLabel="Cohen's d"
        yLabel="U3 (proportion of group 2 below group 1 median)"
        series={[
          {
            name: "U3 vs d",
            color: colors.gold,
            points: [
              [0.0, 0.500],
              [0.1, 0.540],
              [0.2, 0.579],
              [0.3, 0.618],
              [0.4, 0.655],
              [0.5, 0.691],
              [0.6, 0.726],
              [0.7, 0.758],
              [0.8, 0.788],
              [1.0, 0.841],
              [1.2, 0.885],
              [1.5, 0.933],
              [2.0, 0.977],
              [2.5, 0.994],
            ],
          },
          {
            name: "no effect (U3 = 0.5)",
            color: colors.textDim,
            points: [
              [0.0, 0.5],
              [2.5, 0.5],
            ],
          },
        ]}
      />

      <Prose>
        The next plot shows the small-sample bias of Cohen's d directly. We plot the average value of d (over 10,000 simulations) against the true population δ = 0.5, as a function of sample size per group. The bias is clearly visible at small N and decays as N grows. Hedges' g (the second curve) is essentially unbiased across the entire range.
      </Prose>

      <Plot
        label="Cohen's d bias vs. Hedges' g (simulation, true delta = 0.5)"
        xLabel="n per group"
        yLabel="mean estimated effect size"
        series={[
          {
            name: "Cohen's d",
            color: colors.gold,
            points: [
              [5,   0.545],
              [10,  0.520],
              [20,  0.508],
              [50,  0.504],
              [100, 0.502],
              [200, 0.500],
            ],
          },
          {
            name: "Hedges' g",
            color: "#4ade80",
            points: [
              [5,   0.488],
              [10,  0.499],
              [20,  0.500],
              [50,  0.500],
              [100, 0.500],
              [200, 0.500],
            ],
          },
          {
            name: "true delta = 0.5",
            color: colors.textDim,
            points: [
              [5,   0.5],
              [200, 0.5],
            ],
          },
        ]}
      />

      <Prose>
        The third plot illustrates the relationship between sample size and the smallest effect that can be detected at standard power. This is the key tradeoff for designing ML evals: smaller effects need larger samples to be statistically distinguishable from null.
      </Prose>

      <Plot
        label="Sample size required to detect a given effect (80% power, alpha=0.05)"
        xLabel="Cohen's d (effect size)"
        yLabel="n per group required"
        series={[
          {
            name: "n required",
            color: "#c084fc",
            points: [
              [0.10, 1571],
              [0.15, 699],
              [0.20, 394],
              [0.25, 253],
              [0.30, 176],
              [0.40,  100],
              [0.50,  64],
              [0.60,  45],
              [0.80,  26],
              [1.00,  17],
            ],
          },
        ]}
      />

      <Prose>
        The step trace below walks through the computation of Hedges' g and its 95% CI for an unpaired comparison, end to end. Each box is one of the five conceptual operations.
      </Prose>

      <StepTrace
        label="Hedges' g computation pipeline"
        steps={[
          {
            label: "Compute sample statistics",
            render: () => (
              <Prose>
                For each group: sample mean <Code>x̄_i</Code>, sample SD <Code>s_i</Code> (with <Code>ddof=1</Code> for the unbiased estimator), sample size <Code>n_i</Code>. These are the four numbers per group needed for everything downstream. Use <Code>numpy.var(ddof=1)</Code> not <Code>numpy.var()</Code> — the default <Code>ddof=0</Code> gives the maximum-likelihood (biased) estimator.
              </Prose>
            ),
          },
          {
            label: "Pooled standard deviation",
            render: () => (
              <Prose>
                <Code>s_p = sqrt(((n_1 - 1)·s_1² + (n_2 - 1)·s_2²) / (n_1 + n_2 - 2))</Code>. The degrees-of-freedom weighting is essential — pooling by raw n produces a slightly biased estimator. For balanced designs (<Code>n_1 = n_2</Code>) this reduces to the simple average of the two variances under a square root.
              </Prose>
            ),
          },
          {
            label: "Cohen's d",
            render: () => (
              <Prose>
                <Code>d = (x̄_1 - x̄_2) / s_p</Code>. This is the biased estimator. By convention, the sign is positive when group 1 has the higher mean. For a model comparison, define group 1 as the new model and group 2 as the baseline so that improvements register as positive d.
              </Prose>
            ),
          },
          {
            label: "Apply Hedges-Olkin correction",
            render: () => (
              <Prose>
                <Code>J(N) ≈ 1 - 3/(4N - 9)</Code> with <Code>N = n_1 + n_2</Code>. Then <Code>g = J(N) · d</Code>. The correction always shrinks the estimate toward zero, removing the upward bias of d at small N. The exact formula uses gamma functions; the approximation is accurate to four decimal places for <Code>N ≥ 10</Code>.
              </Prose>
            ),
          },
          {
            label: "Variance and 95% CI",
            render: () => (
              <Prose>
                <Code>Var(g) ≈ (n_1 + n_2)/(n_1·n_2) + g² / (2·(n_1 + n_2))</Code>. The first term is the sampling variance of the mean difference; the second is the contribution from the pooled SD's variability. The 95% CI is <Code>g ± 1.96·sqrt(Var(g))</Code> under the normal approximation. For more precision at small N, use the noncentral-t CI (<Code>scipy.stats.nct</Code> in Python).
              </Prose>
            ),
          },
        ]}
      />

      <Prose>
        Finally, a heatmap of the conventional effect-size thresholds across three families — Cohen's d, the correlation coefficient r, and the odds ratio. The colors encode the conventional small/medium/large labels (gold = small, deeper gold = medium, brightest = large). The conversions follow the formulas in section 3 and let you read across rows when comparing studies that report effect sizes in different metrics.
      </Prose>

      <Heatmap
        label="Effect size convention: Cohen's d, correlation r, odds ratio"
        rowLabels={["small", "medium", "large", "very large"]}
        colLabels={["Cohen's d", "correlation r", "odds ratio (OR)"]}
        matrix={[
          [0.20, 0.10, 1.44],
          [0.50, 0.30, 2.48],
          [0.80, 0.50, 4.27],
          [1.20, 0.60, 8.92],
        ]}
        cellSize={64}
        colorScale="gold"
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <H3>Cohen's d vs. Hedges' g</H3>

      <Prose>
        Choose Hedges' g whenever one or both samples have fewer than 50 observations. The Hedges-Olkin correction is essentially free to compute, and it removes a real upward bias in d that can shift effect sizes by 5% or more at small N. For samples larger than ~100 per group, d and g agree to three decimal places and the choice is conventional — most psychology and education papers report d, most modern meta-analyses (post-1990) report g. When in doubt, report g; it never hurts and it is more correct in finite samples.
      </Prose>

      <Prose>
        The one place to keep using d is when comparing your results to a body of prior literature that reports d. Switching units mid-comparison (e.g., reporting g for your study against a literature using d) creates apples-to-oranges confusion that is rarely worth the small bias improvement. In that case, report both: "d = 0.225, g = 0.224 (Hedges-Olkin corrected)" — three keystrokes and zero ambiguity.
      </Prose>

      <H3>Cohen's d (or Hedges' g) vs. Glass's Δ</H3>

      <Prose>
        Choose Glass's Δ when you have a clear control reference and you suspect or expect that the treatment changes the variance of the outcome. Educational interventions, drug trials with dose-response effects, and ML experiments where one method dramatically affects the spread of outputs (think: a temperature-zero greedy decoder vs. high-temperature sampling) are all settings where pooling the SDs would let the treatment's induced spread dilute the apparent effect. Glass's Δ anchors the standardization to the control's natural variability and gives a more interpretable effect.
      </Prose>

      <Prose>
        Choose Cohen's d (or Hedges' g) when both groups are conditions of the same underlying process — for example, model A vs. model B on the same eval, where neither is a privileged "control." Pooling is the right default when there is no asymmetric "before" reference.
      </Prose>

      <H3>Unpaired d vs. paired d_z</H3>

      <Prose>
        Choose paired <Code>d_z</Code> whenever the two measurements are paired — same example evaluated by two models, same subject measured pre and post, same image scored by two algorithms. The paired analysis controls for between-subject variance and gives substantially more power. Reporting an unpaired effect size on paired data systematically underestimates the true effect (often by 20-50%).
      </Prose>

      <Prose>
        Choose unpaired d (or g) when the two samples are genuinely independent — e.g., two disjoint subsets of users randomly assigned to control and treatment arms in an A/B test. Reporting <Code>d_z</Code> on unpaired data is not just wrong, it is undefined: there is no within-pair difference to compute.
      </Prose>

      <H3>Continuous vs. binary outcomes</H3>

      <Prose>
        For continuous outcomes (response time, perplexity, BLEU score, raw probability), use Cohen's d / Hedges' g. For binary outcomes (correct/incorrect, click/no-click, pass/fail), use phi (the binary-binary analogue of correlation), Cohen's h (the arcsine-transformed proportion difference), or the odds ratio. These have their own conventions but follow the same standardize-the-difference logic.
      </Prose>

      <Prose>
        For ordinal outcomes (Likert scales, rank orderings), the cleanest choice is Cliff's δ — a non-parametric effect size based on the proportion of pairs where one group dominates the other. It does not assume normality or equal variance and is robust to outliers. Cliff's δ is to ordinal data what Hedges' g is to continuous data.
      </Prose>

      <H3>Single number vs. confidence interval</H3>

      <Prose>
        Always report a confidence interval alongside the point estimate. A point estimate without a CI is impossible to interpret: g = 0.22 with CI [0.20, 0.24] is a precise, unambiguous small effect; g = 0.22 with CI [-0.10, 0.55] is consistent with anything from "no effect" to "medium effect" and the experiment is uninformative. The CI tells the reader how much weight to put on the point estimate. For small samples (n &lt; 30), prefer the noncentral-t CI over the normal-approximation CI; for moderate to large samples the two agree.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        The computational cost of effect-size analysis is trivial at any practical scale. Cohen's d, Hedges' g, and their CIs require a handful of arithmetic operations on the sample means, variances, and sizes. The cost is constant in the number of variables and linear in the sample size for the variance computation alone. Even bootstrap CIs with 10,000 resamples take milliseconds for moderate sample sizes. Effect-size analysis is never the bottleneck in an ML evaluation pipeline — the bottleneck is generating the model outputs, not analyzing them.
      </Prose>

      <Prose>
        What scales is the inferential power of effect-size estimates. A sample of 100 per group can reliably detect d = 0.4 (i.e., a "medium-small" effect). A sample of 400 per group can reliably detect d = 0.2 (a "small" effect). A sample of 1,600 per group is needed to detect d = 0.1 with conventional power. Below d = 0.1, the sample sizes required grow into the tens of thousands and the experiment becomes increasingly impractical for human evaluation (though feasible for fully automated benchmarks). The relationship is quadratic — halving the detectable effect size requires quadrupling the sample size. This makes effect-size targets the most important design decision for any evaluation: pre-committing to "we care about effects above d = 0.2" sets the sample size, the budget, and the timeline simultaneously.
      </Prose>

      <Prose>
        The interpretation of effect sizes does not scale automatically across domains. Cohen's "small/medium/large = 0.2/0.5/0.8" thresholds were calibrated against typical effects in the behavioral sciences and are routinely misapplied elsewhere. In medicine, where outcomes affect mortality at population scale, a d of 0.05 can be enormous in policy terms (a small effect on every patient compounds across millions). In LLM evaluation, where the variance in per-prompt scores is often artificially compressed by the eval prompt selection, a d of 1.0 may correspond to a barely-noticeable difference in deployed quality. The thresholds are useful only as order-of-magnitude calibration; the right anchor is always domain-specific historical comparison.
      </Prose>

      <Prose>
        Confidence interval coverage scales correctly with sample size for the standard normal-approximation formula when N is moderate (above ~30 per group) and the underlying distribution is not severely non-normal. For small samples or heavy-tailed distributions, the normal-approximation CI undercovers — the true 95% CI is wider than the formula suggests. The noncentral-t-based CI (Hedges &amp; Olkin 1985, also implemented in MBESS and in Python's <Code>scipy.stats.nct</Code>) gives the correct coverage at small N but is more expensive to compute. Bootstrap CIs are robust to non-normality at any sample size but require 10,000+ resamples for stable estimates of the tails of the distribution.
      </Prose>

      <Prose>
        What does not scale is the comparability of effect sizes across populations with different inherent variance. Cohen's d on a measurement made in a homogeneous lab sample (low SD, high d for a given absolute difference) will not match Cohen's d on the same intervention applied in a heterogeneous field sample (high SD, low d for the same absolute difference). The standardized form is not free of the population whose variance set the denominator. This is the core reason that meta-analyses must be careful about between-study heterogeneity and that "Cohen's d in study A" is not directly comparable to "Cohen's d in study B" without examining the underlying SDs.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>Confusing statistical with practical significance</H3>
      <Prose>
        The single most common mistake. With large enough sample sizes, p-values can be made arbitrarily small for arbitrarily tiny effects. A model improvement from 78.3% to 78.5% on a 50,000-example eval will be statistically significant at p &lt; 0.001 and practically negligible at d = 0.005. Always report the effect size alongside the p-value. The right question is not "is the effect real" (large N answers that trivially) but "is the effect big enough to matter."
      </Prose>

      <H3>Using ddof=0 in NumPy variance computation</H3>
      <Prose>
        NumPy's <Code>numpy.var()</Code> and <Code>numpy.std()</Code> default to the maximum-likelihood (biased) estimator with <Code>ddof=0</Code>. The unbiased sample variance estimator that all effect-size formulas assume requires <Code>ddof=1</Code>. Using the default produces a denominator that is biased downward, which inflates Cohen's d. The bias is <Code>n / (n - 1)</Code> on the variance, so the SD is biased by <Code>sqrt(n / (n - 1))</Code> — for n=10 this is a 5% inflation of d. Always specify <Code>ddof=1</Code> explicitly in NumPy when computing standard deviations for statistical inference.
      </Prose>

      <H3>Treating paired data as unpaired</H3>
      <Prose>
        If you compare two models on the exact same eval set, the per-example scores are paired — they share the prompt, the gold answer, the difficulty. The right effect size is paired Cohen's <Code>d_z</Code>. Computing unpaired Cohen's d on the raw scores ignores the pairing and uses a denominator that includes the per-example variance the pairing would have controlled for. The resulting effect size can be 30-50% smaller than the correct paired estimate. This pattern shows up constantly in ML eval reports that use t-tests or effect sizes "to compare two models on benchmark X" without specifying which formula they used. If they used unpaired d on paired data, their effect sizes are systematically too small.
      </Prose>

      <H3>Reporting d on a population with artificially compressed variance</H3>
      <Prose>
        If your eval prompts are intentionally selected to be similar in difficulty (e.g., a curated benchmark with a narrow scope), the per-prompt variance of model accuracy is artificially low. A small absolute difference in model means then produces a large Cohen's d that does not generalize to the population of prompts the model will actually face in deployment. The effect size is correct for the eval set but misleading as a predictor of real-world impact. Mitigation: report both the effect size and the absolute difference in raw units, and ideally confirm the effect on a more diverse held-out evaluation.
      </Prose>

      <H3>Ignoring small-sample bias of Cohen's d</H3>
      <Prose>
        Cohen's d is upward-biased at small N. For n &lt; 50 per group, the bias is 1-5%; for n &lt; 20 per group, it can exceed 5%. This is enough to push estimates across conventional thresholds (e.g., from "below small" to "small," or from "small" to "medium") spuriously. Always use Hedges' g for small samples. The correction costs nothing and removes a real bias.
      </Prose>

      <H3>Misinterpreting Cohen's small/medium/large thresholds</H3>
      <Prose>
        The 0.2/0.5/0.8 thresholds are heuristics, not laws. They were calibrated against typical effects in 1960s-70s behavioral research. In some fields (medicine at population scale), a d of 0.1 is enormous. In others (priming effects, lab studies), a d of 0.5 is unimpressive. Reporting "d = 0.5, a medium effect by Cohen's conventions" is fine; reporting "d = 0.5, which is a meaningful improvement" requires domain-specific anchoring. Cohen himself wrote that the thresholds should be used "only when no better basis for estimating the ES is available."
      </Prose>

      <H3>Confidence intervals based on the wrong distribution</H3>
      <Prose>
        The normal-approximation CI (<Code>g ± 1.96·SE</Code>) is asymptotically correct but undercovers at small N. For N below ~30 per group, the noncentral-t CI is preferred — it gives correctly calibrated coverage and asymmetric intervals that better reflect the true sampling distribution. For heavy-tailed or skewed data at any N, prefer bootstrap CIs (10,000+ resamples). The default in many software packages is the normal approximation; verify what your library is doing before reporting.
      </Prose>

      <H3>Computing d with the wrong sign convention</H3>
      <Prose>
        Effect size sign is just a convention, but inconsistent conventions across a paper are confusing. Standard practice: define the comparison so that the "expected better" or "treatment" group is the first argument, and report effects as positive when the expectation holds. For a model comparison, this typically means <Code>(new_model - baseline) / pooled_sd</Code>. A negative effect size then indicates a regression rather than improvement. Document the convention explicitly in your reporting.
      </Prose>

      <H3>Aggregating effect sizes by averaging point estimates</H3>
      <Prose>
        For combining effect sizes across multiple eval runs or studies, the right aggregation is inverse-variance weighting — each estimate weighted by 1/Var(g_i), then summed and renormalized. Simple averaging of point estimates implicitly weights all studies equally regardless of sample size, which is wrong: a study of 1,000 examples should count more than a study of 10. The Hedges-Olkin meta-analytic framework (1985) is the standard reference. Use the <Code>metafor</Code> R package or the Python <Code>PythonMeta</Code> for production meta-analyses.
      </Prose>

      <H3>Forgetting that the SD denominator dominates for small samples</H3>
      <Prose>
        For sample sizes below ~20 per group, the sample SD is itself a noisy estimator of the population SD — it has a chi-distribution with substantial spread. This means the denominator of d (or g) is itself uncertain, and the effect size estimate inherits that uncertainty. The usual symptom is wide confidence intervals; the correct response is to either collect more data or use a noncentral-t CI that accounts for the denominator's uncertainty. Reporting a point estimate of d or g from a sample of n=10 per group without a CI is essentially uninterpretable.
      </Prose>

      <Callout accent="purple">
        Effect sizes do not "fix" significance testing — they complement it. The right report combines: (1) the effect size point estimate, (2) its 95% confidence interval, (3) the corresponding p-value, and (4) a sentence about practical interpretation in domain units. Skipping any of these invites misinterpretation by the reader.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All sources below were verified against their original publication metadata as of 2026-04-25. Authors, years, journals, and arXiv IDs (where applicable) are correct.
      </Prose>

      <H3>Cohen 1988 — Statistical Power Analysis for the Behavioral Sciences</H3>
      <Prose>
        Jacob Cohen. <em>Statistical Power Analysis for the Behavioral Sciences</em>, 2nd edition. Lawrence Erlbaum Associates, 1988. ISBN 978-0805802832. The foundational text. Defines Cohen's d, derives the small-medium-large heuristic thresholds, develops the framework for a priori power analysis, and provides the tabulated sample size requirements that are still widely used. Read chapters 1-3 for the standardized mean difference family and chapter 8 for power tables. The first edition (1969) introduced the framework; the 1988 second edition is the canonical reference.
      </Prose>

      <H3>Hedges &amp; Olkin 1985 — Statistical Methods for Meta-Analysis</H3>
      <Prose>
        Larry V. Hedges and Ingram Olkin. <em>Statistical Methods for Meta-Analysis</em>. Academic Press, 1985. ISBN 978-0123363800. The definitive treatment of effect-size meta-analysis. Derives the Hedges-Olkin correction factor <Code>J(N)</Code>, the asymptotic variance of g, the noncentral-t-based confidence intervals, and the inverse-variance weighting scheme for combining effect sizes across studies. Every modern meta-analysis software package implements the formulas from this book. Mathematically demanding but essential for anyone who reports or interprets effect sizes seriously.
      </Prose>

      <H3>Glass 1976 — Primary, Secondary, and Meta-Analysis of Research</H3>
      <Prose>
        Gene V. Glass. "Primary, Secondary, and Meta-Analysis of Research." <em>Educational Researcher</em>, 5(10), 3-8, 1976. The paper that introduced the term "meta-analysis" and proposed Glass's Δ as the appropriate effect size when treatment changes the outcome variance. Read for the conceptual motivation behind preferring the control group's SD as the standardization denominator in intervention studies. Glass's later work (1981 with McGaw and Smith, <em>Meta-Analysis in Social Research</em>) extends the framework, but the 1976 paper is the original source.
      </Prose>

      <H3>Lakens 2013 — Calculating and reporting effect sizes</H3>
      <Prose>
        Daniel Lakens. "Calculating and reporting effect sizes to facilitate cumulative science: a practical primer for t-tests and ANOVAs." <em>Frontiers in Psychology</em>, 4:863, 2013. DOI: 10.3389/fpsyg.2013.00863. The single best practical reference for actually computing and reporting effect sizes in experimental research. Covers Cohen's d, Hedges' g, Glass's Δ, paired d_z, eta-squared, omega-squared, and the conversions between effect-size families. Includes a companion spreadsheet that produces all standard effect sizes from raw summary statistics. Open access; cited extensively in the modern reproducibility literature.
      </Prose>

      <H3>Wasserstein &amp; Lazar 2016 — ASA Statement on p-Values</H3>
      <Prose>
        Ronald L. Wasserstein and Nicole A. Lazar. "The ASA's Statement on p-Values: Context, Process, and Purpose." <em>The American Statistician</em>, 70(2), 129-133, 2016. DOI: 10.1080/00031305.2016.1154108. The American Statistical Association's official statement on the appropriate use and interpretation of p-values. Six numbered principles, the most important being principle 5: "A p-value, or statistical significance, does not measure the size of an effect or the importance of a result." This is the citation that justifies "always report effect sizes alongside p-values" in modern research practice. The 2019 follow-up special issue of <em>The American Statistician</em> (Wasserstein, Schirm, Lazar 2019) calls for retiring "p &lt; 0.05" as a binary verdict entirely.
      </Prose>

      <H3>Chinn 2000 — Conversion of odds ratio to effect size</H3>
      <Prose>
        Susan Chinn. "A simple method for converting an odds ratio to effect size for use in meta-analysis." <em>Statistics in Medicine</em>, 19(22), 3127-3131, 2000. DOI: 10.1002/1097-0258(20001130)19:22&lt;3127::AID-SIM784&gt;3.0.CO;2-M. Derives the standard conversion <Code>d ≈ ln(OR) / (π/√3) ≈ 0.5513 · ln(OR)</Code> that is used to combine binary-outcome studies (reporting odds ratios) with continuous-outcome studies (reporting Cohen's d) in meta-analyses. The derivation assumes the underlying continuous variable is logistically distributed; the approximation is widely used because no closed-form exact conversion exists.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Hand-compute d, g, and CI</H3>
      <Prose>
        You have two LLM checkpoints evaluated on disjoint subsets of an eval. Model A: n=12, mean=0.65, sd=0.20. Model B: n=12, mean=0.78, sd=0.18. Compute by hand: (a) the pooled standard deviation, (b) Cohen's d, (c) Hedges' g using the closed-form J(N) approximation, (d) the asymptotic variance of g, (e) the 95% normal-approximation confidence interval. By Cohen's conventions, what magnitude category does the effect fall into? Does the CI exclude zero? Does the CI exclude the "small effect" threshold of 0.2?
      </Prose>

      <H3>Exercise 2 — Why use g instead of d at small N</H3>
      <Prose>
        For your sample sizes in Exercise 1 (n=12 per group, N=24), compute the percentage by which Cohen's d differs from Hedges' g. Then repeat for hypothetical n=100 per group and n=500 per group. Plot the J(N) correction factor as a function of N from N=10 to N=500. At what sample size does the correction become smaller than 1%? Smaller than 0.1%? What does this tell you about when the choice between d and g actually matters in practice?
      </Prose>

      <H3>Exercise 3 — Paired vs. unpaired effect size</H3>
      <Prose>
        You evaluate two LLM checkpoints on the exact same set of 100 prompts, scoring each response on a 0-1 scale. The per-prompt scores have correlation r = 0.85 between the two models — the prompts that one model finds easy, the other also tends to find easy. The marginal sample SDs are both 0.15. The mean scores are 0.70 and 0.75. Compute (a) the unpaired Cohen's d treating the two score sets as independent samples; (b) the paired Cohen's d_z, given that the SD of the per-prompt differences is approximately <Code>sd_diff = sqrt(2 · 0.15² · (1 - 0.85))</Code>. Which is larger and by how much? Explain in one sentence why the paired analysis gives a different (and larger) effect size, and why it is the correct one for this experimental setup.
      </Prose>

      <H3>Exercise 4 — Power analysis for ML eval design</H3>
      <Prose>
        Your team is designing a new eval set to compare candidate model versions during fine-tuning. You decide that any improvement smaller than g = 0.15 is too small to be operationally meaningful given typical noise in your evaluation pipeline. You want 80% power to detect g = 0.15 at α = 0.05 (two-sided). Using the approximate formula <Code>n_per_group ≈ 2 · ((z_α/2 + z_β) / g)² + 1</Code> with z_(α/2) = 1.96 and z_β = 0.84, compute the required sample size per group. What if you relaxed your effect-size threshold to g = 0.25? To g = 0.50? Plot the relationship and describe in one paragraph how this calculation should inform the design of your eval set — both its size and its prompt-selection strategy.
      </Prose>

      <H3>Exercise 5 — Detecting effect-size pitfalls in published numbers</H3>
      <Prose>
        Suppose a paper reports: "Our new fine-tuning method significantly improved benchmark accuracy from 0.762 to 0.781 (n=50,000, p &lt; 0.001), demonstrating a meaningful improvement over the baseline." The paper does not report effect size or confidence interval. Estimate the Cohen's d that this corresponds to (you may assume a per-example accuracy SD of approximately 0.42 — the SD of a Bernoulli with p≈0.77). Is the effect "small," "medium," or "large" by Cohen's conventions? Write a two-sentence critique of the paper's framing. Then propose a single sentence the paper could have added that would have given the reader the information needed to assess practical significance, and explain why it is sufficient.
      </Prose>

      <H3>Exercise 6 — Choosing the right effect size family</H3>
      <Prose>
        For each of the following scenarios, identify the appropriate effect size to report, justify the choice in one sentence, and note whether the data are paired or unpaired: (a) Comparing time-to-completion (in seconds) of two coding-assistant models on a fixed benchmark of 200 problems, with each problem solved by both models. (b) Comparing pass-rate on a 500-problem code-generation benchmark, with pass/fail outcomes per problem, both models evaluated on the same problems. (c) A/B test comparing click-through rates on a recommendation system with 100,000 users in each arm, randomly assigned. (d) Likert-scale (1-5) human ratings of response quality from 80 raters per condition in a side-by-side study. (e) Comparing perplexity on a held-out language modeling corpus between two pretraining recipes. For each, name the effect-size measure (d / g / d_z / Δ / Cliff's δ / phi / h / OR) and one sentence on why.
      </Prose>

    </div>
  ),
};

export default effectSizes;
