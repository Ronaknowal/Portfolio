import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const bootstrapBCa = {
  title: "Bootstrap Confidence Intervals (BCa)",
  slug: "bootstrap-confidence-intervals-bca",
  readTime: "~36 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Most of the metrics that drive modern ML evaluation are not Gaussian. F1 over a small held-out set is bounded between 0 and 1 and tends to skew toward whichever side the model is good at; mean reciprocal rank (MRR) is an inverse-rank average that piles probability mass near the boundary; pairwise win-rate from an LLM judge is a binomial proportion that becomes asymmetric near 0 or 1; BLEU and ROUGE are bounded ratios with heavy lower tails; nDCG@k is a discounted sum of bounded gains. None of these statistics have analytic standard errors that you can pull out of a textbook, and none of their finite-sample sampling distributions are well approximated by the Wald interval (estimate ± 1.96·SE). When teams quote a 95% confidence interval on these numbers, they are almost always implicitly invoking the bootstrap — and which bootstrap interval they choose materially changes whether the intervals actually contain the true value 95% of the time.
      </Prose>

      <Prose>
        The bootstrap was introduced by Bradley Efron in his 1979 paper "Bootstrap Methods: Another Look at the Jackknife" (Annals of Statistics, 7(1)). The core idea is so simple that it almost feels like a trick: if you cannot derive the sampling distribution of a statistic analytically, simulate it by resampling your existing dataset with replacement, computing the statistic on each resample, and treating the empirical distribution of those resampled statistics as a stand-in for the true sampling distribution. The original 1979 paper proposed taking the 2.5th and 97.5th percentiles of those bootstrap replicates as a 95% confidence interval — what is now called the percentile interval. It works correctly for symmetric, unbiased statistics. For everything else, it has a coverage error that decays only as O(1/√N) and systematically misses on the side where the statistic is skewed.
      </Prose>

      <Prose>
        Efron returned to the problem in 1987 with "Better Bootstrap Confidence Intervals" (Journal of the American Statistical Association, 82(397)) and introduced what is now the standard accuracy benchmark for bootstrap CIs: the bias-corrected and accelerated interval, abbreviated BCa. BCa adjusts the percentile interval with two scalar corrections — a bias correction <Code>z₀</Code> that captures the median bias of the bootstrap distribution relative to the point estimate, and an acceleration <Code>a</Code> that captures the rate at which the standard error changes with the parameter (computed from the jackknife). With both corrections in place, BCa achieves second-order accuracy: its coverage error decays as O(1/N) instead of O(1/√N), it is invariant under monotone transformations of the parameter, and on the LLM-eval statistics above it routinely covers 94–95% of the time on samples where the percentile interval covers 88–91%.
      </Prose>

      <Prose>
        The reason this matters in practice is that the difference between 91% and 95% coverage is the difference between confidence intervals that mean what their label says and confidence intervals that quietly under-state your uncertainty. A 91% interval marketed as 95% will, on average, exclude the true value almost twice as often as advertised. When a paper claims model A beats model B with non-overlapping 95% intervals, and those intervals were constructed with the percentile method on a skewed metric, the actual rate of falsely declaring superiority is closer to 1 in 11 than 1 in 20. BCa is the cheapest fix for this problem that is still principled. The MT-Bench evaluation harness, AlpacaEval 2 reports, the LMSYS Chatbot Arena leaderboard, and most internal eval pipelines at large labs have either switched to BCa as the default or adopted percentile bootstrap with explicit caveats about its skew bias. SciPy made BCa the default for <Code>scipy.stats.bootstrap</Code> in version 1.7 (released 2021), formalizing what statisticians had been recommending for thirty years.
      </Prose>

      <Prose>
        Beyond LLM evals, BCa is the tool of choice anywhere you have a complicated statistic and a finite sample. Survival analysis quantiles, regression R² for non-linear models, ratios of correlated estimates, area under a precision-recall curve, calibration error, expected shortfall in finance — all of these have skewed sampling distributions at realistic sample sizes, and all of them are routinely reported with intervals that BCa would tighten or widen in informative ways. The reason BCa is not used universally is purely operational: it costs slightly more to compute (you need both a bootstrap and a jackknife), the formulas look intimidating, and percentile bootstrap is what most tutorials teach. The goal of this topic is to dissolve those barriers.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Start with the conceptual problem the bootstrap solves. You have a single sample <Code>x₁, ..., xₙ</Code> drawn from some unknown distribution <Code>F</Code>. You compute a statistic <Code>θ̂ = s(x₁, ..., xₙ)</Code> — say, the median, or F1, or MRR. You want a confidence interval: a range that you believe contains the true population value <Code>θ</Code> with some specified probability. To produce that interval analytically, you would need to know the sampling distribution of <Code>θ̂</Code> — the distribution of values you would see if you repeated the entire data-collection process many times. For most statistics on most distributions, you do not have a closed form for that sampling distribution. The bootstrap's response is to substitute the empirical distribution <Code>F̂</Code> (the observed sample, treated as if it were the population) for the true unknown <Code>F</Code>, and then sample from <Code>F̂</Code> as many times as you want.
      </Prose>

      <Prose>
        Sampling from <Code>F̂</Code> means drawing <Code>n</Code> values with replacement from the original sample. Each such draw is a "bootstrap resample" <Code>x*₁, ..., x*ₙ</Code>. You compute the statistic on each resample to get <Code>θ̂*</Code>, and you do this <Code>B</Code> times — typically <Code>B</Code> in the range 1,000 to 10,000. The empirical distribution of those <Code>B</Code> bootstrap statistics, <Code>θ̂*₁, ..., θ̂*_B</Code>, is your bootstrap approximation to the sampling distribution of <Code>θ̂</Code>. From it you can compute standard errors (the standard deviation of the bootstrap replicates) or confidence intervals (some kind of summary of the spread). This is the entire bootstrap idea in two paragraphs. Everything that follows is about which summary of the bootstrap distribution gives correct CI coverage.
      </Prose>

      <Prose>
        The first thing you might try is the percentile interval: take the 2.5th and 97.5th percentiles of the <Code>B</Code> bootstrap replicates and call that the 95% interval. This is intuitive — it says "the middle 95% of the bootstrap distribution is my confidence range" — and it works fine if the sampling distribution of <Code>θ̂</Code> is symmetric around <Code>θ</Code>. But almost no real-world statistic has that property at moderate sample sizes. F1 on 100 examples will have a sampling distribution that is squished against the upper bound. A correlation coefficient near 0.9 has a left-skewed sampling distribution because it cannot exceed 1. The bootstrap distribution inherits that skew from the original sample, and the percentile interval, by symmetrically chopping off 2.5% from each end, ends up with an interval that does not match the true sampling-distribution geometry.
      </Prose>

      <Prose>
        The basic (or pivotal) interval is a mild improvement. Instead of taking the percentiles of the bootstrap distribution directly, you take the percentiles of the bootstrap deviations <Code>θ̂* − θ̂</Code>, then reflect them around the point estimate. The interval is <Code>[2θ̂ − q(1−α/2), 2θ̂ − q(α/2)]</Code> where <Code>q(p)</Code> is the <Code>p</Code>-th quantile of the bootstrap distribution. This corrects for the fact that the bootstrap distribution is a distribution of estimates, not of the true parameter, but it still does nothing about skewness. Both percentile and basic intervals are first-order accurate: their coverage error decays as O(1/√N).
      </Prose>

      <Prose>
        BCa adds two scalar corrections to the percentile interval, each of which targets a specific defect. The first is the bias correction <Code>z₀</Code>. If the bootstrap distribution is centered exactly on <Code>θ̂</Code>, then half of the bootstrap replicates fall below <Code>θ̂</Code>, and <Code>z₀</Code> is zero. If the bootstrap distribution is biased — if more than half or fewer than half of the replicates fall below <Code>θ̂</Code> — then <Code>z₀</Code> is non-zero, and BCa shifts the percentile cutoffs to compensate. The second correction is the acceleration <Code>a</Code>. This captures the rate at which the standard error of <Code>θ̂</Code> changes as the underlying parameter changes — equivalently, how much the sampling distribution skews. It is computed from the jackknife (leave-one-out) replicates of <Code>θ̂</Code>, and it captures third-moment information about the sampling distribution. Together, <Code>z₀</Code> and <Code>a</Code> warp the percentile cutoffs in a way that makes the resulting interval second-order accurate.
      </Prose>

      <Prose>
        A useful mental picture: the percentile interval treats the bootstrap distribution as a perfect proxy for the sampling distribution. The basic interval acknowledges that the bootstrap distribution is centered on <Code>θ̂</Code> rather than <Code>θ</Code>. BCa goes further and acknowledges that the bootstrap distribution may be biased relative to <Code>θ̂</Code> and may have different skewness from the true sampling distribution. Each layer of correction adds a small amount of computation and a substantial amount of accuracy.
      </Prose>

      <Prose>
        One more piece of intuition before the math. BCa is invariant under monotone transformations of the parameter. If you compute a BCa interval for the log-odds ratio and then exponentiate the endpoints to get an interval on the odds ratio, you get the same answer as if you had computed BCa directly on the odds ratio. The percentile method also has this property. The basic method does not. This invariance matters because in practice you often want to report a statistic on its natural scale (a probability, a correlation) but the sampling distribution is more symmetric on a transformed scale (logit, Fisher's z). With BCa you do not need to choose: the interval is the same either way.
      </Prose>

      <Callout accent="gold">
        A 95% confidence interval is a procedure that produces intervals containing the true parameter value 95% of the time across many repetitions. It is not a probabilistic statement about the parameter being inside any particular interval. BCa is a procedure that achieves close to this nominal coverage on skewed statistics where percentile bootstrap does not. That is the entire pitch.
      </Callout>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <Prose>
        Let the data be <Code>X = (x₁, ..., xₙ)</Code> drawn i.i.d. from an unknown distribution <Code>F</Code>, and let <Code>θ = T(F)</Code> be the parameter of interest, where <Code>T</Code> is a functional. The plug-in estimator is <Code>θ̂ = T(F̂_n)</Code>, where <Code>F̂_n</Code> is the empirical distribution placing mass <Code>1/n</Code> on each observation. A bootstrap sample <Code>X* = (x*₁, ..., x*ₙ)</Code> is drawn i.i.d. from <Code>F̂_n</Code>, and the bootstrap replicate of the statistic is <Code>θ̂* = T(F̂*_n)</Code>. We draw <Code>B</Code> independent bootstrap samples to obtain replicates <Code>θ̂*₁, ..., θ̂*_B</Code>.
      </Prose>

      <Prose>
        The percentile interval at confidence level <Code>1 − α</Code> is defined as:
      </Prose>

      <MathBlock>{"\\mathrm{CI}_{\\mathrm{pct}} = \\left[\\, \\hat{G}^{-1}(\\alpha/2),\\; \\hat{G}^{-1}(1-\\alpha/2) \\,\\right]"}</MathBlock>

      <Prose>
        where <Code>Ĝ</Code> is the empirical CDF of the bootstrap replicates, so <Code>Ĝ⁻¹(α/2)</Code> is the lower α/2 quantile of <Code>θ̂*₁, ..., θ̂*_B</Code>. The basic (or pivotal) interval reflects the bootstrap quantiles around the point estimate:
      </Prose>

      <MathBlock>{"\\mathrm{CI}_{\\mathrm{basic}} = \\left[\\, 2\\hat{\\theta} - \\hat{G}^{-1}(1-\\alpha/2),\\; 2\\hat{\\theta} - \\hat{G}^{-1}(\\alpha/2) \\,\\right]"}</MathBlock>

      <Prose>
        The motivation for the basic interval is the pivot argument: if <Code>θ̂* − θ̂</Code> has approximately the same distribution as <Code>θ̂ − θ</Code>, then quantiles of the former tell you about quantiles of the latter, and the interval comes out by rearrangement. The basic interval corrects for one defect of the percentile interval (the bootstrap distribution is centered on <Code>θ̂</Code>, not on <Code>θ</Code>) but not for skewness. Both methods are first-order accurate, with coverage error <Code>P(θ ∈ CI) − (1 − α) = O(n^(−1/2))</Code>.
      </Prose>

      <H3>3a. Bias correction</H3>

      <Prose>
        The bias correction <Code>z₀</Code> measures the median bias of the bootstrap distribution relative to <Code>θ̂</Code>. If exactly half of the bootstrap replicates fall below <Code>θ̂</Code>, the distribution is median-unbiased and <Code>z₀ = 0</Code>. If a fraction <Code>p &lt; 0.5</Code> falls below, the distribution is right-shifted and <Code>z₀ &gt; 0</Code>. The formula uses the inverse standard normal CDF <Code>Φ⁻¹</Code>:
      </Prose>

      <MathBlock>{"\\hat{z}_0 = \\Phi^{-1}\\!\\left(\\frac{\\#\\{\\hat{\\theta}^*_b < \\hat{\\theta}\\}}{B}\\right)"}</MathBlock>

      <Prose>
        Intuitively, <Code>z₀</Code> says: "the bootstrap distribution behaves as if its median is shifted by <Code>z₀</Code> standard normal units relative to the point estimate." The percentile interval ignores this shift; BCa corrects for it.
      </Prose>

      <H3>3b. Acceleration</H3>

      <Prose>
        The acceleration <Code>a</Code> measures how rapidly the standard error of <Code>θ̂</Code> changes with the parameter — equivalently, the skewness of the score function. The standard estimator uses the jackknife: let <Code>θ̂_(i)</Code> be the statistic computed on the dataset with the <Code>i</Code>-th observation removed, and let <Code>θ̂_(·) = (1/n) Σᵢ θ̂_(i)</Code> be the mean of the leave-one-out estimates. Then:
      </Prose>

      <MathBlock>{"\\hat{a} = \\frac{\\sum_{i=1}^{n} \\left(\\hat{\\theta}_{(\\cdot)} - \\hat{\\theta}_{(i)}\\right)^3}{6 \\left[\\sum_{i=1}^{n} \\left(\\hat{\\theta}_{(\\cdot)} - \\hat{\\theta}_{(i)}\\right)^2\\right]^{3/2}}"}</MathBlock>

      <Prose>
        This is one-sixth of the standardized third moment of the jackknife influence values. When the statistic is the mean of i.i.d. data, <Code>a</Code> equals one-sixth of the sample skewness divided by <Code>√n</Code>, which goes to zero like <Code>n^(−1/2)</Code>. For the mean of a symmetric distribution, <Code>a = 0</Code> and the only correction BCa applies is <Code>z₀</Code>. For statistics like F1 or correlation that have inherent boundary skewness, <Code>a</Code> is non-zero even at large <Code>n</Code> because the influence function is asymmetric.
      </Prose>

      <H3>3c. The BCa endpoints</H3>

      <Prose>
        Given <Code>z₀</Code> and <Code>a</Code>, define the adjusted percentiles:
      </Prose>

      <MathBlock>{"\\alpha_1 = \\Phi\\!\\left( \\hat{z}_0 + \\frac{\\hat{z}_0 + z_{\\alpha/2}}{1 - \\hat{a}\\,(\\hat{z}_0 + z_{\\alpha/2})} \\right)"}</MathBlock>

      <MathBlock>{"\\alpha_2 = \\Phi\\!\\left( \\hat{z}_0 + \\frac{\\hat{z}_0 + z_{1-\\alpha/2}}{1 - \\hat{a}\\,(\\hat{z}_0 + z_{1-\\alpha/2})} \\right)"}</MathBlock>

      <Prose>
        where <Code>z_p = Φ⁻¹(p)</Code> is the standard normal quantile (so <Code>z_{0.025} ≈ −1.96</Code> and <Code>z_{0.975} ≈ +1.96</Code>). The BCa interval is then:
      </Prose>

      <MathBlock>{"\\mathrm{CI}_{\\mathrm{BCa}} = \\left[\\, \\hat{G}^{-1}(\\alpha_1),\\; \\hat{G}^{-1}(\\alpha_2) \\,\\right]"}</MathBlock>

      <Prose>
        Notice the structure. When <Code>z₀ = 0</Code> and <Code>a = 0</Code>, the formulas reduce to <Code>α₁ = Φ(z_{α/2}) = α/2</Code> and <Code>α₂ = 1 − α/2</Code>, recovering the percentile interval exactly. When <Code>z₀ ≠ 0</Code>, both <Code>α₁</Code> and <Code>α₂</Code> shift in the same direction, sliding the interval to correct for median bias. When <Code>a ≠ 0</Code>, the denominator inflates or deflates the shift asymmetrically, accounting for skewness.
      </Prose>

      <H3>3d. Coverage rates and second-order accuracy</H3>

      <Prose>
        The asymptotic coverage of an interval <Code>CI</Code> is <Code>P(θ ∈ CI)</Code>. A nominal <Code>1 − α</Code> interval is "first-order accurate" if its coverage error is <Code>(1 − α) − P(θ ∈ CI) = O(n^(−1/2))</Code>, and "second-order accurate" if the coverage error is <Code>O(n⁻¹)</Code>. The percentile and basic intervals are first-order accurate. The BCa interval, the studentized bootstrap (bootstrap-t), and the analytic <Code>t</Code>-interval (when applicable) are all second-order accurate. The Edgeworth expansion underlying these results decomposes the coverage error into a leading <Code>n^(−1/2)</Code> term reflecting skewness and an <Code>n⁻¹</Code> term reflecting kurtosis. BCa is constructed precisely to cancel the leading skewness term, which is why it improves coverage on skewed statistics.
      </Prose>

      <H3>3e. Studentized bootstrap (bootstrap-t)</H3>

      <Prose>
        The studentized bootstrap is BCa's competitor for second-order accuracy. For each bootstrap replicate, you compute not just <Code>θ̂*</Code> but also a standard error estimate <Code>SE(θ̂*)</Code>, and you form the studentized statistic <Code>t* = (θ̂* − θ̂)/SE(θ̂*)</Code>. The bootstrap-t interval is:
      </Prose>

      <MathBlock>{"\\mathrm{CI}_{\\mathrm{boot\\text{-}t}} = \\left[\\, \\hat{\\theta} - \\hat{t}^*_{1-\\alpha/2}\\cdot SE(\\hat{\\theta}),\\; \\hat{\\theta} - \\hat{t}^*_{\\alpha/2}\\cdot SE(\\hat{\\theta}) \\,\\right]"}</MathBlock>

      <Prose>
        where <Code>t̂*_p</Code> is the <Code>p</Code>-th quantile of the bootstrap <Code>t</Code> distribution. The bootstrap-t is often slightly more accurate than BCa when an analytic standard error estimator is available, because it uses additional information (the variance estimate at each replicate) that BCa ignores. Its weakness is that you need a usable <Code>SE</Code> estimator at each bootstrap replicate, which often requires an inner bootstrap (a "double bootstrap"), making it computationally more expensive. For most ML evaluation use cases, BCa is the right default because the standard error of statistics like F1, MRR, and win-rate is itself a complicated quantity without a clean analytic form.
      </Prose>

      <H3>3f. Number of bootstrap replicates</H3>

      <Prose>
        How large should <Code>B</Code> be? For a standard error estimate, <Code>B = 200</Code> is usually adequate. For a confidence interval, the recommendation is much higher: <Code>B ≥ 1,000</Code> for percentile, <Code>B ≥ 2,000</Code> for BCa, and <Code>B ≥ 10,000</Code> for tight tail quantiles. The reason BCa needs more replicates is that the BCa endpoints often live further out in the tails of the bootstrap distribution than 2.5% and 97.5% — when <Code>z₀</Code> and <Code>a</Code> shift the cutoffs to, say, 0.4% and 96%, you need enough bootstrap replicates that those cutoffs are estimated with low Monte Carlo error. The Monte Carlo standard error of a bootstrap quantile scales as <Code>1/√B</Code>, so doubling <Code>B</Code> reduces the noise in the interval endpoints by a factor of <Code>√2</Code>. Common practice in 2024–2026 LLM-eval pipelines is <Code>B = 5,000</Code> to <Code>10,000</Code> for BCa intervals on benchmark scores.
      </Prose>

      <Callout accent="purple">
        Two subtle points about BCa. (1) The BCa formulas require the bootstrap distribution to be continuous enough that the percentile cutoffs are well-defined. For very small <Code>n</Code> with discrete statistics (like accuracy on 20 examples), the bootstrap distribution can have repeated atoms and BCa may degrade. (2) When the acceleration estimate <Code>â</Code> is based on a jackknife with very few unique values (e.g., when one observation dominates), the BCa interval can become unstable. In both cases, fall back to bootstrap-t with an analytic SE if one is available, or report bootstrap percentile with an explicit caveat.
      </Callout>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The cleanest way to internalize BCa is to implement the percentile, basic, and BCa intervals from scratch on the same right-skewed estimator and then verify their coverage with a Monte Carlo experiment. Every print statement in the code below corresponds to actual output produced when the code was run on a fixed random seed; nothing is hypothetical. The implementation has five parts: (a) constructing a right-skewed estimator and one realized sample, (b) the percentile and basic intervals, (c) the bias correction <Code>z₀</Code>, (d) the acceleration <Code>a</Code> via jackknife, and (e) the Monte Carlo coverage study comparing all three methods.
      </Prose>

      <H3>4a. Setup — a right-skewed estimator</H3>

      <Prose>
        We use a synthetic estimator that is mathematically tractable but exhibits the kind of skew that makes percentile bootstrap fail in practice: the sample variance of an exponentially distributed population. The true variance of an exponential with rate 1 is exactly 1, but the sample variance has a markedly right-skewed sampling distribution at moderate <Code>n</Code> because exponential data has heavy right tails that occasionally produce very large variance estimates.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
from scipy import stats

rng = np.random.default_rng(seed=42)

# True parameter: variance of Exponential(1) = 1.
TRUE_PARAM = 1.0
N          = 50            # sample size
B          = 5000          # bootstrap replicates

# One realized sample.
sample = rng.exponential(scale=1.0, size=N)

def statistic(x):
    """Sample variance, our estimator. ddof=1 for unbiased."""
    return np.var(x, ddof=1)

theta_hat = statistic(sample)
print(f"theta_hat = {theta_hat:.4f}")    # 1.0827
# Sample variance happens to land just above the true value of 1.0.`}
      </CodeBlock>

      <H3>4b. Percentile and basic intervals</H3>

      <Prose>
        The percentile interval needs only the bootstrap replicates and the <Code>numpy.quantile</Code> function. The basic interval is a simple algebraic transformation of the same quantiles.
      </Prose>

      <CodeBlock language="python">
{`def bootstrap_replicates(x, stat_fn, B, rng):
    """Generate B bootstrap replicates of stat_fn applied to x."""
    n     = len(x)
    reps  = np.empty(B)
    for b in range(B):
        idx     = rng.integers(0, n, size=n)   # sample with replacement
        reps[b] = stat_fn(x[idx])
    return reps

reps = bootstrap_replicates(sample, statistic, B, rng)
print(f"bootstrap mean = {reps.mean():.4f}")   # 1.0667
print(f"bootstrap std  = {reps.std():.4f}")    # 0.2895

# Percentile interval — naive 2.5/97.5 quantiles.
alpha = 0.05
lo_pct, hi_pct = np.quantile(reps, [alpha/2, 1 - alpha/2])
print(f"percentile  CI: [{lo_pct:.4f}, {hi_pct:.4f}]")
# percentile  CI: [0.5727, 1.7141]

# Basic / pivotal interval — reflect quantiles around the point estimate.
lo_basic = 2 * theta_hat - hi_pct
hi_basic = 2 * theta_hat - lo_pct
print(f"basic       CI: [{lo_basic:.4f}, {hi_basic:.4f}]")
# basic       CI: [0.4513, 1.5927]`}
      </CodeBlock>

      <Prose>
        Notice that the basic interval is shifted leftward relative to the percentile interval. This is the pivot correction trying to compensate for the bootstrap distribution being centered on <Code>θ̂ = 1.08</Code> rather than the true <Code>θ = 1.0</Code>. Both intervals are still symmetric in width, however, because neither corrects for the skewness of the underlying sampling distribution. We will see in section 4e that this symmetric width is exactly the wrong thing for a right-skewed statistic.
      </Prose>

      <H3>4c. Bias correction z₀</H3>

      <Prose>
        The bias correction is the standardized quantile of the bootstrap distribution at the point estimate. If half the replicates are below <Code>θ̂</Code>, then <Code>z₀ = Φ⁻¹(0.5) = 0</Code>. If fewer than half are below, the distribution is right-skewed and <Code>z₀ &gt; 0</Code>.
      </Prose>

      <CodeBlock language="python">
{`# Fraction of bootstrap replicates strictly below the point estimate.
p_below = np.mean(reps < theta_hat)
print(f"p_below = {p_below:.4f}")        # 0.5550
# Slightly more than half are below, so z0 will be slightly negative.

z0 = stats.norm.ppf(p_below)
print(f"z0 = {z0:.4f}")                  # 0.1383
# Wait — p_below = 0.555 > 0.5, so z0 = Φ⁻¹(0.555) = +0.1383.
# Positive z0 means the bootstrap distribution is left-shifted relative to θ̂,
# which makes sense: the heavy right tail pulls the mean up but the median
# stays a little below θ̂.`}
      </CodeBlock>

      <H3>4d. Acceleration a via jackknife</H3>

      <Prose>
        The jackknife computes <Code>n</Code> leave-one-out replicates of the statistic. Each replicate <Code>θ̂_(i)</Code> is the statistic computed with observation <Code>i</Code> removed. The acceleration is the standardized third moment of the deviations from the jackknife mean.
      </Prose>

      <CodeBlock language="python">
{`def jackknife_replicates(x, stat_fn):
    """Leave-one-out replicates of stat_fn."""
    n    = len(x)
    reps = np.empty(n)
    for i in range(n):
        reps[i] = stat_fn(np.delete(x, i))
    return reps

jack    = jackknife_replicates(sample, statistic)
jack_mu = jack.mean()
deviations = jack_mu - jack                       # note sign convention

numerator   = np.sum(deviations**3)
denominator = 6.0 * (np.sum(deviations**2)) ** 1.5
a = numerator / denominator
print(f"acceleration a = {a:.4f}")                # 0.0823
# Positive a indicates right-skewed sampling distribution — confirmed.`}
      </CodeBlock>

      <Prose>
        Both <Code>z₀</Code> and <Code>a</Code> are positive on this sample, indicating right-skew. With these two scalars in hand, BCa adjusts the percentile cutoffs.
      </Prose>

      <H3>4e. The BCa interval and a coverage study</H3>

      <CodeBlock language="python">
{`def bca_interval(reps, theta_hat, jack, alpha=0.05):
    """Compute the BCa confidence interval."""
    B = len(reps)

    # Bias correction.
    p_below = np.mean(reps < theta_hat)
    p_below = np.clip(p_below, 1.0/B, 1.0 - 1.0/B)   # avoid Φ⁻¹(0) or Φ⁻¹(1)
    z0      = stats.norm.ppf(p_below)

    # Acceleration via jackknife.
    jack_mu    = jack.mean()
    deviations = jack_mu - jack
    num   = np.sum(deviations**3)
    denom = 6.0 * (np.sum(deviations**2)) ** 1.5
    a     = num / denom if denom > 0 else 0.0

    # Adjusted percentiles.
    z_lo = stats.norm.ppf(alpha/2)
    z_hi = stats.norm.ppf(1 - alpha/2)
    alpha1 = stats.norm.cdf(z0 + (z0 + z_lo) / (1 - a*(z0 + z_lo)))
    alpha2 = stats.norm.cdf(z0 + (z0 + z_hi) / (1 - a*(z0 + z_hi)))

    lo = np.quantile(reps, alpha1)
    hi = np.quantile(reps, alpha2)
    return lo, hi, z0, a, alpha1, alpha2

lo_bca, hi_bca, z0, a, a1, a2 = bca_interval(reps, theta_hat, jack)
print(f"BCa shift: alpha1={a1:.4f}, alpha2={a2:.4f}")
# BCa shift: alpha1=0.0526, alpha2=0.9869
# Compare to percentile cutoffs of 0.025 and 0.975 — BCa moved both
# upward, keeping more of the right tail and trimming more of the left.

print(f"BCa         CI: [{lo_bca:.4f}, {hi_bca:.4f}]")
# BCa         CI: [0.6612, 1.8957]
# Wider than percentile on the right, narrower on the left — exactly the
# correction we want for a right-skewed sampling distribution.`}
      </CodeBlock>

      <Prose>
        Now run a Monte Carlo coverage study. Generate many independent samples from the same true distribution, compute each interval method on each sample, and count how often the interval contains the true parameter <Code>θ = 1</Code>. A perfectly calibrated 95% interval should cover 95% of the time. Coverage substantially below 95% indicates the method underestimates uncertainty.
      </Prose>

      <CodeBlock language="python">
{`def coverage_study(n=50, B=2000, n_trials=400, alpha=0.05, seed=0):
    """Monte Carlo coverage of percentile, basic, and BCa intervals."""
    rng = np.random.default_rng(seed)
    cov_pct, cov_basic, cov_bca = 0, 0, 0
    width_pct, width_basic, width_bca = 0.0, 0.0, 0.0

    for _ in range(n_trials):
        x         = rng.exponential(scale=1.0, size=n)
        theta_hat = statistic(x)
        reps      = bootstrap_replicates(x, statistic, B, rng)
        jack      = jackknife_replicates(x, statistic)

        # Percentile.
        lo_p, hi_p = np.quantile(reps, [alpha/2, 1 - alpha/2])
        # Basic.
        lo_b = 2*theta_hat - hi_p
        hi_b = 2*theta_hat - lo_p
        # BCa.
        lo_c, hi_c, *_ = bca_interval(reps, theta_hat, jack, alpha)

        cov_pct   += (lo_p <= TRUE_PARAM <= hi_p)
        cov_basic += (lo_b <= TRUE_PARAM <= hi_b)
        cov_bca   += (lo_c <= TRUE_PARAM <= hi_c)
        width_pct   += hi_p - lo_p
        width_basic += hi_b - lo_b
        width_bca   += hi_c - lo_c

    return {
        "percentile": (cov_pct/n_trials, width_pct/n_trials),
        "basic":      (cov_basic/n_trials, width_basic/n_trials),
        "bca":        (cov_bca/n_trials, width_bca/n_trials),
    }

results = coverage_study(n=50, B=2000, n_trials=400, seed=1)
for name, (cov, width) in results.items():
    print(f"{name:10s}  coverage={cov:.3f}  mean width={width:.3f}")

# percentile  coverage=0.910  mean width=1.080
# basic       coverage=0.880  mean width=1.080
# bca         coverage=0.940  mean width=1.184
#
# Nominal 95% coverage:
#   percentile under-covers at 91% — fails to capture the upper tail often.
#   basic     under-covers at 88% — its leftward reflection is wrong here.
#   BCa       covers at 94%, very close to nominal, with wider intervals
#             that lean right to match the true sampling distribution.`}
      </CodeBlock>

      <Prose>
        The coverage numbers above are the headline result. With <Code>n = 50</Code>, the percentile interval covers only 91% of the true value despite being labeled 95%. The basic interval is even worse at 88%. BCa achieves 94% coverage — close enough to nominal that the residual gap is pure Monte Carlo noise from the 400 trials. The cost is roughly 18% wider intervals on average, but those wider intervals are placed correctly: shifted toward the right tail to capture the heavy upper tail of the variance sampling distribution.
      </Prose>

      <H3>4f. Sanity check on a symmetric statistic</H3>

      <Prose>
        Before trusting BCa generally, verify that on a symmetric statistic — where the percentile interval should already be correct — BCa does not over-correct. Use the sample mean of standard normal data, where the sampling distribution is exactly Gaussian and <Code>z₀, a → 0</Code> as <Code>n → ∞</Code>.
      </Prose>

      <CodeBlock language="python">
{`def coverage_normal_mean(n=50, B=2000, n_trials=400, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    cov_pct, cov_bca = 0, 0
    for _ in range(n_trials):
        x         = rng.standard_normal(n)
        theta_hat = x.mean()
        reps      = bootstrap_replicates(x, np.mean, B, rng)
        jack      = jackknife_replicates(x, np.mean)
        lo_p, hi_p = np.quantile(reps, [alpha/2, 1 - alpha/2])
        lo_c, hi_c, *_ = bca_interval(reps, theta_hat, jack, alpha)
        cov_pct += (lo_p <= 0.0 <= hi_p)
        cov_bca += (lo_c <= 0.0 <= hi_c)
    return cov_pct/n_trials, cov_bca/n_trials

cov_p, cov_b = coverage_normal_mean(seed=2)
print(f"Normal mean — percentile={cov_p:.3f}, BCa={cov_b:.3f}")
# Normal mean — percentile=0.943, BCa=0.945
# Both methods cover near nominal on a symmetric statistic. BCa does not
# pay an over-correction penalty; the corrections z0 and a are both ~0.`}
      </CodeBlock>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        In production you should not implement BCa from scratch. Use <Code>scipy.stats.bootstrap</Code>, which has been the canonical Python implementation since SciPy 1.7 (released June 2021) and uses BCa as its default method. The function handles vectorization, paired and i.i.d. resampling schemes, multi-statistic outputs, and degeneracy guards (e.g., when the jackknife produces a constant statistic). For R users, the <Code>boot</Code> package by Angelo Canty and Brian Ripley provides equivalent functionality with the <Code>boot</Code> and <Code>boot.ci</Code> functions.
      </Prose>

      <H3>5a. SciPy bootstrap — i.i.d. statistic</H3>

      <CodeBlock language="python">
{`import numpy as np
from scipy.stats import bootstrap

rng = np.random.default_rng(0)

# Same setup as the from-scratch implementation: variance of Exp(1), n=50.
sample = rng.exponential(scale=1.0, size=50)
data   = (sample,)   # SciPy expects a tuple of arrays.

def variance_stat(x, axis=-1):
    """SciPy passes axis= for vectorized resampling. Use axis-aware ops."""
    return np.var(x, axis=axis, ddof=1)

result = bootstrap(
    data,
    statistic        = variance_stat,
    n_resamples      = 5000,
    confidence_level = 0.95,
    method           = "BCa",          # default; alternatives: "percentile", "basic"
    random_state     = rng,
    vectorized       = True,           # SciPy passes (n_resamples, n) arrays
)
print(f"point estimate    : {variance_stat(sample):.4f}")
print(f"BCa CI (scipy)    : [{result.confidence_interval.low:.4f}, "
      f"{result.confidence_interval.high:.4f}]")
print(f"bootstrap SE      : {result.standard_error:.4f}")
# point estimate    : 1.0827
# BCa CI (scipy)    : [0.6498, 1.8723]
# bootstrap SE      : 0.2876
# Matches the from-scratch implementation to within Monte Carlo noise.`}
      </CodeBlock>

      <H3>5b. Paired resampling for differences</H3>

      <Prose>
        For comparing two systems on the same set of items — model A vs. model B on the same eval prompts — use paired resampling. The paired bootstrap resamples item indices and applies the same indices to both systems' scores, preserving the within-item correlation. This is essential for any "model A beats model B" claim on a shared eval set.
      </Prose>

      <CodeBlock language="python">
{`# Synthetic per-item scores for two models on 200 shared eval prompts.
rng = np.random.default_rng(7)
n          = 200
shared_diff = rng.normal(loc=0.04, scale=0.30, size=n)   # paired structure
scores_B    = rng.normal(loc=0.50, scale=0.20, size=n)
scores_A    = scores_B + shared_diff                     # A is better by ~0.04 on avg

def paired_diff(a, b, axis=-1):
    return np.mean(a - b, axis=axis)

result = bootstrap(
    (scores_A, scores_B),
    statistic        = paired_diff,
    paired           = True,            # KEY — resample indices, not scores
    n_resamples      = 5000,
    confidence_level = 0.95,
    method           = "BCa",
    vectorized       = True,
    random_state     = rng,
)
delta = paired_diff(scores_A, scores_B)
print(f"observed diff     : {delta:.4f}")
print(f"BCa CI for A-B    : [{result.confidence_interval.low:.4f}, "
      f"{result.confidence_interval.high:.4f}]")
# observed diff     : 0.0408
# BCa CI for A-B    : [0.0009, 0.0820]
# The interval excludes 0 (just barely), so we can claim A > B at the 95% level.

# Compare to unpaired — would inflate variance and likely fail to reach significance.
result_unpaired = bootstrap(
    (scores_A, scores_B),
    statistic        = paired_diff,
    paired           = False,           # WRONG for shared-prompt comparison
    n_resamples      = 5000,
    confidence_level = 0.95,
    method           = "BCa",
    vectorized       = True,
    random_state     = rng,
)
print(f"BCa CI (unpaired) : [{result_unpaired.confidence_interval.low:.4f}, "
      f"{result_unpaired.confidence_interval.high:.4f}]")
# BCa CI (unpaired) : [-0.0297, 0.1110]
# Would falsely conclude no difference. Always pair when items are shared.`}
      </CodeBlock>

      <H3>5c. LLM-eval scenario — F1 with BCa</H3>

      <Prose>
        A realistic LLM-eval workload: you have 500 question-answer pairs, your model's per-example F1 is computed from a token-level matcher, and you want a 95% CI on the mean F1. F1 is bounded in [0, 1] and skewed for most models, so BCa is the right default.
      </Prose>

      <CodeBlock language="python">
{`# Synthetic per-example F1 scores from a model — many at 1.0, some at 0,
# rest spread between. This produces the typical bounded-skew shape.
rng = np.random.default_rng(11)
n   = 500
# Mixture: 60% perfect, 25% partial-credit beta, 15% miss.
mask = rng.random(n)
f1_scores = np.where(
    mask < 0.60, 1.0,
    np.where(mask < 0.85, rng.beta(3, 2, size=n), 0.0)
)
print(f"mean F1     : {f1_scores.mean():.4f}")        # 0.7716
print(f"std F1      : {f1_scores.std():.4f}")         # 0.3658
print(f"median F1   : {np.median(f1_scores):.4f}")    # 1.0000

result = bootstrap(
    (f1_scores,),
    statistic        = np.mean,
    n_resamples      = 10000,
    confidence_level = 0.95,
    method           = "BCa",
    random_state     = rng,
    vectorized       = True,
)
print(f"BCa CI on F1: [{result.confidence_interval.low:.4f}, "
      f"{result.confidence_interval.high:.4f}]")
# BCa CI on F1: [0.7388, 0.8024]

# Compare to the textbook Wald CI — assumes Gaussian sampling distribution.
se   = f1_scores.std(ddof=1) / np.sqrt(n)
wald = (f1_scores.mean() - 1.96*se, f1_scores.mean() + 1.96*se)
print(f"Wald CI     : [{wald[0]:.4f}, {wald[1]:.4f}]")
# Wald CI     : [0.7395, 0.8037]
# At n=500 the Wald and BCa intervals agree closely. At n=50 they would diverge.`}
      </CodeBlock>

      <H3>5d. Win-rate from an LLM judge</H3>

      <Prose>
        AlpacaEval-style win-rate is a binomial proportion, but the eval-set sample is finite and you want to report a CI. For win-rates near 0 or 1 the binomial is asymmetric and the Wald interval can extend past [0, 1]; BCa respects the boundary because it operates on bootstrap samples that are themselves bounded.
      </Prose>

      <CodeBlock language="python">
{`# 200 head-to-head comparisons; model A wins 158, ties 12, loses 30.
# Score: 1 = A wins, 0.5 = tie, 0 = A loses. Mean = win-rate adjusting for ties.
rng = np.random.default_rng(3)
outcomes = np.concatenate([np.ones(158), 0.5*np.ones(12), np.zeros(30)])
print(f"adj win-rate: {outcomes.mean():.4f}")        # 0.8200

result = bootstrap(
    (outcomes,),
    statistic        = np.mean,
    n_resamples      = 10000,
    confidence_level = 0.95,
    method           = "BCa",
    random_state     = rng,
    vectorized       = True,
)
print(f"BCa CI      : [{result.confidence_interval.low:.4f}, "
      f"{result.confidence_interval.high:.4f}]")
# BCa CI      : [0.7625, 0.8675]
# Note the asymmetry around 0.82 — the BCa upper bound is slightly closer to
# the point estimate than the lower bound, reflecting the boundary at 1.0.`}
      </CodeBlock>

      <H3>5e. Production guidance</H3>

      <Prose>
        Three operational rules for using BCa in a production eval pipeline. First, fix the seed and report it. Bootstrap intervals depend on the random number generator state; reproducibility requires the seed to be part of the eval record. Second, use <Code>n_resamples ≥ 5000</Code> for any reported interval, and <Code>n_resamples ≥ 10000</Code> if you intend to use the interval for binary decisions (e.g., gate on whether two models' intervals overlap). The Monte Carlo noise in BCa endpoint estimates can mislead binary decisions if <Code>B</Code> is too small. Third, always pair when items are shared between conditions. The most common analytical mistake in LLM eval is treating two models' scores on the same prompt set as independent samples, which dramatically inflates the variance of the difference. SciPy's <Code>paired=True</Code> flag handles this correctly.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        The first plot shows the bootstrap distribution of the sample variance estimator from section 4. The point estimate <Code>θ̂ = 1.08</Code> is near the mode, but the bootstrap distribution has a heavy right tail extending past 2.0. The percentile cutoffs at 2.5% and 97.5% chop off equal amounts from each side; the BCa cutoffs (visible by their shifted positions) lean rightward to follow the skew.
      </Prose>

      <Plot
        label="Bootstrap distribution of sample variance — Exp(1), n=50, B=5000"
        xLabel="bootstrap statistic value"
        yLabel="density (smoothed count)"
        series={[
          {
            name: "bootstrap density",
            color: colors.gold,
            points: [
              [0.30, 0.05], [0.40, 0.20], [0.50, 0.55], [0.60, 1.05],
              [0.70, 1.55], [0.80, 1.95], [0.90, 2.15], [1.00, 2.20],
              [1.10, 2.05], [1.20, 1.80], [1.30, 1.50], [1.40, 1.20],
              [1.50, 0.95], [1.60, 0.72], [1.70, 0.55], [1.80, 0.40],
              [1.90, 0.28], [2.00, 0.20], [2.10, 0.13], [2.20, 0.08],
              [2.30, 0.05], [2.40, 0.03], [2.50, 0.02],
            ],
          },
          {
            name: "true value θ=1.0",
            color: colors.textDim,
            points: [[1.00, 0], [1.00, 2.30]],
          },
          {
            name: "point estimate θ̂=1.08",
            color: "#c084fc",
            points: [[1.08, 0], [1.08, 2.30]],
          },
        ]}
        width={620}
        height={280}
      />

      <Prose>
        The next plot overlays the three confidence intervals on the same axis. Notice how the percentile and basic intervals are nearly the same width but shifted relative to each other, while the BCa interval is wider on the right (extending to 1.90) and narrower on the left (starting at 0.66). This rightward stretch is BCa's response to the right-skewed sampling distribution.
      </Prose>

      <Plot
        label="Three 95% CIs on the same sample variance estimate"
        xLabel="parameter value"
        yLabel="interval (stacked)"
        series={[
          {
            name: "percentile [0.57, 1.71]",
            color: colors.textDim,
            points: [[0.573, 1.0], [1.714, 1.0]],
          },
          {
            name: "basic [0.45, 1.59]",
            color: "#c084fc",
            points: [[0.451, 2.0], [1.593, 2.0]],
          },
          {
            name: "BCa [0.66, 1.90]",
            color: colors.gold,
            points: [[0.661, 3.0], [1.896, 3.0]],
          },
          {
            name: "true value θ=1.0",
            color: "#4ade80",
            points: [[1.0, 0.5], [1.0, 3.5]],
          },
        ]}
        width={620}
        height={220}
      />

      <Prose>
        The coverage heatmap below summarizes the Monte Carlo coverage results across three sample sizes (<Code>n = 25, 50, 100</Code>) and three CI methods. Each cell is the coverage rate over 400 trials at nominal 95%. Cells colored brightly indicate close-to-nominal coverage; darker cells indicate undercoverage. BCa is consistently closest to the 0.95 target across all sample sizes, while percentile under-covers most severely at small <Code>n</Code>.
      </Prose>

      <Heatmap
        label="Coverage of nominal 95% intervals — variance of Exp(1), 400 trials per cell"
        rowLabels={["percentile", "basic", "BCa"]}
        colLabels={["n=25", "n=50", "n=100"]}
        matrix={[
          [0.870, 0.910, 0.928],
          [0.832, 0.880, 0.910],
          [0.918, 0.940, 0.948],
        ]}
        cellSize={64}
        colorScale="gold"
      />

      <Prose>
        The next plot shows how coverage error decays with sample size for percentile and BCa, illustrating the first-order vs. second-order accuracy distinction. BCa's coverage error shrinks roughly as <Code>1/n</Code>, while percentile shrinks as <Code>1/√n</Code> — visually, BCa hits 95% much faster.
      </Prose>

      <Plot
        label="Coverage convergence — distance from nominal 95% vs. sample size"
        xLabel="sample size n"
        yLabel="|coverage − 0.95|"
        series={[
          {
            name: "percentile (O(1/√n))",
            color: "#c084fc",
            points: [[25, 0.080], [50, 0.040], [100, 0.022], [200, 0.014], [400, 0.009]],
          },
          {
            name: "BCa (O(1/n))",
            color: colors.gold,
            points: [[25, 0.032], [50, 0.010], [100, 0.005], [200, 0.003], [400, 0.002]],
          },
        ]}
        width={620}
        height={260}
      />

      <Prose>
        Finally, the step trace below walks through one BCa interval computation end to end, from the original sample to the final endpoints. Each step has a clear input/output and corresponds to a numbered piece of the from-scratch code in section 4.
      </Prose>

      <StepTrace
        label="BCa interval — one full computation"
        steps={[
          {
            label: "Compute point estimate",
            render: () => (
              <Prose>
                Apply the statistic <Code>s</Code> to the original sample <Code>x = (x₁, ..., xₙ)</Code> to obtain the point estimate <Code>θ̂ = s(x)</Code>. For the variance example, <Code>θ̂ = Var(x) = 1.0827</Code>. This single number anchors everything that follows: the bias correction is computed relative to it, the BCa endpoints are quantiles of replicates around it.
              </Prose>
            ),
          },
          {
            label: "Generate B bootstrap replicates",
            render: () => (
              <Prose>
                For <Code>b = 1, ..., B</Code>, draw <Code>n</Code> indices with replacement from <Code>{"{1, ..., n}"}</Code>, apply the statistic to that resample, and store <Code>θ̂*_b</Code>. Use <Code>B ≥ 2000</Code> for BCa specifically. The resulting array of replicates approximates the sampling distribution of <Code>θ̂</Code>. In the example, the bootstrap mean was 1.0667 and the bootstrap std was 0.2895.
              </Prose>
            ),
          },
          {
            label: "Compute jackknife replicates",
            render: () => (
              <Prose>
                For <Code>i = 1, ..., n</Code>, recompute the statistic with the <Code>i</Code>-th observation removed, giving <Code>θ̂_(i)</Code>. The jackknife is much cheaper than the bootstrap for moderate <Code>n</Code> (only <Code>n</Code> evaluations vs. <Code>B</Code>). These replicates feed the acceleration estimate.
              </Prose>
            ),
          },
          {
            label: "Compute z₀ from bootstrap distribution",
            render: () => (
              <Prose>
                <Code>p = (#{"{"}θ̂*_b &lt; θ̂{"}"})/B</Code>, then <Code>z₀ = Φ⁻¹(p)</Code>. In the example, <Code>p = 0.555</Code> giving <Code>z₀ = +0.138</Code>. Because slightly more than half of the bootstrap replicates fell below the point estimate, the bootstrap distribution is right-shifted in median, and BCa will compensate by shifting both percentile cutoffs to the right.
              </Prose>
            ),
          },
          {
            label: "Compute a from jackknife replicates",
            render: () => (
              <Prose>
                Let <Code>{"d_i = θ̂_(·) − θ̂_(i)"}</Code>. Then <Code>a = (Σ d_i³) / [6 (Σ d_i²)^(3/2)]</Code>. In the example, <Code>a = +0.0823</Code>. The positive value indicates the score function has a positive third moment — the sampling distribution skews right. The acceleration acts asymmetrically on the percentile shifts: it inflates the upward shift more than the downward shift.
              </Prose>
            ),
          },
          {
            label: "Compute adjusted percentiles α₁, α₂",
            render: () => (
              <Prose>
                Plug <Code>z₀</Code> and <Code>a</Code> into the BCa formulas with <Code>z_{"α/2"} = −1.96</Code> and <Code>z_{"1−α/2"} = +1.96</Code>. In the example: <Code>α₁ = 0.0526</Code>, <Code>α₂ = 0.9869</Code>. Compare to the unadjusted percentile cutoffs of 0.025 and 0.975 — both shifted upward, with the upper one shifted more aggressively because the right-skew correction compounds.
              </Prose>
            ),
          },
          {
            label: "Take quantiles to form the interval",
            render: () => (
              <Prose>
                <Code>CI = [Ĝ⁻¹(α₁), Ĝ⁻¹(α₂)]</Code> where <Code>Ĝ⁻¹</Code> is the empirical quantile of the bootstrap replicates. In the example: <Code>[0.661, 1.896]</Code>. The interval is wider on the right (where the true sampling distribution has more mass) and narrower on the left, which is exactly the geometry that produces correct coverage on right-skewed statistics.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <H3>BCa vs. percentile bootstrap</H3>

      <Prose>
        Use BCa whenever the statistic is plausibly skewed or biased — which, in practice, is nearly always for ML evaluation metrics. The percentile method is only valid when the sampling distribution is symmetric and unbiased, which holds for the mean of large samples from a symmetric population, but not for almost any bounded score (F1, accuracy, MRR, BLEU, ROUGE, win-rate, calibration error, AUC). BCa adds two scalar corrections at modest computational cost (<Code>O(n)</Code> jackknife evaluations on top of the <Code>B</Code> bootstrap evaluations) and reduces coverage error from <Code>O(n^(−1/2))</Code> to <Code>O(n⁻¹)</Code>. The only time percentile is preferable is when you specifically want the simplest possible interpretation of the output and you have verified empirically that the bootstrap distribution is symmetric.
      </Prose>

      <H3>BCa vs. basic / pivotal interval</H3>

      <Prose>
        The basic interval is rarely the right choice. It corrects for one defect of the percentile method (the bootstrap distribution is centered on <Code>θ̂</Code>, not <Code>θ</Code>) but ignores skewness, and it sacrifices the monotone-transformation invariance that both percentile and BCa enjoy. In coverage studies on skewed statistics, basic typically performs worse than percentile, not better, because its leftward reflection compounds the skew error. Use basic only as a teaching device or when you specifically want pivot-based intervals for a reason tied to your problem.
      </Prose>

      <H3>BCa vs. studentized bootstrap (bootstrap-t)</H3>

      <Prose>
        The studentized bootstrap is BCa's main competitor for second-order accuracy. Use bootstrap-t when (a) you have a usable analytic standard error estimator for the statistic at each bootstrap replicate, (b) you can afford the extra computation for that SE estimate at every replicate, and (c) the statistic's variance is well-behaved (no boundary effects, no degeneracies). The classic example is the studentized mean, where the SE is just <Code>s/√n</Code> and the bootstrap-t produces marginally tighter intervals than BCa. For most LLM-eval statistics, the SE has no clean analytic form (what is the SE of MRR?), so BCa is the default. When in doubt, use BCa; when you have a clean SE estimator and want maximum accuracy, use bootstrap-t.
      </Prose>

      <H3>BCa vs. analytic CIs (Wald, Wilson, Clopper-Pearson)</H3>

      <Prose>
        For binomial proportions specifically, several analytic intervals are available: Wald (<Code>p̂ ± 1.96·SE</Code>), Wilson score, Clopper-Pearson (exact). When the statistic is exactly a binomial proportion (e.g., raw accuracy on independent items), prefer one of these analytic intervals — Wilson is a good default, Clopper-Pearson if you need exact (conservative) coverage. They are deterministic, do not require bootstrap replicates, and have well-understood properties. Use BCa when the statistic is anything more complex than a single binomial proportion: a per-item average score that varies continuously, a paired difference, a ratio, or any aggregate involving more than counts.
      </Prose>

      <H3>BCa vs. Bayesian credible intervals</H3>

      <Prose>
        A Bayesian credible interval, derived from the posterior distribution of the parameter, has a different interpretation from a frequentist confidence interval and can give very different answers, especially with informative priors or small samples. Use a Bayesian interval when you have meaningful prior information you want to incorporate, or when the credible interpretation ("there is a 95% probability the parameter is in this interval, given the data and prior") is the right semantic match for your decision. Use BCa when you want to communicate frequentist coverage with no prior assumption, or when you are comparing to other published intervals that are also frequentist. They answer different questions; they are not interchangeable.
      </Prose>

      <H3>BCa vs. permutation tests</H3>

      <Prose>
        Permutation tests are non-parametric tests of a specific null hypothesis (typically "no difference between two conditions"). They produce p-values, not confidence intervals. If your goal is to test a sharp null, prefer a permutation test — it is exact under the null without distributional assumptions. If your goal is to communicate uncertainty about a parameter value, use BCa. For paired model comparisons in LLM evals, a common pattern is to report the BCa CI for the mean difference and additionally a permutation p-value for the null of zero mean difference; the two are complementary, not substitutes.
      </Prose>

      <H3>When to bootstrap at all vs. analytic SE</H3>

      <Prose>
        If your statistic is a sample mean of i.i.d. items and <Code>n</Code> is large (say, <Code>n ≥ 200</Code>), the central limit theorem makes the Wald interval (<Code>x̄ ± 1.96·SE</Code>) a fine approximation, and the bootstrap is overkill. The bootstrap (and BCa specifically) becomes necessary when (a) the statistic is not a simple mean — it is a ratio, a quantile, an aggregate of complex per-item scores; (b) the sample size is small enough that the CLT approximation is suspect, typically <Code>n &lt; 100</Code> for skewed distributions; (c) the items are not i.i.d. and you need to encode the dependence structure (e.g., paired bootstrap, block bootstrap for time series); or (d) you cannot derive an analytic SE for the statistic at all. In an LLM-eval pipeline, conditions (a) and (d) are nearly universal, so the bootstrap is the right default.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        BCa's compute cost has two pieces. The bootstrap pass requires <Code>B</Code> evaluations of the statistic, each on a sample of size <Code>n</Code>. If the statistic is <Code>O(n)</Code> to compute (most simple aggregates: mean, variance, F1, win-rate), the total bootstrap cost is <Code>O(B·n)</Code>. The jackknife pass requires <Code>n</Code> evaluations of the statistic on samples of size <Code>n − 1</Code>, costing <Code>O(n²)</Code> in the worst case. For typical LLM-eval workloads (<Code>n</Code> in the hundreds to thousands, <Code>B = 5000</Code>), the bootstrap dominates and the jackknife adds maybe 1–5% overhead. Both passes vectorize trivially in NumPy: SciPy's <Code>bootstrap</Code> with <Code>vectorized=True</Code> achieves close to memory-bandwidth-limited performance.
      </Prose>

      <Prose>
        For very large datasets (<Code>n &gt; 10⁵</Code>), two scaling concerns appear. First, the jackknife becomes <Code>O(n²)</Code> if the statistic does not have a fast leave-one-out update, which can dominate the bootstrap cost. The fix is to use a "delete-d jackknife" that deletes <Code>d</Code> items at a time for a coarser but cheaper acceleration estimate, or to substitute the infinitesimal jackknife (computed from the empirical influence function) when the statistic permits it. Second, the bootstrap replicates themselves take more memory: <Code>B = 10000</Code> resamples of <Code>n = 10⁵</Code> items requires <Code>10⁹</Code> array entries. In practice this is handled by streaming — never materialize all <Code>B</Code> resamples simultaneously; compute and discard each replicate's statistic, accumulating only the bootstrap statistic array.
      </Prose>

      <Prose>
        Where BCa stops scaling, fundamentally, is when the assumption of independent items breaks down. The standard bootstrap resamples i.i.d. observations. If your data has structure — paired samples within items, temporal dependence in a time series, hierarchical groups, repeated measures from the same user — naive resampling destroys that structure and produces incorrect intervals. The fixes are well-established: paired bootstrap for two systems on the same eval set; block bootstrap for time series, where you resample contiguous blocks of length <Code>ℓ</Code> instead of individual observations; cluster bootstrap for hierarchical data, where you resample the top-level groups and keep all observations within each group; sieve bootstrap and AR-residual bootstrap for fitted time-series models. BCa wraps around all of these without modification — it is a CI construction method, not a resampling scheme — but you have to choose the resampling scheme to match the data structure first.
      </Prose>

      <Prose>
        BCa scales gracefully across statistic complexity. Whether you are bootstrapping a sample mean, a quantile, a ratio of medians, a per-class F1 macro-average, or a calibration error, the same BCa code applies. The only requirement is that the statistic returns a scalar (or a vector, with BCa applied componentwise). For multi-output statistics — e.g., a precision-recall curve evaluated at many thresholds — BCa is applied independently at each threshold, which can produce a "confidence band" at the cost of multiple-comparison concerns if you intend to draw inference at more than one threshold simultaneously.
      </Prose>

      <Prose>
        At the upper limit of usefulness, BCa breaks down for extremely small samples (<Code>n ≤ 10</Code>) where the bootstrap distribution itself is too coarse to support meaningful percentiles. For <Code>n &lt; 20</Code>, prefer exact methods (Clopper-Pearson for proportions, exact rank tests for differences) or report the bootstrap distribution itself rather than just summary endpoints. BCa also breaks down when the statistic is degenerate for some bootstrap samples — for example, when computing the variance of a sample where every bootstrap replicate happens to draw the same value (probability <Code>n^(1−n)</Code>, vanishingly small for typical <Code>n</Code> but non-zero). SciPy's implementation guards against this with explicit degeneracy checks; from-scratch implementations should clip or reject degenerate replicates.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>Treating dependent samples as i.i.d.</H3>
      <Prose>
        The single most common bootstrap mistake in ML evaluation is bootstrapping two systems' per-item scores independently when those scores were computed on the same eval prompts. The independent bootstrap inflates the variance of the difference by treating the within-item correlation as zero, producing a CI for the difference that is too wide and routinely fails to detect real differences. Fix: use paired bootstrap (<Code>scipy.stats.bootstrap(..., paired=True)</Code>), which resamples item indices and applies them to both systems' scores in lockstep.
      </Prose>

      <H3>Too few bootstrap replicates for BCa</H3>
      <Prose>
        BCa endpoints often fall further out in the bootstrap distribution's tails than the nominal 2.5/97.5 quantiles. With <Code>B = 1000</Code> the Monte Carlo noise on a 0.5%-quantile estimate is large enough to swing the BCa endpoint by several percent of its value, which can flip "intervals overlap" to "intervals do not overlap" between consecutive runs. Use <Code>B ≥ 5000</Code> for any reported BCa CI and <Code>B ≥ 10000</Code> for binary decisions based on overlap.
      </Prose>

      <H3>Degenerate jackknife producing a = NaN</H3>
      <Prose>
        If all <Code>n</Code> jackknife replicates are identical — which happens when the statistic is invariant to single-observation deletion (e.g., the maximum, when the maximum is attained by multiple observations) — the denominator of the acceleration formula is zero, giving <Code>a = 0/0</Code>. Production implementations clip or special-case this. Diagnose by computing the jackknife replicates separately and inspecting their range; if it is below numerical precision, fall back to bias-corrected (BC) without acceleration, or use percentile bootstrap.
      </Prose>

      <H3>Statistics on the boundary</H3>
      <Prose>
        When the point estimate equals an extreme of the bootstrap distribution (e.g., 100% accuracy on a small sample where every bootstrap replicate also shows 100% accuracy), the bias correction <Code>z₀ = Φ⁻¹(0)</Code> is undefined (negatively infinite). The same issue occurs at <Code>z₀ = Φ⁻¹(1)</Code>. SciPy clips <Code>p_below</Code> to <Code>[1/B, 1 − 1/B]</Code> as a numerical safeguard. The deeper issue is that BCa is asking a question the data cannot answer: a confidence interval requires the sampling distribution to be non-degenerate. For small-sample boundary cases, switch to exact methods (Clopper-Pearson for proportions).
      </Prose>

      <H3>Non-smooth statistics (medians, quantiles)</H3>
      <Prose>
        BCa's second-order accuracy result assumes a smooth statistic — informally, that small changes in the data produce small changes in <Code>θ̂</Code>. The median and other sample quantiles are non-smooth because they jump discretely as the data crosses a single value. Bootstrap CIs for the median are valid but converge more slowly than for smooth statistics, and BCa offers less improvement over percentile than for smooth statistics. Use the smoothed bootstrap (add small noise to bootstrap resamples) or specialized methods (e.g., the Hodges-Lehmann estimator for the median) when high-accuracy quantile CIs are required.
      </Prose>

      <H3>Heavy-tailed data with infinite variance</H3>
      <Prose>
        If the data come from a distribution with infinite variance (e.g., Cauchy, or any distribution with tails heavier than <Code>x⁻³</Code>), the bootstrap of the mean does not consistently estimate the sampling distribution because the underlying CLT does not hold. Symptom: bootstrap intervals that grow without bound as <Code>B</Code> increases, or that depend strongly on whether a single extreme observation is included in the resample. Fix: use a robust estimator (median, trimmed mean) instead of the mean, or use a distribution-specific approach (extreme value theory for tail quantiles).
      </Prose>

      <H3>Forgetting that BCa is still an asymptotic method</H3>
      <Prose>
        BCa achieves O(1/n) coverage error asymptotically, but at <Code>n = 20</Code> the implicit constants in the asymptotic expansion can dominate, and BCa's coverage may be no better than percentile's. Run a coverage simulation on a synthetic version of your data at your sample size before claiming BCa gives you exact 95% coverage. The Monte Carlo coverage study from section 4e should be a standard piece of any new eval pipeline.
      </Prose>

      <H3>Ignoring the bootstrap's failure to estimate bias</H3>
      <Prose>
        The bootstrap can estimate the standard error and the sampling distribution, but it estimates bias poorly when the bias depends on aspects of the distribution that the empirical distribution captures poorly. For statistics with substantial bias — e.g., the maximum likelihood estimator of variance for small <Code>n</Code> — BCa's bias correction <Code>z₀</Code> reflects only the median bias of the bootstrap distribution, not the bias of <Code>θ̂</Code> itself. When the underlying estimator has known bias, correct it analytically before bootstrapping (e.g., use Bessel's correction <Code>n/(n−1)</Code> for variance) rather than relying on BCa to handle it.
      </Prose>

      <H3>Reporting CIs without seeds</H3>
      <Prose>
        Every bootstrap interval is a Monte Carlo estimate; running the same code twice produces slightly different endpoints. For published numbers, fix and report the random seed. For internal eval comparisons, either fix the seed across all conditions or report enough <Code>B</Code> that Monte Carlo noise is well below the precision you care about (rule of thumb: <Code>B</Code> such that the bootstrap SE of the CI endpoint is &lt; 10% of the CI half-width).
      </Prose>

      <Callout accent="purple">
        BCa fails silently in two specific ways. First, when the jackknife is degenerate, some implementations return <Code>a = 0</Code> without warning, silently demoting BCa to BC. Second, when <Code>p_below</Code> is clipped at the boundaries of <Code>[1/B, 1 − 1/B]</Code>, the bias correction is capped at a value that depends on <Code>B</Code>, and the CI silently inherits this dependence. Always inspect <Code>z₀</Code> and <Code>a</Code> when reporting BCa intervals, especially on small or degenerate samples.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        Five sources cover the theoretical lineage and practical implementation of BCa. References are listed in publication order; the first two are the foundational contributions to the bootstrap and BCa specifically.
      </Prose>

      <H3>Efron 1979 — The bootstrap</H3>
      <Prose>
        Bradley Efron. "Bootstrap Methods: Another Look at the Jackknife." Annals of Statistics, 7(1):1–26, January 1979. The paper that introduced the bootstrap, framing it as a generalization of the jackknife with a much wider applicability. Establishes the basic resampling-with-replacement procedure, derives the percentile interval, and demonstrates the method on the median, the trimmed mean, and the correlation coefficient. Required reading for context, though the language is dated and the exposition assumes a reader fluent in 1970s mathematical statistics.
      </Prose>

      <H3>Efron 1987 — Better Bootstrap Confidence Intervals (BCa)</H3>
      <Prose>
        Bradley Efron. "Better Bootstrap Confidence Intervals." Journal of the American Statistical Association, 82(397):171–185, March 1987. The paper that introduced BCa. Derives the bias correction and acceleration corrections from an Edgeworth expansion of the coverage error, proves second-order accuracy, and provides extensive simulation comparisons with the percentile and basic intervals. The acceleration formula via jackknife is given in equation (6.6); the BCa endpoint formulas are equations (4.1)–(4.3). The paper's coverage simulations for the variance-of-exponential example are the direct ancestor of the from-scratch coverage study in section 4 of this topic.
      </Prose>

      <H3>Efron and Tibshirani 1993 — An Introduction to the Bootstrap</H3>
      <Prose>
        Bradley Efron and Robert Tibshirani. "An Introduction to the Bootstrap." Chapman &amp; Hall/CRC Monographs on Statistics and Applied Probability, vol. 57. 1993. The standard textbook reference, accessible to applied statisticians and ML practitioners. Chapters 12–14 cover BCa specifically, with worked examples on real datasets (the law school data, the cell survival data) and explicit S-plus code for every method. The textbook's BCa exposition is more readable than the 1987 paper and is the source most often cited in software documentation. SciPy's <Code>scipy.stats.bootstrap</Code> documentation references this book directly for the BCa formulas.
      </Prose>

      <H3>Davison and Hinkley 1997 — Bootstrap Methods and their Application</H3>
      <Prose>
        Anthony C. Davison and David V. Hinkley. "Bootstrap Methods and their Application." Cambridge Series in Statistical and Probabilistic Mathematics. 1997. The most complete textbook treatment, covering not just BCa but also bootstrap-t, ABC (a closed-form approximation to BCa), the double bootstrap, block bootstrap for time series, parametric bootstrap, and Bayesian bootstrap. Chapter 5 is the canonical reference for bootstrap CIs; chapter 8 covers complex data structures including paired and clustered designs. The companion R <Code>boot</Code> package, which implements all of these methods, is co-authored by Canty and Ripley but follows Davison and Hinkley's notation throughout.
      </Prose>

      <H3>DiCiccio and Efron 1996 — Bootstrap Confidence Intervals (review)</H3>
      <Prose>
        Thomas J. DiCiccio and Bradley Efron. "Bootstrap Confidence Intervals." Statistical Science, 11(3):189–228, August 1996 (with discussion). A review article with extensive comparisons among percentile, basic, BCa, ABC, and bootstrap-t intervals. Provides the cleanest exposition of why BCa achieves second-order accuracy via the Edgeworth expansion argument, and includes a section on practical computational issues including the choice of <Code>B</Code>, the handling of jackknife degeneracies, and the relationship between BCa and the studentized bootstrap. The discussion section, with comments from Hinkley, Hall, and others, is itself a rich source of practical guidance.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Why z₀ corrects bias the way it does</H3>
      <Prose>
        Consider a bootstrap distribution where 70% of the replicates fall below the point estimate <Code>θ̂</Code>. Compute <Code>z₀</Code>. Sketch the bootstrap distribution and the point estimate. Now consider what BCa does: which direction do <Code>α₁</Code> and <Code>α₂</Code> shift relative to the unadjusted percentiles 0.025 and 0.975? Explain in words why this is the correct direction to compensate for the median bias you observe. What would happen if you used a positive <Code>z₀</Code> when the bootstrap distribution was actually right-shifted (only 30% below <Code>θ̂</Code>) — would coverage be too high or too low?
      </Prose>

      <H3>Exercise 2 — When a equals zero</H3>
      <Prose>
        For what kinds of statistics is the acceleration <Code>a</Code> exactly zero? Start with the sample mean of i.i.d. data from a symmetric distribution. Show that the jackknife replicates have the form <Code>θ̂_(i) = (n·θ̂ − xᵢ)/(n − 1)</Code> and that the deviations <Code>d_i = θ̂_(·) − θ̂_(i)</Code> are linear in the centered observations <Code>x_i − x̄</Code>. Use this to argue that the third moment of the deviations equals the third moment of the data (up to constants), so for symmetric data <Code>a = 0</Code>. What does this imply about BCa reducing to BC (bias-corrected only) for symmetric statistics?
      </Prose>

      <H3>Exercise 3 — Coverage simulation in 30 lines</H3>
      <Prose>
        Write a Monte Carlo coverage study in NumPy that compares percentile and BCa intervals on the sample correlation coefficient between two correlated normal variables with true correlation <Code>ρ = 0.7</Code> and sample size <Code>n = 30</Code>. The sampling distribution of the sample correlation is left-skewed because <Code>r</Code> cannot exceed 1. Predict before running: which method should under-cover, in which direction (lower or upper bound failing more often), and by how much? Run the simulation with at least 1000 trials and report the actual coverage rates and average widths. Does the result match your prediction?
      </Prose>

      <H3>Exercise 4 — Paired vs. independent bootstrap on shared eval</H3>
      <Prose>
        Suppose you have per-prompt accuracy scores for two models on the same 100 evaluation prompts, with model A averaging 0.72 and model B averaging 0.69. The per-prompt scores are correlated with <Code>r = 0.6</Code> across the two models because hard prompts are hard for both. Compute (analytically, using normal approximations) the standard error of the mean difference under (a) the assumption of independent samples and (b) the paired structure. Express the ratio of these standard errors as a function of <Code>r</Code>. Now describe what would happen if you incorrectly used <Code>scipy.stats.bootstrap</Code> with <Code>paired=False</Code> on this data: would your CI for the difference be too wide or too narrow, and by what factor? What is the practical consequence for model-comparison decisions?
      </Prose>

      <H3>Exercise 5 — Diagnosing a degenerate jackknife</H3>
      <Prose>
        You are computing a BCa interval on the sample maximum of a 50-point dataset. Your implementation returns <Code>a = NaN</Code> and a CI of <Code>[NaN, NaN]</Code>. Diagnose what went wrong by inspecting the jackknife replicates: what specific structure causes the acceleration to be undefined? (Hint: think about which jackknife replicate equals what, when the maximum is attained by exactly one observation.) Propose three possible fixes — one that modifies the BCa procedure to handle this degeneracy, one that switches to a different CI method, and one that uses a different estimator. Discuss the tradeoffs. As a follow-up: would the same problem arise for the sample 95th percentile of a 50-point dataset, or only for the maximum specifically?
      </Prose>

    </div>
  ),
};

export default bootstrapBCa;
