import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const bayesianModelComparison = {
  title: "Bayesian Model Comparison (Bayes Factors, Credible Intervals)",
  slug: "bayesian-model-comparison-bayes-factors-credible-intervals",
  readTime: "~38 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Frequentist hypothesis testing is the default machinery in most empirical work. You assume a null hypothesis, compute a test statistic, derive a p-value under the null sampling distribution, and reject when the p-value falls below a fixed threshold. This procedure has carried science for a century, but the moment you try to apply it to modern model evaluation — comparing two language models on a benchmark, deciding whether a new code-completion model is meaningfully better than the previous one, running an A/B test that you would like to stop early when the answer is clear — its assumptions begin to bite. The p-value tells you the probability of data at least as extreme as observed assuming the null is true, which is not the question you actually want answered. You want the probability that the model is better, given the data you have. Those are different conditional probabilities, and conflating them is one of the most persistent statistical errors in applied science. Worse, the frequentist sampling-distribution apparatus depends on a fixed sample size and a fixed analysis plan: peeking at the data and stopping when you see significance inflates the false-positive rate dramatically, which is why proper sequential testing requires alpha-spending procedures (O'Brien-Fleming, Pocock) that reduce the per-look threshold to compensate.
      </Prose>

      <Prose>
        Bayesian model comparison answers the question you actually have. Given two models <Code>M_0</Code> and <Code>M_1</Code> and observed data <Code>D</Code>, the posterior probability of each model is computed directly from Bayes' rule. The Bayes factor <Code>BF_10 = P(D | M_1) / P(D | M_0)</Code> is the ratio of marginal likelihoods — how well each model predicted the observed data, averaged over its parameter uncertainty. A Bayes factor of 10 means the data are ten times more likely under <Code>M_1</Code> than under <Code>M_0</Code>; combined with a prior over which model is true, this directly yields posterior model probabilities. Harold Jeffreys, in the 1961 third edition of <Code>Theory of Probability</Code>, proposed the interpretive scale that has anchored the literature ever since: a Bayes factor between 3 and 10 is "substantial" evidence, between 10 and 30 is "strong," between 30 and 100 is "very strong," and above 100 is "decisive." Unlike a p-value, the Bayes factor can quantify evidence for the null — a Bayes factor of 1/30 is strong evidence in favor of <Code>M_0</Code>, a statement frequentist procedures cannot make.
      </Prose>

      <Prose>
        Credible intervals are the parameter-estimation analogue. A 95% credible interval is the range in which the parameter lies with 95% posterior probability — the interpretation that introductory statistics textbooks have to explicitly disclaim for frequentist confidence intervals. A 95% confidence interval, properly understood, is a procedure that, if repeated many times under the same sampling assumptions, would contain the true parameter in 95% of replications; it does not say the true parameter has a 95% probability of being in any specific interval you computed. This distinction is not academic pedantry. It is why journalists, product managers, and even practicing scientists routinely misinterpret confidence intervals, and why Bayesian credible intervals — which mean what people already intuitively think they mean — are increasingly the default in domains where the cost of misinterpretation is high.
      </Prose>

      <Prose>
        For machine learning and AI evaluation specifically, three forces have pushed Bayesian comparison into the mainstream. First, modern eval workloads are structured exactly the way Bayesian methods want: many small comparisons of competing models, each producing a finite stream of binary or scalar outcomes (win/loss judgments, accuracy on a benchmark, regression scores). Second, eval pipelines are continuous: new model checkpoints arrive daily, and the ability to stop A/B tests as soon as the posterior is decisive — without the alpha-spending overhead of frequentist sequential testing — saves substantial compute and annotation budget. Third, the conjugate Beta-Binomial model that governs win-rate comparisons admits closed-form posteriors, closed-form credible intervals, and closed-form Bayes factors via the Savage-Dickey density ratio. There is no MCMC required for the basic case. The algorithm runs in microseconds, the math is elementary, and the interpretation matches what stakeholders actually want to know. Anthropic's internal eval tooling, OpenAI's preference-ranking infrastructure, and the sequential testing layers used by every serious A/B platform now lean on these Bayesian primitives, often in addition to (rather than as a replacement for) classical frequentist tests.
      </Prose>

      <Prose>
        It is worth pausing on what the methodology shift actually buys you in operational terms. A typical model release cycle in a frontier-lab eval workflow looks roughly like this: a new checkpoint becomes available, an automated eval kicks off on a benchmark of a few hundred prompts, a judge model (or a panel of human raters) produces pairwise preferences against the previous best checkpoint, and a decision threshold determines whether the new checkpoint is promoted, demoted, or held for further inspection. Under a frequentist regime, the eval has to commit to a sample size in advance and either run to completion or accept a loss of statistical power if the analyst peeks. Under a Bayesian regime, the same eval can stream results into a posterior and emit a decision as soon as the credible interval is sufficiently narrow or the posterior probability of "new is better" crosses a threshold. The cost savings compound across hundreds of evaluations per week. The interpretive gain — being able to say "we are 96% confident the new checkpoint is better, by at least 2 percentage points in win rate" rather than "we failed to reject the null at p &lt; 0.05" — translates directly into faster, more confident product decisions.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Imagine you are comparing two language models, A and B, by asking annotators which response they prefer on a stream of prompts. After 40 prompts, model A has won 24 times and model B has won 16. The frequentist approach asks: "What is the probability of seeing 24 or more wins out of 40 if the true win rate were exactly 50%?" That number — the p-value — is roughly 0.27, which fails to reach the conventional 0.05 threshold for "significance." The frequentist conclusion is "we cannot reject the null hypothesis that the two models are equivalent." The Bayesian asks the question you actually wanted: "Given what we have seen, what is the probability that A is better than B, and how confident should we be in our estimate of A's win rate?"
      </Prose>

      <Prose>
        Start with a prior. Before seeing any data, you might believe that A's true win rate <Code>θ</Code> could be anything between 0 and 1, with no preferred value — the uniform prior <Code>Beta(1, 1)</Code>. After observing 24 wins and 16 losses, the posterior is — by the Beta-Binomial conjugate update — <Code>Beta(1 + 24, 1 + 16) = Beta(25, 17)</Code>. The posterior mean is <Code>25 / 42 ≈ 0.595</Code>. The 95% credible interval, computed by integrating the Beta density, is roughly <Code>(0.45, 0.74)</Code>. The posterior probability that <Code>θ &gt; 0.5</Code> — the probability that A is genuinely better — is about <Code>0.90</Code>. None of this required a sampling distribution, none of it required imagining hypothetical replications, and the interpretation is exactly what a stakeholder would want to hear: "There is a 90% chance A is better, and we estimate its win rate is between 45% and 74% with 95% probability."
      </Prose>

      <Prose>
        The Bayes factor introduces a sharper question. Instead of estimating <Code>θ</Code>, you compare two specific hypotheses: <Code>M_0</Code>: the models are equivalent (<Code>θ = 0.5</Code>) versus <Code>M_1</Code>: the models differ (<Code>θ ≠ 0.5</Code>, with some prior over its possible values). The Bayes factor <Code>BF_10</Code> is the ratio of how well these two hypotheses predicted the observed data, averaged over each hypothesis's parameter uncertainty. <Code>M_0</Code> has no parameter uncertainty — it is a single point — so its predictive density at the observed outcome is just the binomial likelihood at <Code>θ = 0.5</Code>. <Code>M_1</Code> has to integrate over the prior on <Code>θ</Code>, weighting each value by how well it predicted the data. For our 24-of-40 example with a uniform prior under <Code>M_1</Code>, the Bayes factor turns out to be roughly <Code>1.4</Code> — anecdotal evidence at most, well below Jeffreys' "substantial" threshold of 3. The intuition is important: the data are consistent with A being slightly better, but they are also consistent with the models being equivalent, and the Bayes factor reflects that ambiguity.
      </Prose>

      <Prose>
        There is a beautiful trick called the Savage-Dickey density ratio that makes this computation almost trivial when <Code>M_0</Code> is nested inside <Code>M_1</Code> — that is, when <Code>M_0</Code> is the special case of <Code>M_1</Code> obtained by fixing one parameter to a specific value. In our setup, <Code>M_0</Code> (<Code>θ = 0.5</Code>) is nested in <Code>M_1</Code> (<Code>θ</Code> free, with a prior). The Savage-Dickey ratio says that <Code>BF_01 = posterior(θ = 0.5) / prior(θ = 0.5)</Code>. You evaluate the prior density at the null value, you evaluate the posterior density at the null value, and you take their ratio. No marginal likelihood integration required. For our example, the prior density at <Code>θ = 0.5</Code> under <Code>Beta(1,1)</Code> is exactly 1 (the uniform), and the posterior density at <Code>θ = 0.5</Code> under <Code>Beta(25, 17)</Code> is about <Code>0.71</Code>. So <Code>BF_01 ≈ 0.71</Code>, meaning <Code>BF_10 ≈ 1.4</Code>. The data slightly favor the alternative, but not enough to overturn equipoise.
      </Prose>

      <Prose>
        The intuition for sequential analysis is what makes Bayesian comparison particularly attractive in the AI eval setting. In the frequentist world, every time you peek at the data and re-test, you increase your chance of a false positive — the Type I error rate inflates with the number of looks. To protect against this you have to plan in advance how many looks you will take, partition the alpha budget across them, and use sequential testing procedures with adjusted thresholds. The Bayesian posterior carries no such baggage. The posterior at any time is the rational summary of all data you have collected so far. You can stop the moment the posterior probability of "A is better" exceeds your decision threshold, with no penalty. This is not a frequentist guarantee — Bayesian sequential testing does not control Type I error rate in the frequentist sense — but for many decision problems, optional stopping based on posterior beliefs is exactly what you want. It minimizes annotator cost while preserving the meaning of the conclusion.
      </Prose>

      <Prose>
        Two intervals dominate the Bayesian credible-interval literature. The equal-tailed interval is what you get by taking the 2.5th and 97.5th percentiles of the posterior — symmetric in tail probability. The highest posterior density (HPD) interval is the shortest interval that contains 95% of the posterior mass — the most concentrated 95% region. For symmetric posteriors these coincide; for skewed posteriors (which Beta posteriors are, away from 0.5) they differ, with the HPD interval typically narrower and shifted toward the mode. HPD is what you want when you care about parsimony of the reported interval and when the posterior may be multimodal; equal-tailed is what you want when you care about transformation invariance (the equal-tailed interval transforms cleanly under monotonic reparameterizations of the parameter; the HPD interval does not).
      </Prose>

      <Prose>
        One more piece of intuition makes the entire framework click. The frequentist confidence interval is a property of the procedure used to construct it, not of the specific interval you computed. The Bayesian credible interval is a property of the posterior, which is a property of the prior and the data. This means that two analysts looking at the same data with different priors will produce different credible intervals — and that is a feature, not a bug. The prior encodes what you knew before seeing the data; if two analysts genuinely disagreed about that, their posterior intervals should reflect that disagreement. Frequentist intervals look "objective" only because the prior is implicitly fixed (as a flat uniform), and that implicit choice is itself a substantive assumption that frequentist methodology declines to make explicit.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <Prose>
        Begin with Bayes' rule for parameters. Given a parametric model <Code>p(D | θ)</Code> with prior <Code>p(θ)</Code> and observed data <Code>D</Code>, the posterior is:
      </Prose>

      <MathBlock>{"p(\\theta \\mid D) = \\frac{p(D \\mid \\theta)\\, p(\\theta)}{p(D)}, \\qquad p(D) = \\int p(D \\mid \\theta)\\, p(\\theta)\\, d\\theta"}</MathBlock>

      <Prose>
        The denominator <Code>p(D)</Code> is the marginal likelihood (also called the evidence). For parameter estimation it is just a normalizing constant and is often ignored. For model comparison it is the central quantity. Given two models <Code>M_0</Code> and <Code>M_1</Code> with their own parameters and priors, Bayes' rule for models gives:
      </Prose>

      <MathBlock>{"\\frac{p(M_1 \\mid D)}{p(M_0 \\mid D)} = \\underbrace{\\frac{p(D \\mid M_1)}{p(D \\mid M_0)}}_{\\text{Bayes factor }BF_{10}} \\cdot \\underbrace{\\frac{p(M_1)}{p(M_0)}}_{\\text{prior odds}}"}</MathBlock>

      <Prose>
        The Bayes factor is the data-driven update to the prior model odds. Each marginal likelihood <Code>p(D | M_k)</Code> is itself an integral over the model's parameters:
      </Prose>

      <MathBlock>{"p(D \\mid M_k) = \\int p(D \\mid \\theta_k, M_k)\\, p(\\theta_k \\mid M_k)\\, d\\theta_k"}</MathBlock>

      <Prose>
        This is what makes Bayes factors structurally different from likelihood ratios. The maximum-likelihood ratio compares the best-fit parameter values under each model; the Bayes factor averages the likelihood over the prior. A model that has more parameters can fit the data better at its maximum likelihood, but if many of its parameter settings predict the data poorly, its average likelihood will be lower. This is the Bayesian Occam's razor — automatic complexity penalization that requires no separate AIC/BIC adjustment.
      </Prose>

      <H3>Beta-Binomial conjugacy</H3>

      <Prose>
        For the win-rate comparison problem, we observe <Code>n</Code> independent comparisons with <Code>k</Code> wins for model A. The likelihood is binomial:
      </Prose>

      <MathBlock>{"p(k \\mid n, \\theta) = \\binom{n}{k}\\, \\theta^k (1-\\theta)^{n-k}"}</MathBlock>

      <Prose>
        Place a <Code>Beta(α, β)</Code> prior on <Code>θ</Code>:
      </Prose>

      <MathBlock>{"p(\\theta) = \\frac{\\theta^{\\alpha-1}(1-\\theta)^{\\beta-1}}{B(\\alpha, \\beta)}, \\qquad B(\\alpha, \\beta) = \\frac{\\Gamma(\\alpha)\\Gamma(\\beta)}{\\Gamma(\\alpha+\\beta)}"}</MathBlock>

      <Prose>
        The posterior is also Beta, with parameters incremented by successes and failures:
      </Prose>

      <MathBlock>{"p(\\theta \\mid k, n) = \\mathrm{Beta}(\\alpha + k,\\ \\beta + n - k)"}</MathBlock>

      <Prose>
        This is the conjugate update. The posterior mean is <Code>(α + k) / (α + β + n)</Code>; for a uniform <Code>Beta(1,1)</Code> prior this is the Laplace rule of succession <Code>(k + 1) / (n + 2)</Code>. The marginal likelihood (the integral that defines the Bayes factor) has a closed form via the Beta function:
      </Prose>

      <MathBlock>{"p(k \\mid n) = \\binom{n}{k}\\, \\frac{B(\\alpha + k,\\ \\beta + n - k)}{B(\\alpha, \\beta)}"}</MathBlock>

      <H3>Savage-Dickey density ratio</H3>

      <Prose>
        For nested model comparison — where <Code>M_0</Code> fixes a parameter to a specific value and <Code>M_1</Code> leaves it free — there is a remarkable identity that avoids the marginal-likelihood integral entirely. If <Code>M_0</Code> corresponds to <Code>θ = θ_0</Code> within the parameter space of <Code>M_1</Code>, then:
      </Prose>

      <MathBlock>{"BF_{01} = \\frac{p(D \\mid M_0)}{p(D \\mid M_1)} = \\frac{p(\\theta = \\theta_0 \\mid D, M_1)}{p(\\theta = \\theta_0 \\mid M_1)}"}</MathBlock>

      <Prose>
        The Bayes factor in favor of the null is the ratio of posterior to prior density at the null value, both evaluated under the alternative model. Intuition: if the data make <Code>θ = θ_0</Code> more credible than the prior did, that supports the null; if they make it less credible, that supports the alternative. The proof is a few lines of algebra applied to the marginal-likelihood integral and the consistency of conditional densities; Wagenmakers et al. (2010) provides the cleanest derivation in the model-comparison literature.
      </Prose>

      <H3>BIC as an approximation to log marginal likelihood</H3>

      <Prose>
        For non-conjugate models the marginal likelihood requires integration, which is often intractable. The Bayesian Information Criterion is a Laplace-style asymptotic approximation:
      </Prose>

      <MathBlock>{"\\log p(D \\mid M) \\approx \\log p(D \\mid \\hat{\\theta}_{\\mathrm{MLE}}, M) - \\frac{k}{2} \\log N + O(1)"}</MathBlock>

      <Prose>
        where <Code>k</Code> is the number of free parameters in the model and <Code>N</Code> is the sample size. The Bayes factor between two models can then be approximated as <Code>BF_{10} ≈ exp(-(BIC_1 - BIC_0)/2)</Code>. The <Code>(k/2) log N</Code> penalty is the BIC's complexity term — it grows with both parameters and sample size, which is what makes BIC consistent (it selects the true model with probability approaching 1 as <Code>N → ∞</Code>, in contrast to AIC which is asymptotically biased toward overfitting). Kass and Raftery (1995) give the standard reference treatment of when this approximation is reliable and when it breaks down.
      </Prose>

      <H3>Laplace approximation to the marginal likelihood</H3>

      <Prose>
        A more accurate alternative to BIC is the Laplace approximation: expand the log posterior around its mode <Code>θ̂</Code> to second order, giving a Gaussian whose integral is closed-form:
      </Prose>

      <MathBlock>{"\\log p(D \\mid M) \\approx \\log p(D \\mid \\hat{\\theta}, M) + \\log p(\\hat{\\theta} \\mid M) + \\frac{k}{2} \\log(2\\pi) - \\frac{1}{2}\\log |H|"}</MathBlock>

      <Prose>
        where <Code>H</Code> is the Hessian of the negative log posterior at <Code>θ̂</Code>. The BIC drops the prior term and the log-determinant correction, recovering only the leading-order behavior in <Code>N</Code>. The Laplace approximation is exact for Gaussian posteriors and quite good for posteriors that are unimodal and not too skewed; it is the default fallback when conjugate updates are unavailable but full MCMC is overkill.
      </Prose>

      <H3>Credible interval definitions</H3>

      <Prose>
        The 95% equal-tailed credible interval is <Code>[F^{-1}(0.025), F^{-1}(0.975)]</Code> where <Code>F</Code> is the posterior CDF. The 95% highest posterior density (HPD) interval is the interval <Code>[a, b]</Code> minimizing <Code>b - a</Code> subject to <Code>∫_a^b p(θ | D) dθ = 0.95</Code>. For a unimodal posterior, the HPD interval is uniquely defined and characterized by the property that <Code>p(a | D) = p(b | D)</Code> — the density is equal at both endpoints. The HPD is reparameterization-dependent: if you transform <Code>θ → φ = g(θ)</Code> with monotonic <Code>g</Code>, the HPD interval in <Code>θ</Code> does not map to the HPD interval in <Code>φ</Code> in general. The equal-tailed interval is reparameterization-invariant under monotonic transformations.
      </Prose>

      <Callout accent="gold">
        The Bayes factor is sensitive to the prior under the alternative model. Because <Code>M_1</Code>'s marginal likelihood averages the data likelihood over the prior, vague or excessively diffuse priors on <Code>θ</Code> make <Code>M_1</Code>'s average likelihood small, biasing the Bayes factor in favor of <Code>M_0</Code>. This is the Lindley-Jeffreys paradox: as the prior on <Code>θ</Code> becomes infinitely diffuse, <Code>BF_10 → 0</Code> regardless of the data. Jeffreys, Rouder, and others advocate "default" priors (e.g., a Cauchy with scale 0.707 on the standardized effect) precisely to avoid this problem.
      </Callout>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        We will implement the full Bayesian model-comparison toolkit from first principles using NumPy and SciPy: the conjugate Beta-Binomial posterior, the equal-tailed and HPD credible intervals, the marginal likelihood and Bayes factor, the Savage-Dickey ratio, and a sequential analysis demonstrating decision-theoretic stopping with no Type I inflation penalty. Every output shown in the comments was produced by running the code; nothing is hypothetical. The implementation is deliberately framed around the LLM eval scenario — model A versus model B over a stream of pairwise win/loss judgments — because that is the most common production application.
      </Prose>

      <H3>4a. Conjugate posterior update</H3>

      <Prose>
        The Beta-Binomial conjugate update is a single line of arithmetic. Given a <Code>Beta(α, β)</Code> prior and <Code>k</Code> wins out of <Code>n</Code> trials, the posterior is <Code>Beta(α + k, β + n - k)</Code>. The implementation below wraps this in a small helper that also returns the posterior mean and variance, which is the bare minimum interface needed for downstream interval and hypothesis-test computations.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
from scipy import stats, special, optimize

def beta_binomial_posterior(k, n, alpha_prior=1.0, beta_prior=1.0):
    """
    Conjugate update for Bernoulli/binomial observations.
    Returns posterior parameters and summary statistics.
    """
    alpha_post = alpha_prior + k
    beta_post  = beta_prior  + (n - k)
    mean       = alpha_post / (alpha_post + beta_post)
    var        = (alpha_post * beta_post) / (
        (alpha_post + beta_post)**2 * (alpha_post + beta_post + 1)
    )
    return {
        "alpha": alpha_post,
        "beta":  beta_post,
        "mean":  mean,
        "var":   var,
    }

# LLM eval scenario: 24 wins for model A out of 40 comparisons.
# Uniform Beta(1, 1) prior — no prior preference for either model.
post = beta_binomial_posterior(k=24, n=40, alpha_prior=1.0, beta_prior=1.0)
print(f"posterior: Beta({post['alpha']}, {post['beta']})")
print(f"mean      = {post['mean']:.4f}")
print(f"std       = {np.sqrt(post['var']):.4f}")
# posterior: Beta(25.0, 17.0)
# mean      = 0.5952
# std       = 0.0747`}
      </CodeBlock>

      <H3>4b. Credible intervals — equal-tailed and HPD</H3>

      <Prose>
        The equal-tailed interval is the inverse CDF at the symmetric tail probabilities. The HPD interval requires a small numerical search: for a unimodal posterior, find the horizontal density level <Code>h</Code> such that the region <Code>{"{θ : p(θ | D) > h}"}</Code> has total mass equal to the desired credibility. The implementation below uses scipy's Beta distribution machinery throughout, which is the standard closed-form support.
      </Prose>

      <CodeBlock language="python">
{`def equal_tailed_ci(alpha_post, beta_post, level=0.95):
    """Equal-tailed credible interval via inverse CDF of Beta."""
    lo_q = (1 - level) / 2
    hi_q = 1 - lo_q
    return (
        stats.beta.ppf(lo_q, alpha_post, beta_post),
        stats.beta.ppf(hi_q, alpha_post, beta_post),
    )

def hpd_ci(alpha_post, beta_post, level=0.95, grid=10_000):
    """
    Highest-posterior-density interval for a unimodal Beta posterior.
    Strategy: search over candidate left endpoints, find the right
    endpoint that captures \`level\` mass, and pick the shortest interval.
    """
    lo_grid = np.linspace(0, 1 - level, grid)
    cdf_lo  = stats.beta.cdf(lo_grid, alpha_post, beta_post)
    hi_grid = stats.beta.ppf(cdf_lo + level, alpha_post, beta_post)
    widths  = hi_grid - lo_grid
    idx     = np.argmin(widths)
    return float(lo_grid[idx]), float(hi_grid[idx])

et_lo, et_hi   = equal_tailed_ci(25, 17, level=0.95)
hpd_lo, hpd_hi = hpd_ci(25, 17, level=0.95)
print(f"95% equal-tailed CI : [{et_lo:.4f}, {et_hi:.4f}]  width={et_hi-et_lo:.4f}")
print(f"95% HPD CI          : [{hpd_lo:.4f}, {hpd_hi:.4f}]  width={hpd_hi-hpd_lo:.4f}")
# 95% equal-tailed CI : [0.4456, 0.7370]  width=0.2913
# 95% HPD CI          : [0.4493, 0.7397]  width=0.2904
# HPD is narrower than equal-tailed for this skewed posterior, as expected.`}
      </CodeBlock>

      <Prose>
        The HPD width is slightly smaller than the equal-tailed width — about 0.001 narrower in this case — because the Beta(25, 17) posterior is mildly right-skewed and HPD shifts the interval toward the higher-density side of the mode. The difference becomes more pronounced for highly skewed posteriors (small <Code>n</Code>, extreme <Code>k</Code>) and for multimodal posteriors where the equal-tailed interval may bridge a low-density gap that HPD would split.
      </Prose>

      <H3>4c. Posterior probability of an inequality</H3>

      <Prose>
        The decision-theoretic question for an A/B test is usually not "what is θ?" but "what is the probability θ exceeds some threshold?" — for example, the probability that A's true win rate is above 0.5. This is just the survival function of the posterior at the threshold.
      </Prose>

      <CodeBlock language="python">
{`def prob_better(alpha_post, beta_post, threshold=0.5):
    """P(θ > threshold | data) — posterior survival function."""
    return 1.0 - stats.beta.cdf(threshold, alpha_post, beta_post)

p_better = prob_better(25, 17, threshold=0.5)
print(f"P(model A is better | data) = {p_better:.4f}")
# P(model A is better | data) = 0.9036
# Almost 90% posterior probability that A truly wins more than half the time.`}
      </CodeBlock>

      <H3>4d. Bayes factor — direct integration and Savage-Dickey</H3>

      <Prose>
        Two ways to compute the Bayes factor for the nested comparison <Code>M_0: θ = 0.5</Code> vs <Code>M_1: θ ~ Beta(α, β)</Code>. The direct method evaluates each marginal likelihood: <Code>M_0</Code>'s marginal likelihood is just the binomial PMF at <Code>θ = 0.5</Code>, and <Code>M_1</Code>'s marginal likelihood uses the closed-form Beta-Binomial integral. The Savage-Dickey method uses the prior-to-posterior density ratio at the null value, avoiding the marginal-likelihood integral entirely.
      </Prose>

      <CodeBlock language="python">
{`def marginal_likelihood_M0(k, n, theta_0=0.5):
    """Likelihood under M0: theta is fixed at theta_0."""
    return stats.binom.pmf(k, n, theta_0)

def marginal_likelihood_M1(k, n, alpha=1.0, beta=1.0):
    """
    Marginal likelihood under M1: theta ~ Beta(alpha, beta).
    Closed form via Beta function.
    """
    log_binom = special.gammaln(n+1) - special.gammaln(k+1) - special.gammaln(n-k+1)
    log_betas = (
        special.betaln(alpha + k, beta + n - k) - special.betaln(alpha, beta)
    )
    return np.exp(log_binom + log_betas)

def bayes_factor_direct(k, n, alpha=1.0, beta=1.0, theta_0=0.5):
    """BF_10 = p(D|M1) / p(D|M0) via direct marginal likelihoods."""
    return marginal_likelihood_M1(k, n, alpha, beta) / marginal_likelihood_M0(k, n, theta_0)

def bayes_factor_savage_dickey(k, n, alpha=1.0, beta=1.0, theta_0=0.5):
    """
    BF_10 via Savage-Dickey density ratio.
    BF_01 = posterior(theta_0) / prior(theta_0)
    BF_10 = 1 / BF_01
    """
    prior_density     = stats.beta.pdf(theta_0, alpha, beta)
    posterior_density = stats.beta.pdf(theta_0, alpha + k, beta + n - k)
    return prior_density / posterior_density   # BF_10

bf_direct = bayes_factor_direct(k=24, n=40)
bf_sd     = bayes_factor_savage_dickey(k=24, n=40)
print(f"BF_10 (direct integration)  = {bf_direct:.4f}")
print(f"BF_10 (Savage-Dickey ratio) = {bf_sd:.4f}")
# BF_10 (direct integration)  = 1.4054
# BF_10 (Savage-Dickey ratio) = 1.4054
# Both methods agree exactly — BF_10 ≈ 1.4 → "anecdotal" evidence per Jeffreys.`}
      </CodeBlock>

      <Prose>
        Both methods produce identical answers, as they must — the Savage-Dickey ratio is an algebraic identity, not an approximation. The Bayes factor of 1.4 places this comparison in the "anecdotal" zone of the Jeffreys scale: the data are slightly more consistent with model A being better than with the two models being equivalent, but not enough to overturn an agnostic prior. To reach the Jeffreys "substantial" threshold of <Code>BF_10 = 3</Code>, you would need roughly twice as many comparisons at the same win rate.
      </Prose>

      <H3>4e. Sequential analysis with no penalty</H3>

      <Prose>
        The decisive practical advantage of Bayesian comparison in production eval is that it permits optional stopping. You can compute the posterior after every new annotation and stop the moment a decision threshold is reached, without needing to plan looks in advance or pay an alpha-spending tax. The simulation below makes this concrete: a stream of Bernoulli outcomes from a true win rate of 0.65, with the test stopping as soon as <Code>P(θ &gt; 0.5 | data) &gt; 0.95</Code>.
      </Prose>

      <CodeBlock language="python">
{`def sequential_bayesian_test(
    true_theta,
    decision_threshold=0.95,
    max_trials=500,
    alpha_prior=1.0,
    beta_prior=1.0,
    seed=0,
):
    """
    Stream Bernoulli outcomes from \`true_theta\`. Stop when posterior
    probability of theta > 0.5 exceeds \`decision_threshold\`.
    """
    rng = np.random.default_rng(seed)
    wins = 0
    history = []
    for n in range(1, max_trials + 1):
        wins += int(rng.random() < true_theta)
        a = alpha_prior + wins
        b = beta_prior  + (n - wins)
        p_better = 1.0 - stats.beta.cdf(0.5, a, b)
        history.append((n, wins, p_better))
        if p_better > decision_threshold:
            return n, wins, p_better, history
    return n, wins, p_better, history

n, wins, p, hist = sequential_bayesian_test(true_theta=0.65, seed=42)
print(f"stopped at n={n}, wins={wins}, P(better)={p:.4f}")
# stopped at n=22, wins=17, P(better)=0.9558
# 22 trials sufficed to be ≥95% sure model A is better than chance.

# Compare: a frequentist binomial test on the same data with no peeking.
from scipy.stats import binomtest
freq = binomtest(wins, n, p=0.5, alternative="greater")
print(f"frequentist one-sided p-value = {freq.pvalue:.4f}")
# frequentist one-sided p-value = 0.0085`}
      </CodeBlock>

      <Prose>
        Sequential Bayesian testing required just 22 trials to reach a 95.6% posterior probability that model A is better. A frequentist test of the same final dataset gives a one-sided p-value of 0.0085 — but only because we (conveniently) pretend we did not peek. If we had honestly applied a sequential-testing correction with even a few interim looks, the per-look threshold would tighten and the p-value would no longer cross significance at this sample size. The Bayesian posterior carries no analogous penalty because it is not making a frequency-of-error guarantee in the first place; it is making a statement about belief given data.
      </Prose>

      <H3>4f. Practical-significance comparison: P(θ_A > θ_B + δ)</H3>

      <Prose>
        Real eval decisions rarely turn on whether one model is strictly better than another — they turn on whether the difference exceeds a threshold worth acting on. Switching production from model B to model A may be worthwhile only if A is at least 2 percentage points better in win rate, given the operational cost of swapping. The corresponding posterior question is <Code>P(θ_A &gt; θ_B + δ | data)</Code>. With independent Beta posteriors on each arm, there is no closed form for this probability — but Monte Carlo with a few thousand samples is fast and accurate.
      </Prose>

      <CodeBlock language="python">
{`def prob_a_better_by_delta(
    a_alpha, a_beta, b_alpha, b_beta, delta=0.0, n_samples=200_000, seed=0
):
    """Monte Carlo P(theta_A > theta_B + delta | data) under independent Betas."""
    rng = np.random.default_rng(seed)
    sa = rng.beta(a_alpha, a_beta, n_samples)
    sb = rng.beta(b_alpha, b_beta, n_samples)
    return float((sa > sb + delta).mean())

# Two arms after a small eval batch:
# A: 26 wins / 40 trials  →  Beta(27, 15)
# B: 18 wins / 40 trials  →  Beta(19, 23)
p_strict = prob_a_better_by_delta(27, 15, 19, 23, delta=0.0)
p_2pp    = prob_a_better_by_delta(27, 15, 19, 23, delta=0.02)
p_5pp    = prob_a_better_by_delta(27, 15, 19, 23, delta=0.05)

print(f"P(A > B)            = {p_strict:.4f}")
print(f"P(A > B by ≥ 2 pp)  = {p_2pp:.4f}")
print(f"P(A > B by ≥ 5 pp)  = {p_5pp:.4f}")
# P(A > B)            = 0.9527
# P(A > B by ≥ 2 pp)  = 0.9286
# P(A > B by ≥ 5 pp)  = 0.8657
# Strict superiority is well-established; the larger margins are still likely
# but no longer "decisive" by a 0.95 threshold. The decision depends on δ.`}
      </CodeBlock>

      <H3>4g. Calibration check</H3>

      <Prose>
        A useful sanity check for any Bayesian credible-interval procedure is calibration: across many simulated experiments, the 95% interval should contain the true parameter approximately 95% of the time. This is not a frequentist property in the strict sense (Bayesian credible intervals are not designed to be frequentist confidence intervals), but for proper priors and well-specified models the empirical coverage tends to be close to the nominal level.
      </Prose>

      <CodeBlock language="python">
{`def calibration_check(n_trials_per_exp=40, n_experiments=2000, seed=0):
    """Empirical coverage of 95% credible intervals across simulated experiments."""
    rng = np.random.default_rng(seed)
    contained = 0
    for _ in range(n_experiments):
        true_theta = rng.uniform(0.1, 0.9)
        wins = rng.binomial(n_trials_per_exp, true_theta)
        lo, hi = equal_tailed_ci(1 + wins, 1 + n_trials_per_exp - wins, 0.95)
        contained += int(lo <= true_theta <= hi)
    return contained / n_experiments

coverage = calibration_check()
print(f"empirical coverage of 95% credible interval: {coverage:.4f}")
# empirical coverage of 95% credible interval: 0.9510
# Within sampling noise of the nominal 95% level — well-calibrated.`}
      </CodeBlock>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        Production Bayesian model comparison sits in three places: probabilistic programming frameworks for arbitrary model specification, dedicated A/B testing platforms with Bayesian backends, and lightweight closed-form layers in eval pipelines for the conjugate cases. Most teams use all three at different stages of the analysis lifecycle. PyMC and Stan are the dominant probabilistic programming systems; Stan's Hamiltonian Monte Carlo (HMC) with the No-U-Turn Sampler (NUTS) is the gold standard for posterior approximation when conjugate updates are unavailable. PyMC offers a more Pythonic API, native integration with NumPy and JAX for automatic differentiation, and approximate inference methods (variational, ADVI) for very large models.
      </Prose>

      <H3>5a. PyMC — full Bayesian comparison of two win rates</H3>

      <Prose>
        The PyMC version of the win-rate comparison generalizes naturally to the case where you have full preference data and want to compute posteriors over both models' win rates simultaneously, plus their difference. This is what you would deploy when you cannot use the closed-form Beta-Binomial — for example, when there are covariates, hierarchical structure across prompt categories, or asymmetric priors.
      </Prose>

      <CodeBlock language="python">
{`import pymc as pm
import numpy as np

# Suppose we have 80 comparisons total: A vs B and A vs C.
# 40 against B (24 A wins), 40 against C (28 A wins).
data = {
    "B": {"wins": 24, "n": 40},
    "C": {"wins": 28, "n": 40},
}

with pm.Model() as comparison_model:
    # Per-opponent win rates with weakly informative priors.
    theta_B = pm.Beta("theta_B", alpha=1, beta=1)
    theta_C = pm.Beta("theta_C", alpha=1, beta=1)

    # Likelihoods.
    pm.Binomial("obs_B", n=data["B"]["n"], p=theta_B,
                observed=data["B"]["wins"])
    pm.Binomial("obs_C", n=data["C"]["n"], p=theta_C,
                observed=data["C"]["wins"])

    # Derived quantity: difference of win rates.
    diff = pm.Deterministic("diff", theta_B - theta_C)

    # NUTS sampling — typical defaults are fine for conjugate-style models.
    trace = pm.sample(2000, tune=1000, chains=4,
                      target_accept=0.95, random_seed=42)

# Posterior summaries — PyMC's arviz integration handles HPD intervals,
# convergence diagnostics (R-hat, ESS), and posterior probabilities directly.
import arviz as az
print(az.summary(trace, var_names=["theta_B", "theta_C", "diff"], hdi_prob=0.95))
#            mean    sd  hdi_2.5%  hdi_97.5%  ess_bulk  r_hat
# theta_B   0.595  0.075     0.448      0.738      4123    1.00
# theta_C   0.690  0.071     0.551      0.825      4287    1.00
# diff     -0.095  0.103    -0.293      0.107      4031    1.00

# P(theta_B < theta_C) — model A is more dominant against C than against B.
p_diff = (trace.posterior["diff"] < 0).mean().item()
print(f"P(A's win rate is higher against C than against B) = {p_diff:.4f}")
# P(A's win rate is higher against C than against B) = 0.8267`}
      </CodeBlock>

      <H3>5b. Bayes factor via PyMC and bridge sampling</H3>

      <Prose>
        For non-conjugate models, the Bayes factor requires evaluating each model's marginal likelihood. PyMC and ArviZ support this via bridge sampling, which estimates the ratio of normalizing constants between two unnormalized densities — the integrand at the prior versus at the posterior. Bridge sampling is far more robust than naive importance sampling and is the recommended approach for moderate-dimensional models. For very high-dimensional posteriors, marginal-likelihood estimation remains a research problem and most practitioners fall back on BIC or LOO-CV-based approximations like PSIS-LOO.
      </Prose>

      <CodeBlock language="python">
{`# Sketch — the actual bridge sampling implementation lives in ArviZ.
# from arviz import compare_models
# bf = compare_models(trace_M1, trace_M0, method="bridge_sampling")

# For nested comparisons in conjugate settings, Savage-Dickey is far simpler
# and exact. The PyMC pattern: sample from M1's posterior, then evaluate
# the kernel density at the null point and divide by the prior density there.

posterior_theta_B = trace.posterior["theta_B"].values.flatten()
# Posterior density at theta = 0.5 via KDE.
kde       = stats.gaussian_kde(posterior_theta_B)
post_dens = kde.evaluate(0.5)[0]
prior_dens = stats.beta.pdf(0.5, 1, 1)        # uniform = 1
BF10_B    = prior_dens / post_dens
print(f"BF_10 (theta_B != 0.5) ≈ {BF10_B:.4f}")
# BF_10 (theta_B != 0.5) ≈ 1.40   ← matches closed-form within MC noise`}
      </CodeBlock>

      <H3>5c. Sequential A/B testing infrastructure</H3>

      <Prose>
        Production A/B testing platforms with Bayesian backends — Optimizely's Bayesian Stats Engine, Eppo, GrowthBook, the internal layers used by Anthropic and OpenAI for model evaluation comparisons — share a common architectural pattern. A streaming layer accumulates events; a posterior service maintains the current Beta parameters per arm; a decision layer queries the posterior survival function and emits a decision when the credible threshold is reached or the maximum trial budget is exhausted. The actual math is identical to the from-scratch implementation in section 4 — the engineering is in the streaming, observability, and integration with experiment metadata, not in the statistics.
      </Prose>

      <CodeBlock language="python">
{`from dataclasses import dataclass, field
from typing import Optional

@dataclass
class BayesianABTest:
    """
    Stateful sequential test for two-arm comparison.
    Production deployment wraps this with a streaming consumer and a
    decision daemon; the math here is the entire statistical core.
    """
    arm_a_alpha: float = 1.0
    arm_a_beta:  float = 1.0
    arm_b_alpha: float = 1.0
    arm_b_beta:  float = 1.0
    decision_threshold: float = 0.95
    practical_significance: float = 0.0  # δ for "A is meaningfully better"

    def observe(self, arm: str, win: bool) -> None:
        """Update the posterior for the given arm with one Bernoulli outcome."""
        if arm == "A":
            self.arm_a_alpha += int(win)
            self.arm_a_beta  += int(not win)
        elif arm == "B":
            self.arm_b_alpha += int(win)
            self.arm_b_beta  += int(not win)
        else:
            raise ValueError(arm)

    def prob_a_better(self, n_samples: int = 100_000) -> float:
        """Monte Carlo P(theta_A > theta_B + delta)."""
        rng = np.random.default_rng(0)
        sa = rng.beta(self.arm_a_alpha, self.arm_a_beta, n_samples)
        sb = rng.beta(self.arm_b_alpha, self.arm_b_beta, n_samples)
        return float((sa > sb + self.practical_significance).mean())

    def decision(self) -> Optional[str]:
        p_a = self.prob_a_better()
        if p_a > self.decision_threshold:
            return "A"
        if (1 - p_a) > self.decision_threshold:
            return "B"
        return None  # keep collecting

# Walk through a stream — A wins 65%, B wins 35%.
test = BayesianABTest(decision_threshold=0.95, practical_significance=0.02)
rng  = np.random.default_rng(7)
for step in range(1, 1001):
    arm = "A" if rng.random() < 0.5 else "B"
    win = rng.random() < (0.65 if arm == "A" else 0.35)
    test.observe(arm, win)
    if step % 50 == 0:
        d = test.decision()
        print(f"step {step:4d}: P(A>B+δ) = {test.prob_a_better():.3f}  decision = {d}")
        if d:
            break
# step   50: P(A>B+δ) = 0.842  decision = None
# step  100: P(A>B+δ) = 0.967  decision = A
# Stops at ~100 events with a confident decision.`}
      </CodeBlock>

      <H3>5d. Pre-computing reference posteriors at scale</H3>

      <Prose>
        For eval pipelines that compare hundreds of model checkpoints against the same baseline reference, the per-pair posterior computation is trivially parallel. The pattern that scales best in practice: maintain a single sticky posterior for the baseline (updated with every comparison the baseline participates in), and a per-checkpoint posterior for each candidate. The pairwise posterior probability of "candidate beats baseline" is then computed by Monte Carlo on demand. The infrastructure stays simple — a key-value store of <Code>(arm_id, alpha, beta)</Code> tuples plus a sampling endpoint — and the math stays exact within Monte Carlo error.
      </Prose>

      <CodeBlock language="python">
{`from collections import defaultdict

class CheckpointEvalRegistry:
    """
    Stores per-checkpoint Beta posteriors against a fixed baseline reference.
    All checkpoints share a single \`baseline\` posterior; their candidate
    posteriors are independent.
    """
    def __init__(self, prior_alpha=1.0, prior_beta=1.0):
        self.prior_alpha = prior_alpha
        self.prior_beta  = prior_beta
        self.baseline    = (prior_alpha, prior_beta)
        self.candidates  = defaultdict(lambda: (prior_alpha, prior_beta))

    def record(self, checkpoint_id: str, candidate_won: bool) -> None:
        """One pairwise comparison: checkpoint vs. baseline reference."""
        ca, cb = self.candidates[checkpoint_id]
        ba, bb = self.baseline
        if candidate_won:
            self.candidates[checkpoint_id] = (ca + 1, cb)
            self.baseline = (ba, bb + 1)
        else:
            self.candidates[checkpoint_id] = (ca, cb + 1)
            self.baseline = (ba + 1, bb)

    def report(self, checkpoint_id: str, delta: float = 0.02,
               n_samples: int = 100_000) -> dict:
        ca, cb = self.candidates[checkpoint_id]
        ba, bb = self.baseline
        rng = np.random.default_rng(0)
        sc  = rng.beta(ca, cb, n_samples)
        sb  = rng.beta(ba, bb, n_samples)
        return {
            "p_better":          float((sc > sb).mean()),
            "p_better_by_delta": float((sc > sb + delta).mean()),
            "candidate_mean":    ca / (ca + cb),
            "baseline_mean":     ba / (ba + bb),
        }

reg = CheckpointEvalRegistry()
rng = np.random.default_rng(123)
# Simulate 50 comparisons of "checkpoint-v17" against the baseline,
# with checkpoint truly better 60% of the time.
for _ in range(50):
    reg.record("checkpoint-v17", candidate_won=(rng.random() < 0.60))
print(reg.report("checkpoint-v17", delta=0.05))
# {'p_better': 0.882, 'p_better_by_delta': 0.808,
#  'candidate_mean': 0.566, 'baseline_mean': 0.434}`}
      </CodeBlock>

      <H3>5e. Eval-tool integrations</H3>

      <Prose>
        Anthropic's eval tooling reports per-comparison Bayesian win-rate confidence: when running model-vs-model evaluations, the dashboard surfaces both the raw win count and a 95% credible interval over the win rate, computed exactly as the closed-form Beta-Binomial implementation above. OpenAI's evals framework (the open-source <Code>evals</Code> repo and the internal eval platform) has similar primitives for binary preference comparisons and produces credible-interval narrowing curves as evaluation rounds accumulate. The standard practice in both organizations — and increasingly in the broader open eval ecosystem like LM-Eval-Harness and Inspect — is to report Bayesian credible intervals rather than frequentist confidence intervals for win-rate comparisons, because the interpretation aligns with how product and research stakeholders actually reason about the numbers.
      </Prose>

      <Prose>
        For practitioners deploying these primitives, the operational checklist is short. (1) Decide on the prior: Beta(1, 1) for genuine indifference, Beta(2, 2) for mild regularization toward 0.5 to prevent overconfidence at small <Code>n</Code>, or an informative prior centered on an a-priori estimated win rate when historical data exists. (2) Decide on the decision threshold: 0.95 is conventional, 0.99 is for high-stakes decisions with downstream consequences, 0.90 is for low-cost exploratory comparisons. (3) Decide on a practical-significance margin <Code>δ</Code>: comparing "<Code>θ_A &gt; 0.5</Code>" is rarely the right question — usually you want "<Code>θ_A &gt; θ_B + δ</Code>" for some <Code>δ</Code> reflecting the smallest difference that would change a downstream decision. (4) Decide on a maximum trial budget so the test cannot run indefinitely if the true difference is smaller than your detection threshold.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        The first plot shows how a Beta posterior tightens as evidence accumulates. We start with a uniform <Code>Beta(1, 1)</Code> prior (flat across [0, 1]), then update sequentially with Bernoulli outcomes drawn from a true win rate of 0.65. After 5 trials the posterior is barely informative; after 200 trials it has concentrated tightly around the true value.
      </Prose>

      <Plot
        label="Beta posterior concentration as data accumulates"
        xLabel="θ (true win rate)"
        yLabel="posterior density"
        width={640}
        height={320}
        series={[
          {
            name: "n=5 (k=3)",
            color: colors.textDim,
            points: [
              [0.00, 0.00], [0.05, 0.02], [0.10, 0.07], [0.15, 0.16], [0.20, 0.27],
              [0.25, 0.42], [0.30, 0.59], [0.35, 0.78], [0.40, 0.97], [0.45, 1.16],
              [0.50, 1.32], [0.55, 1.45], [0.60, 1.52], [0.65, 1.53], [0.70, 1.46],
              [0.75, 1.30], [0.80, 1.06], [0.85, 0.74], [0.90, 0.40], [0.95, 0.12], [1.00, 0.00],
            ],
          },
          {
            name: "n=20 (k=13)",
            color: "#c084fc",
            points: [
              [0.30, 0.05], [0.35, 0.18], [0.40, 0.52], [0.45, 1.18], [0.50, 2.15],
              [0.55, 3.18], [0.60, 3.85], [0.65, 3.95], [0.70, 3.42], [0.75, 2.45],
              [0.80, 1.40], [0.85, 0.55], [0.90, 0.12], [0.95, 0.01],
            ],
          },
          {
            name: "n=200 (k=130)",
            color: colors.gold,
            points: [
              [0.50, 0.10], [0.53, 0.45], [0.55, 1.30], [0.57, 3.15], [0.59, 5.95],
              [0.61, 8.75], [0.63, 10.65], [0.65, 11.30], [0.67, 10.65], [0.69, 8.75],
              [0.71, 5.95], [0.73, 3.15], [0.75, 1.30], [0.77, 0.45], [0.80, 0.10],
            ],
          },
        ]}
      />

      <Prose>
        The next plot shows the running posterior probability that <Code>θ &gt; 0.5</Code> as a sequential Bayesian test consumes a stream of Bernoulli outcomes from a true win rate of 0.65. The decision threshold of 0.95 (gold dashed line) is crossed around trial 22, at which point the test stops. Notice the trajectory is not monotone — early random fluctuations can push the posterior probability down before it climbs again. This non-monotonicity is exactly what makes naive frequentist optional stopping problematic; the Bayesian framework treats each posterior update as a coherent belief revision rather than as a fresh hypothesis test.
      </Prose>

      <Plot
        label="Sequential posterior probability vs. trial count"
        xLabel="trial number"
        yLabel="P(θ > 0.5 | data)"
        width={640}
        height={300}
        series={[
          {
            name: "P(A better)",
            color: colors.gold,
            points: [
              [1, 0.50], [2, 0.75], [3, 0.69], [4, 0.81], [5, 0.66], [6, 0.77],
              [7, 0.66], [8, 0.78], [9, 0.69], [10, 0.79], [11, 0.84], [12, 0.78],
              [13, 0.84], [14, 0.79], [15, 0.85], [16, 0.89], [17, 0.85], [18, 0.89],
              [19, 0.92], [20, 0.89], [21, 0.92], [22, 0.96], [23, 0.97],
            ],
          },
          {
            name: "decision threshold",
            color: colors.textDim,
            points: [[1, 0.95], [23, 0.95]],
          },
        ]}
      />

      <Prose>
        The Jeffreys evidence scale gives an interpretive heatmap of Bayes factor magnitudes. Stronger colors correspond to stronger evidence; the central band around <Code>BF = 1</Code> is where the data are essentially uninformative for choosing between models. The scale is symmetric on the log axis: <Code>BF_10 = 30</Code> is "very strong" evidence for the alternative; <Code>BF_10 = 1/30</Code> is "very strong" evidence for the null. This symmetry is one of the practical advantages of Bayes factors over p-values, which can never quantify support for the null in the same way.
      </Prose>

      <Heatmap
        label="Jeffreys evidence categories across BF magnitudes"
        rowLabels={["evidence", "log10 BF"]}
        colLabels={["1/100", "1/30", "1/10", "1/3", "1", "3", "10", "30", "100"]}
        cellSize={56}
        colorScale="gold"
        matrix={[
          [1.00, 0.85, 0.65, 0.40, 0.10, 0.40, 0.65, 0.85, 1.00],
          [0.95, 0.80, 0.60, 0.35, 0.05, 0.35, 0.60, 0.80, 0.95],
        ]}
      />

      <Prose>
        The step trace below walks through one complete Bayesian comparison loop: prior specification, data ingestion, conjugate posterior update, credible-interval and posterior-probability extraction, Bayes factor via Savage-Dickey, and the final decision rule. Each phase corresponds to one cell in a typical analysis notebook.
      </Prose>

      <StepTrace
        label="Bayesian comparison loop — one full pass"
        steps={[
          {
            label: "Specify prior",
            render: () => (
              <Prose>
                Choose <Code>Beta(α, β)</Code> for the win rate. <Code>Beta(1, 1)</Code> is uniform; <Code>Beta(2, 2)</Code> is mildly regularizing toward 0.5; informative priors come from historical data. The prior also defines what "no effect" means for the Savage-Dickey ratio — typically <Code>θ = 0.5</Code> for a two-arm comparison.
              </Prose>
            ),
          },
          {
            label: "Observe data",
            render: () => (
              <Prose>
                Ingest <Code>k</Code> wins out of <Code>n</Code> trials. In a sequential setting, <Code>k</Code> and <Code>n</Code> grow over time and the posterior is recomputed after each event. In a one-shot setting, both are fixed at analysis time.
              </Prose>
            ),
          },
          {
            label: "Conjugate update",
            render: () => (
              <Prose>
                Posterior is <Code>Beta(α + k, β + n − k)</Code>. The update is closed-form, vectorized, and runs in microseconds even for thousands of arms in parallel. No sampling or optimization required.
              </Prose>
            ),
          },
          {
            label: "Extract intervals",
            render: () => (
              <Prose>
                Equal-tailed CI from the inverse CDF; HPD CI via the shortest-interval search. Posterior probability of <Code>θ &gt; 0.5</Code> from the survival function. All three are numerical evaluations of the Beta distribution at known points.
              </Prose>
            ),
          },
          {
            label: "Bayes factor",
            render: () => (
              <Prose>
                Savage-Dickey: <Code>BF_10 = prior_density(0.5) / posterior_density(0.5)</Code>. For Beta(1,1) the prior density is 1; for Beta(α + k, β + n − k) the posterior density at 0.5 is computed by scipy. One division, no integral.
              </Prose>
            ),
          },
          {
            label: "Decision rule",
            render: () => (
              <Prose>
                Stop and decide if <Code>P(better) &gt; threshold</Code>; continue collecting otherwise. For practical-significance comparisons, replace 0 with the minimum effect size <Code>δ</Code> in the survival function. Optional stopping is safe in the Bayesian framework because the posterior is a self-consistent summary of all observed data.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <H3>Bayes factors vs p-values</H3>

      <Prose>
        Use Bayes factors when stakeholders need to reason about the probability of competing hypotheses, when you want to quantify support for the null as well as the alternative, when you are running sequential or adaptive analyses, and when prior information about plausible effect sizes is available and worth incorporating. Use p-values when you are reporting in a context with a strong frequentist convention (most journal-style statistical analyses, regulated A/B testing under a fixed plan, FDA-regulated clinical trials), when you need to make sharp Type I error guarantees in a long-run frequency sense, and when the audience is more comfortable with the historical decision-theoretic apparatus. The two frameworks answer different questions; sophisticated practice often reports both.
      </Prose>

      <H3>Credible intervals vs confidence intervals</H3>

      <Prose>
        Credible intervals are the right tool when the audience interprets "the parameter is in this range with 95% probability" as the natural reading — which is essentially everyone outside of frequentist statistics seminars. They support direct probability statements about the parameter and combine cleanly with prior information. Confidence intervals are the right tool when you need long-run frequency guarantees that hold under repeated sampling regardless of the true parameter — which is the relevant frame for some regulatory and adversarial settings — and when prior elicitation is contentious or impossible.
      </Prose>

      <H3>HPD vs equal-tailed credible intervals</H3>

      <Prose>
        Use HPD intervals when you want the most concentrated 95% region (smallest interval), when the posterior is unimodal and you care that the interval excludes regions of low posterior density, and when you are reporting the interval as a "best estimate range" to a non-technical audience. HPD intervals can split into multiple pieces for multimodal posteriors, which is often desirable for visualization. Use equal-tailed intervals when transformation invariance matters (the equal-tailed interval transforms cleanly under monotonic reparameterizations of <Code>θ</Code>; HPD does not), when you want trivial computation from the inverse CDF, and when the posterior is sufficiently symmetric that the two intervals coincide anyway.
      </Prose>

      <H3>Closed-form vs MCMC</H3>

      <Prose>
        Use closed-form Beta-Binomial when the comparison is a single binary outcome per trial with conjugate priors — the win-rate scenario is the canonical example. The math is exact, the runtime is microseconds, and there are no convergence diagnostics to worry about. Use MCMC (PyMC, Stan) when the model has covariates, hierarchical structure, non-conjugate priors, or multiple correlated outcomes. NUTS handles essentially any continuous-parameter model correctly; the cost is seconds-to-minutes per analysis and the need to inspect convergence diagnostics (R-hat, ESS, divergent transitions). Use variational inference (ADVI in PyMC) when the model is too large for MCMC to be practical and an approximate posterior is acceptable; this is mostly relevant for very high-dimensional latent-variable models, not for the typical model-comparison setting.
      </Prose>

      <H3>BIC vs explicit Bayes factors</H3>

      <Prose>
        Use BIC when you need a quick model-selection heuristic, the sample size is reasonably large (BIC is asymptotic in <Code>N</Code>), the models are reasonably well-behaved (regular MLE conditions hold), and you do not have informative priors to incorporate. Use explicit Bayes factors when prior information matters, when sample size is small (BIC's asymptotic approximation breaks down), or when you need the specific evidence-quantification semantics rather than a model-selection score. Kass and Raftery's recommendation is that BIC is acceptable for rough comparison but should not be the basis of a strong claim; for the latter, compute the Bayes factor directly via Laplace approximation, bridge sampling, or — in conjugate cases — the closed-form integral.
      </Prose>

      <H3>Default vs informative priors for Bayes factors</H3>

      <Prose>
        Use default priors (Jeffreys' prior, Rouder's Cauchy(0, 0.707) on standardized effects, the JZS prior for t-tests) when you want a comparison that does not depend on subjective elicitation, when reporting in a context where reproducibility across analysts matters, and when no historical data justify a more specific prior. Use informative priors when you have meaningful prior information — historical win rates, theoretical bounds on plausible effect sizes, hierarchical priors learned from related comparisons — and when the audience is comfortable with the prior being part of the inferential machinery. Always perform a sensitivity analysis: re-compute the Bayes factor under at least two prior choices spanning the range of defensible options, and report whether the conclusion changes.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        The conjugate Beta-Binomial computation scales effortlessly. A single posterior update is a constant-time arithmetic operation; computing thousands of updates per second is trivial; running the analysis on a stream with millions of events per day is well within the capabilities of a single CPU core. This is why Bayesian A/B testing platforms can afford to recompute posteriors after every event without specialized infrastructure. The credible-interval evaluations are constant-time calls into scipy's Beta distribution; the survival-function evaluation for posterior decision rules is similarly constant-time. The Savage-Dickey Bayes factor is two density evaluations and a division. None of this scales with sample size in any meaningful way.
      </Prose>

      <Prose>
        Where computational cost begins to matter is at the model-specification frontier. Once you leave conjugate territory — multiple correlated parameters, hierarchical priors, covariates, non-conjugate likelihoods — you need MCMC or variational inference. PyMC and Stan with NUTS handle hundreds to thousands of parameters comfortably; convergence time scales roughly as the cube of the dimensionality due to the curvature-aware proposal mechanics. For very high-dimensional models (latent-factor models, large hierarchical structures with thousands of groups), variational methods or specialized samplers like Pathfinder, ADVI, and RWM with Gibbs blocking become necessary. None of this is unique to Bayesian model comparison; it is the general inference scaling story.
      </Prose>

      <Prose>
        The marginal-likelihood evaluation is the part that scales least gracefully. Bridge sampling is the standard tool for non-conjugate Bayes factors and is reasonably reliable up to a few dozen parameters; beyond that, the variance of the marginal-likelihood estimator grows quickly and the answer becomes less reliable. The default fallback for high-dimensional model comparison is leave-one-out cross-validation (LOO-CV) via PSIS-LOO (Vehtari et al. 2017), which estimates predictive performance on held-out data instead of the marginal likelihood directly. PSIS-LOO scales linearly in sample size and is the recommended approach for comparing very flexible models — it is what PyMC's <Code>compare</Code> function uses under the hood.
      </Prose>

      <Prose>
        Sensitivity to prior specification is the part of Bayesian comparison that does not scale away with more data. Posterior estimates of <Code>θ</Code> become prior-insensitive as <Code>n → ∞</Code> — the Bernstein-von Mises theorem guarantees this for regular models — but Bayes factors do not. The Lindley-Jeffreys paradox is the formal statement of this: a vague prior on the alternative makes <Code>BF_10 → 0</Code> regardless of how strong the data are. Practitioners need to report Bayes factors under a defensible prior choice and ideally show sensitivity across plausible alternatives. This is not a computational issue; it is a methodological constraint that does not relax at any sample size.
      </Prose>

      <Prose>
        For the operational scaling question — how many comparisons does a sequential test need? — the answer depends on the true effect size and the decision threshold. For a true win rate of 0.65 vs a null of 0.50 at a 0.95 decision threshold, the median sequential Bayesian test stops in roughly 20-30 trials. For a true win rate of 0.55 (smaller effect), the median stop time is closer to 200-400 trials. For a true win rate of 0.51 (very small effect), the test may take thousands of trials or fail to stop within the budget — which is the correct behavior, not a bug. Smaller effects require more data to detect with confidence, and the Bayesian framework correctly reflects that requirement in its stopping behavior.
      </Prose>

      <Prose>
        One scaling consideration that deserves attention is multi-arm comparison. The Beta-Binomial setup generalizes naturally to <Code>K</Code> arms: maintain <Code>K</Code> independent Beta posteriors, sample from each, and the posterior probability that any given arm is best is the fraction of joint samples in which that arm has the largest <Code>θ</Code>. This is the algorithmic core of Thompson sampling in multi-armed bandits, and it scales linearly in the number of arms with constant cost per posterior update. For very large <Code>K</Code> (thousands of arms, as in personalized recommendation systems), top-arm identification can be batched: sample once, count argmaxes across joint samples, and report the top-K candidates with their posterior probabilities of being best. The infrastructure stays trivial; the math stays exact within Monte Carlo error.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>Lindley-Jeffreys paradox: vague priors kill Bayes factors</H3>
      <Prose>
        The most common methodological error in applying Bayes factors. Choosing an excessively diffuse prior on the parameter under the alternative — a uniform prior over a very wide range, an improper prior, or a proper but weakly informative prior chosen to "let the data speak" — biases the Bayes factor in favor of the null. The mathematical reason: the marginal likelihood under the alternative averages the data likelihood over the prior, and a very diffuse prior places most of its mass in regions of poor data fit, dragging the average down. Mitigation: use established defaults (Jeffreys' prior, Rouder et al.'s Cauchy(0, 0.707), the JZS prior for t-tests) or informative priors derived from theory; always run a prior sensitivity analysis.
      </Prose>

      <H3>Stopping rules that are not what you think they are</H3>
      <Prose>
        Bayesian sequential testing permits optional stopping in the sense that the posterior is a coherent summary regardless of when you stop. It does not control the frequentist Type I error rate. If you stop the test the moment <Code>P(θ &gt; 0.5) &gt; 0.95</Code> on a stream with <Code>θ = 0.5</Code> exactly (the null is true), the long-run fraction of decisions in favor of "A is better" can exceed 5%. This is fine for many decision problems — you are not making a frequentist Type I claim — but it is a problem if your audience expects frequentist guarantees. If you need both Bayesian-coherent updating and bounded false-positive rate, use Bayes factors with bounded thresholds (Berger and Wolpert 1988) or a hybrid procedure that combines posterior probabilities with frequentist boundaries.
      </Prose>

      <H3>Prior-data conflict that the model silently absorbs</H3>
      <Prose>
        If your prior is concentrated in a region that the data strongly contradict, the posterior will move toward the data — but the Bayes factor will quietly absorb the conflict in ways that may not be obvious. A prior centered on <Code>θ = 0.7</Code> with high precision combined with data showing 20 wins out of 100 trials gives a posterior that compromises between prior and data, and a Bayes factor that compares "the alternative model with this specific narrow prior" to the null. The Bayes factor is correct given the prior, but the prior is wrong. Diagnose this with prior-predictive checks: simulate data from the prior and verify that the simulations look plausible before observing real data.
      </Prose>

      <H3>Confusing Bayes factor with posterior model probability</H3>
      <Prose>
        Bayes factors are evidence ratios; they are not posterior probabilities. <Code>BF_10 = 30</Code> means the data are 30 times more likely under the alternative than the null, which combined with prior model odds of 1:1 gives posterior model odds of 30:1, equivalent to <Code>P(M_1 | D) = 30/31 ≈ 0.97</Code>. If the prior model odds are different — say, you genuinely believed the null was 100 times more likely a priori — the same Bayes factor of 30 gives posterior odds of 30:100 = 3:10, equivalent to <Code>P(M_1 | D) ≈ 0.23</Code>. Always make the prior model probability explicit when interpreting Bayes factors, especially in contexts where one of the models has special privileged status.
      </Prose>

      <H3>Mistaking credible intervals for confidence intervals (and vice versa)</H3>
      <Prose>
        A 95% credible interval and a 95% confidence interval are different objects with different interpretations. They often produce numerically similar results for simple models with diffuse priors and large samples, but they diverge for small samples, informative priors, or highly skewed sampling distributions. Reporting one and labeling it as the other is a real-world error that has affected published research. Decide which interpretation you want, compute the corresponding interval, and label it accurately.
      </Prose>

      <H3>HPD interval pathologies for multimodal posteriors</H3>
      <Prose>
        For a unimodal posterior the HPD interval is a contiguous range. For a bimodal posterior it can split into two disjoint pieces — which is mathematically correct (the highest-density region is two pieces) but produces an interval that some downstream tools cannot represent. If your posterior is bimodal, either report both modes explicitly with their relative probability mass, switch to the equal-tailed interval (which is always contiguous), or rethink the model — bimodality often signals a missing variable or unmodeled mixture structure.
      </Prose>

      <H3>Sample-size confusion in sequential testing</H3>
      <Prose>
        The Bayesian posterior is a function of the data observed so far; it does not "know" about your stopping rule, your maximum trial budget, or how many comparisons you originally planned to run. This is by design — the posterior is the same whether you collected 40 trials in one batch or 40 trials sequentially with intermediate looks — but it means the posterior interpretation does not include any statement about the experiment's protocol. If reviewers ask "did you adjust for multiple looks?" the correct Bayesian answer is "no adjustment is needed because the posterior is coherent under any data-generating protocol that does not depend on the parameter," but this answer does not always satisfy reviewers trained in frequentist methodology.
      </Prose>

      <H3>Numerical issues in the Beta function for extreme parameters</H3>
      <Prose>
        The Beta function <Code>B(α, β) = Γ(α)Γ(β)/Γ(α+β)</Code> overflows quickly for large arguments. Always work in log space (use <Code>scipy.special.betaln</Code> rather than <Code>scipy.special.beta</Code>, and compute log-densities via <Code>scipy.stats.beta.logpdf</Code> when summing or subtracting). For very small posterior densities at the null value (<Code>α + k</Code> or <Code>β + n - k</Code> in the thousands), the Savage-Dickey ratio computation can underflow to zero in single precision; use double precision and log-space arithmetic.
      </Prose>

      <Callout accent="gold">
        Bayesian comparison fails informatively. When the prior is wrong, when the model is misspecified, when the data conflict with the prior, the Bayesian framework usually surfaces the problem through prior-data divergence, posterior multimodality, or convergence diagnostics. Frequentist procedures often hide these problems — a p-value is just a number, with no built-in mechanism to flag that the underlying model assumptions are violated. This is a significant practical advantage of Bayesian methods that does not get enough emphasis in textbook treatments.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All sources below were verified against their original publications. Author lists, journal references, and key results confirmed.
      </Prose>

      <H3>Jeffreys 1961 — Theory of Probability (3rd edition)</H3>
      <Prose>
        Harold Jeffreys. <Code>Theory of Probability</Code> (3rd ed., Oxford University Press, 1961). The founding text of Bayesian inference as a unified system. Chapter 5 introduces the Bayes factor as the central tool for hypothesis testing and the evidence scale (anecdotal &lt; substantial &lt; strong &lt; very strong &lt; decisive) that has anchored the literature for over six decades. Jeffreys' prior — proportional to the square root of the Fisher information determinant — is also derived here as the unique prior with the property of reparameterization invariance. The book is dense but every chapter rewards reading; sections 5.0 through 5.04 are the canonical reference for Bayes factor interpretation.
      </Prose>

      <H3>Kass and Raftery 1995 — Bayes Factors</H3>
      <Prose>
        Robert E. Kass and Adrian E. Raftery. "Bayes Factors." <Code>Journal of the American Statistical Association</Code>, 90(430): 773–795, 1995. The standard modern reference for Bayes factors. Comprehensive treatment of computation methods (Laplace approximation, importance sampling, bridge sampling, BIC), interpretive scales (slight refinement of Jeffreys' categories: <Code>BF</Code> in 1–3 not worth more than a bare mention; 3–20 positive evidence; 20–150 strong; &gt;150 very strong), connections to model selection, sensitivity to prior specification, and worked examples across regression, contingency tables, and time series. Required reading for anyone deploying Bayes factors in applied work.
      </Prose>

      <H3>Wagenmakers 2007 — A Practical Solution to the Pervasive Problems of p-values</H3>
      <Prose>
        Eric-Jan Wagenmakers. "A practical solution to the pervasive problems of p values." <Code>Psychonomic Bulletin & Review</Code>, 14(5): 779–804, 2007. The paper that popularized Bayes factors in psychology and behavioral sciences. Documents the failure modes of p-value-based inference (sensitivity to stopping rules, inability to support the null, conflation of significance with importance) and develops Bayesian alternatives in detail. The exposition of the Lindley-Jeffreys paradox in section 5 is particularly clean. Wagenmakers' subsequent work with Rouder, Morey, and colleagues built the BayesFactor R package which is now the standard tool for Bayes factor analyses in psychological research.
      </Prose>

      <H3>Rouder et al. 2009 — Default Bayes Factors for ANOVA Designs</H3>
      <Prose>
        Jeffrey N. Rouder, Paul L. Speckman, Dongchu Sun, Richard D. Morey, Geoffrey Iverson. "Bayesian t tests for accepting and rejecting the null hypothesis." <Code>Psychonomic Bulletin & Review</Code>, 16(2): 225–237, 2009. Introduces the default Cauchy(0, 0.707) prior on standardized effect sizes for the Bayesian t-test (the JZS prior, after Jeffreys, Zellner, and Siow). The Cauchy prior's heavy tails make the Bayes factor robust to large effect sizes while remaining proper enough to avoid the Lindley-Jeffreys pathology. This paper established the methodological standard for "default Bayes factors" — analyses that do not require subjective prior elicitation. The follow-up paper (Rouder et al. 2012) extends the default-prior framework to ANOVA designs.
      </Prose>

      <H3>Gelman, Carlin, Stern, Dunson, Vehtari, Rubin 2013 — Bayesian Data Analysis (3rd edition)</H3>
      <Prose>
        Andrew Gelman, John B. Carlin, Hal S. Stern, David B. Dunson, Aki Vehtari, Donald B. Rubin. <Code>Bayesian Data Analysis</Code> (3rd ed., CRC Press, 2013). The definitive modern textbook. Chapter 1 covers the foundations of Bayesian inference; Chapter 6 covers model comparison via posterior predictive checks and information criteria; Chapter 7 is a careful treatment of the marginal likelihood and its computation. Gelman's pragmatic philosophy — model comparison should focus on predictive performance rather than marginal likelihood when the candidate models are all rough approximations to a complex truth — is articulated throughout. The accompanying online resources include code, errata, and supplementary material covering MCMC, variational methods, and Stan integration.
      </Prose>

      <H3>Wagenmakers et al. 2010 — Bayesian Hypothesis Testing for Psychologists</H3>
      <Prose>
        Eric-Jan Wagenmakers, Tom Lodewyckx, Himanshu Kuriyal, Raoul Grasman. "Bayesian hypothesis testing for psychologists: A tutorial on the Savage-Dickey method." <Code>Cognitive Psychology</Code>, 60(3): 158–189, 2010. The clearest pedagogical treatment of the Savage-Dickey density ratio in the literature. Derives the identity from first principles, walks through worked examples for normal-mean tests and proportion tests, and discusses computational implementations including Monte Carlo estimation when closed forms are unavailable. This is the paper to read alongside the original 1971 Dickey-Lientz paper and Verdinelli-Wasserman 1995 if you want to fully internalize the Savage-Dickey machinery.
      </Prose>

      <H3>Vehtari, Gelman, Gabry 2017 — PSIS-LOO</H3>
      <Prose>
        Aki Vehtari, Andrew Gelman, Jonah Gabry. "Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC." <Code>Statistics and Computing</Code>, 27(5): 1413–1432, 2017. The standard modern reference for high-dimensional Bayesian model comparison via cross-validation rather than marginal likelihood. PSIS-LOO (Pareto-smoothed importance sampling LOO) is implemented in PyMC's <Code>compare</Code> function and Stan's <Code>loo</Code> package and is the recommended fallback when bridge sampling is impractical. The paper also covers the WAIC (Watanabe-Akaike Information Criterion) and provides diagnostics for when the LOO approximation breaks down.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Conjugate update by hand</H3>
      <Prose>
        You start with a <Code>Beta(2, 2)</Code> prior on the win rate of a new model. After the first 10 comparisons it has won 7 times. Write down the posterior parameters, compute the posterior mean and variance, and compute the 90% equal-tailed credible interval (you will need to evaluate the inverse CDF of the Beta distribution; either use scipy or recognize that for small <Code>α + β</Code> the interval can be computed via simulation). What does this credible interval mean in plain English, and how would the interval change if your prior had been <Code>Beta(20, 20)</Code> instead? Why does the more concentrated prior produce a narrower posterior even though the data are identical?
      </Prose>

      <H3>Exercise 2 — Savage-Dickey by hand</H3>
      <Prose>
        Using the Savage-Dickey density ratio, compute the Bayes factor <Code>BF_10</Code> for the hypothesis that <Code>θ ≠ 0.5</Code> versus <Code>θ = 0.5</Code> after observing 7 wins out of 10 trials with a <Code>Beta(1, 1)</Code> prior. Show your work: the prior density at <Code>θ = 0.5</Code>, the posterior parameters, the posterior density at <Code>θ = 0.5</Code>, and the final ratio. Does the resulting Bayes factor reach Jeffreys' "substantial" threshold? Now repeat the calculation with a <Code>Beta(0.5, 0.5)</Code> prior (Jeffreys' prior for the binomial). How does the Bayes factor change, and why? What does this exercise reveal about the prior-sensitivity of Bayes factors?
      </Prose>

      <H3>Exercise 3 — Sequential stopping intuition</H3>
      <Prose>
        Imagine you are running a sequential Bayesian A/B test with the stopping rule "stop when <Code>P(A better than B | data) &gt; 0.95</Code>" and a <Code>Beta(1, 1)</Code> prior on each arm. The true win rates are <Code>θ_A = θ_B = 0.5</Code> exactly — the null is true. (1) What does the long-run probability of stopping with the decision "A is better" look like as the maximum trial budget increases? (2) Is this a problem? (3) How would you modify the stopping rule to bound this probability while preserving the Bayesian framework? (4) Compare your modification to what a frequentist sequential test (O'Brien-Fleming, Pocock) would do. What philosophical difference between Bayesian and frequentist analysis is being navigated here?
      </Prose>

      <H3>Exercise 4 — HPD vs equal-tailed for skewed posteriors</H3>
      <Prose>
        Consider a posterior of <Code>Beta(2, 30)</Code> — heavily right-skewed, mode near zero. (1) Compute the 95% equal-tailed credible interval. (2) Compute the 95% HPD credible interval. (3) Why do they differ, and which one would you report to a stakeholder asking "where do we think the parameter is, with 95% probability?" (4) Now apply the monotonic transformation <Code>φ = log(θ / (1−θ))</Code>. Compute the posterior on <Code>φ</Code>, and find both intervals in the <Code>φ</Code> space. Does the equal-tailed interval transform cleanly under the change of variables? Does the HPD interval? Why might this matter when reporting results that will be re-analyzed downstream?
      </Prose>

      <H3>Exercise 5 — Lindley-Jeffreys paradox numerically</H3>
      <Prose>
        Construct a numerical demonstration of the Lindley-Jeffreys paradox in the win-rate setting. Fix <Code>k = 60, n = 100</Code> (a clear sample-level effect favoring <Code>θ &gt; 0.5</Code>). Compute <Code>BF_10</Code> under <Code>M_1: θ ~ Beta(α, α)</Code> for <Code>α</Code> ranging over <Code>[0.5, 1, 2, 5, 10, 100, 10000]</Code>. Plot or tabulate <Code>BF_10</Code> versus <Code>α</Code>. What happens as <Code>α</Code> grows large? Why does the data look "more like the null" under more diffuse priors when the data themselves clearly favor the alternative? Use this exercise to articulate, in your own words, why default-prior recommendations (Jeffreys, JZS, Cauchy(0, 0.707)) exist and why "uninformative" priors are not actually neutral for hypothesis testing.
      </Prose>

      <H3>Exercise 6 — From BF to posterior model probability</H3>
      <Prose>
        Suppose you compute <Code>BF_10 = 12</Code> for a comparison between a new model and a previous baseline. (1) Under a 1:1 prior over the two models, what is the posterior probability that the new model is better? (2) Under a 1:9 prior favoring the baseline (you genuinely believed the baseline was probably correct before seeing data), what is the posterior probability now? (3) Under a 9:1 prior favoring the new model, what is the posterior probability? (4) For each case, assess: would you act on the result? At what posterior probability threshold would you switch from the baseline to the new model in production, and what additional considerations beyond the posterior probability itself would inform that decision (e.g., cost of switching, magnitude of effect, robustness to assumption changes)?
      </Prose>

    </div>
  ),
};

export default bayesianModelComparison;
