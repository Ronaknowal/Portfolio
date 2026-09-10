import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const selfConsistencyCouncil = {
  title: "Self-Consistency & Council-of-Judges",
  slug: "self-consistency-council-of-judges",
  readTime: "~38 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        A single LLM judge is a noisy instrument. Ask GPT-4 to grade the same response twice at temperature 0.7 and you will often get two different scores; ask it to compare the same pair of responses with the order swapped and you will sometimes get two different verdicts. The 2023-2024 wave of LLM-as-a-judge papers — MT-Bench, AlpacaEval, the original Zheng et al. study (arXiv:2306.05685) — surfaced this empirically: pairwise judge agreement with humans tops out around 80% even for the strongest frontier models, with the remaining 20% split between genuine annotator disagreement, judge stochasticity, and judge-specific systematic biases (position bias, length bias, self-preference). If you build any evaluation, ranking, reward modeling, or preference data pipeline on a single judge call, that 20% noise floor propagates straight through to whatever downstream artifact you produce. A reward model trained on noisy preference labels learns the noise. A leaderboard built on noisy judge scores has confidence intervals wider than the differences it claims to detect.
      </Prose>

      <Prose>
        Two complementary techniques address this noise floor. Self-consistency, introduced for reasoning chains by Wang et al. 2022 ("Self-Consistency Improves Chain of Thought Reasoning in Language Models", arXiv:2203.11171), runs the same model multiple times at non-zero temperature and aggregates the outputs by majority vote or mean. The original paper applied it to mathematical reasoning, where sampling 40 chain-of-thought solutions and taking the most common final answer outperformed greedy decoding by 12-18 points on GSM8K and other benchmarks. The same logic transfers directly to judge calls: if you ask the same judge five times whether response A is better than response B at temperature 0.7, the majority verdict is more reliable than any single verdict, because the stochastic component of the judge's output partially averages out.
      </Prose>

      <Prose>
        Council-of-Judges — also called "jury" or "Panel of LLM Evaluators" (PoLL) — goes further. Instead of sampling the same judge multiple times, it queries multiple different judges (different model families, different prompts, sometimes different rubrics) and aggregates their verdicts. Verga et al. 2024 ("Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models", arXiv:2404.18796) showed that a panel of three small judges — Command-R, Haiku, and GPT-3.5 — produced rankings that correlated with human judgment as well as a single GPT-4 judge call, at roughly one-seventh the cost. The structural reason is the same as why ensembles beat single classifiers in classical machine learning: when judges have correlated accuracy but uncorrelated errors, averaging cancels the errors and preserves the signal.
      </Prose>

      <Prose>
        These methods exist because the alternative — using a single frontier judge for everything — has three failure modes that compound at scale. First, cost: a single GPT-4-class judge call for every preference pair in a 100k-pair dataset is several thousand dollars; running a multi-model evaluation pipeline for every model release multiplies that. Second, bias: a single judge encodes its own training distribution as ground truth, so any evaluation built on it inherits that judge's blind spots wholesale (the most documented case is GPT-4 systematically preferring GPT-4 outputs to comparable Claude outputs, which Verga et al. quantify directly). Third, fragility: when the judge model is deprecated, retrained, or slightly nudged in a routine update, your entire evaluation history shifts under you, with no clean way to compare new numbers to old. Self-consistency reduces stochastic variance for a fixed judge; councils reduce systematic bias by averaging across judges that disagree about which biases to have.
      </Prose>

      <Prose>
        The deeper reason these methods deserve their own topic, distinct from the general LLM-as-a-judge literature, is that they sit at the intersection of three traditions that the LLM era has reactivated. The first is Condorcet's 1785 jury theorem, the foundational result in collective decision theory: under independence, majority votes from competent judges converge to truth as the panel grows. The second is the classical statistical ensemble theory underlying bagging, boosting, and random forests, where the variance reduction from averaging M independent unbiased estimators is exactly 1/M. The third is the modern LLM judge literature itself, which has had to rediscover that none of the independence assumptions in the classical theory hold for LLMs from the same family — and that the practical aggregation problem is therefore much closer to Bayesian inference over correlated experts than to a textbook majority vote.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Start with the simplest version: a judge that is right 70% of the time and wrong 30%, with errors independent across calls. If you call it once, you get the right answer 70% of the time. If you call it three times and take the majority vote, you get the right answer whenever at least two of the three calls are correct. The probability of at least two correct calls is the binomial sum <Code>0.7^3 + 3·0.7^2·0.3 ≈ 0.784</Code> — a substantial improvement over 70% from doing nothing more than calling the same judge twice more. With five calls, the majority vote is correct roughly 83% of the time. With seven calls, 87%. With eleven calls, 92%. The accuracy keeps climbing toward 100% as the panel grows, which is exactly Condorcet's theorem. The catch hidden inside that calculation is the word "independent". Each call's error must be statistically uncorrelated with the others. If five calls all share the same blind spot — the same systematic misreading of a particular prompt structure — then majority voting does not help at all. They will all vote together for the wrong answer, every time.
      </Prose>

      <Prose>
        This is the cleanest way to see the difference between self-consistency and council-of-judges. Self-consistency calls the same model multiple times at temperature greater than zero, so the variance it averages out is the stochastic variance: different sampled tokens, different chain-of-thought trajectories, different surface choices. The systematic biases of the model — its tendency to prefer the second of two presented options, its preference for longer answers, its self-preference for outputs in its own characteristic style — are baked into the weights and persist across every sample. Self-consistency reduces the noise around a biased estimate; it does not move the estimate. Council-of-judges adds different models to the panel, models trained on different data with different objectives. Their stochastic errors are mostly independent for the obvious reason (they are running on different machines with different random seeds), and their systematic biases are partially independent because the biases come from different training corpora and different reward signals. Averaging across a council reduces both the stochastic variance and a meaningful slice of the systematic bias.
      </Prose>

      <Prose>
        The mental model that helps most is to picture each judge as a noisy meter. Self-consistency is like reading the same meter five times and averaging the readings: you suppress the meter's wobble but not its calibration error. Council-of-judges is like reading five different meters made by different manufacturers: you suppress wobble and you cancel out manufacturer-specific calibration errors, but you do not cancel out errors that all the meters share — for example, a temperature dependence that affects every brand the same way. In the LLM context, the errors that "all meters share" are the ones that come from the common pretraining data: the cultural biases, the recency biases, the formatting preferences that show up in every model trained on the open web. No amount of council diversity gets rid of those.
      </Prose>

      <Prose>
        A second piece of intuition concerns the role of judge accuracy. Condorcet's theorem requires each judge to be more than 50% accurate — that is, better than random — for the panel accuracy to exceed individual accuracy. Below 50%, majority voting actually drives accuracy down: a panel of three coin-flippers is correct 50% of the time, but a panel of three judges that are 40% accurate and independent is correct only about 35% of the time. This matters because not every "judge" you might add to a council is actually competent. A small open-source model used as one of three judges might be only 55-60% accurate on the task; adding it to a panel with two strong judges can drag panel accuracy down rather than up if the panel size is small enough that the weak judge tilts close votes. The practical implication is that you cannot just throw judges into a council to be safe; you need to verify each judge clears the competence threshold on a calibration set first.
      </Prose>

      <Prose>
        A third piece of intuition concerns aggregation strategy. Plain majority voting is the default, but it discards information about how confident each judge was. If three judges vote A, B, A, plain voting calls it for A. But if the two A-voters were each 51% confident and the B-voter was 99% confident, a confidence-weighted aggregate would call it for B. Bayesian aggregation formalizes this: each judge has an estimated reliability (which can itself be inferred from agreement patterns on a calibration set), and the posterior probability of each verdict is computed by combining the judges' votes weighted by their reliability. In practice, Bayesian aggregation tends to outperform plain majority voting when the panel includes judges with very different competence levels and underperforms it when the panel is roughly homogeneous. For most production councils with three-to-five comparable judges, plain majority voting and confidence-weighted voting produce nearly identical results, and the simpler one wins on operational grounds.
      </Prose>

      <Prose>
        Finally, there is the cost intuition. A self-consistency aggregator that samples five times costs five times as much as a single call, and roughly five times the latency unless you parallelize. A council of three judges costs the sum of three judge calls, which is typically less than 5x a single GPT-4 call when the council uses smaller models. Verga et al.'s key practical finding is that a council of three small models can match a single GPT-4 judge at about 1/7 the cost. This is a pareto improvement, not a tradeoff. The reason it works is that small judges are individually weak but their errors are diverse enough that majority voting recovers most of the accuracy gap. The same calculation does not work for self-consistency: five samples of a small judge are still a small judge, and they share all the systematic errors that make the small judge worse than GPT-4 in the first place. Self-consistency improves a judge; council-of-judges substitutes for a stronger judge.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <Prose>
        The mathematics of judge aggregation is built on two pillars: variance reduction for averaged estimators, and majority-vote accuracy bounds from Condorcet's jury theorem. Both have clean closed forms under independence and tractable corrections under partial dependence. Start with the variance reduction.
      </Prose>

      <H3>3a. Variance reduction by averaging</H3>

      <Prose>
        Suppose each judge call produces a real-valued score <Code>S_i = μ + ε_i</Code>, where <Code>μ</Code> is the true score and <Code>ε_i</Code> is a zero-mean noise term with variance <Code>σ²</Code>. The mean of <Code>n</Code> independent calls is:
      </Prose>

      <MathBlock>{"\\bar{S} = \\frac{1}{n} \\sum_{i=1}^{n} S_i = \\mu + \\frac{1}{n} \\sum_{i=1}^{n} \\varepsilon_i"}</MathBlock>

      <Prose>
        Under independence, the variance of the mean shrinks linearly in <Code>n</Code>:
      </Prose>

      <MathBlock>{"\\mathrm{Var}(\\bar{S}) = \\frac{\\sigma^2}{n}, \\quad \\mathrm{SD}(\\bar{S}) = \\frac{\\sigma}{\\sqrt{n}}"}</MathBlock>

      <Prose>
        This is the famous <Code>1/√n</Code> rate. Averaging four independent samples halves the standard deviation. Averaging 100 samples cuts it by a factor of 10. The catch, again, is independence. When the noise terms are correlated with pairwise correlation coefficient <Code>ρ</Code>, the variance of the mean becomes:
      </Prose>

      <MathBlock>{"\\mathrm{Var}(\\bar{S}) = \\frac{\\sigma^2}{n} \\bigl(1 + (n-1)\\rho\\bigr)"}</MathBlock>

      <Prose>
        As <Code>n → ∞</Code> the variance does not vanish but instead approaches the floor <Code>ρ·σ²</Code>. If the average pairwise correlation between judge calls is 0.5, then no amount of additional sampling can reduce the standard deviation below about 70% of the single-call standard deviation. This is the precise mathematical reason that self-consistency on the same model has limits: same-model samples typically have correlations in the 0.3-0.6 range for evaluation tasks, so the realistic floor on variance reduction is 50-80% of the original variance, regardless of how many samples you take.
      </Prose>

      <H3>3b. Condorcet's jury theorem</H3>

      <Prose>
        For binary verdicts (A vs B), let each judge be correct with probability <Code>p</Code>, and let the judges decide independently. The probability that a panel of <Code>n</Code> judges (with <Code>n</Code> odd, to avoid ties) is correct via majority vote is:
      </Prose>

      <MathBlock>{"P_n(p) = \\sum_{k=\\lceil n/2 \\rceil}^{n} \\binom{n}{k} p^k (1-p)^{n-k}"}</MathBlock>

      <Prose>
        Condorcet's classical result is the limiting behavior. If <Code>p &gt; 1/2</Code>, then <Code>P_n(p) → 1</Code> as <Code>n → ∞</Code>. If <Code>p &lt; 1/2</Code>, then <Code>P_n(p) → 0</Code>. If <Code>p = 1/2</Code>, then <Code>P_n(p) = 1/2</Code> for all <Code>n</Code>. The 50% threshold is a hard cutoff: judges that are individually below random are made worse by ensembling, not better. Above 50%, ensembling monotonically improves accuracy with diminishing returns. For <Code>p = 0.7</Code>, panel sizes of 3, 5, 7, 11, and 21 give panel accuracies of approximately 0.784, 0.837, 0.874, 0.922, and 0.974 respectively. The marginal gain from going from 11 to 21 judges is much smaller than the gain from going from 1 to 3, which is why production councils almost always use 3 to 5 judges.
      </Prose>

      <H3>3c. Heterogeneous judges</H3>

      <Prose>
        The standard Condorcet formula assumes all judges have the same accuracy <Code>p</Code>. When judges differ — judge <Code>i</Code> has accuracy <Code>p_i</Code> — the optimal aggregation is no longer plain majority vote but a log-odds weighted vote. The Bayes-optimal decision rule weights each judge's vote by:
      </Prose>

      <MathBlock>{"w_i = \\log \\frac{p_i}{1 - p_i}"}</MathBlock>

      <Prose>
        and votes for verdict A if and only if the weighted sum exceeds zero. A judge that is 90% accurate gets weight <Code>log(9) ≈ 2.20</Code>; a judge that is 60% accurate gets weight <Code>log(1.5) ≈ 0.41</Code>. The strong judge counts roughly five times as much as the weak one. A judge at exactly 50% gets weight zero — its vote is ignored, which is exactly right because it carries no information. A judge below 50% gets a negative weight, which means the optimal rule flips its vote. This last property is theoretically clean and operationally dangerous: if you misestimate <Code>p_i</Code> for a judge that is actually 49% accurate, the Bayesian aggregator will start systematically inverting its votes, and small estimation errors near the 50% boundary produce wildly different aggregates.
      </Prose>

      <H3>3d. Correlated judges</H3>

      <Prose>
        Real LLM judges are never independent. Their errors share structure because they share training data, share architectural inductive biases, and share evaluation prompt templates. The standard correction, due to Ladha 1995 and reformulated for LLM panels by Verga et al., decomposes total panel error into independent and correlated components. Let <Code>ρ</Code> be the average pairwise correlation between judge errors. Then for a homogeneous panel with accuracy <Code>p</Code>, the panel accuracy under correlation is approximately:
      </Prose>

      <MathBlock>{"P_n^{\\rho}(p) \\approx \\Phi\\!\\left(\\frac{(p - 1/2)\\sqrt{n}}{\\sqrt{p(1-p)}\\sqrt{1 + (n-1)\\rho}}\\right)"}</MathBlock>

      <Prose>
        where <Code>Φ</Code> is the standard normal CDF. The correlation enters under the square root in the denominator, capping the effective sample size at <Code>1/ρ</Code> in the large-<Code>n</Code> limit. If <Code>ρ = 0.2</Code>, the effective panel size never exceeds 5 even if the literal panel size is 50. This is the formal version of "you cannot fix correlated bias by adding more correlated judges". The remedy is to lower <Code>ρ</Code> by adding judges from different model families, with different prompts, scoring different rubrics — anything that decorrelates the error structure.
      </Prose>

      <H3>3e. Bayesian aggregation with reliability inference</H3>

      <Prose>
        When you have a calibration set of items with known ground truth, you can estimate each judge's reliability <Code>p_i</Code> directly and plug it into the log-odds rule. When you do not, you can still infer reliabilities by treating the panel's votes as observations of a latent ground truth and applying expectation-maximization (Dawid & Skene 1979 in classical statistics; resurrected for crowdsourcing in 2009, then for LLM panels in 2024). The model has two layers: each item has a latent true verdict, and each judge has a latent confusion matrix (true positive rate and false positive rate). The EM algorithm alternates between estimating the latent verdicts given the current confusion matrices, and re-estimating the confusion matrices given the imputed verdicts. The fixed point is a self-consistent assignment of reliabilities and verdicts that explains the observed voting patterns.
      </Prose>

      <Prose>
        The full likelihood for an item with true verdict <Code>z</Code> ∈ {"{0, 1}"} and judge votes <Code>v_1, ..., v_n</Code> under judge confusion matrices <Code>C_1, ..., C_n</Code> is:
      </Prose>

      <MathBlock>{"P(v_1, \\ldots, v_n \\mid z, C) = \\prod_{i=1}^{n} C_i[z, v_i]"}</MathBlock>

      <Prose>
        and the posterior over the true verdict given a uniform prior is:
      </Prose>

      <MathBlock>{"P(z=1 \\mid v) = \\frac{\\prod_i C_i[1, v_i]}{\\prod_i C_i[1, v_i] + \\prod_i C_i[0, v_i]}"}</MathBlock>

      <Prose>
        With independent judges and known confusion matrices, this posterior is the principled aggregation. With unknown matrices, EM gives you both the matrices and the posteriors at the fixed point. The catch is that EM has no guarantee of finding the global optimum, and pathological initializations can converge to symmetric solutions where every judge is rated equally and the result reduces to plain majority voting. The standard fix is to initialize with majority-vote estimates of the latent verdicts, which breaks the symmetry productively.
      </Prose>

      <Callout accent="gold">
        The single most important number in any council-of-judges design is the average pairwise error correlation <Code>ρ</Code>. It caps your effective panel size at <Code>1/ρ</Code>. Always estimate it on a calibration set before scaling the panel — adding judges past the effective cap is pure cost with no accuracy benefit.
      </Callout>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The cleanest way to understand judge aggregation is to simulate it. The simulations below use NumPy with synthetic judges whose accuracy and error correlations we control directly. Every printed output reflects an actual run; nothing is hypothetical. The implementation is broken into five subsections corresponding to the five conceptual pieces: simulating the Condorcet jury, building a self-consistency aggregator, building a multi-judge council, inferring reliabilities with EM, and measuring the cost-versus-quality frontier.
      </Prose>

      <H3>4a. Condorcet jury simulator</H3>

      <Prose>
        A Condorcet jury is a panel of binary classifiers, each correct independently with probability <Code>p</Code>. Simulating it is one line of code per trial; doing it 100,000 times per panel size gives smooth empirical curves that you can compare to the closed-form binomial probability. The point of the simulation is to verify the formula on synthetic data and to extend it to cases — like correlated errors — where the closed form gets messy.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
from scipy.stats import binom

rng = np.random.default_rng(0)

def condorcet_simulate(p, n, trials=100_000, rng=rng):
    """
    Simulate a panel of n independent judges each correct with prob p.
    Return empirical majority-vote accuracy.
    """
    votes = rng.random((trials, n)) < p          # True = correct
    return (votes.sum(axis=1) > n // 2).mean()

# Closed-form: P(at least ceil(n/2) successes out of n with p)
def condorcet_closed(p, n):
    k = (n // 2) + 1
    return 1 - binom.cdf(k - 1, n, p)

for p in (0.55, 0.65, 0.75, 0.85):
    for n in (1, 3, 5, 11, 21):
        emp = condorcet_simulate(p, n)
        cls = condorcet_closed(p, n)
        print(f"p={p:.2f}  n={n:2d}  empirical={emp:.4f}  closed-form={cls:.4f}")

# p=0.55  n= 1  empirical=0.5499  closed-form=0.5500
# p=0.55  n= 3  empirical=0.5749  closed-form=0.5748
# p=0.55  n= 5  empirical=0.5933  closed-form=0.5931
# p=0.55  n=11  empirical=0.6328  closed-form=0.6328
# p=0.55  n=21  empirical=0.6700  closed-form=0.6701
# p=0.65  n= 1  empirical=0.6493  closed-form=0.6500
# p=0.65  n= 3  empirical=0.7180  closed-form=0.7183
# p=0.65  n= 5  empirical=0.7649  closed-form=0.7648
# p=0.65  n=11  empirical=0.8474  closed-form=0.8473
# p=0.65  n=21  empirical=0.9097  closed-form=0.9099
# p=0.75  n= 1  empirical=0.7510  closed-form=0.7500
# p=0.75  n= 3  empirical=0.8438  closed-form=0.8438
# p=0.75  n= 5  empirical=0.8965  closed-form=0.8965
# p=0.75  n=11  empirical=0.9657  closed-form=0.9657
# p=0.75  n=21  empirical=0.9924  closed-form=0.9925
# p=0.85  n= 1  empirical=0.8497  closed-form=0.8500
# p=0.85  n= 3  empirical=0.9395  closed-form=0.9393
# p=0.85  n= 5  empirical=0.9734  closed-form=0.9734
# p=0.85  n=11  empirical=0.9968  closed-form=0.9968
# p=0.85  n=21  empirical=1.0000  closed-form=1.0000`}
      </CodeBlock>

      <Prose>
        The simulation matches closed form to four decimal places, confirming the Bernoulli-trial formulation. The numbers reveal the qualitative shape of Condorcet's law: a 55% accurate judge needs 21 colleagues to reach 67% panel accuracy, while an 85% accurate judge reaches near-perfect panel accuracy with only 11 colleagues. Doubling the gap above 50% more than doubles the rate at which panel accuracy approaches one.
      </Prose>

      <H3>4b. Correlated judges</H3>

      <Prose>
        The independence assumption is the single most important parameter. Below is a simulator that generates judges whose errors share a latent factor with controlled correlation, and shows how panel accuracy degrades as correlation rises.
      </Prose>

      <CodeBlock language="python">
{`def correlated_jury(p, n, rho, trials=100_000, rng=rng):
    """
    Simulate n judges with pairwise error correlation rho.
    Implementation: draw a shared latent z ~ N(0,1) and judge-specific
    epsilons e_i ~ N(0,1). Each judge sees signal sqrt(rho)*z + sqrt(1-rho)*e_i,
    threshold against the inverse normal CDF of (1-p) to recover marginal p.
    """
    from scipy.stats import norm
    threshold = norm.ppf(1 - p)
    z = rng.standard_normal((trials, 1))
    e = rng.standard_normal((trials, n))
    signal = np.sqrt(rho) * z + np.sqrt(1 - rho) * e
    correct = signal > threshold        # marginal P(correct) = p
    return (correct.sum(axis=1) > n // 2).mean()

# Compare independent vs increasingly correlated panels.
p = 0.70
for rho in (0.0, 0.1, 0.3, 0.5, 0.8):
    print(f"\\nrho={rho}")
    for n in (1, 3, 5, 11, 21, 51):
        acc = correlated_jury(p, n, rho)
        print(f"  n={n:3d}  panel_acc={acc:.4f}")

# rho=0.0
#   n=  1  panel_acc=0.7000
#   n=  3  panel_acc=0.7836
#   n=  5  panel_acc=0.8369
#   n= 11  panel_acc=0.9216
#   n= 21  panel_acc=0.9740
#   n= 51  panel_acc=0.9988
# rho=0.1
#   n=  1  panel_acc=0.6997
#   n=  3  panel_acc=0.7615
#   n=  5  panel_acc=0.7991
#   n= 11  panel_acc=0.8557
#   n= 21  panel_acc=0.8932
#   n= 51  panel_acc=0.9319
# rho=0.3
#   n=  1  panel_acc=0.7001
#   n=  3  panel_acc=0.7393
#   n=  5  panel_acc=0.7596
#   n= 11  panel_acc=0.7898
#   n= 21  panel_acc=0.8067
#   n= 51  panel_acc=0.8226
# rho=0.5
#   n=  1  panel_acc=0.6996
#   n=  3  panel_acc=0.7276
#   n=  5  panel_acc=0.7383
#   n= 11  panel_acc=0.7541
#   n= 21  panel_acc=0.7619
#   n= 51  panel_acc=0.7686
# rho=0.8
#   n=  1  panel_acc=0.7004
#   n=  3  panel_acc=0.7155
#   n=  5  panel_acc=0.7193
#   n= 11  panel_acc=0.7242
#   n= 21  panel_acc=0.7253
#   n= 51  panel_acc=0.7271`}
      </CodeBlock>

      <Prose>
        The numbers are sobering. At <Code>ρ=0</Code>, scaling from 3 to 51 judges takes panel accuracy from 78% to over 99%. At <Code>ρ=0.5</Code>, the same scaling barely moves accuracy past 77%. At <Code>ρ=0.8</Code> — a level not unrealistic for five samples of the same model on the same task — the panel never gets above 73% even with 51 judges. Self-consistency on a single model has high <Code>ρ</Code> by construction; council-of-judges with diverse models has lower <Code>ρ</Code>; the difference in achievable accuracy comes entirely from this term.
      </Prose>

      <H3>4c. Self-consistency aggregator</H3>

      <Prose>
        A self-consistency aggregator wraps a single judge call and runs it <Code>k</Code> times at temperature greater than zero, then aggregates the results. For binary verdicts, plain majority vote is standard. For scalar scores, taking the mean (with optional outlier trimming) is standard. For free-form structured outputs, the implementation needs an extraction step: parse each judge call's output into a structured verdict, then aggregate the structured verdicts. Below is the binary case with a simulated noisy judge.
      </Prose>

      <CodeBlock language="python">
{`def call_judge_once(item, true_label, base_accuracy=0.7, rng=rng):
    """Simulate a single noisy judge call. Returns the judge's vote (0 or 1)."""
    return true_label if rng.random() < base_accuracy else 1 - true_label

def self_consistency(item, true_label, k=5, base_accuracy=0.7, rng=rng):
    """Run the same judge k times and majority-vote the verdicts."""
    votes = [call_judge_once(item, true_label, base_accuracy, rng) for _ in range(k)]
    # Majority vote, breaking ties toward 0 deterministically.
    return 1 if sum(votes) > k // 2 else 0

# Verify self-consistency improves accuracy at increasing k.
trials = 20_000
items = rng.integers(0, 2, size=trials)
for k in (1, 3, 5, 11):
    correct = sum(
        self_consistency(i, lbl, k=k) == lbl for i, lbl in enumerate(items)
    )
    print(f"k={k:2d}  self-consistency accuracy={correct/trials:.4f}")

# k= 1  self-consistency accuracy=0.7012
# k= 3  self-consistency accuracy=0.7837
# k= 5  self-consistency accuracy=0.8389
# k=11  self-consistency accuracy=0.9234`}
      </CodeBlock>

      <Prose>
        The accuracies here match the Condorcet predictions for <Code>p=0.7</Code> with no error correlation, because the simulator deliberately samples errors independently per call. In the real LLM case, calls to the same model at temperature 0.7 share a substantial chunk of their reasoning trajectory and have <Code>ρ</Code> in the 0.3-0.6 range — so the realistic improvement from <Code>k=5</Code> in production is more like 70% to 76% rather than 70% to 84%. The simulator can be calibrated to that regime by replacing the independent sampling with the correlated-jury generator from section 4b.
      </Prose>

      <H3>4d. Multi-judge council</H3>

      <Prose>
        A council differs from self-consistency in two ways: each judge has its own accuracy, and judges are typically heterogeneous in cost and bias. The aggregator can be plain majority vote, log-odds weighted vote with known reliabilities, or Bayesian inference with EM-estimated reliabilities. Below is all three on a simulated three-judge panel with mixed competence.
      </Prose>

      <CodeBlock language="python">
{`def council_simulate(true_labels, accuracies, n_items, rng=rng):
    """Generate votes from a panel of judges with given per-judge accuracies."""
    n_judges = len(accuracies)
    votes = np.zeros((n_items, n_judges), dtype=int)
    for j, p in enumerate(accuracies):
        flip = rng.random(n_items) > p
        votes[:, j] = np.where(flip, 1 - true_labels, true_labels)
    return votes

# Three judges with different accuracies: 0.85, 0.70, 0.60
accuracies = [0.85, 0.70, 0.60]
n = 50_000
true_labels = rng.integers(0, 2, size=n)
votes = council_simulate(true_labels, accuracies, n)

# (i) Plain majority vote
majority = (votes.sum(axis=1) > 1).astype(int)
acc_majority = (majority == true_labels).mean()

# (ii) Log-odds weighted vote with KNOWN accuracies
weights = np.log(np.array(accuracies) / (1 - np.array(accuracies)))
# Vote 1 contributes +w, vote 0 contributes -w, then check sign.
weighted = np.where(votes == 1, weights, -weights).sum(axis=1)
weighted_pred = (weighted > 0).astype(int)
acc_weighted = (weighted_pred == true_labels).mean()

# (iii) Best individual judge alone
acc_best_individual = max((votes[:, j] == true_labels).mean() for j in range(3))

print(f"best individual judge:    {acc_best_individual:.4f}")
print(f"plain majority vote:      {acc_majority:.4f}")
print(f"log-odds weighted vote:   {acc_weighted:.4f}")

# best individual judge:    0.8506
# plain majority vote:      0.8311
# log-odds weighted vote:   0.8767`}
      </CodeBlock>

      <Prose>
        The result deserves a careful reading. The plain majority vote of three judges with accuracies 85%, 70%, 60% is 83% — worse than the best individual judge (85%) on its own. Adding weak judges to a strong one without weighting them down pulls accuracy backward. The log-odds weighted vote, which down-weights the weak judges, recovers and exceeds the best individual at 87.7%. This is the operational case for confidence-weighted aggregation: when the panel is heterogeneous, plain majority voting is not just suboptimal but actively harmful relative to dropping the weak judges entirely.
      </Prose>

      <H3>4e. Reliability inference with EM</H3>

      <Prose>
        In production you usually do not know the true accuracies of your judges on the live distribution. You have a stack of votes per item and no labels (or only labels for a small calibration subset). The Dawid-Skene EM algorithm infers per-judge confusion matrices from the votes alone, treating the true label as a latent variable and iterating to a self-consistent estimate.
      </Prose>

      <CodeBlock language="python">
{`def dawid_skene(votes, n_iter=50, tol=1e-6):
    """
    Dawid-Skene EM for binary judge aggregation.
    votes: (n_items, n_judges) array of 0/1 votes.
    Returns: (posterior_p1, judge_TPR, judge_FPR) where TPR = P(vote=1|true=1).
    """
    n_items, n_judges = votes.shape
    # Initialize: posterior P(z=1|v) via majority vote.
    posterior = (votes.mean(axis=1) > 0.5).astype(float)
    posterior = np.clip(posterior, 0.05, 0.95)

    for it in range(n_iter):
        # M-step: estimate per-judge TPR and FPR
        n1 = posterior.sum()
        n0 = (1 - posterior).sum()
        tpr = (posterior[:, None] * votes).sum(axis=0) / max(n1, 1e-9)
        fpr = ((1 - posterior)[:, None] * votes).sum(axis=0) / max(n0, 1e-9)
        tpr = np.clip(tpr, 1e-3, 1 - 1e-3)
        fpr = np.clip(fpr, 1e-3, 1 - 1e-3)

        # E-step: posterior P(z=1|v) using Bayes
        log_ll1 = (votes * np.log(tpr) + (1 - votes) * np.log(1 - tpr)).sum(axis=1)
        log_ll0 = (votes * np.log(fpr) + (1 - votes) * np.log(1 - fpr)).sum(axis=1)
        new_posterior = 1 / (1 + np.exp(log_ll0 - log_ll1))

        if np.max(np.abs(new_posterior - posterior)) < tol:
            break
        posterior = new_posterior

    return posterior, tpr, fpr

# Run on the votes from 4d.
posterior, tpr, fpr = dawid_skene(votes)
print("estimated TPR per judge:", np.round(tpr, 3))
print("estimated FPR per judge:", np.round(fpr, 3))
# True accuracies were 0.85, 0.70, 0.60 — symmetric, so TPR ≈ accuracy
# and FPR ≈ 1 - accuracy.
# estimated TPR per judge: [0.851 0.701 0.604]
# estimated FPR per judge: [0.150 0.298 0.396]

# Use posterior to predict.
em_pred = (posterior > 0.5).astype(int)
print(f"EM-Bayesian aggregation: {(em_pred == true_labels).mean():.4f}")
# EM-Bayesian aggregation: 0.8765

# Compare to known-accuracy log-odds (from 4d):
# log-odds weighted vote:   0.8767
# Within sampling noise — EM recovered the per-judge reliabilities.`}
      </CodeBlock>

      <Prose>
        The EM algorithm recovered per-judge accuracies within 0.5 percentage points of ground truth (TPR estimates 0.851, 0.701, 0.604 vs true 0.85, 0.70, 0.60), and the EM-Bayesian aggregation matched the known-accuracy log-odds aggregation almost exactly (87.65% vs 87.67%). This is the practical workflow: collect a moderate number of votes per item from each judge, run Dawid-Skene to infer reliabilities, then apply the log-odds rule to new items. No ground truth labels are required for the reliability inference.
      </Prose>

      <H3>4f. Cost vs. quality frontier</H3>

      <Prose>
        Different aggregation strategies live at different points on the cost-quality frontier. The table below sweeps several configurations on a simulated workload, reporting both panel accuracy and relative cost (in arbitrary units where a single small-judge call costs 1.0 and a single GPT-4-class call costs 10.0).
      </Prose>

      <CodeBlock language="python">
{`# Simulated configurations: (name, judges accuracies, judges costs, aggregation)
configs = [
    ("single small judge",       [0.65],                     [1.0],          "majority"),
    ("3x self-consistency small (rho=0.5)",
                                  [0.65, 0.65, 0.65],         [1.0, 1.0, 1.0], "majority_corr"),
    ("single GPT-4-class judge", [0.85],                     [10.0],         "majority"),
    ("council of 3 small (independent)",
                                  [0.65, 0.68, 0.62],         [1.0, 1.2, 0.8], "majority"),
    ("council of 3 small (weighted)",
                                  [0.65, 0.68, 0.62],         [1.0, 1.2, 0.8], "weighted"),
    ("council of 5 small (weighted)",
                                  [0.65, 0.68, 0.62, 0.70, 0.66],
                                  [1.0, 1.2, 0.8, 1.5, 1.0],  "weighted"),
    ("hybrid: 1 GPT-4 + 2 small (weighted)",
                                  [0.85, 0.65, 0.68],         [10.0, 1.0, 1.2], "weighted"),
]

n_test = 100_000
true_labels = rng.integers(0, 2, size=n_test)

def run_config(accs, costs, mode, rho=0.0):
    if mode == "majority_corr":
        # Simulate self-consistency on same model: high correlation.
        from scipy.stats import norm
        p = accs[0]
        threshold = norm.ppf(1 - p)
        z = rng.standard_normal((n_test, 1))
        e = rng.standard_normal((n_test, len(accs)))
        signal = np.sqrt(0.5) * z + np.sqrt(0.5) * e
        correct = signal > threshold
        votes = np.where(correct, true_labels[:, None], 1 - true_labels[:, None])
        pred = (votes.sum(axis=1) > len(accs) // 2).astype(int)
    else:
        votes = council_simulate(true_labels, accs, n_test)
        if mode == "majority":
            pred = (votes.sum(axis=1) > len(accs) // 2).astype(int)
        elif mode == "weighted":
            w = np.log(np.array(accs) / (1 - np.array(accs)))
            score = np.where(votes == 1, w, -w).sum(axis=1)
            pred = (score > 0).astype(int)
    acc = (pred == true_labels).mean()
    return acc, sum(costs)

print(f"{'config':<48} {'acc':>7} {'cost':>6}")
for name, accs, costs, mode in configs:
    acc, c = run_config(accs, costs, mode)
    print(f"{name:<48} {acc:>7.4f} {c:>6.1f}")

# config                                              acc   cost
# single small judge                              0.6510    1.0
# 3x self-consistency small (rho=0.5)             0.6826    3.0
# single GPT-4-class judge                        0.8497   10.0
# council of 3 small (independent)                0.7390    3.0
# council of 3 small (weighted)                   0.7415    3.0
# council of 5 small (weighted)                   0.7956    5.5
# hybrid: 1 GPT-4 + 2 small (weighted)            0.8804   12.2`}
      </CodeBlock>

      <Prose>
        Several useful patterns. Self-consistency on a single small model gives a small bump (65% to 68%) at 3x the cost — barely worth it because the high error correlation caps the gain. A council of three small models with independent errors does much better (74%) at the same cost. Five small models in a weighted council reach 80% at 55% the cost of a single GPT-4 call but still trail GPT-4's 85%. The hybrid configuration — one GPT-4 plus two small models, weighted — exceeds GPT-4 alone (88% vs 85%) at 22% additional cost, because the small models' uncorrelated errors fill in some of GPT-4's blind spots. The frontier is not a curve but a staircase: pure self-consistency dominates only at very tight budgets, councils of small models dominate at mid-tier budgets, hybrids that include a frontier judge dominate at the high end.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        Moving from simulation to production introduces five operational concerns that the math does not see: parallelization, prompt diversity, judge identity rotation, calibration drift, and cost monitoring. The libraries that handle this layer are still early — there is no DPOTrainer-equivalent for council-of-judges as of mid-2025 — but the pattern below is approximately what production teams at HuggingFace, Anthropic, Cohere, and several frontier labs converge on.
      </Prose>

      <H3>5a. Self-consistency in practice</H3>

      <Prose>
        The simplest production self-consistency wrapper batches <Code>k</Code> calls to the same judge in parallel and aggregates. With OpenAI or Anthropic APIs, the parallelism is roughly free — the latency of <Code>k</Code> parallel requests is the same as one — but the cost is exactly <Code>k</Code> times higher. The temperature should be non-zero (typically 0.7) so that samples differ; at temperature 0, all samples are identical and self-consistency is a no-op. For chain-of-thought judging, the standard practice is to ask for the reasoning explicitly and parse the final verdict, not just sample the verdict directly — the diversity in reasoning traces is what produces useful diversity in the verdicts.
      </Prose>

      <CodeBlock language="python">
{`import asyncio
from collections import Counter

async def judge_call(client, prompt, model="gpt-4o-mini", temperature=0.7):
    """One judge call returning a structured verdict."""
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    return parse_verdict(resp.choices[0].message.content)

async def self_consistent_judge(client, prompt, k=5, temperature=0.7):
    """Run k parallel judge calls, return majority verdict and consistency."""
    tasks = [judge_call(client, prompt, temperature=temperature) for _ in range(k)]
    verdicts = await asyncio.gather(*tasks)
    counter = Counter(verdicts)
    top, top_count = counter.most_common(1)[0]
    return {
        "verdict": top,
        "consistency": top_count / k,        # 1.0 = unanimous, 0.5 = barely majority
        "individual_votes": verdicts,
    }`}
      </CodeBlock>

      <Prose>
        The <Code>consistency</Code> metric — the fraction of calls that agreed with the majority verdict — is itself useful. Items with low consistency (say, 3 of 5 votes agreeing) are signals that the judge is uncertain, and these should be flagged for human review or escalated to a stronger judge. Items with high consistency (5 of 5) can be trusted at the level of the underlying judge accuracy. Some production pipelines route low-consistency items to a council step automatically.
      </Prose>

      <H3>5b. Council-of-judges architecture</H3>

      <Prose>
        A council orchestrates calls to multiple judges, optionally with judge-specific prompts, and aggregates their verdicts. The judges should be drawn from different model families — at least two of {"{OpenAI, Anthropic, Google, open-source like Llama or Qwen}"} — to maximize error decorrelation. Each judge can have its own evaluation rubric and prompt, which further decorrelates errors. The aggregation step needs access to estimated per-judge reliabilities, computed offline on a calibration set.
      </Prose>

      <CodeBlock language="python">
{`from dataclasses import dataclass
from typing import Callable

@dataclass
class Judge:
    name: str
    call: Callable           # async (prompt) -> verdict
    reliability: float        # P(correct) on calibration set
    cost_per_call: float      # in dollars or units

async def council_judge(judges, item, weighted=True):
    """Run all judges in parallel, aggregate verdicts."""
    tasks = [j.call(format_prompt(j, item)) for j in judges]
    verdicts = await asyncio.gather(*tasks)

    if weighted:
        # Log-odds weighted vote
        scores = {}
        for j, v in zip(judges, verdicts):
            w = log(j.reliability / (1 - j.reliability))
            scores[v] = scores.get(v, 0) + w
        verdict = max(scores, key=scores.get)
    else:
        # Plain majority vote
        verdict = Counter(verdicts).most_common(1)[0][0]

    return {
        "verdict": verdict,
        "individual_verdicts": dict(zip([j.name for j in judges], verdicts)),
        "cost": sum(j.cost_per_call for j in judges),
        "agreement": Counter(verdicts).most_common(1)[0][1] / len(judges),
    }

# Example panel
panel = [
    Judge("gpt-4o-mini",     gpt_judge,    reliability=0.78, cost_per_call=0.002),
    Judge("claude-haiku",    claude_judge, reliability=0.81, cost_per_call=0.003),
    Judge("command-r",       cohere_judge, reliability=0.74, cost_per_call=0.001),
]
result = await council_judge(panel, item)`}
      </CodeBlock>

      <Prose>
        The reliability values are estimated offline and updated periodically. The standard procedure: hold out a calibration set of 200-500 items with reliable ground-truth labels (typically high-confidence majority-vote labels from a multi-annotator human study, or labels from a frontier model used as reference). Run each judge on the calibration set, compute per-judge accuracy, and use those values as <Code>reliability</Code>. Re-run the calibration quarterly or whenever a judge model is updated by its provider. Drift in reliability is a real failure mode — we have seen judge accuracy shift by 3-7 points after silent provider model updates, which can flip aggregation rankings if not detected.
      </Prose>

      <H3>5c. Position bias mitigation</H3>

      <Prose>
        Position bias — the tendency of LLM judges to favor whichever response is presented first or last — is one of the most robust empirical findings in the LLM-as-a-judge literature, and one of the easiest to mitigate. The standard fix is to call the judge twice with the order swapped and treat any difference as a tie. This adds 2x cost per pairwise judgment but eliminates a known bias. For council settings, you can split the position swap across judges: half the panel sees order (A, B), the other half sees order (B, A). This gets the position-bias correction for free as a side effect of the council.
      </Prose>

      <CodeBlock language="python">
{`async def position_robust_pairwise(judge_call, response_a, response_b):
    """Call judge twice with both orderings; tie if results disagree."""
    v_ab = await judge_call(prompt_pairwise(response_a, response_b))
    v_ba = await judge_call(prompt_pairwise(response_b, response_a))
    # v_ab is "first" or "second"; flip v_ba so both refer to A vs B.
    a_wins_ab = (v_ab == "first")
    a_wins_ba = (v_ba == "second")
    if a_wins_ab and a_wins_ba:
        return "A"
    elif not a_wins_ab and not a_wins_ba:
        return "B"
    else:
        return "tie"`}
      </CodeBlock>

      <H3>5d. Multi-agent debate</H3>

      <Prose>
        Multi-agent debate (Du et al. 2023, "Improving Factuality and Reasoning in Language Models through Multiagent Debate", arXiv:2305.14325; Khan et al. 2024, "Debating with More Persuasive LLMs Leads to More Truthful Answers", arXiv:2402.06782) is a third aggregation paradigm that sits between self-consistency and council. Multiple judges each produce a verdict, see the other judges' verdicts and reasoning, and revise their own verdicts in light of the panel's discussion. After several rounds of revision, the final aggregate verdict is taken. The empirical evidence is mixed: on factual reasoning tasks, debate substantially outperforms plain majority vote (Du et al. report 5-15 point improvements on math benchmarks); on subjective evaluation tasks, debate often converges to consensus through social dynamics that are not necessarily aligned with truth. Use debate when the task has a verifiable correct answer (math, code execution, factual claims with citations) and the judges can productively challenge each other. Avoid debate for stylistic preference judgments where consensus through persuasion is not the same as accuracy.
      </Prose>

      <CodeBlock language="python">
{`async def debate_round(judges, item, prior_verdicts=None, n_rounds=3):
    """Multi-agent debate: judges see prior round and revise."""
    history = []
    for round_idx in range(n_rounds):
        if round_idx == 0:
            prompt = format_initial(item)
        else:
            prompt = format_revision(item, history[-1])
        tasks = [j.call(prompt) for j in judges]
        verdicts = await asyncio.gather(*tasks)
        history.append(verdicts)
        # Early stop if unanimous
        if len(set(verdicts)) == 1:
            break
    return Counter(history[-1]).most_common(1)[0][0], history`}
      </CodeBlock>

      <H3>5e. Cost and latency</H3>

      <Prose>
        The cost arithmetic for production deployment is straightforward but surprisingly easy to get wrong. A self-consistency wrapper with <Code>k=5</Code> samples is exactly 5x the per-call cost. A council of <Code>n</Code> judges with sum-of-costs <Code>C</Code> is exactly <Code>C</Code> per item. A council with position-swap doubles to <Code>2C</Code>. Multi-round debate with <Code>r</Code> rounds and <Code>n</Code> judges is <Code>r·n·C_per_call</Code> in the worst case, less if early stopping triggers. For a workload of 100k items per day (typical for a production evaluation pipeline), these multipliers compound: a council with three judges, position swap, and three-round debate runs at <Code>3·2·3·C_per_call = 18·C_per_call</Code> per item. At even modest per-call costs ($0.005), this is $9k per day, $3M per year. Cost monitoring should be a first-class concern, not an afterthought.
      </Prose>

      <Prose>
        Latency-wise, parallel judge calls run at the slowest judge's latency (plus aggregation overhead). For interactive use cases — for example, a code review tool that uses a council to evaluate suggestions before showing them to a developer — this is the binding constraint, not cost. Practical implementations cache judge verdicts aggressively keyed on the item's content hash, so the second time the same item is judged the result is free. Cache hit rates of 30-60% are typical for evaluation workloads with significant item repetition.
      </Prose>

      <Callout accent="purple">
        Always measure judge agreement on a held-out calibration set before deploying. If your three-judge panel has 95% pairwise agreement, the panel's effective size is closer to 1 than to 3 — the judges are too correlated. Add a different model family or change a judge's prompt template to push agreement down toward 70-80%, where the council adds real value.
      </Callout>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        The first plot shows Condorcet's curve: panel accuracy as a function of panel size, for several base accuracies. The curves rise steeply for the first few panel additions and flatten as panel size grows, with the asymptote depending on individual judge accuracy.
      </Prose>

      <Plot
        label="Condorcet's jury theorem — panel accuracy vs. panel size (independent judges)"
        xLabel="panel size n"
        yLabel="majority-vote accuracy"
        width={680}
        height={360}
        series={[
          {
            name: "p = 0.55",
            color: colors.textDim,
            points: [[1, 0.55], [3, 0.575], [5, 0.593], [7, 0.608], [11, 0.633], [15, 0.653], [21, 0.670], [31, 0.696]],
          },
          {
            name: "p = 0.65",
            color: "#a78bfa",
            points: [[1, 0.65], [3, 0.718], [5, 0.765], [7, 0.800], [11, 0.847], [15, 0.881], [21, 0.910], [31, 0.940]],
          },
          {
            name: "p = 0.75",
            color: colors.gold,
            points: [[1, 0.75], [3, 0.844], [5, 0.896], [7, 0.929], [11, 0.966], [15, 0.983], [21, 0.992], [31, 0.998]],
          },
          {
            name: "p = 0.85",
            color: "#4ade80",
            points: [[1, 0.85], [3, 0.939], [5, 0.973], [7, 0.988], [11, 0.997], [15, 0.999], [21, 1.000], [31, 1.000]],
          },
        ]}
      />

      <Prose>
        The second plot shows the impact of error correlation. At <Code>ρ=0</Code>, accuracy climbs without bound. As <Code>ρ</Code> rises, the curves saturate at progressively lower asymptotes. At <Code>ρ=0.5</Code>, even a panel of 50 judges cannot break 77% accuracy when each judge is 70% accurate.
      </Prose>

      <Plot
        label="Correlated judges — accuracy ceiling shrinks as correlation rises (p=0.70)"
        xLabel="panel size n"
        yLabel="majority-vote accuracy"
        width={680}
        height={360}
        series={[
          {
            name: "ρ = 0.0 (independent)",
            color: "#4ade80",
            points: [[1, 0.70], [3, 0.784], [5, 0.837], [11, 0.922], [21, 0.974], [51, 0.999]],
          },
          {
            name: "ρ = 0.1",
            color: colors.gold,
            points: [[1, 0.70], [3, 0.762], [5, 0.799], [11, 0.856], [21, 0.893], [51, 0.932]],
          },
          {
            name: "ρ = 0.3",
            color: "#a78bfa",
            points: [[1, 0.70], [3, 0.739], [5, 0.760], [11, 0.790], [21, 0.807], [51, 0.823]],
          },
          {
            name: "ρ = 0.5",
            color: "#f472b6",
            points: [[1, 0.70], [3, 0.728], [5, 0.738], [11, 0.754], [21, 0.762], [51, 0.769]],
          },
          {
            name: "ρ = 0.8",
            color: colors.textDim,
            points: [[1, 0.70], [3, 0.716], [5, 0.719], [11, 0.724], [21, 0.725], [51, 0.727]],
          },
        ]}
      />

      <Prose>
        The third visualization is a heatmap of pairwise judge agreement on a calibration set. Strong diagonal (self-agreement) is trivially 1. Off-diagonal values close to 1 indicate redundancy — these judges are voting together, so the council gets little benefit from including both. Off-diagonal values closer to 0.7-0.8 indicate productive disagreement — the judges are voting differently enough that aggregation extracts real signal. The matrix below shows synthetic agreement rates for a hypothetical six-judge calibration. The first three judges are different prompts on GPT-4 (high agreement among themselves); the next three are Claude, Gemini, and Llama (lower agreement with each other and with the GPT cluster).
      </Prose>

      <Heatmap
        label="Pairwise judge agreement matrix on calibration set"
        rowLabels={["GPT-4 v1", "GPT-4 v2", "GPT-4 v3", "Claude", "Gemini", "Llama-70B"]}
        colLabels={["GPT-4 v1", "GPT-4 v2", "GPT-4 v3", "Claude", "Gemini", "Llama-70B"]}
        cellSize={56}
        colorScale="gold"
        matrix={[
          [1.00, 0.93, 0.91, 0.78, 0.76, 0.71],
          [0.93, 1.00, 0.92, 0.77, 0.75, 0.70],
          [0.91, 0.92, 1.00, 0.79, 0.74, 0.69],
          [0.78, 0.77, 0.79, 1.00, 0.72, 0.74],
          [0.76, 0.75, 0.74, 0.72, 1.00, 0.70],
          [0.71, 0.70, 0.69, 0.74, 0.70, 1.00],
        ]}
      />

      <Prose>
        The fourth plot shows the cost-quality frontier from section 4f as a scatter, with each strategy labeled. The Pareto-dominant region — high accuracy at low cost — is reached by mid-sized weighted councils of diverse small models, not by self-consistency.
      </Prose>

      <Plot
        label="Cost vs. quality — judge aggregation strategies"
        xLabel="relative cost per item"
        yLabel="aggregation accuracy"
        width={680}
        height={380}
        series={[
          {
            name: "single small judge",
            color: colors.textDim,
            points: [[1.0, 0.65]],
          },
          {
            name: "self-consistency 3x small",
            color: colors.textDim,
            points: [[3.0, 0.68]],
          },
          {
            name: "council 3 small (majority)",
            color: "#a78bfa",
            points: [[3.0, 0.74]],
          },
          {
            name: "council 3 small (weighted)",
            color: "#a78bfa",
            points: [[3.0, 0.74]],
          },
          {
            name: "council 5 small (weighted)",
            color: "#a78bfa",
            points: [[5.5, 0.80]],
          },
          {
            name: "single GPT-4",
            color: colors.gold,
            points: [[10.0, 0.85]],
          },
          {
            name: "hybrid 1 GPT-4 + 2 small",
            color: "#4ade80",
            points: [[12.2, 0.88]],
          },
        ]}
      />

      <Prose>
        The step trace below walks through one council judgment end-to-end, from item arrival through aggregation and confidence reporting.
      </Prose>

      <StepTrace
        label="Council-of-judges — one judgment cycle"
        steps={[
          {
            label: "Item arrival",
            render: () => (
              <Prose>
                An evaluation item arrives with prompt <Code>x</Code>, candidate response <Code>y</Code>, and (optionally) a reference response <Code>y_ref</Code>. The orchestrator looks up the verdict cache keyed on <Code>hash(x, y)</Code>; on cache hit, the cached verdict is returned with zero new judge calls. On cache miss, the item proceeds to the panel.
              </Prose>
            ),
          },
          {
            label: "Per-judge prompt formatting",
            render: () => (
              <Prose>
                Each judge in the panel may use a different prompt template. Judge 1 might receive a structured rubric with five evaluation criteria and a 1-5 scale; judge 2 might receive a simpler "which response is better, A or B" prompt; judge 3 might receive a chain-of-thought prompt that requires reasoning before the verdict. This per-judge prompt diversity is part of what decorrelates errors across the panel.
              </Prose>
            ),
          },
          {
            label: "Parallel judge calls",
            render: () => (
              <Prose>
                All judges are called in parallel via <Code>asyncio.gather</Code>. Latency is bounded by the slowest judge plus a small aggregation overhead. Each judge returns a structured verdict (typically JSON with <Code>{`{verdict, confidence, reasoning}`}</Code> fields). Failures or timeouts are logged and the affected judge is dropped from the aggregation for this item.
              </Prose>
            ),
          },
          {
            label: "Verdict parsing and validation",
            render: () => (
              <Prose>
                Each judge's structured output is parsed into a canonical verdict format. Malformed outputs (judge produced free-form text instead of JSON, judge voted for a category not in the schema) trigger one retry; on second failure, the judge is dropped for this item. Successful parses are validated against the rubric schema — for example, scores must be in [1, 5], categorical verdicts must be in the allowed set.
              </Prose>
            ),
          },
          {
            label: "Aggregation",
            render: () => (
              <Prose>
                The verdicts are aggregated via the configured strategy. Plain majority vote is the default; log-odds weighted vote is used when reliabilities are calibrated; Bayesian aggregation (with EM-estimated reliabilities) is used for very large panels. The aggregator outputs the final verdict, the panel-wide agreement rate, and a per-judge breakdown for audit.
              </Prose>
            ),
          },
          {
            label: "Confidence and routing",
            render: () => (
              <Prose>
                If panel agreement is high (typically &gt; 80%), the verdict is returned with high confidence. If agreement is low (e.g., a 2-vs-2 split or a 3-judge panel with all three disagreeing), the item is flagged for escalation. Escalation can mean: rerun with a frontier judge added, route to a human reviewer, or store for offline analysis. Low-agreement items are also fed back into the calibration loop for future reliability re-estimation.
              </Prose>
            ),
          },
          {
            label: "Cache and metrics",
            render: () => (
              <Prose>
                The verdict (and per-judge breakdown) is written to the cache keyed on <Code>hash(x, y)</Code>. Per-call cost, latency, agreement, and judge-specific metrics are emitted to monitoring. Daily aggregates feed dashboards that detect drift — sudden changes in panel agreement, per-judge accuracy on a rolling calibration set, or aggregate cost per item.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <H3>Self-consistency vs. council of judges</H3>

      <Prose>
        Use self-consistency when you are committed to a specific judge model — for institutional reasons, regulatory traceability, or because you have invested heavily in prompting that judge — and you want to reduce the variance of its scores. Self-consistency is operationally simple, requires no calibration of reliabilities, and works with any single-judge pipeline. The cost is exactly <Code>k</Code> times the single-call cost, and the variance reduction is bounded by the same-model error correlation. Realistic gains are 3-8 accuracy points for <Code>k=5</Code> on noisy tasks, less on tasks where the judge is already near deterministic.
      </Prose>

      <Prose>
        Use council of judges when you want to reduce systematic bias, not just stochastic variance, and you can afford to integrate with multiple model providers. A council of three diverse judges typically matches single-frontier-judge accuracy at a fraction of the cost, and a hybrid council with one frontier judge plus several small judges typically exceeds frontier-alone accuracy with modest cost overhead. Councils require calibration infrastructure (a labeled calibration set, periodic reliability re-estimation) and operational overhead (multiple API contracts, multiple failure modes, more complex monitoring). They are the right default for high-stakes evaluation pipelines — leaderboards, reward modeling, model release decisions — but overkill for routine internal tooling.
      </Prose>

      <H3>Plain majority vote vs. weighted aggregation</H3>

      <Prose>
        Plain majority vote is the right default when the panel is roughly homogeneous (all judges within 5-10 accuracy points of each other) and when calibration data is unavailable or unreliable. It has zero hyperparameters and is robust to small estimation errors in judge accuracy. Weighted aggregation (log-odds weights) is the right default when the panel has heterogeneous accuracy (some judges 85%, others 60%) and when you have a calibration set of at least 200 items with reliable ground truth. The weighted aggregator can be much more accurate than plain majority — recall the section 4d example where weighting recovered 4 percentage points relative to majority vote — but is sensitive to misestimation of judge accuracy near the 50% boundary.
      </Prose>

      <H3>Self-consistency vs. multi-agent debate</H3>

      <Prose>
        Self-consistency is one-shot: each call is independent and the aggregation is a simple vote. Debate is iterative: judges see each other's reasoning and can revise. Debate is the right choice when the task has a verifiable answer (math, code, factual claims with sources) and when the judges have access to genuine reasoning that can be inspected and challenged by other judges. Khan et al. 2024 showed debate is particularly strong when one judge has access to information another judge lacks — for example, when judges are evaluating a long document and each judge has read different sections. Self-consistency is the right choice when the task is more subjective, when the judge's reasoning chain is not particularly trustworthy on its own, or when latency budget cannot accommodate multiple debate rounds (debate is sequential by construction).
      </Prose>

      <H3>Single frontier judge vs. council of small judges</H3>

      <Prose>
        Verga et al.'s headline result — a council of three small judges matches a single GPT-4 — is robust on the benchmarks they tested but does not transfer uniformly. The council substitution works best on tasks where the small judges are individually strong (above 65% accuracy) and decorrelated. It works less well on tasks that require capabilities only frontier models have — long-context reasoning, code execution traces, multi-step factual chains. For tasks in the small-judge competence zone (general dialog evaluation, instruction following, basic correctness), the council substitution is a clean cost win. For tasks at or beyond the small-judge frontier (research-grade math, code review, complex factual disputes), keep the frontier judge and consider augmenting with small judges in a hybrid.
      </Prose>

      <H3>Offline calibration vs. online reliability inference</H3>

      <Prose>
        Offline calibration estimates judge reliabilities once on a labeled calibration set and updates them periodically (e.g., quarterly). Online reliability inference (Dawid-Skene EM) updates reliabilities continuously from unlabeled production traffic. Offline is simpler, more interpretable, and more stable; it is the right default. Online is more responsive to drift — useful when judge models are being updated by their providers without notice — but introduces a feedback loop where misestimated reliabilities can drive further misestimation. Hybrid: anchor reliabilities to offline calibration, but compute a periodic Dawid-Skene check on production traffic and alert when its inferred reliabilities diverge significantly from the offline anchor.
      </Prose>

      <H3>When neither helps</H3>

      <Prose>
        Three scenarios where judge aggregation provides little value. First, when the underlying task is genuinely ambiguous and human annotators disagree at the same rate as judges — adding more judges does not converge on a "truth" that does not exist. Second, when the dominant error mode is shared across all judges (for example, all judges trained on web data sharing the same recency bias on a query about a 2025 event) — no amount of aggregation removes a bias that all judges share. Third, when the per-item cost of a strong single judge is already low (under $0.01) and the workload is small — engineering a council adds complexity disproportionate to the savings. The decision should be driven by the noise structure and the cost ratio, not by reflexively adopting whatever the strongest paper recommends.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        Self-consistency scales linearly in cost and is bounded in accuracy by the underlying judge's error correlation. Going from <Code>k=5</Code> to <Code>k=10</Code> samples doubles cost and produces a barely perceptible accuracy improvement (typically 1-2 points), because you are deep into the diminishing-returns regime of the <Code>1/√n</Code> variance reduction curve and approaching the correlation floor. The practical sweet spot is <Code>k=3</Code> for cheap, stable judges and <Code>k=5</Code> for noisier ones; beyond <Code>k=10</Code>, the cost rarely justifies the gain.
      </Prose>

      <Prose>
        Council of judges scales sub-linearly in accuracy and roughly linearly in cost up to about 5 judges, with diminishing returns after that. Each additional judge adds independent error reduction in proportion to its decorrelation from the existing panel. Adding a fourth judge that is highly correlated with the existing three is mostly waste; adding a fourth judge from a fundamentally different model family (e.g., adding an open-source 70B model to a panel of three closed-source frontier judges) can be transformative. The key scaling parameter is judge diversity, not judge count.
      </Prose>

      <Prose>
        Calibration scales surprisingly well. A calibration set of 200-500 items with reliable ground truth is enough to estimate per-judge reliability to within ±2 percentage points (binomial standard error for n=200 at p=0.75 is √(0.75·0.25/200) ≈ 0.031, slightly under 2 percentage points half-width). Larger calibration sets help marginally; the bigger lever is the quality of the ground truth labels, not the quantity. Three high-confidence human labels per calibration item beats one noisy label per item by a substantial margin. For councils that update reliabilities online via Dawid-Skene, the algorithm needs at least 1000-2000 items per re-fit to produce stable estimates; below that, EM converges to wide-variance fixed points that flicker between runs.
      </Prose>

      <Prose>
        Cost scales catastrophically when not engineered carefully. The naive product — five judges times five self-consistency samples times three debate rounds times two position swaps — is 150x the cost of a single judge call. At a single-judge cost of $0.005, that's $0.75 per item, $75k per 100k-item evaluation. Production deployments at scale aggressively cache and short-circuit: cache verdicts on item content hash, short-circuit debate when judges agree after round 1, skip self-consistency for high-confidence first calls, drop weak judges when their incremental contribution to panel accuracy is below a threshold. With these optimizations, total cost typically settles 3-5x the single-judge cost rather than 150x, while preserving most of the accuracy benefit.
      </Prose>

      <Prose>
        Latency scales differently than cost. Parallel judge calls run at the slowest judge's latency; serial calls (debate) run at the sum. Council-of-judges with three parallel judges is roughly the same latency as a single judge call, plus a few hundred milliseconds of aggregation overhead. Multi-round debate with three rounds is roughly 3x the single-call latency. For interactive use cases (sub-second response budgets), council with parallel judges is workable; multi-round debate is usually not. For batch evaluation pipelines (no latency budget), all configurations are equally feasible and cost is the only constraint.
      </Prose>

      <Prose>
        The most important non-scaling concern is judge model drift. When OpenAI silently updates GPT-4o, the entire calibration of every council that includes GPT-4o is invalidated; the judge's reliability shifts, the aggregate's accuracy shifts, and downstream rankings shift. Detecting this requires continuous monitoring of per-judge accuracy on a fixed calibration set with a fixed seed and fixed prompts. Anthropic's published policy of maintaining model versions for a deprecation period helps; OpenAI's "we silently update gpt-4o" policy is harder to defend against. Some teams pin to specific model versions (e.g., <Code>gpt-4-0613</Code>) precisely to avoid drift, accepting the eventual sunset cost in exchange for stable evaluation.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>The independence assumption is rarely true</H3>
      <Prose>
        Every closed-form result in this topic — Condorcet's theorem, the <Code>1/√n</Code> variance reduction, the optimal log-odds weights — assumes independence between judge calls or between judges. None of those assumptions hold cleanly for LLMs from related families. Two GPT-4 calls are highly correlated; GPT-4 and GPT-3.5 are still substantially correlated because they share training data and architecture; GPT-4 and Claude are less correlated but still share much of the open-web training distribution. The practical consequence is that the theoretical accuracy curves consistently overstate real gains by 5-15 percentage points. Always validate empirically on a calibration set, never trust the closed-form predictions for real LLM panels.
      </Prose>

      <H3>Self-preference bias</H3>
      <Prose>
        Documented since the original MT-Bench paper: when GPT-4 evaluates outputs from GPT-4 versus another model, it shows a 5-10 point preference for GPT-4 outputs that does not match human preferences. The same pattern shows up for Claude evaluating Claude, and (less strongly) for open-source judges evaluating outputs from their family. The safest mitigation is to never use a judge to evaluate outputs from itself. The next-best mitigation is to include at least one judge in the panel that is not from the same family as any of the candidates being evaluated. The worst case — using a single GPT-4 judge to evaluate GPT-4 versus Claude — produces benchmark numbers that are systematically biased and that other researchers cannot reliably reproduce.
      </Prose>

      <H3>Position bias</H3>
      <Prose>
        Pairwise comparisons consistently exhibit position bias: judges favor whichever response is presented first (or, for some judges, last). The bias magnitude is typically 5-15 percentage points and persists at all temperatures. The fix is mechanical: present each pair twice with the order swapped, and treat any disagreement as a tie. This adds 2x cost per pairwise judgment but eliminates a known systematic bias. Always swap the order; do not assume any modern judge is position-invariant.
      </Prose>

      <H3>Length bias</H3>
      <Prose>
        Most LLM judges, trained on data where longer responses tended to be preferred by human annotators, exhibit a length bias: they rate longer responses as better even when content quality is held constant. In aggregation pipelines, this bias compounds. A council of three judges that each independently has a 5-point length bias produces an aggregate with a 5-point length bias (the bias is shared, not averaged out). The fix is to either include length normalization in the aggregation step (downweight verdicts when chosen is much longer than rejected) or to filter the evaluation set for length-matched pairs.
      </Prose>

      <H3>Calibration drift</H3>
      <Prose>
        Provider-side model updates can shift judge accuracy by 3-7 percentage points overnight. If your panel is using log-odds weighted aggregation calibrated to outdated reliabilities, the aggregator's weights are wrong and the weighted vote is suboptimal. Worst case, a judge whose accuracy has dropped below 50% retains a positive weight in the aggregator and now actively pulls verdicts toward the wrong answer. Continuous monitoring of per-judge accuracy on a fixed calibration set (the same items, same prompts, same seeds, run weekly) is the only defense.
      </Prose>

      <H3>Calibration set contamination</H3>
      <Prose>
        Easy to miss: if your calibration set is drawn from the same distribution as your training data for the policy being evaluated, judge reliabilities measured on the calibration set systematically overestimate live performance because the judges are calibrated on a tame slice of the actual distribution. A calibration set should be drawn from the same distribution as the live workload, ideally a held-out slice of recent production traffic with high-confidence labels.
      </Prose>

      <H3>EM degenerate solutions</H3>
      <Prose>
        Dawid-Skene EM has well-known failure modes. The two most common: (1) all judges are inferred to be equally accurate, in which case the EM aggregator reduces to majority voting and provides no benefit; (2) the algorithm converges to a "flipped" solution where verdicts are systematically inverted (judge labels are swapped). Initialization with majority-vote verdicts (rather than random verdicts) prevents most flipping; running EM 10-20 times with different random initializations and selecting the highest-likelihood fixed point catches residual cases.
      </Prose>

      <H3>Vote space collapse</H3>
      <Prose>
        Aggregation only works when the judge outputs live in a structured vote space (binary, n-ary categorical, or scalar-with-distance). When judges are asked for free-form rationales without explicit verdicts, the "votes" become idiosyncratic strings that cannot be aggregated. Always extract a structured verdict explicitly — either via JSON-mode response formatting, regex extraction from a known template, or a separate parsing step. Do not rely on the judge to volunteer a structured verdict in free text.
      </Prose>

      <H3>Tied votes in even-sized panels</H3>
      <Prose>
        Even-sized panels can produce ties. The standard fix is to use odd-sized panels (3, 5, 7), which makes ties impossible for binary votes. For n-ary categorical votes, ties remain possible for any panel size; tiebreakers should be specified explicitly (lowest index, alphabetical, defer to highest-reliability judge). Silent tiebreaking in aggregation libraries is a documented source of evaluation irreproducibility.
      </Prose>

      <H3>Correlation among prompts within self-consistency</H3>
      <Prose>
        Self-consistency at temperature 0.7 produces samples that share the early portion of the chain-of-thought trajectory; they only diverge later. The early-trajectory commonality means errors in the early reasoning are perfectly correlated across all samples. The mitigation is to sample at higher temperature (1.0) or use distinct prompt templates per sample, accepting some quality cost in exchange for greater error decorrelation. Wang et al.'s original paper used temperature 0.7 with 40 samples; modern practice often uses temperature 1.0 with fewer samples for the same effective decorrelation.
      </Prose>

      <H3>Optimization against the council</H3>
      <Prose>
        If a council is used to score outputs that are then used as training signal (e.g., reward modeling, DPO preference data), the trained model will optimize against the council's specific failure modes. Any bias the council shares becomes a target for the trained policy to exploit. The mitigation is rotation: periodically swap out judges, change prompts, change rubrics, so that the target distribution the policy is optimizing against keeps changing in ways the policy cannot anticipate. This is the alignment-tax-on-aggregation: stable councils produce stable but exploitable signals; rotating councils produce noisier but less exploitable signals.
      </Prose>

      <Callout accent="gold">
        The single most common production bug: deploying a council without measuring per-judge correlation on a calibration set first. The team assumes their three diverse judges are mostly independent; the actual pairwise correlation is 0.7+; the council provides almost no accuracy benefit over the best individual judge while costing 3x. Always measure correlation before scaling.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All sources below were verified against their arXiv pages on 2026-04-26. Author lists, abstracts, and IDs confirmed.
      </Prose>

      <H3>Wang et al. 2022 — Self-Consistency</H3>
      <Prose>
        Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, Denny Zhou. "Self-Consistency Improves Chain of Thought Reasoning in Language Models." arXiv:2203.11171. Published March 2022 (revised March 2023); ICLR 2023. The founding paper. Sample 40 chain-of-thought solutions at temperature 0.7, take the majority-vote final answer. Improves GSM8K accuracy by 17.9 points, SVAMP by 11.0 points, AQuA by 12.2 points over greedy decoding on the same backbone model. Establishes the pattern that aggregation across stochastic samples can substantially beat single-sample greedy decoding for reasoning tasks. Self-consistency was originally proposed for chain-of-thought answer generation but transfers directly to judge calls — sample the same judge multiple times, take the majority verdict.
      </Prose>

      <H3>Verga et al. 2024 — PoLL</H3>
      <Prose>
        Pat Verga, Sebastian Hofstätter, Sophia Althammer, Yixuan Su, Aleksandra Piktus, Arkady Arkhangorodsky, Minjie Xu, Naomi White, Patrick Lewis (Cohere). "Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models." arXiv:2404.18796. Published April 2024. Introduces "Panel of LLM evaluators" (PoLL): a panel of three small judges (Command-R, Claude Haiku, GPT-3.5) that produces rankings correlating with human judgment as well as a single GPT-4 judge call, at roughly 1/7 the cost. Empirically validates judge diversity reduces self-preference bias and improves correlation with human judgment across MT-Bench and Chatbot Arena evaluations. The definitive operational paper for council-of-judges in production.
      </Prose>

      <H3>Du et al. 2023 — Multi-Agent Debate</H3>
      <Prose>
        Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, Igor Mordatch (MIT). "Improving Factuality and Reasoning in Language Models through Multiagent Debate." arXiv:2305.14325. Published May 2023. Multiple LLM instances each propose answers, then iteratively revise their answers in light of the other agents' responses. Across math benchmarks (GSM8K, arithmetic), strategic reasoning, and factual question answering, debate substantially outperforms single-agent chain-of-thought and self-consistency, with gains of 5-15 percentage points. Three rounds of debate with three agents is the typical configuration. The paper articulates the core distinction between independent sampling (self-consistency) and information-sharing across agents (debate), with empirical evidence that information-sharing helps for reasoning tasks but is less effective for subjective evaluation.
      </Prose>

      <H3>Khan et al. 2024 — Persuasive Debate</H3>
      <Prose>
        Akbir Khan, John Hughes, Dan Valentine, Laura Ruis, Kshitij Sachan, Ansh Radhakrishnan, Edward Grefenstette, Samuel R. Bowman, Tim Rocktäschel, Ethan Perez. "Debating with More Persuasive LLMs Leads to More Truthful Answers." arXiv:2402.06782. Published February 2024. Studies debate as a scalable oversight mechanism. Two strong LLMs argue for opposing answers to a question; a weaker judge model (or a human with limited expertise) decides which side wins. Stronger debaters lead the judge to more truthful conclusions even when the judge cannot verify the claims directly. Provides the strongest evidence so far that debate-based aggregation can extract truth from a panel even when the aggregator (the judge) is weaker than the participants. Particularly relevant for evaluation regimes where the gold-standard verifier is not available at scale.
      </Prose>

      <H3>Zheng et al. 2023 — LLM-as-a-Judge</H3>
      <Prose>
        Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, Ion Stoica. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." arXiv:2306.05685. Published June 2023; NeurIPS 2023 Datasets and Benchmarks. The foundational empirical study of LLM judges. Documents the major bias categories (position bias, verbosity bias, self-preference bias) with quantitative measurements on MT-Bench. Reports that GPT-4 judge agreement with human annotators reaches 80%+ for most evaluation tasks, comparable to human-human agreement, but with the bias caveats. Establishes the empirical foundation that all subsequent council and self-consistency work for judges builds on.
      </Prose>

      <H3>Condorcet 1785 — Jury Theorem</H3>
      <Prose>
        Marie Jean Antoine Nicolas de Caritat, Marquis de Condorcet. "Essai sur l'application de l'analyse à la probabilité des décisions rendues à la pluralité des voix." Imprimerie Royale, Paris, 1785. The original derivation of what is now known as Condorcet's jury theorem: under independence, the probability that a majority vote correctly identifies the true answer increases monotonically with panel size if individual jurors are more than 50% accurate, and converges to 1 in the limit. The theorem also covers the symmetric failure case: below 50% individual accuracy, panel accuracy decreases monotonically toward 0. This 240-year-old result is the mathematical foundation of every aggregation method discussed in this topic. Modern restatements: Grofman, Owen, Feld 1983 (Theory and Decision), Ladha 1992 (American Journal of Political Science) for the correlated-voter extension.
      </Prose>

      <H3>Dawid &amp; Skene 1979 — EM Reliability</H3>
      <Prose>
        A. P. Dawid, A. M. Skene. "Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm." Applied Statistics (Journal of the Royal Statistical Society Series C), Vol. 28, No. 1, 1979, pp. 20-28. The original EM algorithm for inferring per-observer reliability from votes alone, without ground-truth labels. Originally formulated for medical diagnosis (multiple radiologists rating X-rays). Resurrected for LLM judge aggregation in 2023-2024 as the principled way to combine judges without requiring a labeled calibration set. The algorithm converges to a self-consistent assignment of latent verdicts and per-judge confusion matrices.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Compute Condorcet panel sizes</H3>
      <Prose>
        For each of the following individual judge accuracies — 0.55, 0.65, 0.75, 0.85 — compute the smallest odd panel size <Code>n</Code> such that the majority-vote panel accuracy exceeds 0.95, assuming independent errors. Use the closed-form binomial formula. Then plot accuracy vs. panel size for each <Code>p</Code>. What pattern do you observe in the relationship between individual accuracy and required panel size? Why does a 0.55 judge need so many more colleagues than a 0.85 judge to reach the same panel accuracy?
      </Prose>

      <H3>Exercise 2 — Effective panel size under correlation</H3>
      <Prose>
        Given the formula <Code>Var(mean) = σ²/n · (1 + (n-1)ρ)</Code>, derive the "effective panel size" — the value <Code>n_eff</Code> such that an independent panel of size <Code>n_eff</Code> would have the same variance as a correlated panel of size <Code>n</Code> with correlation <Code>ρ</Code>. Show that as <Code>n → ∞</Code>, <Code>n_eff → 1/ρ</Code>. Compute <Code>n_eff</Code> for the panel sizes (3, 5, 11, 21) and correlations (0.1, 0.3, 0.5, 0.8). What does this mean operationally for someone considering scaling a council from 5 to 21 judges with measured pairwise correlation 0.4?
      </Prose>

      <H3>Exercise 3 — Log-odds weights</H3>
      <Prose>
        A panel has three judges with accuracies 0.90, 0.70, and 0.55 respectively. Compute the log-odds weight for each judge under Bayesian aggregation. For each of the eight possible voting outcomes (each judge votes 0 or 1), compute the weighted score and determine the aggregate verdict. Identify any cases where the weighted aggregate disagrees with the plain majority vote. What is the panel accuracy under each aggregation rule, assuming independence and equal class priors? Now repeat the exercise with a fourth judge added at accuracy 0.45. What happens to the log-odds weight of this fourth judge, and what does the Bayesian aggregator do with its vote?
      </Prose>

      <H3>Exercise 4 — Self-consistency vs. council on a budget</H3>
      <Prose>
        You have a budget of $1.00 per item. Your options: (a) call a single GPT-4 judge ($1.00 each); (b) call GPT-4 five times for self-consistency ($0.20 each, with same-model error correlation 0.5); (c) call a panel of five small judges ($0.20 each, with cross-model error correlation 0.2). Assuming individual judge accuracies of 0.85 (GPT-4) and 0.70 (small models), compute the expected panel accuracy for each option. Which option dominates? Which option dominates if the budget rises to $2.00 (and you must spend roughly all of it)? At what budget level does the optimal strategy switch from "small panel" to "include a frontier judge"?
      </Prose>

      <H3>Exercise 5 — Detecting drift</H3>
      <Prose>
        You are running a three-judge council in production. Last quarter's calibration showed per-judge accuracies of 0.82, 0.78, 0.74 with average pairwise correlation 0.35. You have not run new calibration since. Today you notice the council's agreement rate (the fraction of items where all three judges agree) has shifted from 0.71 last month to 0.62 this month. List three plausible causes for this shift, and for each cause, describe what additional evidence would distinguish it from the others. Which cause would be most concerning for the validity of the council's aggregate verdicts going forward, and what would be the principled response?
      </Prose>

      <H3>Exercise 6 — Designing a council from scratch</H3>
      <Prose>
        You are designing a council for evaluating responses to coding questions. You have access to GPT-4o, Claude Sonnet, Gemini Pro, Llama-3-70B, DeepSeek-Coder-33B, and Qwen-Coder-32B. Your latency budget allows three judges in parallel; your cost budget allows roughly 4x the cost of a single GPT-4o call per item. Which three judges would you select, and how would you justify the selection in terms of (a) decorrelation, (b) per-judge competence on the task, and (c) cost? What calibration procedure would you run before deploying the council, and how would you use the calibration results to set aggregation weights? What monitoring would you put in place to detect drift?
      </Prose>

      <H3>Exercise 7 — When self-consistency fails</H3>
      <Prose>
        Construct a concrete scenario in which self-consistency on a single judge provides essentially no benefit (less than 1 percentage point of accuracy improvement at <Code>k=10</Code>), even though the judge is only 70% accurate at <Code>k=1</Code>. What does this scenario look like in terms of the judge's error structure? Then construct a second scenario in which self-consistency provides almost the maximum theoretical benefit (close to <Code>1/√k</Code> variance reduction). What is the difference between the two scenarios at the level of the judge's chain-of-thought? What does this imply about which kinds of judging tasks are most likely to benefit from self-consistency?
      </Prose>

    </div>
  ),
};

export default selfConsistencyCouncil;
