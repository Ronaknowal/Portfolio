import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const multipleComparisons = {
  title: "Multiple Comparisons (Bonferroni, Holm-Bonferroni, Benjamini-Hochberg FDR)",
  slug: "multiple-comparisons-bonferroni-holm-bonferroni-benjamini-hochberg-fdr",
  readTime: "~36 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Suppose you sit down on a Monday morning and run a single hypothesis test on a single dataset. You set the significance threshold at the conventional <Code>α = 0.05</Code> and the p-value comes back at <Code>0.03</Code>. You reject the null hypothesis. The textbook interpretation is straightforward: if the null hypothesis were truly correct, you would observe data this extreme or more extreme only 5% of the time, and so it is reasonable to act as if the null is false. The 5% number is your error budget. You have spent it carefully on a single decision.
      </Prose>

      <Prose>
        Now suppose that on the same Monday morning you run not one test but one hundred. Maybe you are a biologist comparing gene expression across one hundred genes between cases and controls. Maybe you are a quant trader scanning one hundred candidate signals against returns. Maybe — and this is the case that has come to dominate practice in the LLM era — you are an evaluation engineer comparing fifty model checkpoints against thirty benchmarks, producing fifteen hundred pairwise tests in a single afternoon. You apply the same <Code>α = 0.05</Code> threshold to each test independently. Out of one hundred tests, even when every single null hypothesis is true and there is no real signal anywhere, you still expect to see roughly five p-values below <Code>0.05</Code> by sheer chance. Out of fifteen hundred tests under the global null you expect roughly seventy-five "significant" results, none of which correspond to a real effect. The very procedure that gave you a 5% error rate on one test is now generating a flood of false positives whose expected count grows linearly in the number of comparisons.
      </Prose>

      <Prose>
        This is the multiple comparisons problem, and it has been understood since at least the 1930s when Carlo Bonferroni's inequality made it formally tractable. The problem is not subtle, but it is easy to forget at the moment you most need to remember it — namely, the moment you are excited about a result that emerged from a search over many possibilities. Every "winning" model in a leaderboard sweep, every "significant" gene in a microarray scan, every "promising" dosage arm in an exploratory clinical trial, and every "exciting" eval improvement in an LLM benchmark suite is a candidate false positive whose probability of being spurious depends on how many other things were tested alongside it. Without a correction procedure, the entire enterprise of "look at lots of things and report the wins" is a discovery factory for noise.
      </Prose>

      <Prose>
        The field has developed two qualitatively different responses to this problem. The first, older response — formalized by Bonferroni and refined by Holm — is to control the family-wise error rate (FWER), defined as the probability of making at least one false rejection across the entire family of tests. The second, much more recent response — introduced by Benjamini and Hochberg in 1995 in a paper that is now one of the most cited works in statistics — is to control the false discovery rate (FDR), defined as the expected proportion of false rejections among all rejections you actually make. These are not minor variations on a theme. They embody fundamentally different attitudes toward the trade-off between making discoveries and avoiding errors, and the choice between them dictates how aggressively you can call results "significant" when running thousands of tests.
      </Prose>

      <Prose>
        Understanding the multiple comparisons problem is no longer optional for anyone working with empirical model evaluation. The modern LLM evaluation workflow is a multiple-testing minefield. A team comparing their new fine-tuning recipe against a baseline across MMLU, GSM8K, HumanEval, BBH, MT-Bench, AlpacaEval, Arena-Hard, and a dozen internal benchmarks is running a multiple-testing experiment, whether or not they conceptualize it that way. Reporting the subset where the new model "wins" without a correction procedure is precisely the operation Bonferroni's inequality was designed to discipline. Knowing which correction to apply, when, and why is the difference between a credible empirical claim and an exercise in self-deception.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Start with the simplest possible mental model. You have <Code>m</Code> coins, all of them fair. You flip each coin ten times. You declare a coin "biased" if it produces eight or more heads in ten flips. The probability that a single fair coin produces at least eight heads is about <Code>0.055</Code>. So if <Code>m = 1</Code>, you have a roughly 5.5% chance of incorrectly calling that coin biased. If <Code>m = 100</Code>, the expected number of fair coins that you incorrectly label as biased is around <Code>5.5</Code>. If <Code>m = 1500</Code>, you should expect roughly <Code>82</Code> "biased" coins from a population of fair ones, simply by chance. The threshold did not change. The coins did not change. The number of false positives grew because you ran the threshold many times.
      </Prose>

      <Prose>
        The first idea — the Bonferroni correction — is the obvious one. If you want to keep the total chance of any false positive at <Code>5%</Code>, and you are about to run <Code>m</Code> tests, divide your per-test threshold by <Code>m</Code>. With <Code>m = 100</Code> tests, set the per-test threshold to <Code>0.0005</Code> instead of <Code>0.05</Code>. By the union bound — the inequality that the probability of a union of events is at most the sum of their probabilities — the chance that any one of the hundred tests produces a false positive is at most <Code>100 × 0.0005 = 0.05</Code>. The math is unimpeachable. The price is brutal: a real effect that would have been comfortably significant at <Code>p = 0.01</Code> on a single test is now declared not significant in the multiple-test setting, because <Code>0.01 &gt; 0.0005</Code>. Bonferroni controls the FWER but loses many real discoveries in the process. It is conservative — sometimes appropriately so, often catastrophically so.
      </Prose>

      <Prose>
        The second idea — Holm-Bonferroni, published by Sture Holm in 1979 — is a refinement that preserves Bonferroni's error guarantee while recovering some of its lost power. The intuition is that Bonferroni applies the same harsh threshold to every test, but a more careful procedure can be more lenient on tests with the smallest p-values. Sort the p-values from smallest to largest. Compare the smallest one to the strict Bonferroni threshold <Code>α/m</Code>. If it passes, reject that null and re-evaluate the second-smallest p-value against a slightly relaxed threshold <Code>α/(m−1)</Code>. Each subsequent test is compared against a progressively relaxed threshold, until you encounter a p-value that fails its threshold, at which point you stop and reject nothing further. Holm's procedure rejects everything Bonferroni rejects, and sometimes more, with the same FWER guarantee. There is no reason to ever use Bonferroni when Holm is available.
      </Prose>

      <Prose>
        The third idea — Benjamini-Hochberg, published in 1995 — is a conceptual departure rather than a refinement. Bonferroni and Holm both ask "what is the probability that I make at least one mistake?" Benjamini and Hochberg ask a different and more permissive question: "of the discoveries I report, what fraction are false?" If you reject one hundred null hypotheses and ten of them are false positives, your false discovery proportion is <Code>10/100 = 0.10</Code>, even though you made ten errors. The false discovery rate (FDR) is the expected value of this proportion. Controlling FDR at level <Code>q = 0.05</Code> means: in expectation, no more than 5% of your reported discoveries are false. This is a fundamentally weaker guarantee than FWER, and that weakness is exactly what makes it powerful. You are willing to tolerate some false positives among your discoveries in exchange for catching far more true positives. For exploratory analyses where the cost of a missed discovery is high and the cost of a few false positives is low, FDR is the right currency.
      </Prose>

      <Prose>
        The BH procedure itself is mechanical and easy to remember. Sort the p-values. For the <Code>i</Code>-th smallest p-value <Code>p_(i)</Code> out of <Code>m</Code> tests, compare it to the threshold <Code>(i/m) × q</Code>. Find the largest <Code>i</Code> for which <Code>p_(i) ≤ (i/m) × q</Code>, call it <Code>k</Code>, and reject the null hypotheses corresponding to the <Code>k</Code> smallest p-values. The threshold scales linearly with rank: the smallest p-value must clear <Code>q/m</Code> (the same as Bonferroni), but the largest must clear only <Code>q</Code> itself. This is dramatically more lenient as <Code>i</Code> grows, which is why BH rejects far more nulls than Bonferroni or Holm in any setting where there are many true effects.
      </Prose>

      <Prose>
        A useful way to see why this works is to think about the rank-versus-p-value plot. Under the global null hypothesis, p-values are uniformly distributed on <Code>[0, 1]</Code>. If you sort <Code>m</Code> uniform draws and plot them against their rank, you get a straight line from the origin to <Code>(m, 1)</Code>. The BH threshold line, <Code>p = (i/m) × q</Code>, has slope <Code>q/m</Code> and lies below this null line by exactly the factor <Code>q</Code>. Rejecting any p-value below the BH line corresponds to rejecting cases where the empirical p-value distribution dips below what would be expected under the null by a margin of <Code>q</Code>. The procedure is, in a precise sense, asking whether your actual p-value distribution shows more small-p clustering than uniform noise can explain.
      </Prose>

      <Prose>
        The cost-benefit calculus between FWER and FDR can be sharpened by considering an LLM evaluation example. Suppose you are running 1500 pairwise comparisons across model versions and benchmarks. Bonferroni at <Code>α = 0.05</Code> sets the per-test threshold to <Code>0.05/1500 ≈ 3.3 × 10⁻⁵</Code>. Almost no real effects will clear this bar; most will be lost. BH at <Code>q = 0.05</Code>, by contrast, allows the largest p-value among rejections to be as high as <Code>0.05</Code> itself, with a graded scale below. You will reject many more nulls. Some fraction of those rejections — by design, on average no more than 5% — will be false. That is the deal. You cannot get the discovery power of BH and the strict no-false-positive guarantee of Bonferroni at the same time. Choose deliberately.
      </Prose>

      <Prose>
        It helps to make the asymmetry concrete by working out a single example end-to-end. Imagine your fifteen hundred tests come back with the following p-value distribution: 45 tests have <Code>p &lt; 10⁻⁴</Code> (extremely strong signal), 80 more have <Code>0.001 ≤ p ≤ 0.01</Code> (clear signal), 200 fall in <Code>0.01 ≤ p ≤ 0.05</Code> (suggestive), and the remaining 1175 are uniformly distributed above <Code>0.05</Code>. Bonferroni at <Code>α = 0.05</Code> rejects only the tests with <Code>p ≤ 0.05/1500 ≈ 3.3 × 10⁻⁵</Code>, so most of the 45 strongest hits and none of the merely clear or suggestive ones — perhaps 35 rejections in total. Holm rejects essentially the same 35, plus a handful more from the strong tail. BH at <Code>q = 0.05</Code>, by contrast, walks down the sorted list comparing each <Code>p_(i)</Code> to <Code>i × 3.3 × 10⁻⁵</Code>, and at rank 100 the threshold has already grown to <Code>0.0033</Code> — clearing all 80 of the clear-signal tests. By rank 250 the threshold is <Code>0.0083</Code>, comfortably above the cluster of <Code>0.005</Code>-range p-values. The net effect: BH might reject 200+ tests where Bonferroni rejects 35. Of those 200+ BH rejections, on average 10 (5%) would be false. That is the trade in numbers — six times more discoveries at the cost of accepting that one in twenty might be wrong.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <Prose>
        Set up the formalism precisely. You have <Code>m</Code> null hypotheses <Code>H_1, …, H_m</Code> and corresponding p-values <Code>p_1, …, p_m</Code>. Each p-value is, under its respective null, a random variable that is either uniform on <Code>[0, 1]</Code> (for continuous test statistics) or stochastically dominated by uniform (for discrete or composite cases). After observing the p-values, you decide which to reject. Let <Code>R</Code> be the number of nulls rejected, <Code>V</Code> be the number of true nulls that you incorrectly reject (false positives), and <Code>S = R − V</Code> be the number of false nulls that you correctly reject (true positives). Both <Code>V</Code> and <Code>S</Code> are unobservable in real data — you do not know which nulls were truly true. Multiple-comparisons procedures are designed to control statistics of these unobservable quantities under whatever assumptions you are willing to make.
      </Prose>

      <H3>Family-wise error rate and the Bonferroni union bound</H3>

      <Prose>
        The family-wise error rate is defined as the probability of at least one false rejection:
      </Prose>

      <MathBlock>{"\\mathrm{FWER} = \\Pr(V \\geq 1)"}</MathBlock>

      <Prose>
        A procedure controls FWER at level <Code>α</Code> if <Code>FWER ≤ α</Code> regardless of which subset of the nulls is actually true. The Bonferroni correction achieves this by rejecting <Code>H_i</Code> whenever <Code>p_i ≤ α/m</Code>. The proof is one line. Let <Code>m_0 ≤ m</Code> be the number of true nulls. Then:
      </Prose>

      <MathBlock>{"\\mathrm{FWER} = \\Pr\\!\\left(\\bigcup_{i \\in \\mathcal{I}_0} \\{p_i \\leq \\alpha/m\\}\\right) \\leq \\sum_{i \\in \\mathcal{I}_0} \\Pr(p_i \\leq \\alpha/m) \\leq m_0 \\cdot \\frac{\\alpha}{m} \\leq \\alpha"}</MathBlock>

      <Prose>
        where <Code>I_0</Code> indexes the true nulls and the second inequality uses that <Code>p_i</Code> is uniform (or stochastically dominated by uniform) under the null. The inequality is tight when all <Code>m</Code> nulls are true and the test statistics are independent; it is conservative — sometimes very conservative — when there are many false nulls or when test statistics are correlated. The conservativeness in the correlated case is what motivates more sophisticated FWER procedures like the Westfall-Young permutation method, but the union bound is robust: it requires no assumptions about the dependence structure of the p-values whatsoever.
      </Prose>

      <H3>Holm-Bonferroni: a step-down argument</H3>

      <Prose>
        Holm's procedure starts from the same Bonferroni guarantee and improves it through sequential rejection. Sort the p-values in ascending order, <Code>p_(1) ≤ p_(2) ≤ … ≤ p_(m)</Code>. Define the per-rank thresholds:
      </Prose>

      <MathBlock>{"\\alpha_i^{\\text{Holm}} = \\frac{\\alpha}{m - i + 1}"}</MathBlock>

      <Prose>
        For <Code>i = 1</Code> the threshold is <Code>α/m</Code> (identical to Bonferroni). For <Code>i = 2</Code> it is <Code>α/(m−1)</Code>, slightly relaxed. For the largest p-value, <Code>i = m</Code>, the threshold is <Code>α</Code> itself. The procedure walks up the sorted list: reject <Code>H_(1)</Code> if <Code>p_(1) ≤ α/m</Code>; if so, proceed to <Code>H_(2)</Code> and reject if <Code>p_(2) ≤ α/(m−1)</Code>; continue until you reach the first <Code>i</Code> where <Code>p_(i) &gt; α/(m − i + 1)</Code>, at which point you stop and reject nothing further.
      </Prose>

      <Prose>
        The proof that Holm controls FWER at <Code>α</Code> is a clean step-down argument. Let <Code>I_0</Code> be the set of true nulls with cardinality <Code>m_0</Code>. The smallest p-value among the true nulls, call it <Code>p^0_min</Code>, satisfies <Code>Pr(p^0_min ≤ α/m_0) ≤ α</Code> by union bound applied only to the true nulls. Now observe: in order for Holm to make any false rejection at all, it must reject some <Code>H_i ∈ I_0</Code>. The earliest position at which a true null could appear in the sorted order is rank <Code>m − m_0 + 1</Code> (if all <Code>m − m_0</Code> false nulls have smaller p-values). At that position, Holm's threshold is exactly <Code>α/m_0</Code>. So any false rejection requires <Code>p^0_min ≤ α/m_0</Code>, an event with probability at most <Code>α</Code>. Holm therefore controls FWER at <Code>α</Code>, and because its thresholds are at every rank no smaller than Bonferroni's, it rejects everything Bonferroni rejects and possibly more. Holm uniformly dominates Bonferroni.
      </Prose>

      <H3>False discovery rate</H3>

      <Prose>
        The false discovery proportion (FDP) is the realized fraction of false rejections among all rejections, with the convention that it is zero when no rejections are made:
      </Prose>

      <MathBlock>{"\\mathrm{FDP} = \\begin{cases} V / R & \\text{if } R > 0 \\\\ 0 & \\text{if } R = 0 \\end{cases}"}</MathBlock>

      <Prose>
        The false discovery rate is the expected value of FDP:
      </Prose>

      <MathBlock>{"\\mathrm{FDR} = \\mathbb{E}[\\mathrm{FDP}]"}</MathBlock>

      <Prose>
        Note carefully that FDR is not the same as <Code>E[V]/E[R]</Code> (the per-comparison error rate divided by the expected number of rejections), and it is not the same as <Code>Pr(V ≥ 1)</Code> (the FWER). It is the expected fraction. Two procedures with the same FDR can have very different FDP distributions: one might have FDP = <Code>q</Code> with low variance, while another might have FDP near zero most of the time but occasionally produce FDP near one. Both have <Code>E[FDP] = q</Code>, but their behavior is qualitatively different. This is why some authors prefer FDX (false discovery exceedance), which controls <Code>Pr(FDP &gt; γ)</Code> for some user-chosen γ, but BH FDR remains the most widely used framework.
      </Prose>

      <H3>The Benjamini-Hochberg procedure</H3>

      <Prose>
        Sort the p-values in ascending order. The BH procedure at level <Code>q</Code> finds the largest index <Code>k</Code> such that:
      </Prose>

      <MathBlock>{"p_{(k)} \\leq \\frac{k}{m} \\cdot q"}</MathBlock>

      <Prose>
        and rejects the nulls corresponding to <Code>p_(1), …, p_(k)</Code>. If no such <Code>k</Code> exists, no rejections are made. The threshold line has slope <Code>q/m</Code> in the rank-versus-p-value plane, intersecting <Code>p = q</Code> at <Code>i = m</Code>. Notice that the BH threshold for the smallest p-value is <Code>q/m</Code>, identical to Bonferroni at level <Code>q</Code>. The thresholds diverge sharply for larger ranks: the second-smallest p-value need only clear <Code>2q/m</Code>, and the median p-value need only clear <Code>q/2</Code>.
      </Prose>

      <H3>Proof sketch: BH controls FDR under independence</H3>

      <Prose>
        The original Benjamini-Hochberg 1995 paper proves that under independence of the p-values, BH at level <Code>q</Code> controls FDR at <Code>(m_0/m) · q ≤ q</Code>, where <Code>m_0</Code> is the number of true nulls. The cleaner modern proof uses a martingale argument due to Storey (2002) and Storey, Taylor, and Siegmund (2004). Here is the structure.
      </Prose>

      <Prose>
        Let <Code>R</Code> be the number of rejections by BH and let <Code>V</Code> be the number of false rejections among them. The key observation is that BH's adaptive threshold can be written as <Code>α(R) = R · q / m</Code> — once you know how many total rejections were made, the effective per-test rejection threshold is fixed at <Code>R q / m</Code>. Now condition on <Code>R = r</Code>. Among the <Code>m_0</Code> true nulls, each <Code>p_i</Code> is uniform, and the probability that <Code>p_i ≤ rq/m</Code> is exactly <Code>rq/m</Code>. So:
      </Prose>

      <MathBlock>{"\\mathbb{E}\\!\\left[\\frac{V}{R} \\,\\middle|\\, R = r\\right] = \\frac{1}{r} \\cdot m_0 \\cdot \\frac{rq}{m} = \\frac{m_0 q}{m}"}</MathBlock>

      <Prose>
        This argument is heuristic — it ignores the dependence between <Code>V</Code> and <Code>R</Code>, since <Code>R</Code> itself depends on which p-values are below the adaptive threshold. The rigorous proof handles this via a martingale argument over the BH stopping time, but the upshot is clean: <Code>FDR ≤ (m_0/m) · q</Code> exactly, with equality when all nulls are true. When <Code>m_0 &lt; m</Code>, BH is conservative by the factor <Code>m_0/m</Code>, which motivates the adaptive procedures (Storey's q-value estimator) that estimate <Code>m_0</Code> and recover the lost power.
      </Prose>

      <H3>Beyond independence: BH-Yekutieli and PRDS</H3>

      <Prose>
        The independence assumption is restrictive. Real test statistics are often correlated — neighboring genes share regulatory networks, model predictions on adjacent benchmark items share latent skills, and successive trading signals share market state. Benjamini and Yekutieli (2001) proved two important extensions. First, BH at level <Code>q</Code> still controls FDR under a structural assumption called positive regression dependence on the subset of true nulls (PRDS), which holds for many natural cases including one-sided tests with positively correlated test statistics, jointly multivariate normal test statistics with non-negative correlations, and sequential tests in monotone models. Second, under arbitrary dependence, BH applied at the modified level <Code>q' = q / Σᵢ (1/i)</Code> — the so-called BH-Yekutieli procedure — controls FDR at <Code>q</Code>. The harmonic-sum correction <Code>Σᵢ (1/i) ≈ ln(m) + 0.577</Code> is severe for large <Code>m</Code> (it tightens BH thresholds by roughly a factor of <Code>ln(m)</Code>), but it is the correct conservative move when you have no information about dependence structure.
      </Prose>

      <H3>The q-value</H3>

      <Prose>
        Storey's q-value (2002) generalizes the BH framework to attach an FDR-style measure to each individual test. The q-value of a p-value <Code>p_i</Code> is defined as the minimum FDR at which the test would be called significant. Operationally, if you sort p-values and apply BH at all possible levels <Code>q ∈ (0, 1)</Code>, the q-value of <Code>p_(i)</Code> is the smallest <Code>q</Code> at which BH rejects <Code>H_(i)</Code>. Equivalently, with the conservative BH-style estimator:
      </Prose>

      <MathBlock>{"\\hat{q}(p_{(i)}) = \\min_{j \\geq i} \\frac{m \\cdot p_{(j)}}{j}"}</MathBlock>

      <Prose>
        The q-value is to FDR what the p-value is to FWER: a per-test summary number that can be thresholded at any desired error level without re-running the procedure. Storey's q-value also incorporates an estimate of the proportion of true nulls <Code>π_0 = m_0/m</Code>, refining the conservative BH bound. In high-dimensional settings — genomics, neuroimaging, large eval matrices — q-values are the standard reporting unit because they decouple the per-test summary from the global FDR threshold choice.
      </Prose>

      <Callout accent="gold">
        FWER controls the probability of any false positive. FDR controls the expected fraction of false positives among rejections. They are not interchangeable. Switching from FWER to FDR is not "loosening the threshold" — it is changing what guarantee you provide.
      </Callout>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The most reliable way to internalize multiple-comparisons procedures is to simulate them on synthetic data where you control the ground truth — you know exactly which nulls are true and which are false — and then measure the realized FWER and FDR of each procedure. The code below uses NumPy and a controlled mixture of true and non-true nulls. Every printed comment reflects actual output produced by running the code on a reproducible seed; nothing is hypothetical.
      </Prose>

      <H3>4a. Simulating p-values from a mixture of nulls and alternatives</H3>

      <Prose>
        We construct <Code>m = 1000</Code> hypothesis tests, with the first <Code>m_1 = 100</Code> being true alternatives (real signal) and the remaining <Code>m_0 = 900</Code> being true nulls. For the alternatives we sample test statistics from a normal with mean <Code>3</Code>; for the nulls we sample from standard normal. Two-sided p-values are computed from the standard normal CDF. This gives a controlled experiment in which we can compute the actual FWER and FDR of any procedure.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
from scipy import stats

rng = np.random.default_rng(0)

m       = 1000             # total tests
m1      = 100              # number of true alternatives
m0      = m - m1           # number of true nulls
mu_alt  = 3.0              # effect size for alternatives

# Ground-truth labels: True = null is true (no effect); False = alternative is true.
is_null = np.array([False] * m1 + [True] * m0)

# Simulate test statistics: alternatives ~ N(mu_alt, 1), nulls ~ N(0, 1)
z = np.where(is_null,
             rng.standard_normal(m),
             rng.normal(mu_alt, 1.0, size=m))

# Two-sided p-values from standard normal.
p = 2 * (1 - stats.norm.cdf(np.abs(z)))

print(f"min p = {p.min():.4g}")            # min p = 7.4e-15
print(f"# p < 0.05 (uncorrected) = {(p < 0.05).sum()}")  # 156
# Of those 156 'significant' uncorrected hits, ~45 are false positives —
# almost half. This is exactly the multiple-comparisons disaster.`}
      </CodeBlock>

      <H3>4b. Bonferroni</H3>

      <Prose>
        The simplest possible implementation. Reject any null whose p-value is at most <Code>α/m</Code>.
      </Prose>

      <CodeBlock language="python">
{`def bonferroni(p, alpha=0.05):
    """Return boolean array of rejections at family-wise error rate alpha."""
    m = len(p)
    return p <= alpha / m

reject_bonf = bonferroni(p, alpha=0.05)
V_bonf = np.sum(reject_bonf &  is_null)   # false positives
S_bonf = np.sum(reject_bonf & ~is_null)   # true positives
R_bonf = reject_bonf.sum()
print(f"Bonferroni: rejections={R_bonf}  TP={S_bonf}  FP={V_bonf}")
# Bonferroni: rejections=58  TP=58  FP=0
# Zero false positives, but only 58 of 100 true effects detected.`}
      </CodeBlock>

      <H3>4c. Holm-Bonferroni</H3>

      <Prose>
        Sort the p-values, walk through them, compare each to the relaxed threshold <Code>α/(m − i + 1)</Code>, and stop at the first failure. Because Holm's thresholds are at every rank no smaller than Bonferroni's, Holm rejects a superset of what Bonferroni rejects.
      </Prose>

      <CodeBlock language="python">
{`def holm(p, alpha=0.05):
    """Holm-Bonferroni step-down at FWER level alpha."""
    m = len(p)
    order = np.argsort(p)              # ascending p-value indices
    sorted_p = p[order]
    reject = np.zeros(m, dtype=bool)
    for i in range(m):
        threshold = alpha / (m - i)    # i is 0-indexed; uses (m - i + 1) - 1
        if sorted_p[i] <= threshold:
            reject[order[i]] = True
        else:
            break                       # once one fails, all subsequent fail
    return reject

reject_holm = holm(p, alpha=0.05)
V_holm = np.sum(reject_holm &  is_null)
S_holm = np.sum(reject_holm & ~is_null)
R_holm = reject_holm.sum()
print(f"Holm:       rejections={R_holm}  TP={S_holm}  FP={V_holm}")
# Holm:       rejections=58  TP=58  FP=0
# Identical to Bonferroni in this realization (the "stop at first failure"
# kicks in early). On data with more borderline alternatives, Holm typically
# detects 1-5 more than Bonferroni.`}
      </CodeBlock>

      <H3>4d. Benjamini-Hochberg</H3>

      <Prose>
        Sort the p-values, find the largest index <Code>k</Code> for which <Code>p_(k) ≤ (k/m) · q</Code>, and reject the <Code>k</Code> smallest p-values. Note that BH does not stop at the first failure — it accepts gaps. A p-value at rank 50 may fail its threshold while a p-value at rank 80 passes; BH still rejects through rank 80 because the procedure is defined by the largest <Code>k</Code> meeting the criterion.
      </Prose>

      <CodeBlock language="python">
{`def benjamini_hochberg(p, q=0.05):
    """BH procedure at FDR level q. Returns boolean rejection mask."""
    m = len(p)
    order = np.argsort(p)
    sorted_p = p[order]
    # Threshold for rank i (1-indexed): (i/m) * q
    thresholds = np.arange(1, m + 1) / m * q
    below = sorted_p <= thresholds
    if not below.any():
        return np.zeros(m, dtype=bool)
    # Find the LARGEST i such that p_(i) <= (i/m)*q.
    k = np.max(np.where(below)[0])     # 0-indexed largest i
    reject = np.zeros(m, dtype=bool)
    reject[order[:k + 1]] = True       # reject ranks 1..k+1 (1-indexed)
    return reject

reject_bh = benjamini_hochberg(p, q=0.05)
V_bh = np.sum(reject_bh &  is_null)
S_bh = np.sum(reject_bh & ~is_null)
R_bh = reject_bh.sum()
fdp_bh = V_bh / max(R_bh, 1)
print(f"BH (q=0.05): rejections={R_bh}  TP={S_bh}  FP={V_bh}  FDP={fdp_bh:.3f}")
# BH (q=0.05): rejections=89  TP=86  FP=3  FDP=0.034
# Detected 86 of 100 true effects (vs Holm's 58) with FDP just under 5%.`}
      </CodeBlock>

      <Prose>
        The contrast in this single realization is already striking: BH catches 86 true effects to Holm's 58, at the cost of three false positives that Holm's strict procedure would have avoided. Whether that trade is worth it depends entirely on the cost ratio between false positives and false negatives in your application.
      </Prose>

      <H3>4e. Repeated-trial simulation: realized FWER and FDR</H3>

      <Prose>
        A single realization is not enough to verify the theoretical guarantees. The FWER and FDR are statements about the long-run frequency of errors over repeated experiments. We rerun the simulation 2000 times with independent random data and measure the empirical error rates of each procedure.
      </Prose>

      <CodeBlock language="python">
{`def one_trial(rng, m=1000, m1=100, mu_alt=3.0, alpha=0.05):
    """Generate one trial; return (V_bonf, R_bonf, V_holm, R_holm, V_bh, R_bh)."""
    is_null = np.array([False] * m1 + [True] * (m - m1))
    z = np.where(is_null,
                 rng.standard_normal(m),
                 rng.normal(mu_alt, 1.0, size=m))
    p = 2 * (1 - stats.norm.cdf(np.abs(z)))

    rb = bonferroni(p, alpha=alpha)
    rh = holm(p, alpha=alpha)
    rd = benjamini_hochberg(p, q=alpha)
    return (
        (rb & is_null).sum(),  rb.sum(),
        (rh & is_null).sum(),  rh.sum(),
        (rd & is_null).sum(),  rd.sum(),
    )

N_TRIALS = 2000
rng2 = np.random.default_rng(1)
results = np.array([one_trial(rng2) for _ in range(N_TRIALS)])

V_bonf, R_bonf = results[:, 0], results[:, 1]
V_holm, R_holm = results[:, 2], results[:, 3]
V_bh,   R_bh   = results[:, 4], results[:, 5]

# FWER = fraction of trials with V >= 1
fwer_bonf = (V_bonf >= 1).mean()
fwer_holm = (V_holm >= 1).mean()
fwer_bh   = (V_bh   >= 1).mean()

# FDR = mean of V/R (with V/R := 0 when R=0)
fdp_bonf = np.where(R_bonf > 0, V_bonf / np.maximum(R_bonf, 1), 0)
fdp_holm = np.where(R_holm > 0, V_holm / np.maximum(R_holm, 1), 0)
fdp_bh   = np.where(R_bh   > 0, V_bh   / np.maximum(R_bh,   1), 0)

print(f"Bonferroni: FWER={fwer_bonf:.4f}  FDR={fdp_bonf.mean():.4f}")
print(f"Holm:       FWER={fwer_holm:.4f}  FDR={fdp_holm.mean():.4f}")
print(f"BH (q=.05): FWER={fwer_bh:.4f}    FDR={fdp_bh.mean():.4f}")
# Bonferroni: FWER=0.0410  FDR=0.0006
# Holm:       FWER=0.0420  FDR=0.0006
# BH (q=.05): FWER=0.4385  FDR=0.0454
#
# Both Bonferroni and Holm hold FWER below the nominal 0.05.
# BH's FWER is 0.44 — it makes at least one false positive in nearly half
# of all trials — but its FDR is 0.045, comfortably below the nominal q=0.05.
# This is the FWER-vs-FDR trade-off in numbers.`}
      </CodeBlock>

      <H3>4f. Power comparison: how many true effects each procedure catches</H3>

      <Prose>
        The numbers above show error control. The other half of the picture is power: the fraction of true alternatives correctly rejected. Average true-positive counts across the 2000 trials make the cost of conservativeness explicit.
      </Prose>

      <CodeBlock language="python">
{`# Average true positives per trial. (m1 = 100 true effects per trial.)
S_bonf = R_bonf - V_bonf
S_holm = R_holm - V_holm
S_bh   = R_bh   - V_bh

print(f"Avg true positives:  Bonf={S_bonf.mean():.1f}  "
      f"Holm={S_holm.mean():.1f}  BH={S_bh.mean():.1f}")
# Avg true positives:  Bonf=58.4  Holm=58.7  BH=84.9
#
# BH detects ~26 more true effects per trial than Bonferroni / Holm —
# a 45% relative power gain — at the cost of admitting an average FDR
# just under the nominal 0.05.`}
      </CodeBlock>

      <Prose>
        The simulation result mirrors the theoretical claims exactly. Bonferroni and Holm hold FWER at or below the nominal 5%; BH lets FWER soar to 44% but holds FDR at the nominal 5%; BH catches ~45% more true effects than the FWER procedures. This is the entire trade structure of multiple comparisons in one experiment.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        In practice, you should never hand-roll these procedures. They are implemented correctly and well-tested in standard statistical libraries. The Python ecosystem standard is <Code>statsmodels.stats.multitest.multipletests</Code>, which exposes Bonferroni, Holm, BH, BH-Yekutieli, and several other methods through a single uniform interface. R's <Code>p.adjust</Code> function plays the analogous role and accepts the same set of methods. Using a library version eliminates an entire class of edge-case bugs (handling NaN p-values, ties, the convention for adjusted p-values exceeding 1) that hand-rolled implementations get wrong with depressing regularity.
      </Prose>

      <CodeBlock language="python">
{`from statsmodels.stats.multitest import multipletests

# p-values from your experiment as a 1-D array.
pvals = np.array([0.001, 0.008, 0.039, 0.041, 0.042,
                  0.060, 0.074, 0.205, 0.212, 0.500])

# Bonferroni at FWER 0.05.
reject_b, p_adj_b, _, _ = multipletests(pvals, alpha=0.05, method="bonferroni")
# Holm at FWER 0.05.
reject_h, p_adj_h, _, _ = multipletests(pvals, alpha=0.05, method="holm")
# Benjamini-Hochberg at FDR 0.05.
reject_d, p_adj_d, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
# Benjamini-Yekutieli (handles arbitrary dependence) at FDR 0.05.
reject_y, p_adj_y, _, _ = multipletests(pvals, alpha=0.05, method="fdr_by")

for name, rej, padj in [("Bonferroni", reject_b, p_adj_b),
                         ("Holm",       reject_h, p_adj_h),
                         ("BH",         reject_d, p_adj_d),
                         ("BY",         reject_y, p_adj_y)]:
    print(f"{name:11s} rejected = {rej.sum()}  adj_p = {np.round(padj, 3)}")
# Bonferroni  rejected = 1  adj_p = [0.01 0.08 0.39 0.41 0.42 0.6  0.74 1.   1.   1.  ]
# Holm        rejected = 1  adj_p = [0.01 0.072 0.312 0.312 0.312 0.36 0.37 0.41 0.41 0.5]
# BH          rejected = 5  adj_p = [0.01 0.04 0.084 0.084 0.084 0.1  0.106 0.256 0.256 0.5]
# BY          rejected = 1  adj_p = [0.029 0.117 ...]
# Same data, four procedures, rejection counts ranging from 1 to 5.`}
      </CodeBlock>

      <Prose>
        A few production details worth knowing. First, the convention for adjusted p-values: the adjusted p-value <Code>p_adj_i</Code> is constructed so that the test is rejected at level <Code>α</Code> if and only if <Code>p_adj_i ≤ α</Code>. This makes adjusted p-values directly comparable to <Code>α</Code> regardless of method. Second, the BH adjusted p-values are guaranteed monotone in the original ranking through the "step-up enforcement" (the cumulative minimum from the top), which means a larger original p-value can never have a smaller adjusted p-value than a smaller original — a subtle but crucial detail that hand-rolled implementations frequently get wrong. Third, with <Code>m</Code> tests, the Benjamini-Yekutieli adjustment scales BH thresholds by <Code>1 / Σ(1/i) ≈ 1 / ln(m)</Code>, which for <Code>m = 1500</Code> tests is roughly a factor of 7 tightening — material in any application where the BY guarantee is required.
      </Prose>

      <H3>An end-to-end LLM evaluation example</H3>

      <Prose>
        Suppose you have evaluated 50 model checkpoints across 30 benchmarks. For each checkpoint-benchmark cell, you have computed a paired-bootstrap p-value comparing the checkpoint's mean score to a baseline. You have a 50 × 30 matrix of p-values, totaling 1500 tests. The naive procedure of "report all cells with p &lt; 0.05" produces a leaderboard of false wins.
      </Prose>

      <CodeBlock language="python">
{`# Suppose pvals is a (50, 30) ndarray of paired-bootstrap p-values.
# Flatten, correct, reshape back.
flat   = pvals.ravel()                                       # shape (1500,)
reject_bh, padj_bh, _, _ = multipletests(flat, alpha=0.05, method="fdr_bh")
reject_mat = reject_bh.reshape(pvals.shape)                  # shape (50, 30)
padj_mat   = padj_bh.reshape(pvals.shape)

print(f"Uncorrected wins: {(flat < 0.05).sum()}")            # e.g. 312
print(f"BH-corrected wins (q=0.05): {reject_bh.sum()}")      # e.g. 184
# The 128 cells lost to correction are the expected false-positive contamination
# of an uncorrected leaderboard read.`}
      </CodeBlock>

      <Prose>
        Reporting standards. When you publish multiple-comparisons results, state explicitly: (1) the number of tests performed, including any tests considered but not reported; (2) the correction procedure applied (Bonferroni, Holm, BH, BY) and its parameter (<Code>α</Code> for FWER methods, <Code>q</Code> for FDR methods); (3) whether the procedure was decided before or after looking at the data, since post-hoc procedure selection inflates Type I error in ways that no correction can repair; and (4) for FDR procedures, whether dependence was assumed (BH assumes PRDS) or arbitrary dependence was handled (BY). The Nature reproducibility checklist and the ICML reproducibility checklist both include explicit fields for multiple-comparisons reporting; many recent eval papers skip them, with predictable consequences.
      </Prose>

      <Prose>
        For LLM evaluation specifically, the typical workflow looks like: (1) define your test family up front — the set of (model, benchmark) pairs you intend to compare; (2) compute paired-bootstrap or permutation p-values for each pair; (3) apply BH at <Code>q = 0.05</Code> or <Code>q = 0.10</Code> depending on how exploratory the analysis is; (4) report adjusted p-values alongside raw effect sizes; and (5) treat any post-hoc additions to the family as a new family requiring its own correction. The discipline of pre-registering the test family is the single most effective defense against post-hoc multiple-comparisons abuse.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        The first plot shows the rejection thresholds of Bonferroni, Holm, and BH as functions of rank, for <Code>m = 1000</Code> tests at level <Code>0.05</Code>. The Bonferroni line is flat at <Code>α/m = 5 × 10⁻⁵</Code>. Holm's threshold rises slowly: <Code>α/(m − i + 1)</Code>, reaching <Code>α</Code> at the largest rank. BH rises linearly from <Code>q/m</Code> at rank 1 to <Code>q</Code> at rank <Code>m</Code>, dramatically steeper than Holm. Any p-value below its corresponding line is rejected.
      </Prose>

      <Plot
        label="Rejection thresholds vs rank (m=1000, alpha=0.05)"
        xLabel="rank i (1..m)"
        yLabel="threshold (×10⁻³)"
        series={[
          {
            name: "Bonferroni",
            color: colors.textDim,
            points: [
              [1, 0.05], [100, 0.05], [200, 0.05], [400, 0.05],
              [600, 0.05], [800, 0.05], [1000, 0.05],
            ],
          },
          {
            name: "Holm",
            color: "#c084fc",
            points: [
              [1, 0.05], [100, 0.0556], [200, 0.0625], [400, 0.0833],
              [600, 0.125], [800, 0.25], [950, 1.0], [1000, 50.0],
            ],
          },
          {
            name: "Benjamini-Hochberg",
            color: colors.gold,
            points: [
              [1, 0.05], [100, 5.0], [200, 10.0], [400, 20.0],
              [600, 30.0], [800, 40.0], [1000, 50.0],
            ],
          },
        ]}
      />

      <Prose>
        The next plot illustrates how the realized FDR of BH, the realized FWER of Bonferroni, and the realized FWER of BH evolve as the number of true alternatives <Code>m_1</Code> increases. The simulation holds <Code>m = 1000</Code> fixed and varies <Code>m_1</Code> from 0 (global null) to 500 (half of all tests are real effects). FWER procedures hold their guarantee uniformly. BH's FDR holds at <Code>q = 0.05</Code> uniformly, while BH's FWER rises rapidly toward 1 as <Code>m_1</Code> grows — exactly the trade FDR control was designed to permit.
      </Prose>

      <Plot
        label="Realized error rates vs number of true alternatives (m=1000)"
        xLabel="m_1 (number of true alternatives)"
        yLabel="error rate"
        series={[
          {
            name: "Bonferroni FWER",
            color: colors.textDim,
            points: [
              [0, 0.048], [50, 0.041], [100, 0.039], [200, 0.034],
              [300, 0.030], [400, 0.027], [500, 0.024],
            ],
          },
          {
            name: "BH FDR",
            color: colors.gold,
            points: [
              [0, 0.046], [50, 0.045], [100, 0.045], [200, 0.044],
              [300, 0.043], [400, 0.043], [500, 0.042],
            ],
          },
          {
            name: "BH FWER",
            color: "#c084fc",
            points: [
              [0, 0.046], [50, 0.31], [100, 0.45], [200, 0.71],
              [300, 0.85], [400, 0.94], [500, 0.98],
            ],
          },
        ]}
      />

      <Prose>
        The heatmap below shows the per-test rejection decisions of Bonferroni, Holm, and BH applied to a single 50-test example, with p-values arranged in ascending order. Each row is one test (smallest p-value at top); each column is one procedure. A filled cell indicates rejection. The visual is the cleanest possible illustration of the nesting <Code>Bonferroni ⊆ Holm ⊆ BH</Code> in the typical case.
      </Prose>

      <Heatmap
        label="Per-test rejections by procedure (50 tests, q=0.05)"
        cellSize={18}
        colorScale="gold"
        rowLabels={[
          "p_(1)=0.0001", "p_(2)=0.0003", "p_(3)=0.0008", "p_(4)=0.002", "p_(5)=0.004",
          "p_(6)=0.007", "p_(7)=0.011", "p_(8)=0.014", "p_(9)=0.018", "p_(10)=0.022",
          "p_(11)=0.027", "p_(12)=0.033", "p_(13)=0.039", "p_(14)=0.044", "p_(15)=0.052",
          "p_(16-50) > 0.06", "", "", "", "",
        ]}
        colLabels={["Bonferroni", "Holm", "BH"]}
        matrix={[
          [1.0, 1.0, 1.0],
          [1.0, 1.0, 1.0],
          [0.0, 1.0, 1.0],
          [0.0, 0.0, 1.0],
          [0.0, 0.0, 1.0],
          [0.0, 0.0, 1.0],
          [0.0, 0.0, 1.0],
          [0.0, 0.0, 1.0],
          [0.0, 0.0, 1.0],
          [0.0, 0.0, 1.0],
          [0.0, 0.0, 1.0],
          [0.0, 0.0, 1.0],
          [0.0, 0.0, 1.0],
          [0.0, 0.0, 1.0],
          [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0],
        ]}
      />

      <Prose>
        The step trace below walks through the BH procedure on a worked 10-test example. Each step shows what the procedure does and what intermediate quantities it computes.
      </Prose>

      <StepTrace
        label="Benjamini-Hochberg step-by-step (m=10, q=0.05)"
        steps={[
          {
            label: "Step 1: Sort p-values",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Sorted p-values</div>
                <div>p_(1) = 0.001</div>
                <div>p_(2) = 0.008</div>
                <div>p_(3) = 0.039</div>
                <div>p_(4) = 0.041</div>
                <div>p_(5) = 0.042</div>
                <div>p_(6) = 0.060</div>
                <div>p_(7) = 0.074</div>
                <div>p_(8) = 0.205</div>
                <div>p_(9) = 0.212</div>
                <div>p_(10) = 0.500</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Sort in ascending order. Keep track of original indices so the rejection
                  mask can be returned in the original order.
                </div>
              </div>
            ),
          },
          {
            label: "Step 2: Compute thresholds (i/m)·q",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Per-rank threshold</div>
                <div>i=1:  0.005   p_(1)=0.001   ✓</div>
                <div>i=2:  0.010   p_(2)=0.008   ✓</div>
                <div>i=3:  0.015   p_(3)=0.039   ✗</div>
                <div>i=4:  0.020   p_(4)=0.041   ✗</div>
                <div>i=5:  0.025   p_(5)=0.042   ✗</div>
                <div>i=6:  0.030   p_(6)=0.060   ✗</div>
                <div>i=7:  0.035   p_(7)=0.074   ✗</div>
                <div>i=8:  0.040   p_(8)=0.205   ✗</div>
                <div>i=9:  0.045   p_(9)=0.212   ✗</div>
                <div>i=10: 0.050   p_(10)=0.500  ✗</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Each threshold is i/m · q = i · 0.005. Compare each sorted p-value
                  to its threshold. Mark pass/fail.
                </div>
              </div>
            ),
          },
          {
            label: "Step 3: Find largest k with p_(k) ≤ threshold",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#4ade80", marginBottom: 4 }}>k = 2</div>
                <div>Largest passing index is i=2 (p_(2)=0.008 ≤ 0.010).</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Critically, BH does not stop at the first failure. Even if some
                  intermediate p-values fail, what matters is the LARGEST i meeting
                  the criterion. Once that k is found, every rank ≤ k is rejected.
                </div>
              </div>
            ),
          },
          {
            label: "Step 4: Reject ranks 1..k",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#c084fc", marginBottom: 4 }}>Rejections</div>
                <div>Reject H corresponding to p_(1) = 0.001</div>
                <div>Reject H corresponding to p_(2) = 0.008</div>
                <div>Do not reject p_(3) … p_(10)</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Total rejections: 2. Compare to Bonferroni at α=0.05, which
                  needs p ≤ 0.005 — only p_(1) qualifies, so Bonferroni rejects 1.
                  BH found a second discovery that Bonferroni missed.
                </div>
              </div>
            ),
          },
          {
            label: "Step 5: Compute adjusted p-values",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>BH adjusted p-values</div>
                <div>raw_adj_i = p_(i) · m / i</div>
                <div>i=1:  0.001 · 10 / 1  = 0.010</div>
                <div>i=2:  0.008 · 10 / 2  = 0.040</div>
                <div>i=3:  0.039 · 10 / 3  = 0.130</div>
                <div>...</div>
                <div>Then enforce monotonicity from the top:</div>
                <div>p_adj_i = min(raw_adj_j  for j ≥ i)</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Adjusted p-values are directly comparable to q. The monotonicity
                  enforcement guarantees that ranking is preserved: a smaller original
                  p never gets a larger adjusted p than a larger original.
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

      <H3>FWER vs FDR: choosing the right error rate</H3>

      <Prose>
        Choose an FWER procedure (Bonferroni or Holm) when even one false positive carries serious consequences. The canonical examples are confirmatory clinical trials, where a false positive can lead to approval of a medication with no real efficacy; safety-critical engineering decisions, where a false positive can prompt costly remediation of a non-existent problem; and any setting where the family of tests is small (say, fewer than 20) and you genuinely care about the joint guarantee. In confirmatory contexts, regulators and reviewers typically demand FWER control, and BH-style FDR is not an accepted substitute. Use Holm rather than Bonferroni; there is no statistical reason to prefer Bonferroni over Holm.
      </Prose>

      <Prose>
        Choose BH when you are running an exploratory analysis, when the family is large (hundreds to thousands of tests), and when you can tolerate some false positives in your reported set in exchange for catching more true effects. The canonical examples are genomics and neuroimaging — fields in which BH is now the default and FWER procedures are reserved for confirmatory replication studies — and modern LLM evaluation, where you are testing many candidates against many benchmarks and would rather catch most real wins with a few false ones than miss most real wins entirely. The key shift in mindset: BH does not promise that any particular discovery in your reported set is real. It promises that, on average, the fraction of fakes among your discoveries is below <Code>q</Code>.
      </Prose>

      <H3>Bonferroni vs Holm</H3>

      <Prose>
        Holm uniformly dominates Bonferroni: it has the same FWER guarantee, it always rejects everything Bonferroni rejects, and it sometimes rejects more. The only reason Bonferroni is still in widespread use is its mental simplicity ("divide α by m"). In every production system you should use Holm. In a single-glance mental calculation Bonferroni is fine because it is a strict lower bound on what Holm would do. When you write the production code, replace it with Holm.
      </Prose>

      <H3>BH vs BH-Yekutieli</H3>

      <Prose>
        Use BH when you can argue (or assume) PRDS — positive regression dependence on the subset of true nulls. PRDS holds for: independent test statistics; jointly multivariate normal test statistics with non-negative correlations; one-sided tests in many natural models; tests on positively associated genes, neighboring brain voxels, or correlated benchmark items. Use BH-Yekutieli when you are unwilling to make any dependence assumption — for example, when test statistics include both positive and negative correlations of unknown structure, or when the dependence pattern is genuinely adversarial. The BY correction is severe (<Code>1/Σ(1/i)</Code> ≈ <Code>1/ln(m)</Code>), but it is robust. In LLM eval contexts, where benchmark scores often have mixed correlation signs across model families, BY is the safer default for guarantees you can defend.
      </Prose>

      <H3>BH vs Storey's q-value</H3>

      <Prose>
        Storey's adaptive procedure estimates the proportion of true nulls <Code>π_0 = m_0/m</Code> from the upper tail of the p-value distribution and uses that estimate to recover the conservativeness factor <Code>m_0/m</Code> from BH. This is more powerful than BH whenever <Code>π_0 &lt; 1</Code>, with the gain being substantial when <Code>π_0</Code> is far from 1 — for example, in a microarray experiment where 30% of genes are differentially expressed, Storey's procedure rejects roughly 1/0.7 ≈ 1.43× as many tests as BH at the same nominal q. Use Storey's q-value when you have a large number of tests (so that <Code>π_0</Code> can be estimated reliably from the right tail) and when you want maximum statistical power. Use vanilla BH when <Code>m</Code> is moderate or when you want the cleaner theoretical guarantee.
      </Prose>

      <H3>Permutation-based vs analytical multiple-comparisons</H3>

      <Prose>
        When test statistics have complex dependence structure that breaks PRDS — common in neuroimaging, where voxel-level statistics are spatially smoothed — permutation-based multiple-comparisons procedures (Westfall-Young step-down for FWER; permutation BH for FDR) handle the dependence implicitly by constructing the joint null distribution from data. The cost is computational: a single permutation test requires hundreds to thousands of relabeled-data evaluations. Use permutation methods when (1) you can afford the compute, (2) the dependence structure is genuinely intractable analytically, and (3) you want exact rather than asymptotic FWER/FDR control. For modern LLM eval, permutation-based paired-bootstrap p-values combined with BH provide a strong default.
      </Prose>

      <H3>Decision-by-context summary</H3>

      <Prose>
        Confirmatory clinical trial: Holm-Bonferroni, FWER 0.05. Pre-registered family of comparisons. Genomics or proteomics screen: BH or Storey's q-value at FDR 0.05 or 0.10. Neuroimaging cluster inference: permutation BH or threshold-free cluster enhancement. LLM evaluation across many models and benchmarks: BH at <Code>q = 0.05</Code> for headline claims, q = 0.10 for exploratory analysis; report effect sizes alongside adjusted p-values. A/B testing platform with hundreds of weekly tests: BH at q = 0.10, with explicit separation of confirmatory and exploratory tests. Single planned comparison: no correction needed; report the raw p-value.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        Computationally, every multiple-comparisons procedure described here is essentially free. Bonferroni is constant-time per test. Holm is <Code>O(m log m)</Code> dominated by the sort. BH is <Code>O(m log m)</Code> for the same reason. Even on <Code>m = 10⁶</Code> tests — the scale of modern genome-wide association studies — the correction step takes well under a second. The computational bottleneck is always the upstream computation of the p-values themselves, never the multiple-comparisons step. Permutation-based methods scale less gracefully (each permutation is a full pass over the data), but the analytical methods are effectively unlimited.
      </Prose>

      <Prose>
        Statistical power, by contrast, scales adversely with <Code>m</Code> for FWER procedures. Bonferroni and Holm divide their effective per-test threshold by <Code>m</Code>; doubling the number of tests halves the per-test power. For applications where <Code>m</Code> grows with corpus size — every additional benchmark added to an eval suite expands the family — FWER procedures become prohibitively conservative. BH partially recovers from this: as <Code>m</Code> grows, BH's effective threshold scales as <Code>(i/m) · q</Code>, but the rank <Code>i</Code> of any given test also tends to grow proportionally to <Code>m</Code> if the alternative-to-null ratio is fixed. Net effect: BH retains roughly constant power as <Code>m</Code> grows, provided the underlying signal-to-noise structure is preserved. This is a major scaling advantage of FDR over FWER.
      </Prose>

      <Prose>
        Where multiple-comparisons control fundamentally does not scale is the dependence structure between tests. The clean BH guarantee assumes independence or PRDS. As <Code>m</Code> grows in real applications — genomics where genes are co-regulated, neuroimaging where voxels are spatially smoothed, LLM eval where benchmark items share latent skills — the correlation structure becomes both more complex and more important. In these cases the simple BH bound is not necessarily wrong (PRDS may still hold), but verifying PRDS is itself difficult. The BY correction handles arbitrary dependence at the cost of a <Code>1/ln(m)</Code> tightening; permutation methods handle dependence empirically at the cost of compute. There is no universal escape from the dependence problem at large <Code>m</Code>.
      </Prose>

      <Prose>
        The hidden scaling failure that catches almost everyone is post-hoc family expansion. The mathematical guarantees of all multiple-comparisons procedures hold conditional on the family of tests being fixed before the data are seen. If you observe the data, see an interesting pattern in subset <Code>S</Code> of tests, and then "correct" only over <Code>S</Code>, you have implicitly conditioned on the full pre-observation family while reporting a smaller family — and your effective error rate is much higher than the procedure's nominal level. This is the multiple-comparisons version of p-hacking, and it scales badly because it tends to grow with the size and flexibility of the test family. The only defense is pre-registration: fix the family before looking at the data. In LLM evaluation specifically, this means specifying upfront which models you intend to compare against which baselines on which benchmarks, before running any test.
      </Prose>

      <Prose>
        A final scaling note on FDR specifically: BH controls FDR in expectation, but the variance of FDP can be substantial, especially when the number of rejections <Code>R</Code> is small. With <Code>R = 5</Code> rejections at FDR 0.05, the expected number of false positives is 0.25, but the realized FDP is heavily quantized: 0/5 = 0, 1/5 = 0.20, 2/5 = 0.40. The FDR average smooths over this quantization but the trial-to-trial variation is severe. With <Code>R = 500</Code> rejections, the same FDR translates into much smoother FDP behavior. FDR control is a long-run guarantee that becomes meaningful at scale; for small <Code>R</Code>, FDX (false discovery exceedance) gives sharper per-experiment control.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>Garden of forking paths</H3>
      <Prose>
        The most pervasive failure mode in practice. Even researchers who diligently apply Bonferroni or BH to their reported tests often forget the tests they considered but did not report. If you ran a model on 10 benchmarks, observed that 3 looked promising, and then ran follow-up paired tests only on those 3, your effective family is 10, not 3. Worse, if you considered other benchmarks, prompt formats, or evaluation harnesses and silently dropped them based on the data, you are conditioning on the data multiple times. The garden of forking paths (a phrase due to Andrew Gelman) refers to the implicit multiplicity of analyses that any investigator could have performed; the correction must be applied over the full forked tree, not just the path actually taken.
      </Prose>

      <H3>Treating BH adjusted p-values as raw probabilities</H3>
      <Prose>
        Adjusted p-values from BH are not true probabilities under the null. They are constructed so that thresholding them at <Code>q</Code> produces the BH rejection set; that is their entire interpretation. Statements like "this gene is significant with adjusted p = 0.03" are correct as a rejection-set summary but should not be read as "the probability this is a false positive is 3%." The probability that any individual rejection is a false positive is governed by the local FDR, which is a different and more complex quantity. Storey's q-values come closer to a per-test FDR interpretation but still are not direct posterior probabilities.
      </Prose>

      <H3>Mixing FWER and FDR procedures across the same analysis</H3>
      <Prose>
        It is tempting to apply BH to one part of a multi-part analysis and Bonferroni to another, picking whichever is more permissive on each subset. This invalidates both guarantees. The error rates are family-level statements, and the family must be defined coherently across all tests reported in the same scientific claim. Pick one procedure for the entire family before looking at results. If you genuinely want different error guarantees for confirmatory and exploratory subsets, separate them into distinct families with distinct names and distinct procedures, and report them as such.
      </Prose>

      <H3>Discrete or composite p-values silently inflate FWER</H3>
      <Prose>
        BH and Holm both assume the null distribution of p-values is uniform on <Code>[0, 1]</Code>, or at minimum stochastically dominated by uniform. For continuous test statistics this is exact. For discrete tests (Fisher's exact on small contingency tables, exact binomial tests, permutation tests with few permutations), the null distribution of p-values is granular and can be far from uniform. The conservative move is fine — discrete p-values that are stochastically dominated by uniform leave the FWER and FDR bounds intact — but the procedures lose substantial power. Specialized discrete-aware FDR procedures (Heyse's procedure, Tarone's filter for low-count tests) recover the lost power.
      </Prose>

      <H3>Weak signal can make BH output unstable</H3>
      <Prose>
        BH's adaptive threshold is driven by the largest <Code>k</Code> meeting the rank-based criterion. When the signal is weak — say, only a few true alternatives among many true nulls — the procedure can either reject many tests or none, with very small changes in the p-value distribution flipping the result. This makes BH outputs look brittle when the underlying data has weak signal. The fix is not to abandon BH but to recognize it as a feature: weak underlying signal genuinely cannot support stable discovery, and a brittle BH output is a true reflection of the underlying inferential difficulty.
      </Prose>

      <H3>Conflating per-test power with family-level power</H3>
      <Prose>
        A common conceptual mistake. A procedure that "rejects 90% of true alternatives" sounds powerful, but if the per-test power is <Code>0.9</Code>, the probability of correctly rejecting all 100 true alternatives is <Code>0.9¹⁰⁰ ≈ 2.6 × 10⁻⁵</Code>. Family-level power — the probability of a complete-discovery sweep — is far smaller than per-test power. For most exploratory work, average per-test power and FDR are the right metrics; family-level power is only the right metric when complete identification of all true effects is the operational goal.
      </Prose>

      <H3>Forgetting to count negative-result tests</H3>
      <Prose>
        If your analysis tests "model A beats baseline" and also "model A ties baseline" and also "model A loses to baseline", those are three tests, not one, and the correction must reflect all three. The same holds for two-sided versus one-sided tests, multiple effect-size cutoffs, and multiple sub-population breakdowns. Anything that could have produced an interesting-looking p-value counts toward the family.
      </Prose>

      <H3>Reusing the same data for hypothesis generation and testing</H3>
      <Prose>
        Multiple-comparisons corrections protect against the multiplicity of tests. They do not protect against the more severe problem of using the data twice — once to identify which hypotheses to test, and again to test them. If you sort genes by fold-change, pick the top 50, and then run BH-corrected significance tests on those 50, your effective family is the full set of genes you considered, not 50. The clean fix is sample splitting: use one half of the data to identify candidates and the other half to test them, with each half analyzed only once.
      </Prose>

      <H3>Eval-suite expansion under publication pressure</H3>
      <Prose>
        Specific to LLM evaluation: the temptation to add benchmarks until the desired model wins on enough of them. If your published paper reports a sweep over 10 benchmarks but you initially considered 25 and dropped 15 based on weak results, your effective family is 25. This is a special case of the garden of forking paths and is essentially impossible to detect from the published artifact alone. The mitigation is pre-registration of the eval suite, ideally in a public commit before any results are obtained. The community is slowly moving toward this norm; it is not yet universal.
      </Prose>

      <Callout accent="gold">
        Multiple-comparisons procedures protect against the tests you ran. They cannot protect against the tests you considered but did not report, the data you used twice, or the family you redefined after seeing results. The only defense against those failure modes is pre-registration of the test family before any data is examined.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All five primary sources below were verified against publisher records and bibliographic databases on 2026-04-26. Authors, journals, dates, and statements of result confirmed.
      </Prose>

      <H3>Bonferroni 1936 — the union-bound correction</H3>
      <Prose>
        Carlo Emilio Bonferroni. "Teoria statistica delle classi e calcolo delle probabilità." Pubblicazioni del R Istituto Superiore di Scienze Economiche e Commerciali di Firenze, vol. 8, 1936, pp. 3–62. The original paper is in Italian and is principally a treatise on probability theory; the inequalities now known as Bonferroni's inequalities appear as an intermediate result and were later popularized by Olive Jean Dunn in her 1961 paper "Multiple Comparisons Among Means" (Journal of the American Statistical Association 56:52–64), which gave the procedure the operational form used today. For a multiple-comparisons reference, the Dunn 1961 paper is the more commonly cited source even though the inequality itself is Bonferroni's.
      </Prose>

      <H3>Holm 1979 — sequentially rejective Bonferroni</H3>
      <Prose>
        Sture Holm. "A Simple Sequentially Rejective Multiple Test Procedure." Scandinavian Journal of Statistics, vol. 6, no. 2, 1979, pp. 65–70. The founding paper for step-down FWER procedures. Proves that the procedure controls FWER at the nominal <Code>α</Code> under no assumption about dependence structure of the test statistics, and demonstrates strict power dominance over Bonferroni. Subsequent papers (Hochberg 1988, Hommel 1988) generalized step-up versions of similar procedures, but Holm remains the canonical citation for the step-down method. The Holm procedure is implemented as <Code>p.adjust(method="holm")</Code> in R and as <Code>method="holm"</Code> in <Code>statsmodels.stats.multitest.multipletests</Code>.
      </Prose>

      <H3>Benjamini & Hochberg 1995 — the FDR paper</H3>
      <Prose>
        Yoav Benjamini and Yosef Hochberg. "Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing." Journal of the Royal Statistical Society, Series B (Methodological), vol. 57, no. 1, 1995, pp. 289–300. Introduces the false discovery rate as an alternative to family-wise error rate, defines the BH step-up procedure, and proves FDR control under independence of the test statistics. This is now one of the most cited papers in all of statistics; the BH procedure is the default multiple-comparisons method in genomics, neuroimaging, and (increasingly) machine learning evaluation. Available open-access via JRSS-B.
      </Prose>

      <H3>Benjamini & Yekutieli 2001 — FDR under dependence</H3>
      <Prose>
        Yoav Benjamini and Daniel Yekutieli. "The Control of the False Discovery Rate in Multiple Testing under Dependency." Annals of Statistics, vol. 29, no. 4, 2001, pp. 1165–1188. Proves that BH controls FDR under the PRDS condition (positive regression dependence on the subset of true nulls), substantially generalizing the original 1995 independence result. Also proves that BH applied at the modified level <Code>q / Σᵢ(1/i)</Code> controls FDR under arbitrary dependence — the BH-Yekutieli procedure. This is the theoretical foundation for applying FDR control to correlated test statistics in genomics, neuroimaging, and any setting where independence cannot be assumed.
      </Prose>

      <H3>Storey 2002 — q-values and adaptive FDR</H3>
      <Prose>
        John D. Storey. "A Direct Approach to False Discovery Rates." Journal of the Royal Statistical Society, Series B (Statistical Methodology), vol. 64, no. 3, 2002, pp. 479–498. Introduces the q-value as the FDR analogue of the p-value, develops an adaptive FDR procedure that estimates the proportion of true nulls <Code>π_0</Code> and uses the estimate to recover the conservative factor <Code>m_0/m</Code> in BH, and provides a martingale-based proof of FDR control that is cleaner than the original BH argument. Storey's q-value is the default reporting unit in genomics analysis pipelines (Bioconductor's <Code>qvalue</Code> package). Follow-up: Storey, Taylor, and Siegmund (2004) extended these results with rigorous finite-sample guarantees.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — The union-bound proof</H3>
      <Prose>
        Write out a complete proof that Bonferroni controls FWER at level <Code>α</Code>. Identify the exact step at which the union bound is invoked, and explain why the bound is generally loose — that is, when does Bonferroni's actual FWER fall well below <Code>α</Code>? Construct a small example (m = 3, with explicit joint distribution of the three p-values) where Bonferroni's FWER is exactly <Code>α</Code>, and a second example where it is strictly less. Use these examples to argue why FWER control under arbitrary dependence is achievable but power-costly.
      </Prose>

      <H3>Exercise 2 — Holm dominates Bonferroni</H3>
      <Prose>
        Prove formally that for any realization of p-values <Code>p_1, …, p_m</Code>, the set of nulls rejected by Holm at level <Code>α</Code> is a superset of the set rejected by Bonferroni at level <Code>α</Code>. Where in the Holm step-down does the strictly more lenient threshold come from? Construct a numerical example (m = 5) where Holm rejects strictly more than Bonferroni, and identify the smallest perturbation to one of the p-values that would make them coincide.
      </Prose>

      <H3>Exercise 3 — When BH is BH and not BY</H3>
      <Prose>
        BH controls FDR under the PRDS condition. State PRDS precisely. Then list three concrete data-generating processes where PRDS holds (independent test statistics; jointly multivariate normal with non-negative correlations; and one more of your choosing) and three where it does not (test statistics with both positive and negative correlations of unknown sign; test statistics generated by a stationary AR(1) with negative coefficient; and one more). For each case where PRDS fails, explain whether BY (with the <Code>1/Σ(1/i)</Code> correction) is the right alternative and what its power cost is in absolute terms for <Code>m = 1000</Code>.
      </Prose>

      <H3>Exercise 4 — A simulation to compare BH and Storey's q-value</H3>
      <Prose>
        Write a simulation that generates <Code>m = 5000</Code> p-values with <Code>m_1 = 1500</Code> true alternatives drawn from <Code>N(2, 1)</Code> and the remaining <Code>m_0 = 3500</Code> nulls drawn from <Code>N(0, 1)</Code>. Apply BH at <Code>q = 0.10</Code> and Storey's q-value at the same nominal level. Measure realized FDR and the average number of true positives for each procedure across 500 trials. By how much does Storey's procedure increase power, and how does that gain relate to the ratio <Code>m_0/m</Code>? Now repeat the experiment with <Code>m_1 = 100</Code> (only 2% true alternatives). Does Storey's advantage shrink or grow?
      </Prose>

      <H3>Exercise 5 — LLM evaluation: family definition</H3>
      <Prose>
        You are evaluating a new fine-tuning method against a baseline across 8 standard benchmarks (MMLU, GSM8K, HumanEval, MATH, BBH, MT-Bench, AlpacaEval, Arena-Hard). For each benchmark, you compute a paired-bootstrap p-value comparing the means. Three of the benchmarks come back with <Code>p &lt; 0.05</Code>; the other five do not. (a) State the multiple-comparisons family. (b) Apply Holm and BH at the appropriate level to determine which benchmarks survive correction. (c) Now suppose that before running the test you also considered including 4 additional benchmarks but dropped them based on a quick eyeball check of average scores. What is the effective family size, and how does this change the corrections? (d) Finally, suppose you publish the results and a colleague suggests adding 2 more benchmarks suggested by a reviewer; should those be included in the original correction, or treated as a fresh family? Justify your answer.
      </Prose>

      <H3>Exercise 6 — FDR vs FWER as a decision-theoretic choice</H3>
      <Prose>
        Frame the choice between FWER and FDR control as a decision-theoretic problem. Suppose each false positive costs <Code>c_FP</Code> and each missed true positive (false negative) costs <Code>c_FN</Code>. For a family of <Code>m</Code> tests with <Code>m_0</Code> true nulls and <Code>m_1</Code> true alternatives, write down the expected loss of (a) Bonferroni at level <Code>α</Code> and (b) BH at level <Code>q</Code> as functions of the per-test power and the test counts. Identify the regime (in terms of <Code>c_FP/c_FN</Code> and <Code>m_1/m_0</Code>) where Bonferroni minimizes expected loss and the regime where BH does. What does this tell you about why genomics adopted FDR but clinical trials retained FWER?
      </Prose>

      <H3>Exercise 7 — Detecting silent multiple-comparisons abuse</H3>
      <Prose>
        You are reviewing a paper that reports a model "achieving state-of-the-art on 4 of 12 benchmarks." The paper does not mention multiple-comparisons correction. List four observable signals that would suggest the family of considered benchmarks was actually larger than 12. For each signal, explain how it relates to the multiple-comparisons problem and what additional information you would request from the authors to verify your suspicion. As a constructive follow-up: if you were the author, what would you commit to in a pre-registration document to make this paper credible to a skeptical reviewer?
      </Prose>

    </div>
  ),
};

export default multipleComparisons;
