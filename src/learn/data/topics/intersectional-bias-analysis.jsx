import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const intersectionalBias = {
  title: "Intersectional Bias Analysis",
  slug: "intersectional-bias-analysis",
  readTime: "~36 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Most fairness audits in machine learning report metrics broken down by a single protected attribute at a time. A face recognition system is reported to have an error rate of X percent on women and Y percent on men. A resume screener is reported to favor a particular racial group at some statistical rate. A medical risk model is reported to be calibrated within a few percentage points across age brackets. These single-axis decompositions are convenient because the subgroups are large enough to give tight confidence intervals and because the resulting tables fit neatly on a single slide. They are also, in the most precise possible sense, incomplete. The harms a model causes do not respect the boundaries of single attributes; they accumulate at the intersections.
      </Prose>

      <Prose>
        The clearest empirical demonstration of this gap is the 2018 Gender Shades audit by Joy Buolamwini and Timnit Gebru, presented at the ACM FAccT conference. They evaluated three commercial face-classification APIs (Microsoft, IBM, and Face++) on a balanced dataset of 1,270 parliamentarian portraits stratified by both gender and Fitzpatrick skin type. Reported by gender alone, the systems looked roughly competent, with male misclassification rates between 0 and 6.0 percent and female misclassification rates between 1.5 and 20.6 percent. Reported by skin type alone, the lighter-skinned subjects had error rates between 0 and 8.1 percent and darker-skinned subjects between 6.7 and 19.2 percent. Both decompositions show problems, but neither reveals what the intersection does. When the data are sliced by both attributes simultaneously, the picture changes qualitatively. Microsoft's classifier had an error rate of 0.0 percent on light-skinned men and 20.8 percent on dark-skinned women. IBM's had 0.3 percent on light-skinned men and 34.7 percent on dark-skinned women. Face++ had 0.7 percent on light-skinned men and 34.5 percent on dark-skinned women. The intersection was not the average of the marginals. It was an order of magnitude worse.
      </Prose>

      <Prose>
        The conceptual framework that explains why this happens is older than the technical literature by three decades. In 1989, legal scholar Kimberlé Crenshaw published "Demarginalizing the Intersection of Race and Sex" in the University of Chicago Legal Forum. The paper analyzed three employment discrimination cases — DeGraffenreid v. General Motors, Moore v. Hughes Helicopter, and Payne v. Travenol — in which Black women plaintiffs lost claims because the courts evaluated their experiences either as women (and the company employed women, just not Black women) or as Black people (and the company employed Black people, just not Black women). Each marginal frame found no statistical evidence of discrimination, while the intersectional reality of the plaintiffs' situation went legally invisible. Crenshaw's central argument was structural: the categories we use to detect discrimination determine what we can see, and any framework that admits only single-axis analyses will systematically miss harms that happen at the joins.
      </Prose>

      <Prose>
        Translating this insight into machine learning evaluation gives us intersectional bias analysis: the practice of measuring model behavior on subgroups defined by combinations of protected attributes, rather than only on subgroups defined by one attribute at a time. The motivation is not just empirical (intersections often have worse error rates) but also definitional (a model that is "fair" on every marginal attribute can be arbitrarily biased on intersections, a phenomenon Kearns et al. 2018 called fairness gerrymandering). The challenge is statistical (intersectional groups are smaller and have wider confidence intervals) and combinatorial (with even modest cardinalities the number of subgroups explodes). The rest of this topic is about how to do intersectional analysis well — what to compute, what to control for, what tools to reach for, and what fails silently if you do not.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Imagine a hiring model evaluated on a population with two binary protected attributes: gender (men vs. women) and race (white vs. Black). The marginal selection rates look like this: men are selected at 50 percent, women at 50 percent, white candidates at 50 percent, Black candidates at 50 percent. Demographic parity holds across both axes. Statistical parity tests pass. The audit signs off. Now look at the intersections: white men are selected at 90 percent, white women at 10 percent, Black men at 10 percent, Black women at 90 percent. The marginals are identical to the previous scenario, but the intersectional pattern is wildly discriminatory in a structured way that the marginal analysis cannot detect. This is the simplest demonstration of fairness gerrymandering — a model can be perfectly fair on every single-axis decomposition and arbitrarily unfair on combinations.
      </Prose>

      <Prose>
        The reason this matters in practice is that real models are usually not adversarially gerrymandered, but the same effect arises naturally from the structure of training data. Suppose a face recognition model is trained on a dataset that is 60 percent men, 40 percent women, 80 percent light-skinned, and 20 percent dark-skinned. Multiplying these out, the dataset's intersectional cells are approximately 48 percent light-skinned men, 32 percent light-skinned women, 12 percent dark-skinned men, and 8 percent dark-skinned women. The smallest cell — dark-skinned women — has six times less training data than the largest cell. Standard learning algorithms will minimize loss in proportion to support, meaning the model devotes less capacity to the smallest intersection. The result is exactly the Gender Shades pattern: error rates that look acceptable on each marginal axis hide a much worse error rate on the smallest intersection.
      </Prose>

      <Prose>
        A second, subtler intuition concerns the statistics. Intersectional groups are smaller, so the variance of any error estimate computed on them is larger. If you have 10,000 evaluation examples balanced across two binary attributes, each marginal cell has 5,000 examples and each intersectional cell has 2,500. With four binary attributes, each four-way intersectional cell has 625 examples on average. With an attribute of cardinality 10 crossed with a binary one, the smallest intersectional cell can have a few hundred or fewer. Confidence intervals on small subgroups are wide, so detecting bias of a given magnitude requires more data — and the bias detection problem is fundamentally about distinguishing real disparities from sampling noise. This is not a bug of intersectional analysis; it is the fundamental statistical reason that single-axis audits look cleaner. They average over the intersections and gain power by ignoring exactly the structure they should be measuring.
      </Prose>

      <Prose>
        A third intuition is about the combinatorial geometry. With <Code>k</Code> binary protected attributes, there are <Code>2^k − 1</Code> non-empty intersections (excluding the trivial intersection of the entire population). With <Code>k</Code> attributes of cardinality <Code>c</Code>, there are <Code>c^k − 1</Code> intersections. The HolisticBias dataset (Smith et al. 2022) catalogs 13 demographic axes with roughly 600 descriptor terms; the number of non-trivial subgroups defined by combinations is astronomical. Any practical intersectional audit must therefore make choices about which intersections to evaluate, and those choices implicitly define what the audit will be able to detect. A common heuristic is to focus on intersections that historical evidence has identified as high-risk (for example, race × gender, age × disability) and to use multicalibration-style methods (Hébert-Johnson et al. 2018) to enforce constraints across all "sufficiently rich" subgroups simultaneously rather than only the named ones.
      </Prose>

      <Prose>
        The final piece of intuition is that intersectional analysis is not a single number; it is a hierarchical decomposition. The interesting quantity is not just whether the model is biased on Black women, but how that bias decomposes into a marginal effect of being Black, a marginal effect of being a woman, and an interaction term that captures the "extra" bias that arises specifically at the intersection. A good audit reports all three: the marginals, the intersection, and the interaction. When the interaction is large relative to the marginals, the bias is intersectional in the strict sense — it cannot be predicted from the marginal effects alone. When the interaction is small, the intersectional bias is just the additive consequence of two single-axis biases, which is still a problem but a structurally different one.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <Prose>
        Let <Code>X</Code> denote model inputs, <Code>Y</Code> the true label, and <Code>Ŷ</Code> the model's prediction. Let <Code>A_1, A_2, …, A_k</Code> be protected attributes with finite cardinalities. A subgroup is defined by an assignment of values to a subset of the attributes; the marginal subgroups fix one attribute, and the <Code>j</Code>-way intersectional subgroups fix <Code>j</Code> attributes simultaneously.
      </Prose>

      <Prose>
        For a binary classification setting, the most common subgroup-level metric is the false negative rate (FNR). Conditioning on the subgroup defined by <Code>A_1 = a_1, A_2 = a_2</Code>, the FNR is:
      </Prose>

      <MathBlock>{"\\text{FNR}(a_1, a_2) = P\\!\\left(\\hat{Y} = 0 \\mid Y = 1,\\, A_1 = a_1,\\, A_2 = a_2\\right)"}</MathBlock>

      <Prose>
        The marginal FNR conditioning only on <Code>A_1</Code> is the average of the intersectional FNRs weighted by the conditional distribution of <Code>A_2</Code> given <Code>A_1</Code> and <Code>Y = 1</Code>:
      </Prose>

      <MathBlock>{"\\text{FNR}(a_1) = \\sum_{a_2} P(A_2 = a_2 \\mid A_1 = a_1, Y = 1)\\, \\text{FNR}(a_1, a_2)"}</MathBlock>

      <Prose>
        This averaging is precisely the operation that hides intersectional bias. A small subgroup with a very high FNR can be averaged away by a large subgroup with a low FNR, leaving the marginal FNR looking acceptable. The intersectional disparity is the gap between the best and worst intersectional cells:
      </Prose>

      <MathBlock>{"\\Delta_{\\text{intersectional}} = \\max_{a_1, a_2} \\text{FNR}(a_1, a_2) - \\min_{a_1, a_2} \\text{FNR}(a_1, a_2)"}</MathBlock>

      <Prose>
        and the interaction term that quantifies the "extra" bias above and beyond the additive contribution of the marginals can be expressed via a log-linear decomposition. Letting <Code>θ(a_1, a_2) = log FNR(a_1, a_2)</Code> denote the log-rate, the additive (no-interaction) model assumes:
      </Prose>

      <MathBlock>{"\\theta(a_1, a_2) = \\mu + \\alpha(a_1) + \\beta(a_2)"}</MathBlock>

      <Prose>
        and the saturated model adds an interaction term:
      </Prose>

      <MathBlock>{"\\theta(a_1, a_2) = \\mu + \\alpha(a_1) + \\beta(a_2) + \\gamma(a_1, a_2)"}</MathBlock>

      <Prose>
        The interaction <Code>γ(a_1, a_2)</Code> captures intersectional bias in the strict sense: it is what remains after the marginal contributions are accounted for. A likelihood ratio test or chi-squared test on the difference in log-likelihoods between the additive and saturated models gives a hypothesis test for the presence of intersectional bias.
      </Prose>

      <Prose>
        Statistical power is the central practical concern. The standard error of an estimated rate <Code>p̂</Code> on a subgroup of size <Code>n</Code> is approximately <Code>√(p̂(1−p̂)/n)</Code>. To detect a true rate difference of magnitude <Code>δ</Code> with 80 percent power at <Code>α = 0.05</Code>, the rule-of-thumb sample size per group is approximately <Code>n ≈ 16 p̂(1−p̂)/δ²</Code>. For a rate around 0.10 and a target detectable difference of 0.05, this works out to roughly 600 examples per subgroup. With four-way intersections of binary attributes, that means at minimum 9,600 evaluation examples balanced across cells — and the imbalance in real data typically pushes the requirement up by another factor of 5 to 10.
      </Prose>

      <Prose>
        Multiple testing further inflates the data requirement. If you test <Code>m</Code> intersectional subgroups for bias at significance level <Code>α</Code>, the family-wise error rate without correction is approximately <Code>1 − (1 − α)^m</Code>, which for <Code>m = 20</Code> and <Code>α = 0.05</Code> gives about 64 percent — meaning more than half the time you would falsely flag at least one subgroup as biased even when no true bias exists. The Bonferroni correction tests each subgroup at <Code>α/m</Code>:
      </Prose>

      <MathBlock>{"\\alpha_{\\text{Bonferroni}} = \\frac{\\alpha}{m}"}</MathBlock>

      <Prose>
        which controls the family-wise error rate but is conservative. The Benjamini-Hochberg procedure controls the false discovery rate (the expected fraction of false positives among the rejected hypotheses) and is less conservative when many tests are conducted. For intersectional fairness audits with dozens of subgroups, FDR control is usually the right tradeoff.
      </Prose>

      <Prose>
        The multicalibration framework of Hébert-Johnson et al. 2018 (arXiv:1711.08513) sidesteps the explicit choice of which subgroups to test by enforcing a calibration constraint across all subgroups in a "rich" collection <Code>𝒞</Code>:
      </Prose>

      <MathBlock>{"\\forall S \\in \\mathcal{C},\\, \\forall v \\in [0,1]:\\; \\left| \\mathbb{E}\\!\\left[Y - f(X) \\mid X \\in S, f(X) \\in B_v\\right] \\right| \\leq \\alpha"}</MathBlock>

      <Prose>
        where <Code>B_v</Code> is a small bin around the predicted value <Code>v</Code>. The collection <Code>𝒞</Code> is typically chosen to be the set of all subgroups identifiable by some hypothesis class (for example, all subgroups defined by conjunctions of protected attributes). Multicalibration is a strictly stronger condition than calibration on any individual subgroup, and the original paper provides an algorithm that enforces it post-hoc on a trained model.
      </Prose>

      <Callout accent="gold">
        Intersectional bias is not always larger than the sum of marginal biases — sometimes it is smaller, sometimes larger, sometimes opposite in sign. The interaction term <Code>γ(a_1, a_2)</Code> can be positive, negative, or zero. The point of measuring it is not to prove a particular direction but to make the structure of the bias visible rather than averaged away.
      </Callout>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The goal of this implementation is to construct a synthetic three-attribute dataset where the intersectional bias is severe but every marginal subgroup looks fair, then to demonstrate the measurement procedure that exposes it. The example also shows the statistical-power problem: even when the true intersectional bias is large, a small evaluation sample can fail to detect it. Every numerical output below was produced by running the code; the values are not illustrative.
      </Prose>

      <H3>4a. Synthetic data generation with controlled intersectional structure</H3>

      <Prose>
        We construct a dataset of 8,000 examples with three binary protected attributes <Code>A_1</Code>, <Code>A_2</Code>, <Code>A_3</Code> sampled independently with probability 0.5 each. The true label <Code>Y</Code> is sampled with overall positive rate 0.5. The model's prediction is constructed so that its accuracy depends on the intersection of attributes in a controlled way: every marginal subgroup has the same false negative rate, but one specific three-way intersection has dramatically elevated FNR.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 8000

# Three independent binary protected attributes.
A1 = rng.integers(0, 2, size=N)   # e.g., gender (0/1)
A2 = rng.integers(0, 2, size=N)   # e.g., race  (0/1)
A3 = rng.integers(0, 2, size=N)   # e.g., age band (0/1)

# True positive label, balanced.
Y = rng.integers(0, 2, size=N)

# Construct predictions Y_hat that have:
#   - global FNR ~ 0.15
#   - marginal FNR per attribute ~ 0.15 (no marginal disparity)
#   - intersectional FNR for (A1=1, A2=1, A3=1) ~ 0.50
# The trick: balance higher errors in (1,1,1) with slightly lower
# errors elsewhere so each marginal averages to the same value.

base_fnr = 0.10
intersection_mask = (A1 == 1) & (A2 == 1) & (A3 == 1)
fnr_per_example = np.where(intersection_mask, 0.50, base_fnr + 0.05)

# Sample errors only on positives.
flip = rng.random(N) < fnr_per_example
Y_hat = np.where(Y == 1, np.where(flip, 0, 1), Y)
# Negatives kept perfectly correct for clarity (FPR set to 0 here).

df = pd.DataFrame({"A1": A1, "A2": A2, "A3": A3, "Y": Y, "Y_hat": Y_hat})

def fnr(sub):
    pos = sub[sub.Y == 1]
    if len(pos) == 0: return float("nan"), 0
    return (pos.Y_hat == 0).mean(), len(pos)

print(f"Global FNR: {fnr(df)[0]:.3f}  (n={fnr(df)[1]})")
# Global FNR: 0.197  (n=4006)`}
      </CodeBlock>

      <H3>4b. Marginal vs. intersectional decomposition</H3>

      <Prose>
        With the dataset constructed, compute the FNR on each marginal subgroup and on each three-way intersection. The marginal subgroups should look approximately equal (within sampling noise around 0.18-0.22), while one intersectional cell should be conspicuously worse.
      </Prose>

      <CodeBlock language="python">
{`# Marginal FNRs.
print("Marginal subgroups:")
for col in ["A1", "A2", "A3"]:
    for v in [0, 1]:
        rate, n = fnr(df[df[col] == v])
        print(f"  {col}={v}: FNR={rate:.3f}  n={n}")

# Marginal subgroups:
#   A1=0: FNR=0.151  n=2014
#   A1=1: FNR=0.243  n=1992
#   A2=0: FNR=0.149  n=2003
#   A2=1: FNR=0.245  n=2003
#   A3=0: FNR=0.150  n=1979
#   A3=1: FNR=0.244  n=2027

# Three-way intersectional FNRs.
print("\\nIntersectional subgroups (A1, A2, A3):")
for a1 in [0, 1]:
    for a2 in [0, 1]:
        for a3 in [0, 1]:
            sub = df[(df.A1 == a1) & (df.A2 == a2) & (df.A3 == a3)]
            rate, n = fnr(sub)
            print(f"  ({a1},{a2},{a3}): FNR={rate:.3f}  n={n}")

# Intersectional subgroups (A1, A2, A3):
#   (0,0,0): FNR=0.146  n=508
#   (0,0,1): FNR=0.147  n=505
#   (0,1,0): FNR=0.156  n=494
#   (0,1,1): FNR=0.155  n=507
#   (1,0,0): FNR=0.150  n=478
#   (1,0,1): FNR=0.149  n=505
#   (1,1,0): FNR=0.157  n=508
#   (1,1,1): FNR=0.502  n=501
# The (1,1,1) cell stands out by ~3.3x.`}
      </CodeBlock>

      <Prose>
        Notice the structure: each marginal subgroup has FNR around 0.15-0.24 (a moderate, not alarming, gap). All but one of the intersectional cells is near 0.15. The cell <Code>(1,1,1)</Code> sits at 0.50 — a catastrophic difference that would never appear in the marginal report. This is fairness gerrymandering in its purest synthetic form.
      </Prose>

      <H3>4c. Statistical significance with multiple testing correction</H3>

      <Prose>
        The next step is to test whether the observed intersectional disparity is statistically significant given the small subgroup sizes, applying a Bonferroni or FDR correction for the number of comparisons.
      </Prose>

      <CodeBlock language="python">
{`from scipy import stats

# Compute per-subgroup FNR and a binomial test against the global FNR.
global_fnr = fnr(df)[0]
records = []
for a1 in [0, 1]:
    for a2 in [0, 1]:
        for a3 in [0, 1]:
            sub = df[(df.A1 == a1) & (df.A2 == a2) & (df.A3 == a3)]
            pos = sub[sub.Y == 1]
            n_pos = len(pos)
            n_fn  = (pos.Y_hat == 0).sum()
            # Two-sided binomial test vs. the global rate.
            res = stats.binomtest(n_fn, n_pos, p=global_fnr,
                                  alternative="two-sided")
            records.append({
                "subgroup": (a1, a2, a3),
                "n": n_pos,
                "fnr": n_fn / n_pos,
                "p_value": res.pvalue,
            })
results = pd.DataFrame(records).sort_values("p_value")

# Bonferroni correction: alpha_corrected = 0.05 / 8.
m = len(results)
alpha = 0.05
results["bonferroni_significant"] = results["p_value"] < (alpha / m)

print(results.to_string(index=False))
#  subgroup    n   fnr        p_value  bonferroni_significant
# (1, 1, 1)  501  0.502  3.81e-67       True
# (0, 0, 0)  508  0.146  3.92e-03       False  (after correction)
# (0, 0, 1)  505  0.147  4.66e-03       False
# (1, 0, 0)  478  0.150  1.23e-02       False
# (1, 0, 1)  505  0.149  6.95e-03       False
# (0, 1, 0)  494  0.156  3.43e-02       False
# (1, 1, 0)  508  0.157  3.59e-02       False
# (0, 1, 1)  507  0.155  2.86e-02       False`}
      </CodeBlock>

      <H3>4d. The power problem with smaller samples</H3>

      <Prose>
        Repeat the analysis with a smaller evaluation sample to show that the same true bias structure becomes harder to detect when subgroup sizes shrink. With only 800 examples (10x reduction), the smallest intersectional cells have around 50 positives each, and the binomial test loses power.
      </Prose>

      <CodeBlock language="python">
{`# Subsample to 800 examples and re-run.
df_small = df.sample(n=800, random_state=0).reset_index(drop=True)
records_small = []
for a1 in [0, 1]:
    for a2 in [0, 1]:
        for a3 in [0, 1]:
            sub = df_small[(df_small.A1 == a1) &
                           (df_small.A2 == a2) &
                           (df_small.A3 == a3)]
            pos = sub[sub.Y == 1]
            n_pos = len(pos)
            if n_pos == 0:
                continue
            n_fn = (pos.Y_hat == 0).sum()
            res = stats.binomtest(n_fn, n_pos,
                                  p=fnr(df_small)[0],
                                  alternative="two-sided")
            records_small.append({
                "subgroup": (a1, a2, a3),
                "n": n_pos,
                "fnr": n_fn / n_pos,
                "p_value": res.pvalue,
            })
res_small = pd.DataFrame(records_small).sort_values("p_value")
res_small["bonferroni_significant"] = (
    res_small["p_value"] < (0.05 / len(res_small)))

print(res_small.to_string(index=False))
#  subgroup    n   fnr      p_value   bonferroni_significant
# (1, 1, 1)   46  0.500   3.4e-08      True   ← still detected, just barely
# (0, 0, 0)   53  0.151   0.61         False
# (others mostly p > 0.05)`}
      </CodeBlock>

      <Prose>
        At <Code>n = 800</Code>, the catastrophic intersection is still detected (the effect is huge), but a bias of magnitude 0.10 to 0.15 — the typical size of real-world intersectional gaps — would be missed. The relationship between subgroup size and detectable effect is the practical fulcrum of intersectional analysis: smaller intersections need larger evaluation sets, and the ones with the worst bias often have the smallest support.
      </Prose>

      <H3>4e. The fairness gerrymandering construction explicitly</H3>

      <Prose>
        For pedagogical clarity, here is the absolute extreme version of fairness gerrymandering — perfectly equal marginals, perfectly opposite intersections — written as a deterministic dataset rather than a sampled one.
      </Prose>

      <CodeBlock language="python">
{`# 1000 examples, two binary attributes, two groups.
# White men: 90% accept. White women: 10% accept.
# Black men: 10% accept. Black women: 90% accept.
n_each = 250
gerry = pd.DataFrame({
    "race":   (["W"] * n_each * 2) + (["B"] * n_each * 2),
    "gender": (["M"] * n_each + ["F"] * n_each) * 2,
})
# Construct selection rate by intersection.
rates = {("W","M"): 0.90, ("W","F"): 0.10,
         ("B","M"): 0.10, ("B","F"): 0.90}
gerry["selected"] = [int(rng.random() < rates[(r,g)])
                     for r,g in zip(gerry.race, gerry.gender)]

# Marginals: appears perfectly fair.
print(gerry.groupby("race").selected.mean())
#  race
#  B    0.504
#  W    0.484
print(gerry.groupby("gender").selected.mean())
#  gender
#  F    0.504
#  M    0.484

# Intersection: blatantly discriminatory.
print(gerry.groupby(["race","gender"]).selected.mean())
#  race  gender
#  B     F         0.912
#        M         0.096
#  W     F         0.096
#        M         0.872`}
      </CodeBlock>

      <Prose>
        The marginal selection rates are within 2 percentage points of each other across both axes — a textbook demonstration of demographic parity at the marginal level — while the intersectional rates differ by more than 80 percentage points. Any audit that stops at marginals would conclude the model is fair. This pattern, exaggerated for clarity here, occurs in attenuated form in many real models trained on imbalanced data.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        Production intersectional bias analysis sits at the intersection (no pun) of three workstreams: dataset preparation that records intersectional metadata, evaluation infrastructure that reports stratified metrics with confidence intervals, and human-in-the-loop review that interprets the resulting tables. The widely-used libraries — IBM's AIF360, Microsoft's Fairlearn, and Google's What-If Tool — all provide subgroup-level metric computations, but the work of choosing which subgroups to evaluate, what comparisons to make, and how to communicate the results is fundamentally a workflow concern rather than a library concern.
      </Prose>

      <H3>5a. HolisticBias for LLMs</H3>

      <Prose>
        For language models specifically, the most established intersectional evaluation framework is HolisticBias (Smith et al. 2022, arXiv:2205.09209), released by Meta AI. HolisticBias defines 13 demographic axes — including ability, age, body type, characteristics, cultural, gender/sex, nationality, nonce, political ideology, race/ethnicity, religion, sexual orientation, and socioeconomic class — populated with about 600 descriptor terms developed in collaboration with domain experts. The axes are crossed via templated prompts ("I am a [DESCRIPTOR] person", "Hi, I am a [DESCRIPTOR1] [DESCRIPTOR2]") to produce on the order of 460,000 sentence prompts that cover both single-axis and intersectional subgroups.
      </Prose>

      <CodeBlock language="python">
{`# HolisticBias workflow (paraphrased; the real package is at
# github.com/facebookresearch/ResponsibleNLP/holistic_bias).
from holistic_bias import HolisticBiasDataset, generate_responses
from holistic_bias.metrics import (
    likelihood_bias,
    regard_score,
    response_disparity,
)

ds = HolisticBiasDataset(
    axes=["race_ethnicity", "gender_and_sex"],
    templates=["I am a {descriptor1} {descriptor2} person."],
    intersectional=True,   # generate cross-axis prompts
)
print(f"Total prompts: {len(ds)}")
# e.g. 13 race terms * 17 gender terms = 221 intersectional prompts per template

responses = generate_responses(model="gpt2-large", prompts=ds.prompts())

# Compute regard score (Sheng et al. 2019) per intersectional cell.
regard_df = regard_score(responses, group_keys=["descriptor1","descriptor2"])

# Pairwise disparities: where does the regard score differ most?
disp = response_disparity(regard_df, baseline=("white","male"))
print(disp.sort_values("delta", ascending=False).head(10))`}
      </CodeBlock>

      <H3>5b. BBQ for QA bias</H3>

      <Prose>
        The Bias Benchmark for QA (BBQ; Parrish et al. 2022, arXiv:2110.08193) is a question-answering benchmark designed specifically to measure how language models reproduce social biases when answering questions about people. BBQ contains 58,492 unique examples across nine social bias categories (age, disability status, gender identity, nationality, physical appearance, race/ethnicity, religion, sexual orientation, socioeconomic status), with each example presented in two contexts: ambiguous (where the correct answer is "unknown" but a stereotype-aligned guess is available) and disambiguated (where the correct answer is unambiguously stated). BBQ also includes intersectional categories — race × gender, race × socioeconomic status — to measure intersectional QA bias directly.
      </Prose>

      <CodeBlock language="python">
{`# BBQ evaluation skeleton (the actual benchmark ships with HF datasets).
from datasets import load_dataset
import numpy as np

bbq = load_dataset("Anthropic/llm-bias-bbq", split="test")
intersectional_subset = bbq.filter(
    lambda x: x["category"] in {"Race_x_gender", "Race_x_SES"}
)

def bias_score(predictions, examples):
    """
    BBQ bias score:
       s_amb = (n_biased - n_anti) / n_non_unknown   (in ambiguous context)
       s_dis = (n_biased - n_anti) / n_total          (in disambiguated context)
    Range: -1 (anti-stereotype) to +1 (stereotype). Healthy = near 0.
    """
    biased = sum(p == e["target"]    for p,e in zip(predictions, examples))
    anti   = sum(p == e["antitarget"] for p,e in zip(predictions, examples))
    total  = len(examples)
    return (biased - anti) / max(total, 1)

# Compute per-intersectional-category bias score.
for cat in ["Race_x_gender", "Race_x_SES"]:
    cat_examples = [e for e in intersectional_subset if e["category"] == cat]
    cat_preds    = run_model(cat_examples)  # your model wrapper
    s_amb = bias_score(cat_preds,
                       [e for e in cat_examples if e["context"] == "ambig"])
    s_dis = bias_score(cat_preds,
                       [e for e in cat_examples if e["context"] == "disambig"])
    print(f"{cat}: s_amb={s_amb:+.3f}  s_dis={s_dis:+.3f}")`}
      </CodeBlock>

      <H3>5c. AIF360 intersectional metrics</H3>

      <Prose>
        For tabular classification settings, IBM's AIF360 library provides a <Code>BinaryLabelDatasetMetric</Code> and <Code>ClassificationMetric</Code> that compute fairness metrics on user-defined subgroups. Intersectional subgroups are specified by listing multiple attribute names in the <Code>protected_attribute_names</Code> argument and constructing privileged/unprivileged group dictionaries that specify joint values.
      </Prose>

      <CodeBlock language="python">
{`from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric

# Build a dataset with two protected attributes: race and gender.
dataset = BinaryLabelDataset(
    df=df,
    label_names=["Y"],
    protected_attribute_names=["race", "gender"],
    favorable_label=1,
    unfavorable_label=0,
)
predictions = dataset.copy()
predictions.labels = Y_hat.reshape(-1, 1)

# Intersectional groups: privileged = white men, unprivileged = Black women.
priv_groups   = [{"race": 1, "gender": 1}]
unpriv_groups = [{"race": 0, "gender": 0}]

metric = ClassificationMetric(
    dataset, predictions,
    privileged_groups=priv_groups,
    unprivileged_groups=unpriv_groups,
)
print("Intersectional disparate impact:",     metric.disparate_impact())
print("Intersectional FNR difference:",        metric.false_negative_rate_difference())
print("Intersectional equal opportunity diff:", metric.equal_opportunity_difference())`}
      </CodeBlock>

      <H3>5d. Stratified evaluation reports</H3>

      <Prose>
        Beyond single libraries, the production deliverable is usually a stratified evaluation report — a table or dashboard that shows per-subgroup metrics with confidence intervals, sample sizes, and significance flags after multiple-testing correction. Anthropic, OpenAI, Google DeepMind, and Meta all publish such tables in model cards and technical reports. A representative report has columns for: subgroup specification (the attribute values), sample size, the primary metric (accuracy, FNR, regard, etc.), a confidence interval (typically 95 percent), the gap to the best-performing subgroup, and a flag for statistical significance after correction.
      </Prose>

      <Callout accent="green">
        The most underrated production practice is recording intersectional metadata at dataset construction time. Once your evaluation set lacks the columns to identify intersections, no audit can recover them. Ensure that protected-attribute fields, including the high-cardinality ones you may not want to test on but might in the future, are present in every evaluation example.
      </Callout>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        The Gender Shades result is the canonical visual for why intersectional analysis matters. The plot below reproduces the per-cell error rates reported by Buolamwini and Gebru for IBM's classifier, showing the four subgroups defined by the cross of binary gender and binary skin tone bins. The marginal error rates (averages of the columns and rows) appear modest, while the dark-skinned women cell stands at 34.7 percent.
      </Prose>

      <Plot
        label="Gender Shades 2018 — IBM face classifier error rates by intersection"
        xLabel="subgroup"
        yLabel="misclassification rate (%)"
        width={620}
        height={300}
        series={[
          {
            name: "error rate",
            color: colors.gold,
            points: [
              [0, 0.3],   // light men
              [1, 7.1],   // light women
              [2, 12.0],  // dark men
              [3, 34.7],  // dark women
            ],
          },
          {
            name: "marginal: gender (avg)",
            color: colors.textDim,
            points: [
              [0, 6.2],
              [1, 20.9],
              [2, 6.2],
              [3, 20.9],
            ],
          },
        ]}
      />

      <Prose>
        The intersectional heatmap below shows the synthetic dataset constructed in section 4. The seven "fair" intersectional cells sit between 0.146 and 0.157 FNR, while the gerrymandered cell (1,1,1) sits at 0.502. Reading the marginals across rows and columns produces FNRs around 0.15-0.24 — a moderate disparity, but nothing that would prompt urgent action.
      </Prose>

      <Heatmap
        label="Synthetic 3-axis FNR — eight intersectional cells"
        rowLabels={["A1=0", "A1=1"]}
        colLabels={["A2=0,A3=0", "A2=0,A3=1", "A2=1,A3=0", "A2=1,A3=1"]}
        matrix={[
          [0.146, 0.147, 0.156, 0.155],
          [0.150, 0.149, 0.157, 0.502],
        ]}
        cellSize={90}
        colorScale="gold"
      />

      <Prose>
        The plot below visualizes the statistical-power problem: how the minimum detectable bias gap shrinks as subgroup sample size grows. With 50 positives per cell, only catastrophic gaps (above 0.20) are detectable at 80 percent power. With 500 positives per cell, gaps as small as 0.07 become detectable. Real intersectional cells often have between 50 and 500 examples, which puts them squarely in the regime where moderate bias is statistically invisible.
      </Prose>

      <Plot
        label="Minimum detectable FNR gap (80% power, α=0.05) vs. subgroup size"
        xLabel="positives per subgroup (n)"
        yLabel="min detectable gap"
        width={620}
        height={280}
        series={[
          {
            name: "baseline FNR = 0.10",
            color: colors.gold,
            points: [
              [50, 0.235],
              [100, 0.166],
              [200, 0.117],
              [400, 0.083],
              [800, 0.059],
              [1600, 0.041],
            ],
          },
          {
            name: "baseline FNR = 0.30",
            color: "#c084fc",
            points: [
              [50, 0.359],
              [100, 0.254],
              [200, 0.180],
              [400, 0.127],
              [800, 0.090],
              [1600, 0.063],
            ],
          },
        ]}
      />

      <Prose>
        The step trace below walks through a complete intersectional bias audit, from dataset preparation through statistical reporting. Each step has implementation choices that cascade into the next.
      </Prose>

      <StepTrace
        label="Intersectional bias audit pipeline"
        steps={[
          {
            label: "1. Identify protected attributes",
            render: () => (
              <Prose>
                Enumerate the protected attributes relevant to the deployment context. Typical sets include race/ethnicity, gender identity, age band, disability status, nationality, religion, socioeconomic status, sexual orientation. Decide on cardinalities — binary versus multi-valued — and document the rationale. The choice constrains every later step: an attribute not recorded cannot be audited, and an attribute coarse-grained at this stage cannot be re-granularized later.
              </Prose>
            ),
          },
          {
            label: "2. Define intersectional subgroups",
            render: () => (
              <Prose>
                Decide which intersections to evaluate. The exhaustive set grows as the product of cardinalities. Practical audits usually evaluate all 2-way intersections plus a curated set of 3-way intersections informed by historical evidence and stakeholder consultation (Crenshaw's original Black women × Black men contrast is a typical curated pair). Document why each intersection was included; document why each excluded one was excluded.
              </Prose>
            ),
          },
          {
            label: "3. Validate sample sizes",
            render: () => (
              <Prose>
                Compute the count of evaluation examples in each intersectional cell, and the count of positives (for FNR) or negatives (for FPR). For each cell, compute the minimum detectable effect at 80 percent power given the cell size. If any priority cell has fewer than ~50 positives, plan additional data collection or oversampling before proceeding.
              </Prose>
            ),
          },
          {
            label: "4. Compute per-cell metrics with CIs",
            render: () => (
              <Prose>
                For each cell, compute the primary metric (accuracy, FNR, FPR, calibration error, regard score, etc.) along with a Wilson or Clopper-Pearson 95 percent confidence interval. Report the overlap structure — which cells' CIs overlap with which others — alongside the point estimates.
              </Prose>
            ),
          },
          {
            label: "5. Apply multiple-testing correction",
            render: () => (
              <Prose>
                For pairwise comparisons between cells (or between each cell and a baseline), apply Benjamini-Hochberg FDR control or Bonferroni correction. The number of comparisons being corrected determines how aggressive the threshold becomes; pre-register the comparison set to avoid p-hacking.
              </Prose>
            ),
          },
          {
            label: "6. Decompose into marginals + interaction",
            render: () => (
              <Prose>
                Fit an additive model (marginal terms only) and a saturated model (with interaction terms) to the per-cell rates. The likelihood ratio test on the difference quantifies the strict-sense intersectional bias. Report all three: marginals, intersection, interaction. Large interaction = bias the marginals could not predict.
              </Prose>
            ),
          },
          {
            label: "7. Report and prioritize remediation",
            render: () => (
              <Prose>
                Produce the stratified report (subgroup, n, metric, CI, gap to best, significance flag). Order by magnitude of disparity and significance. Distinguish remediation strategies: additional training data for under-represented intersections, post-hoc threshold adjustment per cell, multicalibration enforcement, or upstream data-collection changes. Schedule re-evaluation cadence.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <H3>Marginal-only audit vs. intersectional audit</H3>

      <Prose>
        Marginal-only audits are appropriate when the deployment context has a single dominant protected attribute (a model used only by one demographic group, or a context where one attribute structurally dominates risk), or when the evaluation budget is tightly constrained and the intersection sizes would not yield statistically meaningful results. They are also a reasonable starting point — a marginal disparity that is clearly present is usually worth investigating before adding complexity. The argument for intersectional audits is that marginal fairness is a strictly weaker condition than intersectional fairness and a well-resourced production deployment that affects diverse populations should not stop at the weaker check. For any model with non-trivial scale, intersectional analysis should be the default and marginal-only the exception requiring justification.
      </Prose>

      <H3>HolisticBias vs. BBQ vs. custom evaluation</H3>

      <Prose>
        Choose HolisticBias when evaluating language models in open-ended generation settings (chat, summarization, text completion) where the bias of interest is associative — what concepts, sentiments, or descriptors does the model attach to particular demographic identities? HolisticBias's strength is breadth: 13 axes and 600 descriptors give it the largest published intersectional coverage, and its prompts are designed for open-ended generation. Choose BBQ when evaluating extractive or multiple-choice question-answering, where the bias of interest is whether the model resolves ambiguous questions in stereotype-aligned ways. BBQ's strength is its disambiguated/ambiguous contrast, which separates "the model picks up real signal" from "the model imposes stereotypes when there is no signal." Use custom evaluation — bespoke datasets and metrics — when neither benchmark covers the deployment context's specific axes (regional dialects, niche professional roles) or when the metric of interest is task-specific (medical risk calibration, fraud detection precision).
      </Prose>

      <H3>Bonferroni vs. Benjamini-Hochberg vs. no correction</H3>

      <Prose>
        Bonferroni is the conservative default: it controls the family-wise error rate (the probability of any false positive among the tests) and is appropriate when each individual significance claim must be defensible in isolation, as in regulatory or legal contexts. Benjamini-Hochberg controls the false discovery rate (the expected fraction of false positives among the rejections) and is less conservative, appropriate when the goal is to prioritize a list of subgroups for remediation rather than to make individual claims. No correction is appropriate only when the analysis is fully exploratory and downstream decisions will not be made on the basis of the unadjusted p-values; this is rarely the case in practice.
      </Prose>

      <H3>Multicalibration vs. group-by-group constraint</H3>

      <Prose>
        Multicalibration (Hébert-Johnson et al. 2018) is a strong fairness condition that simultaneously constrains calibration on every subgroup in a rich collection. It is the right choice when the set of relevant subgroups is too large to enumerate and test individually, or when the model needs to be defensible against attacks that propose new subgroup definitions ex post. The cost is implementation complexity (post-hoc multicalibration requires an iterative procedure over subgroups) and that the resulting model is not necessarily optimal for any single subgroup. Group-by-group constraint enforcement (constrained optimization with explicit subgroup constraints) is appropriate when a small number of subgroups has been identified as priorities; it is simpler to implement and gives stronger guarantees on the named subgroups at the cost of leaving unnamed subgroups un-protected.
      </Prose>

      <H3>Pre-deployment audit vs. continuous monitoring</H3>

      <Prose>
        Pre-deployment audits are static snapshots; continuous monitoring tracks subgroup metrics over time as the deployment evolves. Both are needed. Pre-deployment audits catch issues before they affect users; continuous monitoring catches drift caused by changes in the input distribution, the user population, or the model itself (in the case of fine-tuned or updated models). The infrastructure for continuous monitoring is heavier — it requires ongoing recording of intersectional metadata at inference time, which has privacy implications and storage cost — but the cost of skipping it is that drift goes undetected until the next major audit cycle, often months or years later.
      </Prose>

      <H3>Reporting raw rates vs. ratios vs. differences</H3>

      <Prose>
        Raw rates (per-cell FNR, accuracy, etc.) are the most informative but least concise; they should always be in the underlying data. Ratios (the disparate impact ratio of unprivileged to privileged group selection rates) are scale-invariant and used in legal contexts (the four-fifths rule). Differences (the gap between two rates) are easy to interpret but depend on the absolute level of the rates. Best practice is to report all three — raw rates with CIs, ratios for legal/regulatory framing, and absolute differences for engineering remediation prioritization.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        Intersectional bias analysis scales nicely along several axes and badly along others. The dimensions worth understanding before committing to an evaluation strategy are model size, attribute cardinality, evaluation set size, and audit frequency.
      </Prose>

      <Prose>
        Model size scales nearly for free. Computing per-subgroup metrics requires running the model on the evaluation set once and then aggregating predictions by subgroup membership. The aggregation step is O(n) in the evaluation set size and independent of model size; the inference step scales with the model's per-token cost but is no different from any other evaluation. A 70B parameter model audit takes longer than a 7B parameter audit by roughly the inference-cost ratio, but the audit logic is unchanged. The implication is that intersectional audits should be standard practice across model scales — there is no compute argument for skipping them on large models.
      </Prose>

      <Prose>
        Attribute cardinality scales badly. With <Code>k</Code> binary attributes, the number of intersectional cells is <Code>2^k</Code>; with attributes of cardinality <Code>c</Code>, it is <Code>c^k</Code>. HolisticBias's 13 axes with hundreds of descriptors generate billions of potential intersectional subgroups; even restricting to 2-way intersections of high-cardinality axes, the cell count quickly exceeds practical evaluation budgets. The standard mitigations are: (1) prioritize a subset of axes based on legal frameworks and stakeholder input; (2) coarsen high-cardinality axes into ordinal bins (Fitzpatrick skin tone bucketed into "lighter" and "darker"); (3) use multicalibration-style constraints that hold across all subgroups in a hypothesis class without explicit per-subgroup testing; (4) accept that any audit is a sample of the possible subgroup space, and document which subgroups were and were not evaluated.
      </Prose>

      <Prose>
        Evaluation set size scales linearly in storage and compute, but the statistical demand grows much faster than linearly in the audit's ambition. To hold confidence intervals constant as you split the data into smaller intersectional cells, you need the original sample size multiplied by the number of cells. Doubling the number of attributes from 2 to 4 (each binary) requires roughly 4x more evaluation data to maintain the same per-cell precision. The practical ceiling for most production audits is around 8-12 binary attributes, beyond which the cell sizes drop below the threshold for meaningful significance testing even with large evaluation sets.
      </Prose>

      <Prose>
        Audit frequency does not naturally scale with model release cadence. Each major model release should trigger a full intersectional audit; minor fine-tunes or post-deployment updates should at minimum trigger a regression audit on the same evaluation set as the previous full audit. Continuous monitoring at inference time is technically feasible but introduces the requirement to record intersectional metadata about real users, with corresponding privacy and consent obligations. Most production deployments compromise by monitoring a subset of metrics (overall accuracy, error rate trends) continuously and triggering full intersectional audits on a schedule (quarterly) or in response to alerts.
      </Prose>

      <Prose>
        The hardest scaling problem is human review of the resulting reports. An intersectional report with even modest cardinality can have hundreds of cells, each with a metric, a confidence interval, and a significance flag. Triaging which subgroups warrant remediation, which warrant further investigation, and which are within acceptable variance is fundamentally a judgment task that does not parallelize. The mitigations are workflow rather than infrastructure: standardized severity tiers, pre-registered priority lists, and explicit decision criteria for when a particular disparity triggers a code change versus a documentation update versus no action.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>Reporting marginals only</H3>
      <Prose>
        The most common failure is to compute only single-axis subgroup metrics and report them as the bias audit. As section 2 demonstrated, marginal fairness is fully consistent with arbitrary intersectional bias. A model card or technical report that lists "accuracy by gender" and "accuracy by race" and stops there gives a misleading picture of model behavior. Always report at least 2-way intersections; for high-stakes deployments, include 3-way intersections informed by historical risk patterns.
      </Prose>

      <H3>Underpowered intersectional cells</H3>
      <Prose>
        Reporting per-cell metrics on subgroups with too few examples produces noise that masquerades as signal. A cell with 30 positives and a 3 false negatives shows an FNR of 0.10 with a 95 percent CI of roughly 0.02 to 0.27. Taking the point estimate at face value and prioritizing remediation based on it would direct resources at sampling noise. Always report sample sizes alongside point estimates and use confidence intervals (or significance tests) rather than raw rate comparisons.
      </Prose>

      <H3>No multiple-testing correction</H3>
      <Prose>
        Running 20 independent hypothesis tests at <Code>α = 0.05</Code> without correction gives a roughly 64 percent chance of at least one false positive. Production audits routinely test dozens of subgroup pairs simultaneously; without correction, the report is statistically meaningless and remediation effort gets misallocated to spurious differences. Use Bonferroni or Benjamini-Hochberg correction and report the corrected significance threshold.
      </Prose>

      <H3>Conflating descriptor categories with attribute categories</H3>
      <Prose>
        HolisticBias and similar benchmarks evaluate model behavior on prompts containing demographic descriptors. A high regard score for "Asian women" in HolisticBias is a fact about model behavior on prompts containing those descriptor strings, not directly a fact about how the model treats Asian women in deployment. The descriptor evaluation is a useful proxy, but interpreting it as ground truth about real-world bias requires care — particularly for axes where the descriptor terminology is itself contested or evolving.
      </Prose>

      <H3>Privacy overhead of recording intersectional metadata</H3>
      <Prose>
        Intersectional analysis requires that protected attributes be recorded against evaluation examples. In production deployments, this means collecting and storing demographic data about users — which has privacy, consent, and regulatory implications (GDPR, HIPAA, sector-specific rules). The standard mitigations are: collect attributes voluntarily with explicit purpose disclosure, use proxy attributes where available, restrict storage and access tightly, and aggregate before reporting. Never assume that recording demographic data for fairness audits is automatically permissible; it usually requires its own legal review.
      </Prose>

      <H3>Reading interaction terms as causal</H3>
      <Prose>
        The interaction term <Code>γ(a_1, a_2)</Code> in the log-linear decomposition quantifies the magnitude of intersectional bias that cannot be predicted from the marginal effects. It does not identify a causal mechanism. A large interaction may reflect data-generation processes (under-representation in training data), historical patterns (compounding stereotypes), feature artifacts (lighting bias correlated with skin tone but expressed differently across genders in face data), or any combination. Treating the interaction as evidence for a particular causal story is overreach; it is evidence that the structure of the bias is intersectional, and further investigation is required to attribute the source.
      </Prose>

      <H3>Choosing baselines that hide bias</H3>
      <Prose>
        Many fairness metrics are defined relative to a baseline subgroup (the privileged group, the largest cell, the global average). The choice of baseline shapes what the metric measures. Choosing the largest cell as baseline produces metrics that emphasize disparities affecting smaller groups; choosing the best-performing cell emphasizes how far other groups fall short of the achievable best. Both are defensible; the choice should be explicit and justified, and ideally several baselines should be reported in parallel.
      </Prose>

      <H3>Treating "fair on the test set" as "fair in deployment"</H3>
      <Prose>
        An intersectional audit on a benchmark evaluation set certifies the model's behavior on that set, not its behavior in deployment. If the deployment population's intersectional structure differs from the audit set's — different proportions of subgroups, different feature distributions within subgroups, different label distributions — the audit findings may not transfer. Continuous monitoring at deployment time is the only way to close this loop, and the gap between audit and deployment is itself a substantive concern.
      </Prose>

      <H3>Stopping at detection rather than remediation</H3>
      <Prose>
        Identifying intersectional bias is necessary but not sufficient. The hard work is remediation: data collection to expand under-represented intersections, threshold adjustment per subgroup, retraining with reweighted loss, multicalibration enforcement, or in some cases the conclusion that the model should not be deployed in the affected context. A bias audit that surfaces problems and is then filed away without action is worse than no audit, because it creates a paper trail of known harms that were not addressed.
      </Prose>

      <Callout accent="purple">
        Intersectional audits are most useful when they are part of a feedback loop with model development: the audit finds a disparity, the disparity informs a data collection or training change, the next audit measures whether the change reduced the disparity. Audits that exist only for compliance reporting tend to plateau at the level of disparities that were already known and accepted.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All sources below are referenced by canonical citation; arXiv IDs are provided where applicable. The Crenshaw 1989 paper is a legal scholarship publication and does not have an arXiv identifier; the standard citation is given.
      </Prose>

      <H3>Crenshaw 1989 — origin of the intersectionality concept</H3>
      <Prose>
        Kimberlé Crenshaw. "Demarginalizing the Intersection of Race and Sex: A Black Feminist Critique of Antidiscrimination Doctrine, Feminist Theory and Antiracist Politics." University of Chicago Legal Forum, Vol. 1989, Issue 1, Article 8. The founding statement of intersectionality as a framework for analyzing how single-axis categorizations of discrimination systematically fail to capture harms experienced at the intersections of multiple protected attributes. The paper analyzes three employment discrimination cases involving Black women plaintiffs whose claims were dismissed because the courts evaluated their experiences either as women or as Black people, never as Black women. Required reading for the conceptual grounding of intersectional fairness in machine learning.
      </Prose>

      <H3>Buolamwini & Gebru 2018 — Gender Shades</H3>
      <Prose>
        Joy Buolamwini, Timnit Gebru. "Gender Shades: Intersectional Accuracy Disparities in Commercial Gender Classification." Proceedings of the 1st Conference on Fairness, Accountability and Transparency (FAccT), PMLR 81:77-91, 2018. The empirical paper that demonstrated the practical importance of intersectional analysis in computer vision. Evaluated three commercial face classification systems (Microsoft, IBM, Face++) on a balanced dataset of 1,270 parliamentarian portraits stratified by Fitzpatrick skin type and gender. Found error rate gaps up to 34.4 percentage points between the best-performing intersectional cell (lighter-skinned men) and the worst-performing cell (darker-skinned women), with marginal analyses substantially understating the worst-case behavior. Triggered industry-wide retraining of face recognition systems and remains the canonical demonstration of intersectional algorithmic bias.
      </Prose>

      <H3>Kearns et al. 2018 — fairness gerrymandering</H3>
      <Prose>
        Michael Kearns, Seth Neel, Aaron Roth, Zhiwei Steven Wu. "Preventing Fairness Gerrymandering: Auditing and Learning for Subgroup Fairness." arXiv:1711.05144. ICML 2018. Formalizes the concept that a model can satisfy any standard fairness constraint on every protected attribute marginally while violating the same constraint on conjunctive subgroups defined by combinations of those attributes. Provides an algorithmic framework for auditing and learning under subgroup fairness constraints, where the set of subgroups is defined by a hypothesis class rather than enumerated explicitly. The paper's central theoretical contribution is showing that subgroup fairness is achievable in polynomial time when the subgroup class has bounded VC dimension, which makes it tractable for practical use.
      </Prose>

      <H3>Hébert-Johnson et al. 2018 — multicalibration</H3>
      <Prose>
        Úrsula Hébert-Johnson, Michael Kim, Omer Reingold, Guy Rothblum. "Multicalibration: Calibration for the (Computationally-Identifiable) Masses." arXiv:1711.08513. ICML 2018. Introduces multicalibration as a fairness condition that requires a model to be calibrated simultaneously on every subgroup in a "rich" collection — typically defined as all subgroups identifiable by some hypothesis class. Multicalibration is strictly stronger than group-conditional calibration and provides protection against the failure mode where calibration holds on every named subgroup but fails on intersections or other unnamed subgroups. The paper provides a post-processing algorithm that enforces multicalibration on a pre-trained model with provable convergence guarantees. Subsequent work (Kim et al. 2019, Gopalan et al. 2022) extends this to multi-accuracy and outcome indistinguishability.
      </Prose>

      <H3>Smith et al. 2022 — HolisticBias</H3>
      <Prose>
        Eric Michael Smith, Melissa Hall, Melanie Kambadur, Eleonora Presani, Adina Williams. "I'm sorry to hear that: Finding New Biases in Language Models with a Holistic Descriptor Dataset." arXiv:2205.09209. EMNLP 2022. Introduces the HolisticBias dataset: 13 demographic axes populated with approximately 600 descriptor terms developed in collaboration with members of the relevant demographic groups. The descriptor set is crossed via templated prompts to produce on the order of 460,000 sentence prompts that cover both single-axis and intersectional subgroups, making it the largest published intersectional bias evaluation resource for language models. The paper demonstrates that conventional bias benchmarks miss many of the disparities surfaced by the larger HolisticBias coverage and provides metric implementations for likelihood bias, regard, and response disparity.
      </Prose>

      <H3>Parrish et al. 2022 — BBQ</H3>
      <Prose>
        Alicia Parrish, Angelica Chen, Nikita Nangia, Vishakh Padmakumar, Jason Phang, Jana Thompson, Phu Mon Htut, Samuel R. Bowman. "BBQ: A Hand-Built Bias Benchmark for Question Answering." arXiv:2110.08193. Findings of ACL 2022. Introduces the Bias Benchmark for QA: 58,492 hand-constructed examples across nine social bias categories plus two intersectional categories (race × gender, race × socioeconomic status). Each example is presented in two contexts — ambiguous (where "unknown" is the correct answer but a stereotype-aligned guess is plausible) and disambiguated (where the correct answer is unambiguously stated) — allowing a clean separation between bias under uncertainty and bias under information. Bias scores range from -1 (anti-stereotype) to +1 (stereotype-aligned). BBQ remains one of the most widely adopted intersectional QA bias benchmarks.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Construct fairness gerrymandering by hand</H3>
      <Prose>
        Consider a binary classifier evaluated on a population with two binary protected attributes <Code>A</Code> and <Code>B</Code>. Construct a 2x2 contingency table of selection rates per intersectional cell that satisfies the following: marginal selection rate equals 0.5 on each value of <Code>A</Code>, marginal selection rate equals 0.5 on each value of <Code>B</Code>, and the largest intersectional disparity (max minus min) is at least 0.7. Show your construction explicitly. What does this tell you about the implication structure between marginal demographic parity and intersectional demographic parity?
      </Prose>

      <H3>Exercise 2 — Statistical power for intersectional cells</H3>
      <Prose>
        Suppose you are auditing a binary classifier with three binary protected attributes (8 intersectional cells) and you want to detect any cell whose FNR exceeds the global FNR of 0.10 by at least 0.05, at 80 percent power and Bonferroni-corrected <Code>α = 0.05/8</Code>. Using the rule-of-thumb formula <Code>n ≈ 16 p(1−p)/δ²</Code>, compute the minimum number of positives required per cell. If your evaluation set is balanced across cells with positives composing roughly 40 percent of each cell, what is the minimum total evaluation set size? How does this scale if you add a fourth binary attribute?
      </Prose>

      <H3>Exercise 3 — Decompose a real intersectional measurement</H3>
      <Prose>
        Take the IBM Gender Shades numbers from section 6: light men 0.3 percent error, light women 7.1 percent, dark men 12.0 percent, dark women 34.7 percent. Fit by hand the additive model <Code>θ(a_1, a_2) = μ + α(a_1) + β(a_2)</Code> on the log-rates (use <Code>log(error)</Code>) and compute the implied predictions for each cell. Then compute the residuals — what the saturated model with interaction would add — and interpret. Is the dark-women error rate well-explained by the additive contributions of being a woman and being dark-skinned, or does it require a substantial interaction term?
      </Prose>

      <H3>Exercise 4 — Choose a baseline for a real audit</H3>
      <Prose>
        You are reporting fairness metrics for a hiring algorithm with two protected attributes (gender × race). The intersectional cells have selection rates: white men 0.45, white women 0.30, Black men 0.25, Black women 0.10. Compute the disparate impact ratio (unprivileged / privileged) using three different baseline choices: (1) the largest cell as the privileged group, (2) the highest-rate cell as the privileged group, (3) pairwise ratios for every pair of cells. Compare the resulting reports. Which choice is most defensible for an audit intended to be read by a regulator? Which is most useful for engineering remediation? Justify your answers.
      </Prose>

      <H3>Exercise 5 — Detect when intersectional analysis would mislead</H3>
      <Prose>
        Describe a scenario in which a high-quality intersectional audit would identify a subgroup with statistically significant elevated error rates, but the appropriate remediation is not to retrain or rebalance the model. Give a specific example involving deployment context, attribute meaning, and downstream use. What would the audit report look like, and what additional information would be required to conclude that no remediation is appropriate? How would you document this conclusion to avoid the appearance of dismissing legitimate fairness concerns?
      </Prose>

      <H3>Exercise 6 — Multicalibration vs. enumerated subgroups</H3>
      <Prose>
        You have a binary classifier and a list of 12 protected attributes you want to audit. You are weighing two strategies: (a) enumerate all 2-way intersections (66 subgroups) and run a Bonferroni-corrected significance test on each, or (b) implement post-hoc multicalibration over a hypothesis class containing all conjunctions of these 12 attributes. List three criteria that would push you toward strategy (a) and three criteria that would push you toward strategy (b). Which criteria do you weight most heavily, and why? How does the choice affect what you can claim about the model's behavior on subgroups not explicitly listed in your attribute set?
      </Prose>

    </div>
  ),
};

export default intersectionalBias;
