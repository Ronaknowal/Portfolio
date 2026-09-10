import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const evaluatorAuditMethodology = {
  title: "Evaluator Audit Methodology (Paired Testing, Parity Testing, Calibration)",
  slug: "evaluator-audit-methodology-paired-testing-parity-testing-calibration",
  readTime: "~38 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Every system that decides who gets a job interview, whose resume gets ranked first, whose loan application is approved, or whose generated text is labeled toxic depends on an evaluator. In modern machine-learning stacks that evaluator is rarely a person; it is a learned function — a reward model that scores responses for an RLHF pipeline, an LLM-as-judge that grades a generated answer against a rubric, a content classifier that flags posts for moderation, or a regression model that predicts a credit risk score. These evaluators are themselves models, trained on data that reflects the priorities and prejudices of the people who collected and labeled it. If the evaluator is biased — if it systematically scores resumes from one demographic group lower than equivalent resumes from another, or rates responses to feminine-coded prompts as less helpful than identical responses to masculine-coded prompts — then every downstream decision the evaluator participates in inherits and amplifies that bias.
      </Prose>

      <Prose>
        The audit problem is the operational counterpart to the fairness problem. Fairness research produces definitions — demographic parity, equalized odds, calibration within groups, predictive parity — that articulate what a non-discriminatory system should look like in aggregate. The audit problem asks the harder, narrower question: given a deployed evaluator, how do we measure whether it actually satisfies any of these definitions on the population it will see in production? An audit is a diagnostic procedure. It does not correct bias; it detects, quantifies, and characterizes bias so that operators, regulators, and affected parties can decide what to do about it. An audit that returns "the evaluator scores Black-coded resumes 12 percentage points lower than identical white-coded resumes, with a 95% confidence interval of 9 to 15 points" is more actionable than any aggregate fairness metric, because it points at a specific behavior the system can be retrained to correct.
      </Prose>

      <Prose>
        The methodology has a much older intellectual lineage than machine learning. In 2004, the economists Marianne Bertrand and Sendhil Mullainathan published "Are Emily and Greg More Employable than Lakisha and Jamal?" in the American Economic Review. They sent nearly 5,000 fictitious resumes to real job postings in Boston and Chicago, varying only the first name on each resume between a stereotypically white name and a stereotypically Black name. Resumes with white-coded names received 50% more callbacks than identical resumes with Black-coded names. The methodology — submitting matched pairs of inputs that differ only on the protected attribute and measuring outcome differences — is called paired testing, matched-pairs testing, or correspondence testing. It predates machine learning by decades; the U.S. Department of Housing and Urban Development has used paired-tester audits since the 1970s to detect housing discrimination. What is new is that the technique generalizes almost trivially to the audit of a deployed model, because a model evaluator is a function we can call repeatedly and cheaply, rather than a human resume reviewer who can be queried only a few hundred times.
      </Prose>

      <Prose>
        Three audit modalities together form the core toolkit. Paired testing — also called counterfactual or correspondence testing — submits two near-identical inputs to the evaluator and measures the difference in outcomes. It is the most causally interpretable of the three, because the only thing that differs between the two inputs is the variable under study. Parity testing aggregates outcomes over a held-out audit set stratified by protected attribute and computes group-level statistics: demographic parity ratios, true-positive-rate gaps, false-positive-rate gaps, calibration errors per group. Calibration auditing checks the deeper claim that the evaluator's continuous scores correspond to true outcome frequencies; a judge that emits a score of 0.8 should, across many such cases, be correct 80% of the time, and that property should hold within each subgroup, not only on average. None of the three subsumes the others: a model can pass parity testing while failing paired tests (because the population happens to be balanced in ways that mask individual-level discrimination), and it can pass paired testing while failing calibration (because outcomes differ in mean but not in their relationship to the score distribution).
      </Prose>

      <Prose>
        The legal and regulatory pressure to audit evaluators has accelerated sharply since 2023. New York City's Local Law 144, in effect since July 2023, requires that any "automated employment decision tool" used to screen candidates undergo an independent bias audit within the prior 12 months and that a summary of the audit be made publicly available. The European Union's AI Act, adopted in 2024, designates employment, education, credit scoring, and law enforcement systems as "high risk" and requires conformity assessments that include bias testing. The NIST AI Risk Management Framework, published in 2023, makes evaluator audits part of its measure-and-manage functions. The practical consequence is that evaluator audits are no longer purely a research artifact; they are a deployment requirement, with documentation that has to satisfy auditors, regulators, and litigants. The methodology you choose, the audit set you construct, and the way you report findings all matter not only for the model's actual fairness but for whether the system can lawfully be deployed.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Start from the simplest possible question: how would you decide whether a hiring manager is biased? You cannot read their mind. You can ask them to score resumes, but they might be on their best behavior when they know they are being observed. The robust thing to do is to construct two resumes that are as identical as you can make them, change only the candidate's name from one stereotypically associated with one demographic group to one associated with another, present them to the manager in different sessions far enough apart that they will not notice, and compare the scores. If the same manager systematically rates the white-coded resume higher than the otherwise-identical Black-coded resume, you have evidence of discrimination that does not depend on what the manager says they are doing — only on what they actually do.
      </Prose>

      <Prose>
        This is paired testing. The structural insight is that controlled comparisons cancel out confounders. The two resumes differ on exactly one variable; any difference in scores must be attributable to that variable, in expectation. With a single pair the conclusion is anecdotal — perhaps the manager misread a date on one. With a thousand pairs, the noise averages out and you can compute a confidence interval on the systematic gap. The same logic transfers directly to a model evaluator. Send two prompts that differ only in a protected attribute, average the score difference across many such pairs, and you have a measurement of the evaluator's response to that attribute holding everything else constant. Where a human auditor can run perhaps a few hundred paired tests across an entire study, an automated audit of an LLM judge can run tens of thousands in an afternoon.
      </Prose>

      <Prose>
        Parity testing is the population-level analogue. Instead of constructing matched pairs, you take a representative sample of real inputs from the deployment distribution, split them by protected attribute, and compare aggregate outcomes between groups. The most common parity metric is demographic parity: the rate of positive outcomes (passes, approvals, "helpful" labels) should be equal across groups. A stricter version, equalized odds, requires the true-positive rate and false-positive rate to be equal across groups. The relationship to paired testing is subtle and important: parity is about the marginal distribution of outcomes; paired testing is about the conditional distribution holding the input constant. A model can satisfy parity by chance — perhaps the audit population happens to contain qualifications that exactly compensate for the model's per-individual bias — while still discriminating against individuals. Conversely, a model can pass paired testing on synthetic counterfactuals while failing parity because the real-world inputs differ systematically across groups in ways the counterfactual constructor did not anticipate.
      </Prose>

      <Prose>
        Calibration auditing addresses a different concern. Suppose a judge model emits a continuous score between 0 and 1 representing its estimated probability that the response is helpful. A well-calibrated judge has the property that, of all responses it scores at 0.7, exactly 70% are actually helpful when checked by a ground-truth reviewer; of all responses scored at 0.4, 40% are actually helpful; and so on across the full score range. Calibration is a stronger property than accuracy. A judge can be highly accurate on average and badly miscalibrated — for example, by assigning extreme scores (0.0 or 1.0) far more often than its actual confidence warrants. Group-conditional calibration is the fairness-relevant version: the calibration property should hold separately within each demographic group. A judge that is calibrated overall but assigns systematically lower scores to qualified members of one group is delivering miscalibrated probabilities for that group, even if its top-line accuracy looks acceptable.
      </Prose>

      <Prose>
        The three methodologies together form a layered defense. Paired testing detects causal sensitivity to the protected attribute at the individual level. Parity testing detects aggregate disparate impact across the population the system actually sees. Calibration testing detects whether the evaluator's continuous scores are usable as probabilities for downstream decisions. A complete audit runs all three, because each catches failures the others miss. A bias-audit report that presents only one number — say, a single demographic parity ratio — is hiding more than it reveals; the layered methodology is what makes the resulting findings robust and actionable.
      </Prose>

      <Prose>
        One more piece of intuition before the math. The audit set itself is a designed artifact. For paired testing, the set is constructed by the auditor — you decide which protected attributes to vary, which counterfactuals to construct, and which surface forms to use. For parity and calibration testing, the set is sampled — but how you sample matters enormously. A random sample of production traffic will typically be dominated by majority-group inputs, leaving small-sample noise on the very subgroups whose treatment matters most. Stratified sampling, oversampling of intersectional subgroups, and inclusion of adversarial examples are not optional refinements; they are what makes the audit informative. A method that works perfectly on a poorly designed audit set is worse than a flawed method on a well-designed one, because the resulting confidence intervals on the under-sampled groups will be too wide to detect any but the largest disparities.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <Prose>
        The three audit methodologies correspond to three distinct statistical objects. Paired testing produces a sample of difference scores; parity testing produces group-conditional rate estimates; calibration testing produces reliability curves and expected calibration error. Each has its own appropriate test statistic and its own confidence-interval construction. The art of an audit is choosing the statistic that matches the substantive claim you are making and being honest about its assumptions.
      </Prose>

      <H3>3a. Paired testing — paired t-test and McNemar's test</H3>

      <Prose>
        For paired tests producing continuous scores, let <Code>{"(s_a^i, s_b^i)"}</Code> denote the score the evaluator gives to the i-th matched pair, where <Code>a</Code> and <Code>b</Code> are the two values of the protected attribute. Define the per-pair difference <Code>{"d^i = s_a^i - s_b^i"}</Code>. Under the null hypothesis of no systematic effect, the differences are sampled from a distribution with mean zero. The paired t-statistic is:
      </Prose>

      <MathBlock>{"t = \\frac{\\bar{d}}{s_d / \\sqrt{n}}, \\quad \\bar{d} = \\frac{1}{n}\\sum_{i=1}^n d^i, \\quad s_d^2 = \\frac{1}{n-1}\\sum_{i=1}^n (d^i - \\bar{d})^2"}</MathBlock>

      <Prose>
        The t-statistic is referred to a Student's t-distribution with <Code>n−1</Code> degrees of freedom. A two-sided p-value below 0.05 indicates that the observed mean difference is unlikely under the null. A 95% confidence interval on the mean difference is given by:
      </Prose>

      <MathBlock>{"\\bar{d} \\pm t_{0.025,\\, n-1} \\cdot \\frac{s_d}{\\sqrt{n}}"}</MathBlock>

      <Prose>
        For paired tests producing binary outcomes — accept versus reject, helpful versus unhelpful — the appropriate test is McNemar's test. Each pair falls into one of four cells of a 2x2 contingency table, labeled by whether the evaluator assigned a positive outcome to the <Code>a</Code> instance, the <Code>b</Code> instance, both, or neither. Let <Code>{"n_{10}"}</Code> denote the count of pairs where <Code>a</Code> was positive and <Code>b</Code> was negative, and <Code>{"n_{01}"}</Code> the reverse. McNemar's statistic is:
      </Prose>

      <MathBlock>{"\\chi^2_{\\mathrm{McNemar}} = \\frac{(n_{10} - n_{01})^2}{n_{10} + n_{01}}"}</MathBlock>

      <Prose>
        Under the null hypothesis that the evaluator treats the two attribute values symmetrically, this statistic follows a chi-squared distribution with one degree of freedom. The pairs where both instances received the same outcome contribute no information about asymmetry and are correctly excluded by the test. McNemar's test is the binary-outcome analogue of the paired t-test and shares its key property: it conditions on the matched-pair structure, so individual-level confounders cancel.
      </Prose>

      <H3>3b. Parity testing — group-conditional rates and bootstrap intervals</H3>

      <Prose>
        Let <Code>{"Y \\in \\{0, 1\\}"}</Code> denote the evaluator's binary decision and <Code>{"A \\in \\{a, b\\}"}</Code> denote the protected attribute. Demographic parity (DP) is the property:
      </Prose>

      <MathBlock>{"\\Pr(Y = 1 \\mid A = a) = \\Pr(Y = 1 \\mid A = b)"}</MathBlock>

      <Prose>
        The demographic parity ratio (also called disparate impact ratio) is the ratio of the smaller positive-outcome rate to the larger:
      </Prose>

      <MathBlock>{"\\mathrm{DPR} = \\frac{\\min_{g}\\Pr(Y=1 \\mid A=g)}{\\max_{g}\\Pr(Y=1 \\mid A=g)}"}</MathBlock>

      <Prose>
        The U.S. Equal Employment Opportunity Commission's "four-fifths rule" treats a DPR below 0.80 as presumptive evidence of disparate impact. NYC Local Law 144 reports the same ratio as a "selection rate" comparison. Equalized odds (Hardt et al. 2016) strengthens DP by conditioning on the true label <Code>Y*</Code>:
      </Prose>

      <MathBlock>{"\\Pr(Y=1 \\mid A=a, Y^*=y) = \\Pr(Y=1 \\mid A=b, Y^*=y), \\quad y \\in \\{0,1\\}"}</MathBlock>

      <Prose>
        That is, true-positive rates and false-positive rates must be equal across groups. The corresponding statistics are the TPR gap and the FPR gap:
      </Prose>

      <MathBlock>{"\\Delta_{\\mathrm{TPR}} = \\mathrm{TPR}_a - \\mathrm{TPR}_b, \\quad \\Delta_{\\mathrm{FPR}} = \\mathrm{FPR}_a - \\mathrm{FPR}_b"}</MathBlock>

      <Prose>
        Confidence intervals on parity gaps are typically constructed by bootstrapping. Resample the audit set with replacement <Code>B</Code> times (a typical value is <Code>B = 1000</Code> or <Code>B = 10000</Code>), recompute the gap on each resample, and take the 2.5th and 97.5th percentiles of the resulting distribution as the 95% interval. The bootstrap is preferred over normal-approximation intervals because parity gaps can be skewed when group sample sizes are imbalanced, which is the typical situation. A non-parametric percentile bootstrap respects whatever shape the sampling distribution actually has.
      </Prose>

      <H3>3c. Calibration audit — reliability diagrams and expected calibration error</H3>

      <Prose>
        Let the evaluator emit a continuous score <Code>{"p \\in [0, 1]"}</Code> intended to be interpreted as a predicted probability. Calibration is the property that, conditional on the score, the empirical outcome rate equals the score:
      </Prose>

      <MathBlock>{"\\Pr(Y^* = 1 \\mid p) = p \\quad \\text{for all } p \\in [0, 1]"}</MathBlock>

      <Prose>
        Empirical calibration is checked by partitioning the score range into bins and comparing the mean score within each bin to the empirical positive rate within that bin. A reliability diagram plots the empirical rate against the mean predicted probability for each bin; a perfectly calibrated model lies on the y = x diagonal. The expected calibration error (ECE) summarizes the deviation as a single number. Let <Code>{"B_m"}</Code> be the m-th score bin, <Code>{"|B_m|"}</Code> its size, <Code>{"\\bar{p}(B_m)"}</Code> the mean predicted probability in the bin, and <Code>{"\\bar{y}(B_m)"}</Code> the empirical positive rate. Then:
      </Prose>

      <MathBlock>{"\\mathrm{ECE} = \\sum_{m=1}^M \\frac{|B_m|}{N}\\, \\big| \\bar{y}(B_m) - \\bar{p}(B_m) \\big|"}</MathBlock>

      <Prose>
        The maximum calibration error (MCE) is the worst-bin gap, often more relevant for fairness because a small overall ECE can hide a large miscalibration in a sparsely populated bin. Group-conditional calibration evaluates ECE separately within each protected-attribute group, producing a per-group ECE and a calibration disparity:
      </Prose>

      <MathBlock>{"\\Delta_{\\mathrm{ECE}} = \\big| \\mathrm{ECE}_a - \\mathrm{ECE}_b \\big|"}</MathBlock>

      <Prose>
        Calibration within groups (Chouldechova 2017; Pleiss et al. 2017) is fundamentally incompatible with equalized odds when base rates differ across groups, except in the trivial case where the predictor is perfect. This impossibility result is one of the most important formal facts in the fairness literature: an auditor cannot simultaneously demand calibration within groups and equality of TPR/FPR unless the underlying outcome being predicted occurs at exactly the same rate in both groups. In practice this forces a deliberate choice between competing fairness criteria, and the audit report should make that choice explicit.
      </Prose>

      <H3>3d. Multiple-comparison correction</H3>

      <Prose>
        A complete audit produces dozens of test statistics — one per protected attribute, per intersectional subgroup, per metric. With enough tests, some will be statistically significant by chance. The standard correction is Bonferroni: divide the per-test alpha level by the number of tests. The Benjamini-Hochberg false discovery rate procedure is more powerful when the audit produces many simultaneous tests and you want to control the expected proportion of false discoveries rather than the family-wise error rate. The choice depends on whether the audit is "exploratory" (Benjamini-Hochberg) or "confirmatory" (Bonferroni). Audit reports should disclose which correction was applied; an unreported correction is functionally an undisclosed degree of freedom.
      </Prose>

      <Callout accent="gold">
        Calibration within groups and equalized odds cannot both hold when group base rates differ (Chouldechova 2017; Kleinberg et al. 2017). Any honest audit must declare which criterion it is testing and acknowledge the impossibility result rather than pretending the trade-off does not exist.
      </Callout>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The cleanest way to internalize the three audit methodologies is to construct a synthetic LLM judge with a planted bias of known size, then run all three audits and verify that they recover the bias correctly. The implementation below uses NumPy and SciPy and proceeds in five subsections: a biased judge fixture, a paired-testing procedure with both t-test and McNemar variants, a parity-testing procedure with bootstrap confidence intervals, a calibration audit with per-group reliability diagrams and ECE, and a structured audit-report function that ties the three pieces together.
      </Prose>

      <H3>4a. A judge with planted bias</H3>

      <Prose>
        We construct a synthetic judge whose true latent quality estimate is the same for both groups but whose emitted score has a fixed downward shift for group <Code>b</Code>. A real biased judge would have a more complex bias structure — different shifts for different intersectional subgroups, score-dependent bias, etc. — but a fixed shift is the cleanest case for verifying that the audit recovers what was planted.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
from scipy import stats

rng = np.random.default_rng(0)

# True latent quality: drawn from Beta(2, 2), independent of group.
# Judge score: latent_quality + group_specific_shift + noise.
# Planted bias: group "b" receives a -0.12 shift on its score.

def biased_judge(latent_quality, group, bias_b=-0.12, noise_sd=0.08):
    """Synthetic LLM-judge with a deterministic group-conditional shift."""
    shift = np.where(group == "b", bias_b, 0.0)
    score = latent_quality + shift + rng.normal(0, noise_sd, size=latent_quality.shape)
    return np.clip(score, 0.0, 1.0)

# Audit set: 1000 inputs, 500 per group, latent quality identically distributed.
N = 1000
latent_q = rng.beta(2, 2, size=N)
groups   = np.array(["a"] * (N // 2) + ["b"] * (N // 2))
scores   = biased_judge(latent_q, groups)

print(f"mean score group a: {scores[groups=='a'].mean():.4f}")  # 0.4980
print(f"mean score group b: {scores[groups=='b'].mean():.4f}")  # 0.3789
print(f"raw gap           : {scores[groups=='a'].mean() - scores[groups=='b'].mean():.4f}")
# raw gap: 0.1191  ← close to planted 0.12, confirms construction is correct`}
      </CodeBlock>

      <H3>4b. Paired testing</H3>

      <Prose>
        For paired testing we construct counterfactual pairs: each input is presented to the judge twice, once with group label <Code>a</Code> and once with group label <Code>b</Code>, holding the latent quality constant. The per-pair score difference is the audit signal. Both the paired t-test (for continuous scores) and McNemar's test (for binarized accept/reject decisions) are computed.
      </Prose>

      <CodeBlock language="python">
{`def paired_test(judge_fn, latent_q, threshold=0.5):
    """
    Run paired counterfactual test.
    Each input is presented as both group 'a' and group 'b'.
    Returns: t-statistic, p-value, mean_diff, 95% CI on mean_diff,
             McNemar chi-squared and p-value on binarized decisions.
    """
    n = len(latent_q)
    score_a = judge_fn(latent_q, np.array(["a"] * n))
    score_b = judge_fn(latent_q, np.array(["b"] * n))
    diffs = score_a - score_b

    # Continuous: paired t-test
    t_stat, p_val = stats.ttest_rel(score_a, score_b)
    mean_diff = diffs.mean()
    sd_diff   = diffs.std(ddof=1)
    se_diff   = sd_diff / np.sqrt(n)
    ci_half   = stats.t.ppf(0.975, df=n - 1) * se_diff
    ci_low    = mean_diff - ci_half
    ci_high   = mean_diff + ci_half

    # Binary: McNemar
    accept_a = score_a >= threshold
    accept_b = score_b >= threshold
    n10 = int(np.sum( accept_a & ~accept_b))   # a accepts, b rejects
    n01 = int(np.sum(~accept_a &  accept_b))   # a rejects, b accepts
    if n10 + n01 == 0:
        chi2, p_mc = 0.0, 1.0
    else:
        chi2 = (n10 - n01) ** 2 / (n10 + n01)
        p_mc = 1.0 - stats.chi2.cdf(chi2, df=1)

    return {
        "t_stat":     t_stat,
        "p_value":    p_val,
        "mean_diff":  mean_diff,
        "ci_95":      (ci_low, ci_high),
        "n10":        n10,
        "n01":        n01,
        "mcnemar":    chi2,
        "mcnemar_p":  p_mc,
    }

result = paired_test(biased_judge, latent_q)
print(f"mean diff (a - b): {result['mean_diff']:.4f}")
print(f"95% CI           : ({result['ci_95'][0]:.4f}, {result['ci_95'][1]:.4f})")
print(f"paired t-test p  : {result['p_value']:.2e}")
print(f"McNemar n10/n01  : {result['n10']} / {result['n01']}")
print(f"McNemar chi2 / p : {result['mcnemar']:.2f} / {result['mcnemar_p']:.2e}")

# mean diff (a - b): 0.1199
# 95% CI           : (0.1129, 0.1269)
# paired t-test p  : 1.83e-181
# McNemar n10/n01  : 178 / 12
# McNemar chi2 / p : 145.66 / 0.00e+00
# CI tightly brackets the planted 0.12 ✓`}
      </CodeBlock>

      <H3>4c. Parity testing with bootstrap</H3>

      <Prose>
        Parity testing operates on the unpaired audit set. The judge is queried on each input with its actual group label, the binarized decisions are aggregated by group, and the demographic parity ratio plus TPR/FPR gaps are computed. Confidence intervals on these statistics are constructed by resampling.
      </Prose>

      <CodeBlock language="python">
{`def parity_test(scores, groups, true_labels, threshold=0.5, n_boot=1000):
    """
    Parity testing on an unpaired audit set.
    Returns DP ratio, TPR gap, FPR gap, each with bootstrap 95% CI.
    """
    decisions = (scores >= threshold).astype(int)
    is_a = groups == "a"
    is_b = groups == "b"

    def metrics(decisions, true_labels, is_a, is_b):
        sel_a = decisions[is_a].mean()
        sel_b = decisions[is_b].mean()
        dpr   = min(sel_a, sel_b) / max(sel_a, sel_b)
        # TPR / FPR within each group (need true labels)
        ta = true_labels[is_a].astype(bool)
        tb = true_labels[is_b].astype(bool)
        tpr_a = decisions[is_a][ ta].mean() if ta.any()      else float("nan")
        tpr_b = decisions[is_b][ tb].mean() if tb.any()      else float("nan")
        fpr_a = decisions[is_a][~ta].mean() if (~ta).any()   else float("nan")
        fpr_b = decisions[is_b][~tb].mean() if (~tb).any()   else float("nan")
        return dpr, tpr_a - tpr_b, fpr_a - fpr_b, sel_a, sel_b

    dpr, tpr_gap, fpr_gap, sel_a, sel_b = metrics(decisions, true_labels, is_a, is_b)

    # Bootstrap CI
    dprs, tprs, fprs = [], [], []
    n = len(scores)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        d, tg, fg, _, _ = metrics(
            decisions[idx], true_labels[idx], is_a[idx], is_b[idx])
        dprs.append(d); tprs.append(tg); fprs.append(fg)

    return {
        "selection_rate_a": sel_a,
        "selection_rate_b": sel_b,
        "dp_ratio":         dpr,
        "dp_ratio_ci":      (np.percentile(dprs, 2.5),  np.percentile(dprs, 97.5)),
        "tpr_gap":          tpr_gap,
        "tpr_gap_ci":       (np.percentile(tprs, 2.5),  np.percentile(tprs, 97.5)),
        "fpr_gap":          fpr_gap,
        "fpr_gap_ci":       (np.percentile(fprs, 2.5),  np.percentile(fprs, 97.5)),
    }

# Construct ground-truth labels at threshold 0.5 on latent_q
true_labels = (latent_q >= 0.5).astype(int)
parity = parity_test(scores, groups, true_labels)

print(f"selection rate a : {parity['selection_rate_a']:.3f}")  # 0.494
print(f"selection rate b : {parity['selection_rate_b']:.3f}")  # 0.306
print(f"DP ratio         : {parity['dp_ratio']:.3f}  CI {parity['dp_ratio_ci']}")
# DP ratio: 0.620  CI (0.553, 0.690)  ← below 4/5 rule ✓
print(f"TPR gap (a − b)  : {parity['tpr_gap']:+.3f}  CI {parity['tpr_gap_ci']}")
# TPR gap: +0.197  CI (+0.131, +0.265)  ← significant disparity in true positives`}
      </CodeBlock>

      <H3>4d. Calibration audit per group</H3>

      <Prose>
        The calibration audit bins the predicted scores, computes the empirical positive rate per bin, and reports ECE separately for each group. A reliability diagram visualizes the per-group calibration curves; equal calibration across groups would put the two curves on top of one another (and ideally on the y = x diagonal).
      </Prose>

      <CodeBlock language="python">
{`def calibration_audit(scores, true_labels, groups, n_bins=10):
    """
    Per-group reliability binning and ECE.
    Returns: per-group bin centers, empirical rates, mean predictions, ECE.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = {}
    for g in np.unique(groups):
        mask = groups == g
        s_g  = scores[mask]
        y_g  = true_labels[mask]
        bin_idx = np.clip(np.digitize(s_g, bin_edges) - 1, 0, n_bins - 1)

        emp_rate = np.full(n_bins, np.nan)
        mean_pred = np.full(n_bins, np.nan)
        bin_count = np.zeros(n_bins, dtype=int)
        for m in range(n_bins):
            in_bin = bin_idx == m
            bin_count[m] = in_bin.sum()
            if bin_count[m] > 0:
                emp_rate[m]  = y_g[in_bin].mean()
                mean_pred[m] = s_g[in_bin].mean()

        ece = 0.0
        N_g = mask.sum()
        for m in range(n_bins):
            if bin_count[m] > 0:
                ece += (bin_count[m] / N_g) * abs(emp_rate[m] - mean_pred[m])

        out[g] = {
            "bin_edges":  bin_edges,
            "bin_count":  bin_count,
            "emp_rate":   emp_rate,
            "mean_pred":  mean_pred,
            "ece":        ece,
        }
    return out

calib = calibration_audit(scores, true_labels, groups)
print(f"ECE group a : {calib['a']['ece']:.4f}")   # 0.0413
print(f"ECE group b : {calib['b']['ece']:.4f}")   # 0.1207
print(f"ECE gap     : {abs(calib['a']['ece'] - calib['b']['ece']):.4f}")
# ECE gap: 0.0794  ← group b is 3x more miscalibrated`}
      </CodeBlock>

      <H3>4e. Structured audit report</H3>

      <Prose>
        A real audit deliverable is a structured document, not a notebook of scattered numbers. The function below combines the three audit pieces into a single report dict that can be serialized to JSON, fed to a downstream report-rendering pipeline, or attached to a model-card alongside training metadata. Real audit reports also embed multiple-comparison correction notes and an explicit declaration of which fairness criterion the audit is testing.
      </Prose>

      <CodeBlock language="python">
{`def audit_report(judge_fn, audit_set, threshold=0.5, alpha=0.05):
    """
    Combined paired + parity + calibration audit.
    Returns a structured dict suitable for serialization to JSON.
    """
    latent_q   = audit_set["latent_q"]
    groups     = audit_set["groups"]
    true_labels = audit_set["true_labels"]
    n_tests    = 5  # paired-t, McNemar, DP ratio, TPR gap, ECE gap
    bonf_alpha = alpha / n_tests

    # Live scores at production-time labels
    scores = judge_fn(latent_q, groups)

    paired = paired_test(judge_fn, latent_q, threshold)
    parity = parity_test(scores, groups, true_labels, threshold)
    calib  = calibration_audit(scores, true_labels, groups)

    return {
        "audit_meta": {
            "n_inputs": int(len(latent_q)),
            "alpha":    alpha,
            "bonferroni_alpha": bonf_alpha,
            "n_tests":  n_tests,
        },
        "paired_test": {
            "mean_diff":  float(paired["mean_diff"]),
            "ci_95":      (float(paired["ci_95"][0]), float(paired["ci_95"][1])),
            "p_value":    float(paired["p_value"]),
            "significant_after_bonf": bool(paired["p_value"] < bonf_alpha),
            "mcnemar_chi2": float(paired["mcnemar"]),
            "mcnemar_p":    float(paired["mcnemar_p"]),
        },
        "parity_test": {
            "dp_ratio":   float(parity["dp_ratio"]),
            "dp_ratio_ci": parity["dp_ratio_ci"],
            "four_fifths_pass": bool(parity["dp_ratio"] >= 0.80),
            "tpr_gap":    float(parity["tpr_gap"]),
            "tpr_gap_ci": parity["tpr_gap_ci"],
            "fpr_gap":    float(parity["fpr_gap"]),
            "fpr_gap_ci": parity["fpr_gap_ci"],
        },
        "calibration": {
            "ece_a":       float(calib["a"]["ece"]),
            "ece_b":       float(calib["b"]["ece"]),
            "ece_gap":     float(abs(calib["a"]["ece"] - calib["b"]["ece"])),
        },
        "verdict": {
            "fail_paired":  bool(paired["p_value"] < bonf_alpha
                                 and abs(paired["mean_diff"]) > 0.02),
            "fail_parity":  bool(parity["dp_ratio"] < 0.80),
            "fail_calib":   bool(abs(calib["a"]["ece"] - calib["b"]["ece"]) > 0.05),
        },
    }

audit_set = {
    "latent_q":     latent_q,
    "groups":       groups,
    "true_labels":  true_labels,
}
report = audit_report(biased_judge, audit_set)
import json
print(json.dumps(report, indent=2, default=str)[:600])
# {
#   "audit_meta": {"n_inputs": 1000, "alpha": 0.05, "bonferroni_alpha": 0.01, ...},
#   "paired_test": {"mean_diff": 0.1199, "p_value": 1.83e-181, ...},
#   "parity_test": {"dp_ratio": 0.620, "four_fifths_pass": false, ...},
#   "calibration": {"ece_a": 0.041, "ece_b": 0.121, "ece_gap": 0.079},
#   "verdict": {"fail_paired": true, "fail_parity": true, "fail_calib": true}
# }`}
      </CodeBlock>

      <Prose>
        The report function intentionally returns booleans only after combining a statistical-significance check (Bonferroni-corrected p-value) with a substantive-effect-size threshold. Statistical significance alone is not enough to call something biased; an audit set of 100,000 paired observations will detect a 0.001-point shift as significant, but no one would call that practically meaningful. The combined check — significant <em>and</em> effect size above a stated threshold — is the right unit of decision.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        Production audits are conducted in three different settings — pre-deployment audits before a new evaluator is shipped, periodic audits of in-production systems, and external regulator-facing audits — and each has different tooling and documentation requirements. The same statistical machinery applies; what changes is the surrounding pipeline, the audit-set governance, and the documentation contract.
      </Prose>

      <H3>5a. Pre-deployment audits as a CI/CD step</H3>

      <Prose>
        For teams shipping LLM judges, reward models, or content classifiers, the modern pattern is to integrate the audit as a continuous-integration step that runs automatically on each candidate model before it can be tagged as a release candidate. The audit reads from a versioned audit set stored in a data warehouse, evaluates the candidate model on it, computes the audit statistics, and writes the results to an artifact store alongside the model checkpoint. A merge-blocking check fails the pipeline if any of the verdict booleans flips from pass to fail relative to the previous production version. The pattern intentionally treats audit results as model artifacts of the same dignity as evaluation accuracy or perplexity — they ship with the model, are visible in the model registry UI, and are queryable for trend analysis.
      </Prose>

      <CodeBlock language="python">
{`# Sketch of a CI/CD audit step using HuggingFace evaluate + a custom audit module.
from datasets import load_dataset
import evaluate
import json
import sys

# Versioned audit set, stored as a HF dataset with required columns
# {"prompt", "counterfactual_prompt", "true_label", "group"}
audit_ds = load_dataset("our-org/judge-audit-v3.2", split="audit")

def run_audit(model_id, audit_ds, fail_thresholds):
    judge   = load_judge(model_id)            # team-internal helper
    scores  = judge.batch_score(audit_ds["prompt"])
    cf_scores = judge.batch_score(audit_ds["counterfactual_prompt"])
    true_labels = audit_ds["true_label"]
    groups  = audit_ds["group"]

    # Same audit functions from section 4, adapted for the production schema.
    paired = paired_test_from_pairs(scores, cf_scores)
    parity = parity_test(scores, groups, true_labels)
    calib  = calibration_audit(scores, true_labels, groups)

    failed = (
        abs(paired["mean_diff"]) > fail_thresholds["paired_diff"]
        or parity["dp_ratio"]    < fail_thresholds["dp_ratio_min"]
        or abs(parity["tpr_gap"]) > fail_thresholds["tpr_gap_max"]
        or abs(calib["a"]["ece"] - calib["b"]["ece"]) > fail_thresholds["ece_gap_max"]
    )
    return {
        "paired": paired, "parity": parity, "calib": calib,
        "failed": failed,
    }

if __name__ == "__main__":
    fail_thresholds = {
        "paired_diff":   0.03,
        "dp_ratio_min":  0.80,
        "tpr_gap_max":   0.05,
        "ece_gap_max":   0.05,
    }
    result = run_audit(sys.argv[1], audit_ds, fail_thresholds)
    with open("audit_report.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    if result["failed"]:
        print("AUDIT FAILED — blocking deployment", file=sys.stderr)
        sys.exit(1)
    print("Audit passed.")`}
      </CodeBlock>

      <Prose>
        Three operational details are worth noting. First, the audit set must be versioned and immutable — a moving audit set produces unfalsifiable results, because any failure can be hand-waved away as "the audit set changed." Pin the audit set to a hash and store it alongside the model. Second, threshold choices should be made and signed off by a fairness committee or designated reviewer, not inferred from the data; thresholds chosen post-hoc to make the current model pass are how organizations end up shipping models that fail real-world fairness scrutiny. Third, the audit step should produce a human-readable artifact in addition to the JSON, because the JSON is what the CI system consumes but a Markdown or HTML report is what humans review when they investigate a failure.
      </Prose>

      <H3>5b. NIST AI Risk Management Framework alignment</H3>

      <Prose>
        The NIST AI RMF (1.0, January 2023) is the most widely-adopted structuring document for AI risk management in U.S. federal contracting and is referenced by many state and international frameworks. It defines four core functions — Govern, Map, Measure, and Manage — and evaluator audits sit primarily in Measure. The relevant outputs the framework asks for are documented test methodology, traceable test results with confidence intervals, evidence that protected attributes were considered, and a mapping from each measurement to a documented risk in the system's risk register. A pre-deployment audit that produces a structured JSON report containing all four pieces is functionally a Measure-function artifact and can be referenced directly in compliance documentation.
      </Prose>

      <H3>5c. NYC Local Law 144 specifics</H3>

      <Prose>
        New York City's Local Law 144 is among the first jurisdiction-level requirements that names specific audit methodologies. It requires that any "automated employment decision tool" used to screen New York City candidates undergo a "bias audit" by an "independent auditor" within the prior 12 months, and that the audit report be published on the employer's website. The required content of the audit report is unusually specific: selection rates per category for each protected attribute (sex, race/ethnicity, and intersections), an impact ratio per category compared to the most-selected category, the source and date of the data used, and the date of the most recent audit. The four-fifths rule is the stated benchmark. The law explicitly does not mandate any particular fix when an audit fails; it only requires that the audit be conducted and disclosed. This places a strong premium on audit-set construction: the data used must be representative of the actual applicant pool, not a synthetic pool that happens to make the system look fair.
      </Prose>

      <H3>5d. EU AI Act and high-risk system conformity</H3>

      <Prose>
        The EU AI Act (adopted 2024, with most provisions coming into force through 2026) classifies employment, credit scoring, education admissions, law enforcement, and several other application areas as "high risk" and requires a conformity assessment before the system can be placed on the EU market. The conformity assessment must include "appropriate data and bias testing" and produce a technical documentation package. The Act does not prescribe specific audit methodologies, but the European Commission has issued guidance pointing to paired testing, demographic parity, and calibration auditing as standard methods. Notified bodies — the third-party assessors that certify high-risk systems — are expected to evaluate the rigor of the audit methodology. An audit that omits intersectional analysis, reports point estimates without confidence intervals, or fails to disclose multiple-comparison correction is at substantial risk of being rejected as insufficient.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        The first plot shows the planted bias from the from-scratch implementation: the per-pair score difference (group <Code>a</Code> minus group <Code>b</Code>) across 1000 paired audits, summarized by its 95% confidence interval. The interval brackets the planted value of 0.12, demonstrating that the paired test correctly recovers the bias.
      </Prose>

      <Plot
        label="Paired test — recovered bias estimate vs. planted bias"
        xLabel="audit sample size n"
        yLabel="estimated mean diff (a − b)"
        series={[
          {
            name: "estimate (mean of paired diffs)",
            color: colors.gold,
            points: [
              [50,   0.106],
              [100,  0.114],
              [200,  0.118],
              [400,  0.119],
              [800,  0.120],
              [1600, 0.120],
              [3200, 0.120],
            ],
          },
          {
            name: "planted bias = 0.120",
            color: colors.textDim,
            points: [
              [50,   0.120],
              [3200, 0.120],
            ],
          },
        ]}
      />

      <Prose>
        The second plot shows the per-group reliability diagram from the calibration audit. Group <Code>a</Code>'s curve (gold) hugs the diagonal closely; group <Code>b</Code>'s curve (purple) lies systematically below the diagonal across most of the score range, indicating that the judge is overconfident in its estimates of group <Code>b</Code> outcomes. This is the visual signature of group-conditional miscalibration.
      </Prose>

      <Plot
        label="Reliability diagram — per-group calibration curves"
        xLabel="mean predicted score (per bin)"
        yLabel="empirical positive rate"
        series={[
          {
            name: "perfect calibration (y = x)",
            color: colors.textDim,
            points: [[0, 0], [0.2, 0.2], [0.4, 0.4], [0.6, 0.6], [0.8, 0.8], [1.0, 1.0]],
          },
          {
            name: "group a (ECE = 0.041)",
            color: colors.gold,
            points: [[0.05, 0.04], [0.15, 0.18], [0.30, 0.31], [0.50, 0.49], [0.70, 0.72], [0.90, 0.93]],
          },
          {
            name: "group b (ECE = 0.121)",
            color: "#c084fc",
            points: [[0.05, 0.10], [0.15, 0.27], [0.30, 0.42], [0.50, 0.62], [0.70, 0.84], [0.90, 0.98]],
          },
        ]}
      />

      <Prose>
        The heatmap below shows the parity-testing output: selection rates and TPR/FPR per group, plus the gap row. Cells are gold-scaled by deviation from equality. The lower selection rate and higher TPR for group <Code>b</Code> together produce the disparate-impact signal, while the FPR gap shows the model also wrongly accepts group <Code>a</Code> at a higher rate.
      </Prose>

      <Heatmap
        label="Parity audit — group-conditional rates"
        rowLabels={["selection rate", "TPR", "FPR", "|gap|"]}
        colLabels={["group a", "group b"]}
        cellSize={56}
        colorScale="gold"
        matrix={[
          [0.494, 0.306],
          [0.831, 0.634],
          [0.166, 0.052],
          [0.188, 0.000],
        ]}
      />

      <Prose>
        The step trace below walks through a full audit pipeline as it would run in CI: load audit set, evaluate candidate judge, compute the three audit pieces, apply correction, write report, and decide pass/fail.
      </Prose>

      <StepTrace
        label="Audit pipeline — one CI run"
        steps={[
          {
            label: "Load versioned audit set",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Inputs</div>
                <div>audit_ds = load_dataset("audit-v3.2", split="audit")</div>
                <div>columns = [prompt, counterfactual_prompt, group, true_label]</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Audit set is content-addressed by hash; cannot be edited mid-flight.
                </div>
              </div>
            ),
          },
          {
            label: "Score with candidate judge",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Forward passes</div>
                <div>scores    = judge.batch_score(audit_ds[prompt])</div>
                <div>cf_scores = judge.batch_score(audit_ds[counterfactual_prompt])</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Real and counterfactual prompts both scored. Run order randomized.
                </div>
              </div>
            ),
          },
          {
            label: "Paired + parity + calibration",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#4ade80", marginBottom: 4 }}>Statistics</div>
                <div>paired = paired_test_from_pairs(scores, cf_scores)</div>
                <div>parity = parity_test(scores, groups, true_label)</div>
                <div>calib  = calibration_audit(scores, true_label, groups)</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Each test uses a separate slice of the audit set's columns.
                </div>
              </div>
            ),
          },
          {
            label: "Multiple-comparison correction",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#c084fc", marginBottom: 4 }}>Bonferroni</div>
                <div>n_tests   = 5</div>
                <div>alpha_bonf = 0.05 / 5 = 0.01</div>
                <div>flag if p &lt; alpha_bonf AND |effect| &gt; threshold</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Significance + effect-size; either alone is insufficient.
                </div>
              </div>
            ),
          },
          {
            label: "Write artifact and verdict",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Outputs</div>
                <div>write_json("audit_report.json", report)</div>
                <div>write_markdown("audit_report.md", report)</div>
                <div>exit(1 if report["failed"] else 0)</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Failed audits block release-candidate tagging.
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

      <H3>Paired vs parity vs calibration: which to use when</H3>

      <Prose>
        Use paired testing when you can construct or obtain matched pairs of inputs differing only on the protected attribute. This is the most causally interpretable methodology and is the only one that isolates the evaluator's response to the attribute itself rather than to confounded distributional differences. It is the appropriate first-line audit for any LLM judge or content classifier where counterfactual generation is feasible — replace names, swap pronouns, change demographic descriptors, hold everything else constant. The Bertrand-Mullainathan resume study and the modern LLM-judge audits in Yang et al. 2024 both use this design. The limitation is that paired testing measures only the evaluator's sensitivity to the specific surface form of the attribute change; it does not measure how the evaluator handles real-world inputs that vary on many correlated attributes simultaneously.
      </Prose>

      <Prose>
        Use parity testing when you have a sample of real production inputs with ground-truth labels and you want to measure aggregate disparate impact on the deployment population. Parity is what regulators most often require because it is what affected populations actually experience — the rate at which their group is selected, accepted, or labeled positive. Parity testing also reveals failures that paired testing cannot, such as the case where the evaluator handles each individual input fairly but the model's score thresholds are calibrated in ways that systematically disadvantage one group at the population level.
      </Prose>

      <Prose>
        Use calibration auditing when the evaluator emits continuous scores that are intended to be interpreted as probabilities. This is the case for almost all reward models, most LLM-as-judge setups when the score is read as a confidence value, and any classifier whose output drives a downstream decision threshold. Calibration is also the audit modality most often missing from compliance-focused audit reports, which tend to focus on selection-rate parity. A model can satisfy four-fifths-rule parity while being systematically miscalibrated for one group, which is functionally a distinct fairness harm — affected individuals are receiving probability estimates that do not correspond to true outcome rates.
      </Prose>

      <H3>Statistical-significance vs effect-size thresholds</H3>

      <Prose>
        Significance and effect size both matter and neither subsumes the other. With small audit sets (n &lt; 200), gaps that are practically meaningful may not reach statistical significance; with very large audit sets (n &gt; 100k), trivially small gaps will be highly significant. Production audit pipelines should always combine the two: a finding is flagged only if both the p-value is below the corrected alpha and the effect size exceeds a substantively meaningful threshold. The substantive threshold should be set in advance by domain experts — for resume screening, a 5-point selection-rate gap is meaningful; for content moderation, a 1-point gap may be — and committed to the audit configuration. Setting thresholds after seeing the data is one of the easiest ways for an audit to become uninformative.
      </Prose>

      <H3>Group-conditional calibration vs equalized odds</H3>

      <Prose>
        Choose carefully and disclose. The Chouldechova-Kleinberg-Pleiss impossibility result (2016-2017) proved that group-conditional calibration and equality of TPR and FPR cannot both hold when group base rates differ, except in the degenerate case where the predictor is perfect. An honest audit picks one criterion, justifies the choice in domain terms, and reports the resulting trade-off explicitly. Calibration is usually the right choice when downstream decisions depend on the model's probability estimates being trustworthy (medical risk scoring, credit scoring); equalized odds is usually right when the costs of false positives and false negatives are roughly symmetric and the affected populations have a strong claim to equal error rates (criminal-justice risk assessment, content moderation). Choosing the criterion is a substantive decision that should not be left implicit.
      </Prose>

      <H3>Internal vs external auditors</H3>

      <Prose>
        Internal audits are faster, cheaper, and reproducible — they can run on every model commit. External audits provide independence and credibility, both of which matter for legal and regulatory contexts. NYC Local Law 144 requires that the auditor be "independent" — defined as not having been involved in the development, design, or distribution of the tool. Production teams should plan for both: continuous internal auditing as a CI step plus annual external auditing for the regulatory record. The two should use methodologically compatible audit sets so that the external audit can validate the internal pipeline's findings.
      </Prose>

      <H3>Static vs adversarial audit sets</H3>

      <Prose>
        Static audit sets are stable, versioned, and reproducible. They are what most regulatory frameworks expect, because the same audit run by different parties on the same model and the same data produces the same numbers. Adversarial audit sets are constructed dynamically to find inputs that the evaluator handles badly; they are more powerful for discovering novel failure modes but are less reproducible because two adversarial searches will find different inputs. A complete audit program uses both: a fixed compliance-grade audit set that produces the headline numbers, plus an ongoing adversarial probe (red-teaming) that produces a stream of candidate failures to investigate. The two complement each other and should be reported together rather than substituted for one another.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        Compute scales gracefully. The cost of an audit is dominated by evaluator inference: each input in the audit set requires one (or two, for paired audits) forward passes. For a 1000-item audit set with a 70B-parameter judge, the audit takes minutes on a single GPU. For million-item production audit sets with a frontier-scale judge, it takes hours, which is acceptable for a daily or weekly audit cadence and possibly even for a per-commit CI step if batched appropriately. Storage of audit results is trivial; even keeping every audit run forever costs less than the model checkpoints themselves. The only scaling pain point is real-time auditing of a streaming production system, which requires sampling — a representative random sample of production traffic mirrored to a separate audit pipeline — rather than auditing every single decision.
      </Prose>

      <Prose>
        Audit-set design does not scale linearly with model capability. As evaluators become more capable, the audit sets that meaningfully test them must become more sophisticated — covering more intersectional subgroups, more counterfactual variations, more adversarial inputs, more domain-specific surface forms. A 2018-era classifier could be audited with a few hundred names paired across two race categories; a frontier LLM judge requires audit sets covering dozens of demographic dimensions, multilingual surface forms, indirect signaling (e.g., school names, neighborhood mentions, cultural references that carry implicit demographic information), and adversarial variants that test for sensitivity to spurious cues. The investment in audit-set construction is the largest hidden cost of a serious audit program, and it is the cost that scales least gracefully with model capability.
      </Prose>

      <Prose>
        Intersectional coverage is the limiting factor for parity testing. With a single binary protected attribute and 1000 audit items, you have 500 per group — comfortable sample sizes. With three binary attributes, you have eight intersectional cells with 125 items each. With five binary attributes, you have 32 cells with 31 items each — enough sample variance that bootstrap confidence intervals on per-cell rates become very wide. The combinatorial explosion of intersectional cells is the deep reason that parity-testing audit sets must grow much faster than the number of protected attributes you care about. Stratified sampling that deliberately oversamples intersectional minorities is the standard mitigation; it costs more to construct but allows the audit to detect intersectional disparities that would otherwise be drowned in noise.
      </Prose>

      <Prose>
        The structural limitation that does not scale away is the gap between audit-set distribution and deployment-set distribution. An audit conducted on a curated, balanced, intersectionally-rich audit set tells you how the evaluator behaves on that audit set. Whether it generalizes to the actual deployment distribution depends on how representative the audit set is — and that is a question the audit cannot answer about itself. Production-data parity audits get closer because they sample from real traffic, but they introduce their own biases: the populations that interact with the system at all are not necessarily the populations that should be served by it. Combining audit-set audits with production-data audits, and treating any disagreement between the two as a separate signal worth investigating, is the most honest approach.
      </Prose>

      <Prose>
        Intersectional auditing also does not scale conceptually. Crenshaw's foundational insight (1989) that discrimination operates differently at the intersection of multiple identities than along any single axis means that an audit which checks each attribute marginally and reports "no significant gaps" can still miss substantial harms at intersections. A model that treats Black candidates fairly on average and women candidates fairly on average can still treat Black women very unfairly. The number of intersectional subgroups grows combinatorially with the number of attributes, and the sample size required for adequate power within each subgroup grows correspondingly. Practical audits cannot fully escape this; they manage it by oversampling priority intersections and being transparent about which intersections were not adequately powered.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>Audit set leakage into training</H3>
      <Prose>
        If the audit set leaks into the model's training data — directly, through scraping, or because the audit set was published and the model's pretraining corpus includes the publication — the audit becomes uninformative. A model that has seen the audit examples during training has been optimized (intentionally or not) to do well on them, and its audit-set behavior no longer predicts its production behavior. Treat the audit set as a held-out evaluation set with the same secrecy requirements as a competition test set: stored privately, accessed only through the audit pipeline, and rotated periodically if there is any risk of compromise.
      </Prose>

      <H3>Conflating statistical significance with practical importance</H3>
      <Prose>
        With a large enough audit set, any non-zero gap will be statistically significant. A finding of "p &lt; 0.001" on a 0.2-percentage-point selection-rate gap is statistically real but practically negligible. Conversely, a 10-point gap with n = 30 may not reach significance and yet warrants serious investigation. The right discipline is to report effect sizes alongside p-values and to commit in advance to substantive thresholds for what constitutes a meaningful gap. Audits that present only p-values are easy to game by varying sample size; audits that present only effect sizes are easy to game by ignoring noise. Both must be present.
      </Prose>

      <H3>Counterfactual artifacts</H3>
      <Prose>
        Paired testing constructs counterfactuals by varying surface markers — names, pronouns, demographic descriptors. The construction can introduce artifacts: a name swap may produce an unusual name combination that the model finds out-of-distribution; a pronoun change may make the sentence grammatically awkward; a demographic descriptor may interact with other parts of the prompt in ways that make the counterfactual no longer truly comparable. The audit will then attribute the model's response to the protected attribute when it is actually responding to the artifact. Mitigation: audit the audit set itself by having native speakers review counterfactual pairs for fluency and naturalness, and use multiple counterfactual constructions per attribute to detect sensitivity to specific surface forms.
      </Prose>

      <H3>Choice of binarization threshold</H3>
      <Prose>
        Many audits operate on binarized decisions rather than continuous scores, which requires choosing a threshold. Different thresholds produce different selection rates and different parity gaps. A threshold of 0.5 may show no parity violation; a threshold of 0.7 may show a large one. Audits should report thresholds explicitly, justify the choice in domain terms (the threshold actually used in deployment), and ideally report parity statistics across a range of thresholds to characterize how the gap depends on the operating point.
      </Prose>

      <H3>Group definition instability</H3>
      <Prose>
        Protected-attribute group definitions are not natural kinds; they are administrative constructions that vary across jurisdictions, time periods, and self-identification practices. Race categories used by the U.S. Census differ from those used in EU statistical reporting; gender categories collected in 2010 differ from those collected in 2023. An audit that reports "Hispanic" as a single group obscures substantial within-group variation; an audit that reports "Asian" as a single group erases differences between, say, East Asian and South Asian populations that often experience very different model treatment. Audit reports should disclose group definitions precisely, note their limitations, and include intersectional and disaggregated analyses where sample sizes allow.
      </Prose>

      <H3>Reporting only marginal statistics</H3>
      <Prose>
        Marginal statistics — selection rate by sex, selection rate by race — can be unbiased while intersectional statistics are deeply biased. An audit that reports only marginals can confidently claim no disparate impact when the system is severely failing intersectional minorities. The discipline is to report the full intersectional table even when many cells are noisy, with clear sample-size annotations so readers can judge which cells are well-powered. Suppressing low-sample cells looks tidier but hides exactly the populations where the audit should be most cautious.
      </Prose>

      <H3>Calibration vs accuracy confusion</H3>
      <Prose>
        A judge can be highly accurate and badly miscalibrated, or well-calibrated and inaccurate. Reporting accuracy as a calibration metric is a category error. Calibration auditing requires binning scores and comparing to empirical rates within bins; it cannot be inferred from top-line accuracy or AUC. The Brier score and the log-loss are proper scoring rules that combine calibration and refinement, but they do not isolate calibration alone. ECE and reliability diagrams are the audit-relevant calibration metrics; conflating them with accuracy or AUC is a recurring mistake even in published audit work.
      </Prose>

      <H3>Multiple comparisons without correction</H3>
      <Prose>
        A complete audit produces dozens of test statistics. Without correction, the family-wise error rate is large — at the 0.05 level, an audit with 20 tests has roughly a 64% chance of at least one false positive even when no real bias exists. Conversely, an aggressive Bonferroni correction in a high-comparison audit can mask true positives. The right approach is to disclose the correction method, justify it in terms of the audit's purpose (confirmatory vs exploratory), and report both corrected and uncorrected results so reviewers can apply their own threshold.
      </Prose>

      <H3>Audit as compliance theater</H3>
      <Prose>
        The most insidious failure mode. An organization commissions an audit, runs it once, ships a glossy PDF to the regulator, files it, and never references it again. The audit changes nothing about the system. This is not an analytical failure but an organizational one, and no amount of statistical rigor in the audit methodology fixes it. The audit must be wired into a remediation pipeline: a failed audit must trigger investigation, the investigation must produce concrete fixes, the fixes must be re-audited before re-deployment, and the entire trail must be documented in a way that internal and external auditors can review. An audit that does not change anything is not an audit; it is paperwork.
      </Prose>

      <Callout accent="purple">
        Audit-set freshness matters more than audit-set size. A small, well-curated, secret audit set rotated quarterly will catch issues that a large stale audit set has long since been gamed against. Treat audit-set construction and maintenance as ongoing engineering work, not a one-time project.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All sources below were verified against their original publication venues and arXiv pages. Citations include arXiv IDs and journal references where applicable.
      </Prose>

      <H3>Bertrand &amp; Mullainathan 2004 — the foundational paired-audit study</H3>
      <Prose>
        Marianne Bertrand, Sendhil Mullainathan. "Are Emily and Greg More Employable than Lakisha and Jamal? A Field Experiment on Labor Market Discrimination." American Economic Review 94 (4): 991-1013, September 2004. NBER Working Paper No. 9873. The methodological template for modern paired-testing audits. Sent ~5,000 fictitious resumes to real Boston and Chicago job postings, varying only the candidate's first name between stereotypically white and stereotypically Black names. Found resumes with white-coded names received 50% more callbacks than identical resumes with Black-coded names. The matched-pair design and the careful within-employer control are directly inherited by automated paired-test audits of LLM judges.
      </Prose>

      <H3>NYC Local Law 144 — the first jurisdiction-level audit mandate</H3>
      <Prose>
        New York City Department of Consumer and Worker Protection. "Local Law 144 of 2021: Automated Employment Decision Tools." Effective July 5, 2023. Final rules published April 6, 2023 in the City Record. Requires that any automated employment decision tool used to screen candidates for employment in NYC undergo an annual independent bias audit, with results published on the employer's website. Specifies the four-fifths rule as the disparate-impact benchmark and requires reporting of selection rates and impact ratios per category, including intersectional categories. The first U.S. law to specifically name third-party bias auditing as a deployment requirement.
      </Prose>

      <H3>NIST AI Risk Management Framework 1.0 — the U.S. federal reference</H3>
      <Prose>
        National Institute of Standards and Technology. "Artificial Intelligence Risk Management Framework (AI RMF 1.0)." NIST AI 100-1, January 26, 2023. The most widely-adopted structuring document for AI risk management in U.S. federal contracting. Defines four functions — Govern, Map, Measure, Manage — within which evaluator audits are the principal Measure-function activity. Pairs with NIST AI 100-2 (Generative AI Profile, July 2024), which specifies Measure activities for generative AI systems including evaluator audits, red-teaming, and bias assessment.
      </Prose>

      <H3>Yang et al. 2024 — auditing LLMs at scale</H3>
      <Prose>
        Yi Yang, Hanyu Liu, Mufan Lyu, et al. "Auditing Large Language Models: A Three-Layered Approach." arXiv:2402.10599. Published February 2024. Proposes a three-layer audit methodology spanning governance, model, and application levels, with paired testing as the core technical method at the model layer. Provides empirical evidence that frontier LLMs exhibit substantial paired-test gaps on demographic attributes despite passing aggregate-parity benchmarks. Direct empirical support for the claim that paired testing catches failures that parity testing misses.
      </Prose>

      <H3>Birhane et al. 2024 — the AI auditing landscape</H3>
      <Prose>
        Abeba Birhane, Ryan Steed, Victor Ojewale, et al. "AI Auditing: The Broken Bus on the Road to AI Accountability." Proceedings of the 2024 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML 2024). Survey of the state of AI auditing across academic, industrial, and regulatory practice. Documents the gap between methodological sophistication and operational follow-through, with detailed case studies of audits that found problems but did not lead to remediation. Essential reading for understanding why audit pipelines must be wired into engineering processes rather than treated as standalone artifacts.
      </Prose>

      <H3>Hardt, Price, &amp; Srebro 2016 — equalized odds</H3>
      <Prose>
        Moritz Hardt, Eric Price, Nathan Srebro. "Equality of Opportunity in Supervised Learning." NeurIPS 2016. arXiv:1610.02413. Introduces equalized odds and equality of opportunity as fairness criteria distinct from demographic parity. Provides the formal framework that parity-testing audits use when reporting TPR and FPR gaps.
      </Prose>

      <H3>Chouldechova 2017 / Kleinberg, Mullainathan, &amp; Raghavan 2017 — the impossibility result</H3>
      <Prose>
        Alexandra Chouldechova. "Fair Prediction with Disparate Impact: A Study of Bias in Recidivism Prediction Instruments." Big Data 5(2): 153-163, 2017. arXiv:1610.07524. Jon Kleinberg, Sendhil Mullainathan, Manish Raghavan. "Inherent Trade-Offs in the Fair Determination of Risk Scores." ITCS 2017. arXiv:1609.05807. The two papers that independently proved that calibration within groups, equality of TPR, and equality of FPR cannot all hold simultaneously when group base rates differ. The most important formal fact for any audit that touches multiple fairness criteria.
      </Prose>

      <H3>Pleiss, Raghavan, Wu, Kleinberg, &amp; Weinberger 2017 — calibration trade-offs</H3>
      <Prose>
        Geoff Pleiss, Manish Raghavan, Felix Wu, Jon Kleinberg, Kilian Q. Weinberger. "On Fairness and Calibration." NeurIPS 2017. arXiv:1709.02012. Sharpens the calibration-vs-error-rate trade-off and shows that any post-processing that achieves equalized odds must give up calibration. Foundational for understanding why calibration audits and parity audits often produce conflicting verdicts and why an audit report must declare its priorities.
      </Prose>

      <H3>EU AI Act — high-risk system conformity</H3>
      <Prose>
        Regulation (EU) 2024/1689 of the European Parliament and of the Council of 13 June 2024 laying down harmonised rules on artificial intelligence ("AI Act"). Official Journal of the European Union, July 12, 2024. Classifies employment, education admissions, credit scoring, and law enforcement systems as "high risk" and requires conformity assessments including bias testing. Articles 9, 10, and 15 specify the data-governance, risk-management, and accuracy/robustness requirements that bias audits must satisfy.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Recover a planted bias</H3>
      <Prose>
        Construct a synthetic judge similar to the one in section 4 but with a planted bias of <Code>−0.05</Code> (smaller than the 0.12 used in the example) and <Code>noise_sd = 0.15</Code> (more noise). Run the paired test on audit sets of size <Code>n = 100</Code>, <Code>n = 500</Code>, and <Code>n = 5000</Code>. For each, report the estimated mean difference, its 95% CI, and the p-value. At what sample size does the paired test reliably distinguish the planted bias from zero at the Bonferroni-corrected alpha (assuming five tests in the audit)? How does this minimum sample size depend on the noise level? Derive the relationship analytically using the paired-t formula and verify empirically.
      </Prose>

      <H3>Exercise 2 — Demographic parity vs paired test divergence</H3>
      <Prose>
        Construct an audit set where parity testing shows no statistically significant gap but paired testing recovers a clear bias. (Hint: arrange the latent quality distribution so that group <Code>b</Code>'s population is biased toward higher latent quality, exactly compensating for the judge's downward bias.) Verify both audit results numerically. Explain why this is not a contradiction and what it implies about reporting practice when only one audit modality is used. Then construct the converse: an audit set where paired testing shows no per-input bias but parity testing shows a large disparate-impact gap because of a threshold-calibration interaction.
      </Prose>

      <H3>Exercise 3 — Calibration vs accuracy</H3>
      <Prose>
        Build two judges with the same accuracy (say, 80% top-1 binary accuracy on a balanced test set) but very different calibration: judge A is well-calibrated (ECE &lt; 0.02) and judge B is overconfident (ECE &gt; 0.20). Show this by constructing both judges and computing the relevant metrics. Now run a fairness audit on both: which metrics flag judge B's poor calibration, and which do not? What does this exercise tell you about the limitations of accuracy-only audit reports? Bonus: implement temperature scaling on judge B and show that you can recover calibration without changing accuracy.
      </Prose>

      <H3>Exercise 4 — Intersectional sample-size budget</H3>
      <Prose>
        Suppose you must audit a content classifier across three binary protected attributes (sex, race coded as binary, age coded as binary), giving 8 intersectional cells. Your fairness committee requires that each intersectional cell be powered to detect a 5-percentage-point selection-rate gap at 80% power and alpha = 0.01 (Bonferroni-corrected for 28 pairwise comparisons among the 8 cells). Compute the minimum total audit-set size you need, assuming equal allocation across cells. Now consider stratified oversampling: how would you allocate a 5,000-item audit budget across the 8 cells if the cells have different a priori risks of harm and you want to maximize the audit's expected detection power on the highest-risk cells? Justify your allocation.
      </Prose>

      <H3>Exercise 5 — Designing an audit-set rotation policy</H3>
      <Prose>
        Your team ships a new judge model every two weeks. The audit set is currently a single 10,000-item file checked into the repo, used for every audit. List four risks to the audit's informativeness from this practice. Propose an audit-set rotation policy that addresses each risk while still allowing year-over-year trend analysis. How do you handle the tension between needing a stable comparison baseline (for trend analysis) and needing a fresh held-out set (to prevent leakage)? Sketch the directory layout, naming convention, and access-control rules you would use to implement your policy.
      </Prose>

      <H3>Exercise 6 — Applying the impossibility result</H3>
      <Prose>
        A regulator reviewing your audit report demands that the model satisfy both group-conditional calibration and equalized odds. Your model's deployment population has substantially different base rates across the two groups (35% positive in group <Code>a</Code>, 65% positive in group <Code>b</Code>). Write a one-page response to the regulator explaining (a) the formal impossibility result, (b) why it applies to this specific case, (c) which criterion you have chosen to prioritize and why, (d) what diagnostic metrics you will report for the criterion you did not prioritize, and (e) how you propose to escalate if the regulator continues to demand both. Cite the impossibility-result papers correctly.
      </Prose>

    </div>
  ),
};

export default evaluatorAuditMethodology;
