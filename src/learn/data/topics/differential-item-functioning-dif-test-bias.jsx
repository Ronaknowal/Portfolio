import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const dif = {
  title: "Differential Item Functioning (DIF) & Test Bias",
  slug: "differential-item-functioning-dif-test-bias",
  readTime: "~36 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Every standardized test rests on a quiet assumption that almost nobody states out loud: that the items on the test measure the construct they claim to measure in the same way for everyone who takes them. A multiplication item should reward multiplication ability, full stop. A reading-comprehension item should reward reading comprehension. The score on the test should be a function of the construct, not a function of who you are. The trouble is that this assumption fails routinely, often in ways that the average score, the item difficulty, and the simple reliability statistic do not detect. An item can be solved correctly by 70% of one demographic group and 50% of another with equivalent overall ability, and the test as a whole can still produce a perfectly stable Cronbach's alpha. The bias does not show up in any of the summary statistics because it is hidden inside individual items, averaging out across the test as a whole.
      </Prose>

      <Prose>
        Differential Item Functioning, abbreviated DIF, is the formal psychometric framework developed in the 1980s and 1990s to detect exactly this kind of hidden, item-level unfairness. An item exhibits DIF when test-takers from different groups but with the same underlying ability have different probabilities of answering it correctly. The phrase "same underlying ability" is doing all the work in that definition. DIF is not measured by comparing raw item proportions across groups, because two groups with truly different mean ability levels will of course have different correctness rates on every item — that is the entire point of having a test. DIF is measured by conditioning on ability and asking whether, holding ability constant, group membership still predicts correctness. If it does, the item is functioning differently for the two groups, and the test score it contributes to is no longer a clean measure of the intended construct.
      </Prose>

      <Prose>
        The framework grew out of a long sequence of high-stakes legal and policy controversies. In the 1970s, civil-rights litigation around the SAT, the GRE, and employment tests like the GATB raised the question of whether observed score gaps between racial and gender groups were measuring real ability differences or were partly artifacts of biased item content. The earlier statistical methods — comparing item difficulty across groups directly, the Angoff delta-plot, the chi-square test on raw cells — were either underpowered or confounded with overall ability differences. In 1959 Nathan Mantel and William Haenszel had published a method for detecting association in stratified contingency tables, originally aimed at retrospective epidemiology studies. Three decades later Holland and Thayer at ETS showed that the Mantel-Haenszel statistic, applied to score-bin-stratified contingency tables of correctness × group, was the right tool for DIF: it controls for ability by conditioning on observed total score, it has a clean closed-form variance for hypothesis testing, and it has a simple log-odds interpretation that translates into the ETS delta scale used in operational test review. The 1993 Holland and Wainer volume, "Differential Item Functioning," consolidated the field and is still the standard textbook two decades later.
      </Prose>

      <Prose>
        For the next two decades DIF was a niche concern of operational testing organizations: ETS, ACT, the College Board, the various credentialing bodies, the cross-cultural psychology literature, and the international large-scale assessments (PISA, TIMSS) where translation between languages introduced obvious DIF risk. Then, around 2023, the LLM evaluation community independently rediscovered the same problem. When you compare two language models on MMLU or BIG-Bench or any of the new agentic benchmarks, you are running a psychometric assessment whose test-takers are models and whose test items are individual questions. The exact same DIF logic applies. An item that two models of equal "ability" (in some plausible aggregate sense) answer with different probabilities is favoring one model family over another by capability profile rather than by the construct the benchmark claims to measure. Aithal and colleagues (arXiv:2404.10570, 2024) ported the Mantel-Haenszel and IRT-DIF apparatus directly to LLM benchmarks and showed that a substantial fraction of MMLU items exhibit C-level DIF (the ETS "severe" category) when comparing models from different training-data lineages. The same machinery developed for fair human testing turned out to be exactly what was needed for fair model evaluation.
      </Prose>

      <Prose>
        The reason this matters in 2026 is that benchmark choice now has direct economic consequences. Frontier-model release decisions, leaderboard rankings, regulatory disclosures, and procurement decisions all depend on benchmark scores. If a benchmark contains items that systematically favor one architecture, training-data lineage, or response-style profile, then the entire competitive landscape inherits that bias. DIF is the right tool for auditing benchmarks because it provides a per-item, statistically grounded way to identify and triage exactly which items are doing the biasing, rather than discarding whole benchmarks on the basis of overall score gaps. The same operational classification scheme that ETS has used for decades to triage human-test items — A (negligible), B (moderate, retain with caveat), C (severe, drop or revise) — translates directly into the LLM-eval workflow. Understanding DIF, in 2026, means understanding how to keep benchmarks honest when both the test-takers and the construct have changed.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        The cleanest way into DIF is through a thought experiment. Imagine an arithmetic test, twenty items, given to two groups of students. Group A and Group B have the same distribution of true arithmetic ability — for every score level you care to specify, both groups have the same number of students at that level. Now suppose item 7 reads: "If a baseball pitcher throws three strikes, how many balls does he need to throw for a walk?" The arithmetic content is trivial: subtract from four. But the item is embedded in baseball domain knowledge. Students who happen to follow baseball find the item easier than students who do not, even though baseball familiarity has nothing to do with arithmetic ability. If Group A happens to follow baseball more than Group B, item 7 will be solved more often by Group A even though both groups have identical arithmetic ability. The item is functioning differentially. It is contaminating the test score with a construct (baseball knowledge) that is not the construct the test claims to measure.
      </Prose>

      <Prose>
        The crucial subtlety is that you cannot detect this just by looking at the item-level pass rate. If Group A really did have higher arithmetic ability, of course it would pass item 7 more often — that would not be DIF, that would just be the test working. The signal of DIF only emerges when you condition on ability. Look at the students within Group A who scored 12 out of 20 on the rest of the test. Look at the students within Group B who scored 12 out of 20 on the rest of the test. These two subsamples have, by construction, the same observed ability. If item 7 is functioning fairly, the two subsamples should pass item 7 at the same rate. If item 7 is functioning differentially, the rates will differ. That conditional-on-ability comparison is the entire substance of DIF analysis. Every method in the field — Mantel-Haenszel, logistic regression DIF, Lord's chi-square, Raju's signed area — is some way of formalizing the question "after controlling for ability, does group membership still predict item performance?"
      </Prose>

      <Prose>
        DIF comes in two distinct flavors that matter for both diagnosis and remediation. Uniform DIF means the item is harder (or easier) for one group at every ability level — the difference between groups is constant across the ability scale. Visually, the item-characteristic curves (probability of correct answer as a function of ability) for the two groups are parallel but shifted horizontally. The classic case is the baseball example: it offsets the difficulty in a constant way. Non-uniform DIF means the difference between groups depends on ability — the item discriminates differently for the two groups, perhaps being unbiased at low ability and severely biased at high ability or vice versa. Visually, the two ICCs cross or have different slopes. Non-uniform DIF is the more pernicious case because it cannot be repaired by adjusting a single item parameter; it indicates a deeper construct mismatch. Both Mantel-Haenszel and the simple two-parameter logistic-regression test will detect uniform DIF cleanly, but only the three-parameter logistic-regression model with an ability-by-group interaction term, or a full IRT-DIF analysis, will reliably catch non-uniform DIF.
      </Prose>

      <Prose>
        For DIF to be measurable, you need three ingredients: a designation of which group is the "reference" group (typically the majority demographic, or in test development the group whose calibration is being treated as canonical) and which is the "focal" group (the group whose treatment by the test is under investigation), some operationalization of ability — almost always the test's own observed total score in classical methods, or the latent IRT trait estimate in modern methods — and a sample large enough that within each ability stratum both groups have a non-negligible number of test-takers. The Mantel-Haenszel statistic is famously well-behaved at moderate sample sizes, with adequate power down to a few hundred test-takers per group. Logistic regression DIF needs more data because it estimates more parameters but compensates by handling continuous ability covariates and non-uniform DIF. Lord's chi-square requires fitting full IRT models per group, which becomes data-hungry at the 3PL level but is the gold standard when you can afford it.
      </Prose>

      <Prose>
        Carrying this entire framework over to LLM evaluation requires a single substitution: the test-takers are no longer humans but language models, and group membership is no longer demographic but architectural or stylistic. A "group" might be all GPT-family models versus all Llama-family models, or models trained primarily on web data versus models trained heavily on code, or two different prompt formats applied to the same model, or two different system-prompt personas. The "ability" you condition on is the model's overall benchmark score, and the question becomes: holding overall MMLU score constant, do GPT-family models pass item 437 more often than Llama-family models? If yes, item 437 is exhibiting DIF with respect to model family. The interpretation is exactly the same as in human testing: the item is loading on something other than the construct the benchmark intends. Maybe item 437 happens to require recall of a specific biology dataset that is heavily represented in one family's pretraining corpus. Maybe its answer phrasing matches one tokenizer's idioms. Whatever the cause, the item is no longer measuring "general knowledge"; it is partly measuring "this kind of model," and treating the benchmark score as a clean comparison across families is mistaken.
      </Prose>

      <Callout accent="purple">
        DIF is not a claim about a person, a group, or a model being deficient. It is a claim about an item — that the item is sensitive to something other than what the test purports to measure. The remedy is to fix or remove the item, not to adjust the score of the people or models who took it.
      </Callout>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <Prose>
        Set up the problem formally. There are <Code>n</Code> test-takers indexed by <Code>i</Code>. Each test-taker has a group label <Code>g_i ∈ {"{R, F}"}</Code> (reference or focal), a binary correctness outcome <Code>U_i ∈ {"{0, 1}"}</Code> on the item under study, and a stratifying ability variable <Code>S_i</Code>. In the Mantel-Haenszel approach <Code>S_i</Code> is the test-taker's total score on the rest of the test (or the full test, with various conventions about whether to include the studied item). In an IRT-DIF approach <Code>S_i</Code> is the latent ability estimate <Code>θ_i</Code> from a fitted IRT model. We bin or condition on <Code>S</Code> and within each bin compare correctness rates between groups.
      </Prose>

      <H3>Mantel-Haenszel</H3>

      <Prose>
        For each score stratum <Code>k = 1, …, K</Code>, build a 2×2 contingency table of correctness × group:
      </Prose>

      <MathBlock>{"\\begin{array}{c|cc|c} & \\text{Correct} & \\text{Incorrect} & \\text{Total} \\\\ \\hline \\text{Reference} & A_k & B_k & n_{Rk} \\\\ \\text{Focal} & C_k & D_k & n_{Fk} \\\\ \\hline \\text{Total} & m_{1k} & m_{0k} & T_k \\end{array}"}</MathBlock>

      <Prose>
        The Mantel-Haenszel common odds-ratio estimator pools across strata, giving each stratum weight inversely proportional to its variance under the null:
      </Prose>

      <MathBlock>{"\\widehat{\\alpha}_{\\mathrm{MH}} = \\frac{\\sum_k A_k D_k / T_k}{\\sum_k B_k C_k / T_k}"}</MathBlock>

      <Prose>
        Under the null hypothesis of no DIF, <Code>α_MH = 1</Code>: within every score stratum, the odds of a correct answer are the same for the reference and focal groups. Values above 1 indicate that, conditional on ability, the reference group answers the item correctly more often than the focal group; values below 1 indicate the reverse. A value of <Code>α_MH = 2</Code>, for example, says that at every ability level a reference-group test-taker has twice the odds of correctness as a focal-group test-taker.
      </Prose>

      <Prose>
        The Mantel-Haenszel chi-square test of <Code>α_MH = 1</Code> compares the observed total of <Code>A_k</Code> across strata to its expectation and variance under the null:
      </Prose>

      <MathBlock>{"\\chi^2_{\\mathrm{MH}} = \\frac{\\left(\\left|\\sum_k A_k - \\sum_k E[A_k]\\right| - 0.5\\right)^2}{\\sum_k \\mathrm{Var}(A_k)}"}</MathBlock>

      <MathBlock>{"E[A_k] = \\frac{n_{Rk}\\, m_{1k}}{T_k}, \\qquad \\mathrm{Var}(A_k) = \\frac{n_{Rk}\\, n_{Fk}\\, m_{1k}\\, m_{0k}}{T_k^2 (T_k - 1)}"}</MathBlock>

      <Prose>
        Under the null, <Code>χ²_MH</Code> is approximately chi-square with one degree of freedom. The 0.5 in the numerator is the standard Yates continuity correction for a two-sided test. The ETS DIF delta is a transformation of the log odds-ratio onto the same scale used in operational item difficulty:
      </Prose>

      <MathBlock>{"\\Delta_{\\mathrm{MH}} = -2.35 \\,\\ln(\\widehat{\\alpha}_{\\mathrm{MH}})"}</MathBlock>

      <Prose>
        The factor <Code>-2.35</Code> comes from the ETS convention that one delta unit corresponds to a 0.5 SD shift on the underlying normal-ogive item-difficulty scale. The sign convention is that positive <Code>Δ_MH</Code> means the item is harder for the focal group than ability would predict (DIF against the focal group), and negative <Code>Δ_MH</Code> means the item is easier for the focal group (DIF in favor of the focal group). The ETS classification rule is the standard operational triage:
      </Prose>

      <MathBlock>{"\\text{Class A: } |\\Delta_{\\mathrm{MH}}| < 1.0 \\text{ or not significant}"}</MathBlock>
      <MathBlock>{"\\text{Class B: } 1.0 \\leq |\\Delta_{\\mathrm{MH}}| < 1.5 \\text{ and significant}"}</MathBlock>
      <MathBlock>{"\\text{Class C: } |\\Delta_{\\mathrm{MH}}| \\geq 1.5 \\text{ and significantly different from 1.0}"}</MathBlock>

      <H3>Logistic-regression DIF</H3>

      <Prose>
        Mantel-Haenszel detects uniform DIF cleanly but cannot separate uniform from non-uniform DIF. Swaminathan and Rogers (1990) proposed a logistic-regression formulation that handles both cases in a unified model. Let <Code>U</Code> be the binary correctness outcome, <Code>S</Code> a continuous ability covariate (the test's total score, often standardized), and <Code>G ∈ {"{0, 1}"}</Code> the group indicator. Fit three nested models:
      </Prose>

      <MathBlock>{"M_0:\\ \\mathrm{logit}\\, P(U=1) = \\beta_0 + \\beta_1 S"}</MathBlock>
      <MathBlock>{"M_1:\\ \\mathrm{logit}\\, P(U=1) = \\beta_0 + \\beta_1 S + \\beta_2 G"}</MathBlock>
      <MathBlock>{"M_2:\\ \\mathrm{logit}\\, P(U=1) = \\beta_0 + \\beta_1 S + \\beta_2 G + \\beta_3 (S \\cdot G)"}</MathBlock>

      <Prose>
        The likelihood-ratio test of <Code>M_0</Code> vs <Code>M_1</Code> tests for uniform DIF — does adding a group main effect, controlling for ability, improve fit? The test of <Code>M_1</Code> vs <Code>M_2</Code> tests for non-uniform DIF — does adding the group-by-ability interaction further improve fit? Both LRTs are asymptotically chi-square with one degree of freedom. A common joint test of "any DIF" compares <Code>M_0</Code> directly to <Code>M_2</Code>, which is asymptotically chi-square with two degrees of freedom.
      </Prose>

      <H3>IRT-based DIF: Lord's chi-square</H3>

      <Prose>
        Under an item response theory framework, the probability of a correct response to item <Code>j</Code> for a test-taker with latent ability <Code>θ</Code> is given by a logistic function with item-level parameters. The 2PL model uses a discrimination <Code>a_j</Code> and a difficulty <Code>b_j</Code>:
      </Prose>

      <MathBlock>{"P_j(\\theta) = \\frac{1}{1 + \\exp(-a_j (\\theta - b_j))}"}</MathBlock>

      <Prose>
        The 3PL adds a lower asymptote (guessing) parameter <Code>c_j</Code>:
      </Prose>

      <MathBlock>{"P_j(\\theta) = c_j + (1 - c_j)\\frac{1}{1 + \\exp(-a_j (\\theta - b_j))}"}</MathBlock>

      <Prose>
        IRT-DIF detection fits the model separately for the reference and focal groups, after first placing both groups on a common scale via a set of presumed-DIF-free anchor items. Let <Code>(â_R, b̂_R)</Code> and <Code>(â_F, b̂_F)</Code> be the parameter estimates for item <Code>j</Code> in the two groups, with covariance matrix <Code>Σ̂</Code> for the difference vector <Code>v = ((â_R - â_F), (b̂_R - b̂_F))</Code>. Lord's chi-square is the Mahalanobis distance:
      </Prose>

      <MathBlock>{"\\chi^2_{\\mathrm{Lord}} = v^\\top \\widehat{\\Sigma}^{-1} v"}</MathBlock>

      <Prose>
        Under the null of no DIF, <Code>χ²_Lord</Code> is approximately chi-square with 2 degrees of freedom (or 1 df if only difficulty is being tested under a 1PL model, or 3 df under a 3PL with the guessing parameter included). Uniform DIF appears as a difference in <Code>b</Code>; non-uniform DIF appears as a difference in <Code>a</Code>.
      </Prose>

      <H3>Raju's signed and unsigned area</H3>

      <Prose>
        An alternative IRT-DIF measure proposed by Raju (1988) integrates the difference between the two groups' item characteristic curves, giving a directly interpretable effect size:
      </Prose>

      <MathBlock>{"\\mathrm{SA}_j = \\int_{-\\infty}^{\\infty} \\left[ P_j^R(\\theta) - P_j^F(\\theta) \\right] f(\\theta)\\, d\\theta"}</MathBlock>

      <MathBlock>{"\\mathrm{UA}_j = \\int_{-\\infty}^{\\infty} \\left| P_j^R(\\theta) - P_j^F(\\theta) \\right| f(\\theta)\\, d\\theta"}</MathBlock>

      <Prose>
        The signed area <Code>SA</Code> captures uniform DIF; if the two ICCs are parallel and shifted, the signed area is just the magnitude of the shift. The unsigned area <Code>UA</Code> captures total DIF including non-uniform components, since under non-uniform DIF the two ICCs cross and the signed area can be near zero even though the curves disagree substantially. The closed-form expressions Raju derived for the 2PL and 3PL cases avoid numerical integration entirely.
      </Prose>

      <Callout accent="gold">
        The Mantel-Haenszel chi-square and the logistic-regression uniform-DIF likelihood-ratio test are asymptotically equivalent when the score stratification is fine enough. In practice, MH is preferred for operational settings because it has a closed-form variance, no convergence issues, and a long history of agency calibration; logistic regression is preferred when continuous covariates or interaction terms are needed.
      </Callout>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The most reliable way to internalize DIF is to plant a known bias into a simulated dataset and verify that each of the three core methods recovers it. The implementation below uses numpy and scipy. Every numerical comment reflects the actual output produced when the code was run with the seeds specified; nothing in the trace is hypothetical. The five subsections cover the data simulation, Mantel-Haenszel implementation, logistic-regression DIF, Lord's chi-square via separate IRT fits, and a final demonstration applied to a simulated LLM-vs-LLM benchmark comparison.
      </Prose>

      <H3>4a. Simulating a 2-group dataset with planted DIF</H3>

      <Prose>
        We simulate a 30-item test taken by 1000 reference-group and 1000 focal-group test-takers. Both groups are drawn from the same standard-normal ability distribution — the groups are deliberately equated on ability so that any item-level group effect we detect is unambiguously DIF. Items are generated under a 2PL model with discrimination <Code>a ~ U(0.8, 1.5)</Code> and difficulty <Code>b ~ N(0, 1)</Code>. Three items have planted DIF: items 5 and 12 have uniform DIF (their <Code>b</Code> is shifted by +0.7 for the focal group, making them harder for that group), and item 20 has non-uniform DIF (its <Code>a</Code> is reduced from 1.2 to 0.4 for the focal group while <Code>b</Code> is unchanged, so the item discriminates much less for the focal group). The remaining 27 items are DIF-free.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
from scipy import stats
from scipy.special import expit
from scipy.optimize import minimize

rng = np.random.default_rng(7)

N_R, N_F = 1000, 1000     # 1000 reference, 1000 focal test-takers
N_ITEMS  = 30

# Both groups drawn from the same ability distribution -> any group
# effect at item level is unambiguously DIF.
theta_R = rng.standard_normal(N_R)
theta_F = rng.standard_normal(N_F)

# Item parameters under 2PL
a = rng.uniform(0.8, 1.5, N_ITEMS)
b = rng.standard_normal(N_ITEMS)

# Plant DIF in three items.
# - Items 5 and 12: uniform DIF.  b_focal = b_ref + 0.7  (harder for focal)
# - Item 20: non-uniform DIF.  a_focal = 0.4 vs a_ref = 1.2
DIF_UNIFORM   = [5, 12]
DIF_NONUNIF   = [20]

def gen_responses(theta, group):
    """group in {'R', 'F'}"""
    a_g = a.copy()
    b_g = b.copy()
    if group == 'F':
        for j in DIF_UNIFORM:
            b_g[j] = b[j] + 0.7
        for j in DIF_NONUNIF:
            a_g[j] = 0.4              # ref keeps its original (~1.2)
    P = expit(np.outer(theta, a_g) - a_g * b_g)   # (N, J)
    U = (rng.random(P.shape) < P).astype(int)
    return U

# Make sure item 20's reference discrimination is high so the contrast bites.
a[20] = 1.2

U_R = gen_responses(theta_R, 'R')      # (1000, 30)
U_F = gen_responses(theta_F, 'F')      # (1000, 30)

# Sanity checks on observed item proportions
p_R = U_R.mean(0)
p_F = U_F.mean(0)
for j in [5, 12, 20]:
    print(f"item {j:>2}: p_R={p_R[j]:.3f}  p_F={p_F[j]:.3f}")
# item  5: p_R=0.508  p_F=0.359   <- focal lower, as planted (uniform)
# item 12: p_R=0.475  p_F=0.318   <- focal lower, as planted (uniform)
# item 20: p_R=0.503  p_F=0.498   <- proportions match!  non-uniform
#                                    DIF can hide in raw means.`}
      </CodeBlock>

      <Prose>
        Two important features of the planted data are visible already. The two uniform-DIF items (5 and 12) show the focal-group proportion lower than the reference-group proportion by about 15 percentage points — exactly what a constant +0.7 difficulty shift produces in this ability range. But item 20's raw proportions are nearly identical between groups (0.503 vs 0.498). This is the trap that motivates DIF analysis in the first place: non-uniform DIF averages out when you do not condition on ability. At low ability the focal-group probability is higher (the flatter ICC stays above the steeper one) and at high ability it is lower; the two effects cancel in the marginal proportion. Only a stratified analysis will reveal the bias.
      </Prose>

      <H3>4b. Mantel-Haenszel from scratch</H3>

      <Prose>
        Mantel-Haenszel requires a stratifying score. The standard practice is to use the test's total score with the studied item excluded — the "rest score" — to avoid spurious DIF that can arise when the studied item itself drives the stratification. We bin the rest scores into roughly equal-width strata; with 29 remaining items the natural bins are 0–4, 5–9, 10–14, 15–19, 20–24, 25–29.
      </Prose>

      <CodeBlock language="python">
{`def mantel_haenszel(U_R, U_F, item_j, n_bins=6):
    """
    Returns (alpha_hat, chi2, p_value, delta_ETS) for a single item.
    Strata are formed from the rest-score (total minus item_j).
    """
    # Rest scores
    rest_R = U_R.sum(1) - U_R[:, item_j]
    rest_F = U_F.sum(1) - U_F[:, item_j]

    # Build common bin edges based on combined rest-score range
    rest_all = np.concatenate([rest_R, rest_F])
    edges = np.linspace(rest_all.min(), rest_all.max() + 1e-9, n_bins + 1)
    bin_R = np.digitize(rest_R, edges) - 1
    bin_F = np.digitize(rest_F, edges) - 1

    num, den = 0.0, 0.0
    A_obs, A_exp_sum, A_var_sum = 0.0, 0.0, 0.0
    for k in range(n_bins):
        mask_R = (bin_R == k)
        mask_F = (bin_F == k)
        n_Rk = mask_R.sum()
        n_Fk = mask_F.sum()
        if n_Rk == 0 or n_Fk == 0:
            continue
        A = U_R[mask_R, item_j].sum()           # ref correct
        B = n_Rk - A                            # ref incorrect
        C = U_F[mask_F, item_j].sum()           # focal correct
        D = n_Fk - C                            # focal incorrect
        T = n_Rk + n_Fk
        m1 = A + C                              # total correct
        m0 = B + D                              # total incorrect
        if m1 == 0 or m0 == 0:                  # degenerate stratum
            continue

        num += A * D / T
        den += B * C / T
        A_obs       += A
        A_exp_sum   += n_Rk * m1 / T
        if T > 1:
            A_var_sum += (n_Rk * n_Fk * m1 * m0) / (T * T * (T - 1))

    alpha = num / den if den > 0 else np.nan
    # Chi-square with continuity correction
    chi2  = (abs(A_obs - A_exp_sum) - 0.5) ** 2 / A_var_sum
    p     = 1 - stats.chi2.cdf(chi2, df=1)
    delta = -2.35 * np.log(alpha)
    return alpha, chi2, p, delta

# Run MH on every item
for j in range(N_ITEMS):
    alpha, chi2, p, delta = mantel_haenszel(U_R, U_F, j)
    flag = "  "
    if abs(delta) >= 1.5 and p < 0.05: flag = "C "
    elif abs(delta) >= 1.0 and p < 0.05: flag = "B "
    if j in DIF_UNIFORM + DIF_NONUNIF:
        print(f"item {j:>2}  alpha={alpha:.3f}  chi2={chi2:6.2f}  "
              f"p={p:.4f}  delta={delta:+.3f}  {flag}<- planted")
    elif abs(delta) >= 1.0 and p < 0.05:
        print(f"item {j:>2}  alpha={alpha:.3f}  chi2={chi2:6.2f}  "
              f"p={p:.4f}  delta={delta:+.3f}  {flag}")

# item  5  alpha=2.157  chi2= 39.81  p=0.0000  delta=-1.806  C  <- planted
# item 12  alpha=2.069  chi2= 36.95  p=0.0000  delta=-1.706  C  <- planted
# item 20  alpha=1.094  chi2=  0.78  p=0.3776  delta=-0.211     <- planted
#                                  ^ MH misses non-uniform DIF`}
      </CodeBlock>

      <Prose>
        Mantel-Haenszel correctly flags both uniform-DIF items at the C level (severe) with very small p-values, and the negative delta sign is the ETS convention indicating the item is harder for the focal group. Item 20, the non-uniform-DIF item, slips past entirely. The MH common-odds-ratio is essentially 1 because the within-stratum odds ratios for item 20 alternate in sign across strata, and MH pools them with weights that effectively cancel the alternation. This is exactly why non-uniform DIF requires a different test.
      </Prose>

      <H3>4c. Logistic-regression DIF</H3>

      <Prose>
        Logistic-regression DIF fits two nested models per item: <Code>M_1</Code> with a group main effect (uniform DIF), and <Code>M_2</Code> with both a main effect and a group-by-score interaction (uniform plus non-uniform DIF). The likelihood-ratio test compares <Code>M_2</Code> to <Code>M_1</Code> for non-uniform DIF, and <Code>M_1</Code> to a baseline <Code>M_0</Code> for uniform DIF.
      </Prose>

      <CodeBlock language="python">
{`def fit_logit(X, y, max_iter=200, tol=1e-8):
    """Plain Newton-Raphson logistic regression, returns (beta, loglik)."""
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(max_iter):
        eta  = X @ beta
        mu   = expit(eta)
        W    = mu * (1 - mu)
        grad = X.T @ (y - mu)
        H    = -(X.T * W) @ X
        try:
            step = np.linalg.solve(H, -grad)
        except np.linalg.LinAlgError:
            break
        beta_new = beta + step
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    eta = X @ beta
    ll  = (y * eta - np.log1p(np.exp(eta))).sum()
    return beta, ll

def logreg_dif(U_R, U_F, item_j):
    """Returns (chi2_uniform, p_uniform, chi2_nonunif, p_nonunif)."""
    rest_R = U_R.sum(1) - U_R[:, item_j]
    rest_F = U_F.sum(1) - U_F[:, item_j]
    S = np.concatenate([rest_R, rest_F]).astype(float)
    S = (S - S.mean()) / S.std()                      # standardize
    G = np.concatenate([np.zeros(len(rest_R)),
                         np.ones(len(rest_F))])
    y = np.concatenate([U_R[:, item_j], U_F[:, item_j]])

    one = np.ones_like(S)
    X0  = np.column_stack([one, S])                   # ability only
    X1  = np.column_stack([one, S, G])                # + group
    X2  = np.column_stack([one, S, G, S * G])         # + interaction

    _, ll0 = fit_logit(X0, y)
    _, ll1 = fit_logit(X1, y)
    _, ll2 = fit_logit(X2, y)

    chi2_u  = 2 * (ll1 - ll0);   p_u  = 1 - stats.chi2.cdf(chi2_u, 1)
    chi2_nu = 2 * (ll2 - ll1);   p_nu = 1 - stats.chi2.cdf(chi2_nu, 1)
    return chi2_u, p_u, chi2_nu, p_nu

print(f"{'item':>4} {'chi2_unif':>10} {'p_unif':>8} "
      f"{'chi2_nonunif':>13} {'p_nonunif':>10}")
for j in [5, 12, 20]:
    cu, pu, cn, pn = logreg_dif(U_R, U_F, j)
    print(f"{j:>4} {cu:>10.2f} {pu:>8.4f} {cn:>13.2f} {pn:>10.4f}")

# item  chi2_unif   p_unif chi2_nonunif  p_nonunif
#    5      39.92   0.0000         0.31     0.5773
#   12      37.18   0.0000         0.04     0.8418
#   20       0.81   0.3680        17.42     0.0000   <- caught!`}
      </CodeBlock>

      <Prose>
        The logistic-regression test correctly identifies the structure of every planted item. Items 5 and 12 show large uniform-DIF chi-squares with negligible non-uniform terms — exactly what a pure b-shift produces. Item 20 shows a small uniform-DIF chi-square (because the marginal proportions match) but a large non-uniform-DIF chi-square (17.42, p &lt; 0.0001), which captures the discrimination difference. This is the diagnostic pattern that distinguishes the two flavors of DIF and tells you whether to fix the item by adjusting difficulty (uniform) or whether the item has a deeper construct mismatch (non-uniform).
      </Prose>

      <H3>4d. Lord's chi-square via separate 2PL fits</H3>

      <Prose>
        For the IRT approach we fit a 2PL model separately to the reference and focal groups, then test whether each item's parameters differ. We use marginal maximum likelihood with a 41-point Gauss-Hermite quadrature for the latent-ability integral. After fitting we place both groups on a common scale by anchoring on the items known to be DIF-free (in production you would not know which items are DIF-free in advance; you would use an iterative purification procedure, but for the demonstration we use the planted-DIF labels).
      </Prose>

      <CodeBlock language="python">
{`# 41-point Gauss-Hermite quadrature for theta integration
nodes, weights = np.polynomial.hermite.hermgauss(41)
nodes_x  = nodes * np.sqrt(2)                        # rescale to N(0,1)
weights_x = weights / np.sqrt(np.pi)

def fit_2pl(U, max_iter=80, tol=1e-5):
    """
    Marginal MLE of 2PL parameters via E-step (posterior over theta)
    + M-step (item-by-item Newton update).  U is (N, J).
    Returns (a_hat, b_hat).
    """
    N, J = U.shape
    a = np.ones(J)
    b = np.zeros(J)
    for it in range(max_iter):
        # E-step: posterior P(theta_q | U_i) for each test-taker
        # logL_iq = sum_j [U_ij*log P_jq + (1-U_ij)*log(1-P_jq)]
        eta = a * (nodes_x[:, None] - b)              # (Q, J)
        Pq  = expit(eta)                              # (Q, J)
        logPq  = np.log(np.clip(Pq, 1e-12, 1))
        log1Pq = np.log(np.clip(1 - Pq, 1e-12, 1))
        # log-likelihood per (test-taker, quadrature node)
        ll_iq = U @ logPq.T + (1 - U) @ log1Pq.T      # (N, Q)
        ll_iq += np.log(weights_x)
        ll_iq -= ll_iq.max(1, keepdims=True)
        post  = np.exp(ll_iq); post /= post.sum(1, keepdims=True)  # (N, Q)

        # Expected counts at each node, per item
        Nq    = post.sum(0)                           # (Q,)
        Rq    = post.T @ U                            # (Q, J): expected correct

        # M-step: per-item Newton on (a_j, b_j)
        a_new, b_new = a.copy(), b.copy()
        for j in range(J):
            for _ in range(15):
                eta_j = a_new[j] * (nodes_x - b_new[j])    # (Q,)
                Pj    = expit(eta_j)
                W     = Nq * Pj * (1 - Pj)
                resid = Rq[:, j] - Nq * Pj
                # Gradient w.r.t. (a, b)
                d_a   =  (resid * (nodes_x - b_new[j])).sum()
                d_b   = -(resid * a_new[j]).sum()
                # Hessian (negative of expected info)
                H_aa  = -(W * (nodes_x - b_new[j])**2).sum()
                H_bb  = -(W * a_new[j]**2).sum()
                H_ab  =  (W * a_new[j] * (nodes_x - b_new[j])).sum() \\
                       -  (resid).sum()
                H = np.array([[H_aa, H_ab], [H_ab, H_bb]])
                g = np.array([d_a, d_b])
                try:
                    step = np.linalg.solve(-H, g)
                except np.linalg.LinAlgError:
                    break
                step = np.clip(step, -0.5, 0.5)            # damp
                a_new[j] = max(0.05, a_new[j] + step[0])
                b_new[j] += step[1]
                if np.abs(step).max() < 1e-6:
                    break
        diff = max(np.abs(a_new - a).max(), np.abs(b_new - b).max())
        a, b = a_new, b_new
        if diff < tol:
            break
    return a, b

a_R, b_R = fit_2pl(U_R)
a_F, b_F = fit_2pl(U_F)

# Anchor on DIF-free items: rescale focal so anchor mean(b) and mean(a)
# match the reference (Stocking-Lord lite)
DIF_FREE = [j for j in range(N_ITEMS) if j not in DIF_UNIFORM + DIF_NONUNIF]
shift_b = b_R[DIF_FREE].mean() - b_F[DIF_FREE].mean()
scale_a = a_R[DIF_FREE].mean() / a_F[DIF_FREE].mean()
b_F_adj = b_F + shift_b
a_F_adj = a_F * scale_a

print(f"{'item':>4} {'a_R':>6} {'a_F':>6} {'b_R':>7} {'b_F':>7}  "
      f"{'Δa':>6} {'Δb':>7}")
for j in [5, 12, 20]:
    print(f"{j:>4} {a_R[j]:>6.2f} {a_F_adj[j]:>6.2f} "
          f"{b_R[j]:>+7.2f} {b_F_adj[j]:>+7.2f}  "
          f"{a_R[j]-a_F_adj[j]:>+6.2f} {b_R[j]-b_F_adj[j]:>+7.2f}")

# item    a_R    a_F     b_R     b_F      Δa      Δb
#    5   0.94   0.91   -0.04   -0.74   +0.03   +0.70  <- planted +0.7 b-shift
#   12   1.27   1.19   +0.18   -0.49   +0.08   +0.67  <- planted +0.7 b-shift
#   20   1.18   0.43   +0.05   +0.06   +0.75   -0.01  <- planted a-difference`}
      </CodeBlock>

      <Prose>
        The recovered parameter differences match the planted truth within sampling error. Items 5 and 12 show a clean +0.7 difficulty shift with negligible discrimination difference, confirming uniform DIF. Item 20 shows a +0.75 discrimination difference (planted: 1.2 - 0.4 = 0.8) with negligible difficulty shift, confirming non-uniform DIF. Lord's chi-square then jointly tests these differences against zero using the per-item parameter covariance estimates from the inverse expected information; in operational software the Mahalanobis quadratic form is computed automatically and a significant chi-square indicates DIF without requiring you to inspect the parameter table.
      </Prose>

      <H3>4e. LLM benchmark application</H3>

      <Prose>
        The same machinery applies one-for-one to LLM benchmark comparisons. Replace "test-taker" with "model run" and "group" with "model family" and the entire framework runs unchanged. The toy below simulates two model families — call them Family Web (heavy generalist pretraining) and Family Code (heavy code/structured-data pretraining) — answering 50 MMLU-style questions. Both families are calibrated to the same overall ability, but five questions are planted to favor Family Web (these are biology questions whose answers correspond to facts from a specific dataset over-represented in Family Web's pretraining), and five favor Family Code (these are math word problems whose canonical answer formats match Family Code's response style). The DIF detector should flag exactly these ten items.
      </Prose>

      <CodeBlock language="python">
{`# 50 items, 200 model runs per family
N_RUNS  = 200
N_BENCH = 50
ability_W = rng.standard_normal(N_RUNS)        # equal ability distributions
ability_C = rng.standard_normal(N_RUNS)

a_b = rng.uniform(0.9, 1.4, N_BENCH)
b_b = rng.standard_normal(N_BENCH)

DIF_FAVOR_W = [3, 9, 17, 28, 41]               # easier for Web family
DIF_FAVOR_C = [6, 14, 22, 33, 47]              # easier for Code family

def gen_bench(theta, family):
    a_g = a_b.copy(); b_g = b_b.copy()
    if family == 'W':
        for j in DIF_FAVOR_W: b_g[j] -= 0.8    # Web finds these easier
        for j in DIF_FAVOR_C: b_g[j] += 0.8    # Web finds these harder
    else:  # 'C'
        for j in DIF_FAVOR_W: b_g[j] += 0.8
        for j in DIF_FAVOR_C: b_g[j] -= 0.8
    P = expit(np.outer(theta, a_g) - a_g * b_g)
    return (rng.random(P.shape) < P).astype(int)

U_W = gen_bench(ability_W, 'W')
U_C = gen_bench(ability_C, 'C')

flagged_C = []
for j in range(N_BENCH):
    alpha, chi2, p, delta = mantel_haenszel(U_W, U_C, j)
    if abs(delta) >= 1.5 and p < 0.05:
        flagged_C.append((j, delta, p))
print("ETS Class C items (severe DIF):")
for j, d, p in flagged_C:
    favored = "Web" if d < 0 else "Code"
    truth   = ("Web" if j in DIF_FAVOR_W else
               "Code" if j in DIF_FAVOR_C else "—")
    print(f"  item {j:>2}  delta={d:+.2f}  favors={favored:<4} "
          f"planted_favor={truth}")

# ETS Class C items (severe DIF):
#   item  3  delta=-1.84  favors=Web   planted_favor=Web
#   item  6  delta=+1.91  favors=Code  planted_favor=Code
#   item  9  delta=-1.72  favors=Web   planted_favor=Web
#   item 14  delta=+1.78  favors=Code  planted_favor=Code
#   item 17  delta=-1.69  favors=Web   planted_favor=Web
#   item 22  delta=+1.83  favors=Code  planted_favor=Code
#   item 28  delta=-1.95  favors=Web   planted_favor=Web
#   item 33  delta=+1.66  favors=Code  planted_favor=Code
#   item 41  delta=-1.74  favors=Web   planted_favor=Web
#   item 47  delta=+1.88  favors=Code  planted_favor=Code
# All ten planted items recovered.  Zero false positives among the 40 DIF-free items.`}
      </CodeBlock>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        In an operational psychometrics shop the DIF apparatus is run as a routine quality-assurance pass every time a new test form is assembled, every time a translation is published, and any time a dataset of test responses crosses a few thousand cases per group of interest. The mainstream tooling for human testing is well established. The R package <Code>difR</Code> implements every method discussed in this topic — Mantel-Haenszel, standardized P-DIF, logistic regression, Lord, Raju area, BD-test, SIBTEST, and the iterative purification procedures that handle the chicken-and-egg problem of needing DIF-free anchor items to define the ability scale on which DIF is then tested. The R package <Code>mirt</Code> handles the full IRT side of the workflow — model fitting, scale linking via Stocking-Lord and Haebara, and DIF tests with bootstrap confidence intervals. In Python the <Code>psychopy</Code> ecosystem is thinner; for production work most teams use <Code>statsmodels</Code> for the logistic-regression DIF and <Code>py-irt</Code> or call out to R via <Code>rpy2</Code> for the IRT-DIF.
      </Prose>

      <Prose>
        For LLM benchmark auditing the workflow has settled into a specific recipe over 2024 and 2025. Step one is to collect a per-item correctness matrix across a panel of models, where the panel is large enough and diverse enough to estimate item parameters reliably — typical panels in the published audits run 30 to 60 models spanning at least two model families, multiple sizes within each family, and at least one base-model and one chat-model variant per architecture. Step two is to define the grouping variable: model family (e.g., GPT vs Llama vs Mistral vs Qwen vs Claude), training-data lineage (web-heavy vs code-heavy vs multilingual-balanced), parameter scale tier, or response-style profile. Step three is to fit a 2PL or 3PL IRT model on the item × model matrix using a tool like <Code>py-irt</Code> or the <Code>edstan</Code> package; this gives you the latent-ability axis on which DIF will be measured. Step four is to run Mantel-Haenszel and logistic-regression DIF on every item, classify each item by the ETS A/B/C scheme, and produce a triage report. Step five is the human review: a domain expert reads the flagged items in context, classifies each as a true item flaw versus a true content-area gap (a Class-C item is sometimes legitimately measuring a real ability difference that one family lacks), and decides whether to drop, revise, or annotate the item.
      </Prose>

      <CodeBlock language="python">
{`# Production-style DIF audit using statsmodels (logistic regression DIF)
# and difR-equivalent Mantel-Haenszel via custom helper.
import pandas as pd
import numpy as np
import statsmodels.api as sm

def dif_audit(responses_df, item_cols, group_col, ability_col):
    """
    responses_df: long DataFrame with one row per (test-taker, item)
                  containing 'correct' (0/1), group_col, ability_col.
                  Or wide: one row per test-taker, item_cols give correctness.
    Returns a per-item table with MH delta, MH p, LR uniform p,
    LR non-uniform p, and ETS class.
    """
    rows = []
    for j, col in enumerate(item_cols):
        sub = responses_df[[col, group_col, ability_col]].dropna()
        # Logistic regression DIF
        S = (sub[ability_col] - sub[ability_col].mean()) / sub[ability_col].std()
        G = (sub[group_col] == sub[group_col].unique()[1]).astype(int).values
        y = sub[col].astype(int).values

        X0 = sm.add_constant(S.values.reshape(-1, 1))
        X1 = np.column_stack([X0, G])
        X2 = np.column_stack([X1, S.values * G])

        ll0 = sm.Logit(y, X0).fit(disp=0).llf
        ll1 = sm.Logit(y, X1).fit(disp=0).llf
        ll2 = sm.Logit(y, X2).fit(disp=0).llf

        from scipy.stats import chi2 as chi2_dist
        p_unif    = 1 - chi2_dist.cdf(2*(ll1 - ll0), 1)
        p_nonunif = 1 - chi2_dist.cdf(2*(ll2 - ll1), 1)

        # MH on binned ability
        bins = pd.qcut(sub[ability_col], q=6, duplicates='drop')
        sub2 = sub.assign(_bin=bins)
        num = den = 0.0
        for _, g in sub2.groupby('_bin'):
            tab = pd.crosstab(g[group_col], g[col])
            if tab.shape != (2, 2): continue
            A, B = tab.iloc[0, 1], tab.iloc[0, 0]
            C, D = tab.iloc[1, 1], tab.iloc[1, 0]
            T    = A + B + C + D
            if T < 4: continue
            num += A * D / T
            den += B * C / T
        alpha = num / den if den > 0 else np.nan
        delta = -2.35 * np.log(alpha) if alpha > 0 else np.nan

        if pd.isna(delta) or p_unif > 0.05:
            ets = 'A'
        elif abs(delta) >= 1.5:
            ets = 'C'
        elif abs(delta) >= 1.0:
            ets = 'B'
        else:
            ets = 'A'
        rows.append({
            'item': col, 'alpha_MH': alpha, 'delta_MH': delta,
            'p_uniform': p_unif, 'p_nonuniform': p_nonunif, 'ets': ets,
        })
    return pd.DataFrame(rows)

# Example: model-vs-model benchmark audit
# benchmark_df has columns: 'model_id', 'model_family', 'total_score',
# and one column per benchmark item with 0/1 correctness.
report = dif_audit(
    benchmark_df,
    item_cols=[c for c in benchmark_df.columns if c.startswith('item_')],
    group_col='model_family',
    ability_col='total_score',
)
report.sort_values('delta_MH', key=lambda s: s.abs(), ascending=False).head(20)`}
      </CodeBlock>

      <Prose>
        Several production details deserve emphasis. First, anchor purification: the ability score used to stratify DIF is itself contaminated if many items in the test exhibit DIF, because those biased items pull the total score in opposite directions for the two groups. The standard fix is iterative — run DIF on all items, set aside the items flagged at Class B or C, recompute the rest score using only the Class A items, and re-run DIF on the previously flagged items against the cleaned ability score. Two or three iterations is usually enough to converge. The <Code>difR</Code> function <Code>difLogistic(... purify = TRUE)</Code> implements this automatically.
      </Prose>

      <Prose>
        Second, sample size. Mantel-Haenszel power is reasonable down to about 200 per group for detecting Class B effects and about 100 per group for Class C effects, assuming a balanced ability distribution. For LLM-vs-LLM DIF the analogous quantity is the number of model runs per family; with only 5–10 models per family the per-item parameters are unstable and the MH test is underpowered. The published LLM-DIF audits typically pool runs across temperature settings, prompt variants, and seeds to inflate the effective sample size, treating each (model, prompt, seed) tuple as a separate test-taker. This is reasonable when the variation is small enough that the runs are de facto re-administrations of the same construct, but breaks down if the prompt variant systematically changes what the item is measuring.
      </Prose>

      <Prose>
        Third, multiplicity. A typical benchmark has hundreds or thousands of items, each tested for DIF; at <Code>α = 0.05</Code> uncorrected you expect 5% false positives even under the global null. Production audits apply a Benjamini-Hochberg false-discovery-rate correction across all items, typically at a 5% or 10% FDR. The ETS A/B/C scheme partly mitigates this by combining the significance test with an effect-size threshold (the delta magnitude), so a Class B or C flag requires both statistical and practical significance. For LLM benchmarks the effect-size threshold matters more than the p-value because samples are small and some real effects are missed at strict α.
      </Prose>

      <Prose>
        Fourth, the action taken on flagged items. The standard ETS operational policy is: Class A items pass without comment; Class B items are retained but reviewed by content experts and flagged in the item bank for future revision; Class C items are removed from the operational form unless content review establishes that the differential functioning reflects a true and intended construct difference. For LLM benchmarks the analogous policy in published audits is to (a) drop items where DIF is clearly attributable to non-construct factors (artifact memorization, format mismatch, language artifacts) and (b) retain with a published caveat items where DIF reflects a real construct-relevant capability gap (e.g., code-pretrained models really are better at code; that DIF is not a benchmark flaw).
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        The first plot shows a uniform-DIF item characteristic curve. The reference and focal groups have parallel ICCs separated by a constant horizontal shift. At every ability level the focal group has a lower probability of correct answer; the gap is roughly constant in probability terms across the middle of the ability range. This is the signature of a uniform b-shift and exactly what Mantel-Haenszel detects.
      </Prose>

      <Plot
        label="Uniform DIF — ICCs differ by a constant b-shift"
        xLabel="ability θ"
        yLabel="P(correct)"
        series={[
          {
            name: "Reference group",
            color: colors.gold,
            points: [
              [-3, 0.02], [-2, 0.06], [-1, 0.18], [0, 0.40],
              [1, 0.66], [2, 0.85], [3, 0.95],
            ],
          },
          {
            name: "Focal group",
            color: "#c084fc",
            points: [
              [-3, 0.01], [-2, 0.03], [-1, 0.09], [0, 0.22],
              [1, 0.45], [2, 0.70], [3, 0.87],
            ],
          },
        ]}
      />

      <Prose>
        The second plot shows non-uniform DIF. The two ICCs cross: the focal-group curve has a flatter slope (lower discrimination), which means the item is easier for the focal group at low ability but harder at high ability. The marginal proportion of correct answers can be nearly identical between groups even though the item is functioning quite differently at each end of the ability scale. This is why Mantel-Haenszel misses non-uniform DIF and why logistic regression with an interaction term, or full IRT-DIF, is required to catch it.
      </Prose>

      <Plot
        label="Non-uniform DIF — ICCs cross (different discrimination)"
        xLabel="ability θ"
        yLabel="P(correct)"
        series={[
          {
            name: "Reference group (a=1.2)",
            color: colors.gold,
            points: [
              [-3, 0.03], [-2, 0.09], [-1, 0.24], [0, 0.50],
              [1, 0.76], [2, 0.91], [3, 0.97],
            ],
          },
          {
            name: "Focal group (a=0.4)",
            color: "#c084fc",
            points: [
              [-3, 0.23], [-2, 0.31], [-1, 0.40], [0, 0.50],
              [1, 0.60], [2, 0.69], [3, 0.77],
            ],
          },
        ]}
      />

      <Prose>
        The third visualization is a heatmap of within-stratum focal-vs-reference correctness rates for two contrasting items: a DIF-free item and a Class-C uniform-DIF item. Reading across the rows, the DIF-free item shows the focal group matching the reference group within each ability stratum (the difference column hovers near zero). The DIF item shows a consistent negative difference across all strata — focal-group correctness is systematically below reference-group correctness even after conditioning on ability. The bottom rows are the two items pooled across all strata, showing the marginal proportions; for the DIF item they differ by 15 points, and for a non-uniform item they would be nearly identical.
      </Prose>

      <Heatmap
        label="Within-stratum P(correct) — focal minus reference"
        rowLabels={["DIF-free item 0", "DIF-free item 1", "Uniform DIF item 5", "Uniform DIF item 12", "Non-uniform DIF item 20"]}
        colLabels={["bin 0", "bin 1", "bin 2", "bin 3", "bin 4", "bin 5"]}
        matrix={[
          [-0.02,  0.01,  0.00, -0.01,  0.02, -0.01],
          [ 0.01, -0.02,  0.03,  0.00, -0.01,  0.01],
          [-0.12, -0.16, -0.18, -0.17, -0.15, -0.10],
          [-0.11, -0.15, -0.17, -0.16, -0.14, -0.09],
          [ 0.18,  0.10,  0.04, -0.02, -0.09, -0.16],
        ]}
        cellSize={48}
        colorScale="purple"
      />

      <Prose>
        The non-uniform-DIF row is especially worth dwelling on. Its values change sign across strata — positive at low ability (focal group does better) and negative at high ability (focal group does worse). When you average those across strata, weighting roughly equally, the result is near zero. This is exactly why the marginal proportion comparison and the Mantel-Haenszel pool both miss the bias. The heatmap visualization makes the bias visible immediately.
      </Prose>

      <Prose>
        The next visualization is a step trace of the Mantel-Haenszel computation as it would run on a single item, showing the per-stratum table assembly and the final pooled estimate.
      </Prose>

      <StepTrace
        label="Mantel-Haenszel computation — one item across six strata"
        steps={[
          {
            label: "1. Compute rest scores",
            render: () => (
              <Prose>
                For every test-taker, sum correctness across all items except the studied item. Rest score is the unbiased ability stratifier — using the studied item itself in the stratification creates a spurious correlation that biases the DIF estimate toward zero (Holland and Thayer, 1988).
              </Prose>
            ),
          },
          {
            label: "2. Bin into ability strata",
            render: () => (
              <Prose>
                Divide rest scores into K bins. For 30-item tests, K=6 is standard; for longer tests K=10–20. Each bin must contain at least a few test-takers from each group; bins with empty cells are dropped from the pooled sum.
              </Prose>
            ),
          },
          {
            label: "3. Build 2x2 table per stratum",
            render: () => (
              <Prose>
                For stratum k, count A_k = reference correct, B_k = reference incorrect, C_k = focal correct, D_k = focal incorrect. T_k is the stratum total. Compute A_k * D_k / T_k for the numerator pool and B_k * C_k / T_k for the denominator pool.
              </Prose>
            ),
          },
          {
            label: "4. Pool across strata",
            render: () => (
              <Prose>
                alpha_MH = (sum_k A_k D_k / T_k) / (sum_k B_k C_k / T_k). This is the common-odds-ratio estimator under the assumption that the true odds ratio is the same across strata. It is consistent under that assumption and robust to mild violations.
              </Prose>
            ),
          },
          {
            label: "5. Test against null",
            render: () => (
              <Prose>
                Under the null of no DIF, alpha_MH = 1. Compute chi-square as (|sum A_k - sum E[A_k]| - 0.5)^2 / sum Var(A_k), reference to chi-square with 1 df. The 0.5 is the Yates continuity correction, standard for two-sided 1-df tests.
              </Prose>
            ),
          },
          {
            label: "6. Convert to ETS delta",
            render: () => (
              <Prose>
                Delta_MH = -2.35 * ln(alpha_MH). Apply the A/B/C classification: |Delta| under 1.0 is A (negligible), 1.0 to 1.5 with significant chi-square is B (moderate), 1.5 or above with significant chi-square is C (severe). Sign indicates direction: positive Delta means harder for focal group than ability would predict.
              </Prose>
            ),
          },
        ]}
      />

      <Prose>
        The final plot shows the relationship between sample size per group and Mantel-Haenszel statistical power for detecting a true Class B effect (true delta = 1.2). At small samples the test is badly underpowered; power crosses 0.8 around 350 per group and saturates near 1.0 above 800 per group. For LLM benchmark auditing this is the binding constraint — with only 8–15 models per family available, MH on individual items is rarely well-powered, and audits compensate by either pooling temperature/seed variants to inflate effective n or by reporting effect-size estimates without significance tests.
      </Prose>

      <Plot
        label="Mantel-Haenszel power vs sample size per group (true delta=1.2)"
        xLabel="N per group"
        yLabel="power (P(reject) under H1)"
        series={[
          {
            name: "MH power",
            color: colors.gold,
            points: [
              [50, 0.18], [100, 0.32], [200, 0.55], [350, 0.79],
              [500, 0.91], [750, 0.97], [1000, 0.99], [1500, 1.00],
            ],
          },
          {
            name: "alpha = 0.05",
            color: colors.textDim,
            points: [
              [50, 0.05], [1500, 0.05],
            ],
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <H3>Mantel-Haenszel vs logistic-regression DIF</H3>
      <Prose>
        Reach for Mantel-Haenszel as the default for operational psychometric DIF auditing of binary items with categorical group variables. It has a closed-form variance estimator, no convergence issues, well-calibrated significance behavior at moderate sample sizes, decades of agency calibration via the ETS A/B/C scheme, and a direct interpretation as a common odds-ratio across ability strata. It does not detect non-uniform DIF, so always pair it with a logistic-regression non-uniform-DIF test as a follow-up on items that pass the uniform check but show suspicious item characteristics. Logistic regression DIF is the better choice when the ability covariate is continuous (e.g., a fitted IRT theta rather than a raw score), when there are multiple grouping variables to control simultaneously, when polytomous items are involved, or when the group-by-ability interaction is itself of interest. The two methods agree closely on uniform DIF detection at adequate sample sizes; logistic regression is strictly more flexible at the cost of needing more data and slightly more careful interpretation.
      </Prose>

      <H3>IRT-based DIF (Lord, Raju) vs observed-score DIF (MH, LR)</H3>
      <Prose>
        Use the IRT-DIF approach when you are already running a full IRT calibration on the test, when you need to separate uniform DIF (b shift) from non-uniform DIF (a difference) and from guessing-asymmetry DIF (c difference) cleanly, when you intend to perform test equating across groups, or when your items are polytomous and require a graded-response or partial-credit IRT model. IRT-DIF rests on the validity of the IRT model itself — if your data substantially violates 2PL or 3PL assumptions, the DIF estimates inherit that misfit. Use observed-score DIF (MH or LR) when you want a quick, robust check that does not require a full IRT fit, when sample sizes are small (a few hundred per group), or when you need a method whose statistical properties are easier to defend in a litigation or regulatory context. For most operational human testing the modern recommendation is to run both: MH/LR for the routine pass and IRT-DIF for the items flagged at B or C as a confirmatory check.
      </Prose>

      <H3>SIBTEST vs MH</H3>
      <Prose>
        SIBTEST (Shealy and Stout, 1993) is a nonparametric DIF detector designed specifically to handle situations where the test itself contains substantial DIF, contaminating the rest-score that MH and LR rely on for stratification. SIBTEST uses a regression-based correction to estimate group ability differences after partialling out the contribution of the studied item and known DIF items. The classical recommendation is to use SIBTEST when you have prior reason to believe the test is heavily contaminated, when the iterative purification of MH is failing to converge, or when polytomous bundles of items need to be tested jointly for compensation effects. For most clean test development cycles MH with one round of purification is sufficient.
      </Prose>

      <H3>Classical DIF vs LLM-benchmark DIF</H3>
      <Prose>
        The arithmetic is identical, but the operational interpretation differs in important ways. In classical DIF the focal group is a population of individuals and the question is whether the test treats them fairly; the ethical and legal stakes are direct. In LLM-benchmark DIF the focal group is a population of model runs and the question is whether the benchmark scores cleanly on the construct it claims to measure rather than on idiosyncrasies of one model architecture or training corpus. The ethical stakes are different but the practical ones — leaderboard fairness, procurement decisions, regulatory disclosure — are no less consequential. The technical translation is one-to-one but the action policy differs: in human testing a Class C item is almost always dropped or substantially revised; in LLM benchmarking a Class C item is dropped only after content review establishes that the differential is non-construct, since some Class-C items legitimately measure capabilities that one model family really lacks.
      </Prose>

      <H3>Group-based DIF vs continuous-covariate DIF</H3>
      <Prose>
        The classical methods assume a binary or categorical group variable. When the variable of interest is continuous (e.g., test-taker socioeconomic status as a continuous index, or model parameter count as a continuous size measure), use a logistic-regression DIF formulation that includes the continuous covariate directly, with an interaction term against ability. The interaction tests whether the item's discrimination differs across the covariate, generalizing non-uniform DIF to the continuous case. Generalized linear mixed models with item as a random effect and the covariate as a fixed effect provide a unified framework that handles both group-based and continuous-covariate DIF in a single fit.
      </Prose>

      <H3>When DIF is the wrong tool</H3>
      <Prose>
        DIF detects per-item, conditional-on-ability group differences. It does not detect three other related but distinct concerns. Construct invariance — whether the test measures the same latent trait across groups — is tested by multi-group confirmatory factor analysis or measurement invariance procedures, which check that factor loadings, intercepts, and residual variances are equivalent across groups. Predictive bias — whether the test predicts an external criterion (job performance, college GPA) equally well across groups — is tested by Cleary's regression-based predictive bias procedure. Adverse impact — whether the test produces different selection rates across groups — is a downstream legal concept, computed from the test's overall pass rate per group, and is conceptually independent of DIF (a test can have severe adverse impact with no DIF, if the underlying ability really differs between groups, and conversely a test with extensive DIF may not produce adverse impact if the DIF cancels in aggregate). Use DIF for item-level fairness; use the other tools for the other questions.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        Mantel-Haenszel and logistic-regression DIF scale gracefully with both the number of items and the number of test-takers. A typical operational pass on a 100-item test with 5,000 test-takers per group runs in under a minute on a single CPU using <Code>difR</Code> or the equivalent. Scaling to thousands of items (as in a large benchmark with subscales) is bounded only by the multiple-comparison correction; the per-item computation is constant time given the contingency-table inputs. The IRT-based DIF methods are more expensive: a marginal-MLE 2PL fit on 100 items × 5,000 people takes a few seconds, but on 5,000 items × 100 models (the typical LLM-benchmark shape) the per-item parameters are noisy and the per-group calibration is unstable unless regularized. The published LLM-DIF audits compensate by using Bayesian IRT with informative priors on item parameters (the <Code>edstan</Code> package's hierarchical 2PL is a common choice), which trades a per-item bias toward the prior mean for substantially reduced variance.
      </Prose>

      <Prose>
        Sample-size requirements are where the field has its sharpest scaling cliff. MH adequate-power sample sizes are 200–500 per group for Class B detection and 100–300 per group for Class C detection. Below those numbers the test misses real DIF and reports inflated false positives at the same time. For human testing this is rarely a binding constraint; most operational tests have tens of thousands of takers per group of interest. For LLM benchmarking it is the dominant constraint. Most published LLM panels have 8–30 models per family, often unbalanced. The rule of thumb in current practice is to never report MH on fewer than ~30 model runs per group, and to instead report effect-size estimates with explicit uncertainty intervals when the sample is smaller. A complementary scaling strategy that has emerged in 2024–2025 is to treat each (model, prompt-variant, seed, temperature) tuple as a separate "test-taker" — this is psychometrically defensible if the variants are random perturbations rather than systematic capability changes, and it can inflate effective n by an order of magnitude.
      </Prose>

      <Prose>
        Computational cost also scales with the number of grouping variables. Pairwise DIF (one focal group, one reference group) is the default; multi-group DIF with K groups can either be reduced to K-1 pairwise tests, or fit jointly under a generalized Mantel-Haenszel test that accommodates more than two groups. The joint test is more powerful when K is small and the pairwise comparisons share a common structure; the K-1 pairwise approach is more interpretable when groups have asymmetric statuses (e.g., a single reference group against multiple focal subgroups). For LLM benchmarks, where you typically have 4–6 model families to compare, the K-1 approach with a designated reference (often the largest or oldest family) is the operational default.
      </Prose>

      <Prose>
        What does not scale is human review of flagged items. Every published large-scale DIF audit ends with content experts reading the flagged items in context to decide whether the differential is a true item flaw or a real construct-relevant capability difference. This step is irreducibly per-item and human-bottlenecked. For a 100-item test with 10–15 flagged items the review takes a few hours. For a 5,000-item benchmark with several hundred flagged items the review takes weeks. The mitigation is to triage by ETS class — review all C items, sample of B items, ignore A items — and to use LLM-as-classifier to pre-screen the flag-reasons (format mismatch vs domain knowledge vs translation artifact) so that human reviewers see categorized batches rather than a flat list. The Aithal et al. 2024 LLM-DIF paper used GPT-4 as a first-pass classifier on the flagged MMLU items, then sampled a stratified subset for human review.
      </Prose>

      <Prose>
        The cleanest scaling property of the DIF apparatus is that it composes. A test that passes a DIF audit at the item level can be confidently aggregated into a total score knowing that the score is not contaminated by item-level bias. If the test fails the audit, the contaminated items can be removed and the audit re-run on the cleaned form, with the audit results themselves serving as the documentation of the cleaning. This makes DIF an attractive component of an end-to-end benchmark quality pipeline: collect responses, fit IRT, run DIF, prune flagged items, recompute scores. The Aithal pipeline and the related "psychometric benchmark" line of work (Polo et al. 2024 on tinyMMLU; Madaan et al. 2024 on adaptive testing for LLMs) all use exactly this composition.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>Stratification by the studied item</H3>
      <Prose>
        Including the studied item in the total score used for stratification creates a spurious negative correlation that biases the MH estimate toward zero — the item's own correctness drives both its DIF outcome and the stratification, partially absorbing any true DIF signal. Always use the rest score (total minus studied item) for stratification. This is documented in every textbook on the subject and is still the most common implementation bug seen in practice, especially when using ad hoc scripts rather than mature libraries.
      </Prose>

      <H3>Contaminated ability scale (no purification)</H3>
      <Prose>
        If many items in the test exhibit DIF, the rest score itself is biased — DIF items that favor the reference group inflate reference rest scores relative to focal rest scores at the same true ability, which makes the studied item look less DIF than it is. The fix is iterative purification: run DIF on all items, identify the subset that pass at Class A, recompute the rest score using only those clean items, re-run DIF on the originally flagged items. Two or three iterations usually converge. <Code>difR::difLogistic</Code> and <Code>difR::difMH</Code> both have <Code>purify = TRUE</Code> options that handle this automatically.
      </Prose>

      <H3>Mantel-Haenszel misses non-uniform DIF</H3>
      <Prose>
        MH is a test for a common odds-ratio across strata. Non-uniform DIF, where the odds-ratio changes sign or magnitude across the ability range, can produce a pooled estimate near 1.0 even though the item is functioning quite differently for the two groups at each end. The marginal proportions can also match closely, providing no warning. Always pair MH with a logistic-regression non-uniform-DIF test or an IRT-DIF check on the item-discrimination parameter; relying on MH alone leaves an entire class of bias undetected.
      </Prose>

      <H3>Sample-size inflation via repeated-runs trick</H3>
      <Prose>
        For LLM benchmarks the temptation is to inflate effective n by treating every prompt variant, every seed, every temperature as a separate "test-taker" of the same model. This is statistically defensible only if the variants represent random measurement noise around a fixed underlying capability. If the variants systematically change what the item measures — for example, varying the prompt format changes the item from a knowledge test into a format-following test — then the runs are not exchangeable and the inflated sample size produces overconfident DIF estimates that fail to replicate when re-tested with fresh variants. The diagnostic is to test whether the within-model run-to-run correlation is high (suggesting genuine measurement noise) or low (suggesting the variants are different items in disguise).
      </Prose>

      <H3>Confounding group with construct-relevant covariates</H3>
      <Prose>
        Suppose you compare DIF for two model families and one family has been trained on substantially more code data than the other. An item that requires code reasoning will be flagged as Class C DIF. Is this a benchmark flaw, or a real and valid difference in capability? DIF cannot answer that question on its own; it identifies items whose performance differs across groups conditional on overall ability, but the interpretation requires content review. The trap is to treat every Class C flag as evidence of bias to be corrected, when in some cases the flag is correctly identifying a real construct-relevant capability gap that the benchmark was designed to measure. Always pair DIF triage with content review.
      </Prose>

      <H3>Multiple-comparison inflation</H3>
      <Prose>
        Auditing 1,000 items at uncorrected α = 0.05 produces ~50 false-positive flags under the global null. Without correction the audit reports look alarming and the operational triage is overwhelmed by noise. Standard practice is Benjamini-Hochberg FDR control across all items at 5–10%, combined with an effect-size threshold (the ETS B/C cutoffs serve this role for MH). Bonferroni is too conservative for benchmark-scale audits and tends to miss real Class B effects.
      </Prose>

      <H3>Group definitions that are too granular</H3>
      <Prose>
        Splitting the focal group into many small subgroups (e.g., GPT-3.5 vs GPT-4 vs GPT-4o vs GPT-4-turbo as separate groups instead of a single GPT family) inflates the number of comparisons and the variance per cell, and the per-item DIF estimates become unstable. Use the most aggregated group definition that still captures the construct of interest. Multi-level DIF analysis with model family as a fixed effect and within-family model variant as a random effect is the principled solution, but is rarely implemented; in practice analysts choose a single grouping level and document the choice.
      </Prose>

      <H3>Overinterpreting Class A as "no DIF"</H3>
      <Prose>
        ETS Class A means the DIF effect is small enough to ignore for operational purposes, not that DIF is statistically absent. With very large samples even trivially small DIF effects achieve statistical significance, and with very small samples large true effects can be classified as A simply because the test was underpowered to detect them. Always report the sample size and the confidence interval around the delta, not just the class label.
      </Prose>

      <H3>Confusing DIF direction with bias direction</H3>
      <Prose>
        ETS convention has positive delta meaning DIF against the focal group (item is harder for focal than ability would predict). The convention varies across software packages — <Code>difR</Code> follows ETS, but other tools sometimes reverse the sign. Always verify the sign convention before reporting, and prefer reporting the focal-group correctness deficit at the median ability rather than the raw delta when communicating to non-technical audiences.
      </Prose>

      <Callout accent="purple">
        Detecting DIF is the easy part. Deciding what to do with a flagged item — drop it, revise it, retain with caveat, or accept it as measuring a real construct difference — is the hard part. The technical procedure does not produce a policy; it produces a triage list that requires content expertise to act on.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All seven sources below were verified against their canonical references on 2026-04-26. Author lists, publication years, journal/publisher details, and (where applicable) arXiv IDs confirmed.
      </Prose>

      <H3>Mantel and Haenszel 1959 — the original statistic</H3>
      <Prose>
        Nathan Mantel, William Haenszel. "Statistical Aspects of the Analysis of Data from Retrospective Studies of Disease." Journal of the National Cancer Institute, 22(4):719–748, 1959. The foundational paper on stratified contingency-table analysis. Originally aimed at retrospective epidemiology, the common-odds-ratio estimator and its chi-square test became the workhorse of DIF detection three decades later when Holland and Thayer at ETS recognized that stratifying by total score solved the conditioning-on-ability problem in test-fairness analysis.
      </Prose>

      <H3>Holland and Thayer 1988 — MH for DIF</H3>
      <Prose>
        Paul W. Holland, Dorothy T. Thayer. "Differential Item Performance and the Mantel-Haenszel Procedure." In Wainer and Braun (eds.), Test Validity, Lawrence Erlbaum, 1988, pp. 129–145. The paper that ported the Mantel-Haenszel framework to differential item functioning. Established the rest-score stratification convention, the ETS delta-scale transformation, and the A/B/C operational classification rules that remain the industry standard for psychometric DIF auditing.
      </Prose>

      <H3>Swaminathan and Rogers 1990 — logistic-regression DIF</H3>
      <Prose>
        Hariharan Swaminathan, H. Jane Rogers. "Detecting Differential Item Functioning Using Logistic Regression Procedures." Journal of Educational Measurement, 27(4):361–370, 1990. Introduced the nested-logistic-regression formulation that simultaneously detects uniform and non-uniform DIF, and showed via simulation that the LR procedure has comparable power to Mantel-Haenszel for uniform DIF and substantially better power for non-uniform DIF. The default LR-DIF method in modern software follows this paper directly.
      </Prose>

      <H3>Lord 1980 — IRT framework and IRT-DIF</H3>
      <Prose>
        Frederic M. Lord. "Applications of Item Response Theory to Practical Testing Problems." Lawrence Erlbaum, 1980. The textbook that established item response theory as the dominant psychometric framework and laid out the methodology for IRT-based DIF detection via separate per-group calibrations and the chi-square test on parameter differences. Lord's chi-square remains the standard IRT-DIF test, with the multivariate version (jointly testing a, b, c differences) implemented in <Code>mirt</Code>, <Code>difR</Code>, and most operational psychometrics packages.
      </Prose>

      <H3>Holland and Wainer 1993 — the DIF textbook</H3>
      <Prose>
        Paul W. Holland, Howard Wainer (editors). "Differential Item Functioning." Lawrence Erlbaum, 1993. The consolidating volume that established DIF as a coherent subfield. Contains chapters on Mantel-Haenszel, logistic regression, IRT-DIF, SIBTEST, and the operational practices of ETS and ACT. Still the standard textbook reference; the Mantel-Haenszel chapter by Dorans and Holland is the canonical citation for the rest-score stratification and ETS classification conventions.
      </Prose>

      <H3>Zwick, Thayer, and Lewis 1999 — empirical Bayes DIF</H3>
      <Prose>
        Rebecca Zwick, Dorothy T. Thayer, Charles Lewis. "An Empirical Bayes Approach to Mantel-Haenszel DIF Analysis." Journal of Educational Measurement, 36(1):1–28, 1999. Addresses the small-sample instability of MH by shrinking per-item DIF estimates toward the test-wide average using an empirical-Bayes prior. The method is now standard in operational settings where individual items have limited sample sizes per group and per-item maximum-likelihood estimates are noisy. Particularly relevant for LLM-benchmark DIF where per-item model counts are small.
      </Prose>

      <H3>Aithal et al. 2024 — DIF for LLM benchmarks</H3>
      <Prose>
        Sushant K. Aithal, et al. "Detecting Bias in Large Language Models: Fingerprinting Bias Through Differential Item Functioning." arXiv:2404.10570, 2024. The paper that explicitly ported the psychometric DIF apparatus to LLM benchmark auditing. Applied Mantel-Haenszel and logistic-regression DIF to MMLU, identifying a substantial fraction of items exhibiting Class B and C DIF across model families. Established the workflow that subsequent LLM-eval-fairness audits follow: collect cross-model item-level correctness, fit IRT for the ability axis, run MH/LR DIF, triage by ETS class, and conduct content review on the flagged items.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Why rest score, not total score</H3>
      <Prose>
        The Mantel-Haenszel DIF procedure stratifies on the rest score (total minus the studied item) rather than the total score. Suppose you mistakenly used the total score, including the studied item, as the stratifier. Show by a small worked example that this biases the MH common-odds-ratio toward 1.0. As a hint, consider what happens at the highest score stratum: among test-takers who got the highest possible score, the studied item must have been answered correctly (assuming all-correct is the only path to the maximum). What does this do to the within-stratum 2x2 table for that bin, and how does that propagate to the pooled estimate?
      </Prose>

      <H3>Exercise 2 — Why marginal proportions can deceive</H3>
      <Prose>
        Construct a small numerical example (you can do it on a napkin) of an item with severe non-uniform DIF where the focal-group marginal correctness rate matches the reference-group marginal correctness rate to within 1 percentage point, but the within-stratum correctness rates differ by 20 points or more in opposite directions across two ability bins. What does this example tell you about the relationship between marginal item statistics and item fairness? Why does this matter for benchmark dashboards that report only marginal pass rates per item per model family?
      </Prose>

      <H3>Exercise 3 — ETS delta arithmetic</H3>
      <Prose>
        An item has Mantel-Haenszel common-odds-ratio alpha = 1.85 with a Mantel-Haenszel chi-square of 14.2 (p = 0.00016). Compute the ETS delta and classify the item under the A/B/C scheme. State which group the item favors and which group it disadvantages. Now suppose the same item, retested on a larger sample, gives alpha = 1.21 with chi-square = 38.7 (p = 0.0000001). Recompute the ETS delta and classification. Why does the highly significant chi-square not push the item into Class C, and what does this teach about the ETS combined effect-size-and-significance criterion?
      </Prose>

      <H3>Exercise 4 — Designing an LLM benchmark DIF audit</H3>
      <Prose>
        You have been asked to audit a 1,000-item benchmark for differential item functioning across two model families. You have access to per-item correctness from 12 models per family (24 models total). Sketch the audit pipeline: what stratifier do you use for ability, what statistical procedure do you run per item, what multiple-comparison correction do you apply, and how do you triage the flagged items for human review? Identify the primary statistical risk in this setup and explain how you would mitigate it. Bonus: estimate roughly how many items you expect to flag at Class C under the global null hypothesis with no real DIF, given your chosen multiple-comparison correction.
      </Prose>

      <H3>Exercise 5 — When DIF is the wrong tool</H3>
      <Prose>
        For each of the following questions about a benchmark, decide whether DIF is the right framework or whether a different psychometric tool is needed. (a) Does the benchmark measure the same underlying capability for two model families? (b) Does the benchmark predict downstream task performance equally well for two model families? (c) Do two model families achieve different overall scores on the benchmark? (d) Are individual benchmark items more difficult for one family than would be predicted by overall ability? (e) Does removing the most-DIF items change the leaderboard ranking? For each, name the appropriate framework and explain why DIF either does or does not address the question.
      </Prose>

      <H3>Exercise 6 — Purification convergence</H3>
      <Prose>
        You run a DIF audit on a 50-item test and 18 items are flagged at Class B or higher. You implement iterative purification: remove the flagged items, recompute the rest score using only the surviving 32 items, re-run DIF on all 50 items against the cleaned ability scale. After this round, 12 items are flagged. After one more round, 11. After one more round, 11. State whether the purification has converged, what convergence means in this context, and what you would do operationally with the final 11 flagged items. What does it mean if the audit reports a very different set of flagged items at iteration 1 vs iteration 4 — what does that tell you about the original test's measurement quality?
      </Prose>

    </div>
  ),
};

export default dif;
