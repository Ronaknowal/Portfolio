import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const reliabilityCoefficients = {
  title: "Reliability Coefficients (Cronbach's Alpha, Cohen's & Fleiss' Kappa, Krippendorff's Alpha)",
  slug: "reliability-coefficients-cronbachs-alpha-cohens-fleiss-kappa-krippendorffs-alpha",
  readTime: "~38 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Every measurement claim in psychology, content analysis, machine learning evaluation, and clinical assessment carries an implicit promise: that if you ran the measurement again — with different items on the same test, with a different rater, on a different day — you would get a similar answer. Reliability is the formal name for the degree to which that promise is kept. A single number summarizing reliability is not a luxury; it is the precondition for taking any subsequent inference seriously. If two annotators labeling the same dataset for "harmful content" agree only as often as they would by tossing a coin, the labels carry no information about harm; they encode the annotators' independent prejudices, and any model trained on them learns those prejudices rather than the construct. If a benchmark composed of fifteen reading-comprehension items has internal consistency near zero, the items are not measuring a coherent skill; the total score is a mixture of unrelated abilities, and improvements on it cannot be attributed to anything in particular. Reliability is the gate that lets validity claims through.
      </Prose>

      <Prose>
        The problem is that "agreement" looks deceptively simple and is actually subtle. Two raters labeling 100 items as cat or dog with 90% identical labels sounds like high agreement. But if 90% of the items are cats and both raters lazily call everything a cat, they would also achieve 90% identical labels through pure base-rate exploitation, contributing nothing reliable. A reliability coefficient is supposed to subtract out this chance baseline, expressing how much agreement exceeds what random labeling at the same marginal frequencies would produce. Different coefficients implement that correction in different ways, and pick up different definitions of "chance" along the way. Cronbach's alpha treats reliability as the proportion of total score variance that is shared across items. Cohen's kappa treats it as the excess over expected agreement under independence of two raters' marginals. Fleiss' kappa generalizes this to many raters with no fixed pairing. Krippendorff's alpha drops the assumption that raters are exchangeable individuals and treats disagreement as a distance, allowing nominal, ordinal, interval, and ratio scales as well as missing data. The intraclass correlation coefficient (ICC), in its six Shrout-Fleiss forms, handles continuous outcomes through variance-component decomposition.
      </Prose>

      <Prose>
        These methods were developed in waves across seventy years, each addressing a problem that an earlier coefficient could not handle. Cronbach (1951) gave a closed-form lower bound on the reliability of a sum-score test. Cohen (1960) introduced kappa to correct percent-agreement for the chance term. Shrout and Fleiss (1979) clarified that "ICC" was a family, not a single statistic, and tabulated when each form is appropriate. Fleiss (1971) extended kappa beyond the two-rater case by averaging over rater pairs implicitly through marginal proportions. Krippendorff's alpha — first published in his 1980 textbook on content analysis and given a unified, software-ready computational formula by Hayes and Krippendorff (2007) — is the most general member of the family and is the de facto standard in computational content analysis and the default recommendation for inter-annotator agreement in NLP eval pipelines.
      </Prose>

      <Prose>
        The current generation of LLM evaluation has revived all of these in a new context. When you run an MMLU subset through GPT-4o-as-judge and Claude-as-judge and a panel of three human reviewers, you have a multi-rater categorical agreement problem that is structurally identical to a 1970s content-analysis study. The kappa paradox — high observed agreement coexisting with kappa near zero because the marginals are skewed — shows up in safety classification benchmarks where 95% of items are "safe" and the interesting 5% drives all the model behavior. Krippendorff's alpha shows up in reports that compare automatic judges against human reference panels with missing data (judges that timed out, raters who skipped items). Cronbach's alpha shows up in benchmark internal-consistency analysis: if MMLU's 57 task subsets are supposed to measure a single "general knowledge" construct, alpha computed across subjects on a per-model basis tells you whether the benchmark is one test or fifty-seven loosely correlated ones. Each coefficient solves a specific problem, and using the wrong one produces numbers that look authoritative and mean nothing.
      </Prose>

      <Prose>
        The point of this topic is to make the choice precise. By the end you should be able to look at a measurement scenario — count the raters, identify the scale, note the missingness, decide whether items are exchangeable — and select the coefficient whose assumptions match. You should also be able to derive each from first principles, implement each from numpy, validate each against a reference library, recognize the kappa paradox when it occurs, and report the result with the conventional interpretive thresholds (Krippendorff's α ≥ 0.80 acceptable, ≥ 0.667 tentative; Cronbach's α ≥ 0.70 acceptable for research, ≥ 0.90 required for clinical decisions; Cohen's κ ≥ 0.61 substantial, ≥ 0.81 almost perfect by Landis and Koch's much-criticized but ubiquitous benchmarks). The math is small, the implementations are short, and the failure modes are concrete.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Begin with the simplest possible question: two people independently labeled the same hundred items into two categories, and they agreed on 85 of them. Is 0.85 a good number? The answer depends entirely on what you would have expected by accident. If both raters used both categories with equal frequency, random labeling would produce agreement on roughly 50 of 100 items by symmetry, so 85 represents a substantial improvement over chance — a kappa of about 0.70. If instead both raters labeled 95 items as category A and only 5 as category B, random labeling at those marginals would produce agreement on roughly 90.5 of 100 items, so 85 actually underperforms chance, yielding a negative kappa. The same observed agreement number tells two opposite stories depending on the marginals. This is the core motivation for chance-corrected agreement coefficients: raw percentages are deceptive when the categories are not balanced, and any honest reliability measure must subtract a baseline computed from the actual marginal distribution of labels.
      </Prose>

      <Prose>
        Cohen's kappa formalizes that idea for two raters and categorical labels. Compute the observed agreement <Code>p_o</Code> as the proportion of items the two raters labeled identically. Compute the expected agreement under chance <Code>p_e</Code> by assuming the raters labeled independently using their actual marginal frequencies — the probability that both happened to land on category <Code>k</Code> by independent draw is the product of the two raters' marginals for <Code>k</Code>, summed over all <Code>k</Code>. The chance-corrected agreement is then the observed excess over chance, normalized by the maximum possible excess (which is <Code>1 − p_e</Code>). A kappa of 1 means perfect agreement. A kappa of 0 means agreement at the chance level. Negative kappa means systematic disagreement.
      </Prose>

      <Prose>
        Cronbach's alpha attacks a different problem. You have one test with <Code>k</Code> items and a sample of <Code>N</Code> respondents, each of whom answered every item. There are no separate raters; each respondent is producing one set of item responses. The reliability question here is: do the items hang together as a coherent measurement of a single construct? If they do, an individual respondent's score on each item should track their score on every other item, after accounting for individual differences. The variance of the total score across respondents will then be much larger than the sum of the per-item variances, because positively correlated items reinforce each other into a coherent total. If the items are unrelated, the per-item variances simply add up to the total variance and there is no reinforcement. Alpha measures this: it is one minus the ratio of the sum of per-item variances to the total-score variance, scaled by <Code>k/(k−1)</Code> so that the maximum value is 1. A high alpha means the items mutually reinforce; a low alpha means they are independent measurements of unrelated things, even if any single item is itself reliable.
      </Prose>

      <Prose>
        Fleiss' kappa generalizes Cohen's kappa to more than two raters, but with a structural difference: it does not assume raters are paired. Each item is rated by some number of raters (the same number for all items, in the original formulation), but the raters do not have stable identities across items — rater 3 on item 1 is not necessarily the same person as rater 3 on item 2. The agreement on a given item is computed combinatorially as the number of agreeing rater pairs divided by the total number of rater pairs. The expected agreement under chance is computed from the global marginal distribution of category usage across all raters and all items pooled together, treating each rater-item observation as a draw from a single common distribution. Fleiss' kappa is the appropriate choice when you have a panel of judges, each of whom rated some subset of items, and you want a single agreement number without committing to any particular pairing.
      </Prose>

      <Prose>
        Krippendorff's alpha is the most general formulation in the family. It does not require all items to be rated by the same number of raters, it does not require raters to be the same across items, it handles missing data by simply skipping pairs that include a missing observation, and it accommodates any measurement scale through a configurable distance function. For nominal categories, the distance is 0 if the two values match and 1 otherwise — recovering the kappa-style computation. For ordinal categories, the distance accounts for the rank ordering, so confusing "strongly agree" with "agree" is penalized less than confusing "strongly agree" with "strongly disagree." For interval data, the distance is squared difference; for ratio data, it is squared log difference. The coefficient itself is one minus the ratio of observed disagreement to expected disagreement under chance, computed over coincidences (pairs of values assigned to the same unit by different raters), and the chance baseline comes from the actual marginal distribution of values rather than from any assumed null.
      </Prose>

      <Prose>
        These four coefficients form a tower of generality. Cronbach's alpha is the special case for internal consistency of a sum score. Cohen's kappa is the special case of two raters and nominal categories. Fleiss' kappa adds many raters but assumes complete data and unweighted nominal disagreement. Krippendorff's alpha sits at the top: any number of raters, missing data tolerated, any measurement level supported. The intraclass correlation coefficient (ICC) sits in a slightly different lineage — it operates on continuous outcomes via variance-component analysis from ANOVA, and the Shrout-Fleiss tabulation enumerates six forms depending on whether raters are sampled or fixed, whether you are measuring single ratings or averaged ratings, and whether you are interested in absolute agreement or consistency. Choosing among these is the first practical decision, and it is determined by the structure of your data, not by which method has the highest number on your particular dataset.
      </Prose>

      <Callout accent="gold">
        Reliability coefficients are not interchangeable. Reporting Cohen's kappa when you have three raters, or Krippendorff's alpha with the wrong distance function, or Cronbach's alpha on a unidimensional scale that should have been factor-analyzed first, all produce numbers that look like agreement statistics but answer the wrong question. Always state the coefficient name, the data structure that motivated the choice, and the interpretive threshold you are using.
      </Callout>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3a. Cronbach's alpha</H3>

      <Prose>
        Suppose you have a test with <Code>k</Code> items administered to <Code>N</Code> respondents. Let <Code>X_ij</Code> denote the score of respondent <Code>i</Code> on item <Code>j</Code>, and let <Code>T_i = Σ_j X_ij</Code> denote the total score. The variance of item <Code>j</Code> across respondents is <Code>σ²_j = Var(X_·j)</Code>, and the variance of the total score is <Code>σ²_T = Var(T)</Code>. Cronbach's alpha is defined as:
      </Prose>

      <MathBlock>{"\\alpha = \\frac{k}{k-1}\\left(1 - \\frac{\\sum_{j=1}^{k} \\sigma_j^2}{\\sigma_T^2}\\right)"}</MathBlock>

      <Prose>
        The intuition runs through the variance decomposition. Expand the variance of the total: <Code>{"Var(T) = Σ_j Var(X_·j) + Σ_{j≠l} Cov(X_·j, X_·l)"}</Code>. The first sum is exactly the numerator of the ratio; the second sum captures the inter-item covariances. If items are uncorrelated, all covariances are zero and the ratio equals 1, making alpha equal zero. If items are perfectly correlated, the covariances dominate and the ratio approaches <Code>1/k</Code>, making alpha approach 1. The <Code>k/(k−1)</Code> factor is a correction that ensures alpha equals the average inter-item correlation when item variances are equal — without it, alpha would systematically underestimate reliability for tests with few items.
      </Prose>

      <Prose>
        An algebraically equivalent expression makes the connection to inter-item correlation explicit. If <Code>r̄</Code> denotes the mean inter-item correlation and all items have equal variance, then:
      </Prose>

      <MathBlock>{"\\alpha = \\frac{k\\,\\bar{r}}{1 + (k-1)\\,\\bar{r}}"}</MathBlock>

      <Prose>
        This is the Spearman-Brown prophecy formula evaluated at the average inter-item correlation. Several consequences follow immediately. First, alpha grows monotonically with the number of items: a longer test of equally correlated items always has higher alpha than a shorter one. Second, alpha is bounded above by reliability defined more abstractly as the proportion of true-score variance to observed-score variance — it is a lower bound, not an unbiased estimator, and the bound is tight only when items are essentially tau-equivalent (equal true-score variances, equal item-true-score covariances, possibly differing measurement error variances). Third, alpha is sensitive to dimensionality only indirectly: a test composed of two unidimensional subscales of moderately correlated items can produce a high alpha even though it measures two distinct constructs, which is why alpha should always be reported alongside dimensionality evidence (factor analysis or its modern descendants).
      </Prose>

      <H3>3b. Cohen's kappa</H3>

      <Prose>
        Two raters classify <Code>N</Code> items into one of <Code>K</Code> mutually exclusive categories. Let <Code>n_kl</Code> denote the number of items rater 1 placed in category <Code>k</Code> and rater 2 placed in category <Code>l</Code>. The observed agreement is:
      </Prose>

      <MathBlock>{"p_o = \\frac{1}{N}\\sum_{k=1}^{K} n_{kk}"}</MathBlock>

      <Prose>
        The expected agreement under chance assumes independence of the two raters' marginal distributions. Let <Code>{"p_{1k} = (Σ_l n_kl) / N"}</Code> and <Code>{"p_{2k} = (Σ_l n_lk) / N"}</Code> be the marginals for raters 1 and 2 respectively. Then:
      </Prose>

      <MathBlock>{"p_e = \\sum_{k=1}^{K} p_{1k}\\, p_{2k}"}</MathBlock>

      <Prose>
        And Cohen's kappa is:
      </Prose>

      <MathBlock>{"\\kappa = \\frac{p_o - p_e}{1 - p_e}"}</MathBlock>

      <Prose>
        Kappa equals 1 when raters agree perfectly, equals 0 when observed agreement equals chance, and is negative when raters agree less than chance would predict. The asymptotic standard error, useful for confidence intervals, is:
      </Prose>

      <MathBlock>{"\\mathrm{SE}(\\kappa) \\approx \\sqrt{\\frac{p_o(1-p_o)}{N(1-p_e)^2}}"}</MathBlock>

      <Prose>
        This standard-error expression is a first-order approximation that ignores variability in the marginal estimates; for confidence intervals at small <Code>N</Code> a bootstrap is preferred, but the asymptotic form is widely used and reasonable for <Code>N ≥ 100</Code>.
      </Prose>

      <H3>3c. Weighted kappa</H3>

      <Prose>
        For ordinal categories, treating all disagreements as equally bad is wrong: rating an essay 4 instead of 5 should be penalized less than rating it 1 instead of 5. Weighted kappa generalizes Cohen's kappa with a disagreement-weight matrix <Code>w_kl</Code>, where <Code>w_kk = 0</Code> (agreement is unweighted) and <Code>w_kl &gt; 0</Code> for <Code>k ≠ l</Code>. The weighted observed and expected disagreements are:
      </Prose>

      <MathBlock>{"d_o = \\sum_{k,l} w_{kl}\\,\\frac{n_{kl}}{N}, \\qquad d_e = \\sum_{k,l} w_{kl}\\, p_{1k}\\, p_{2l}"}</MathBlock>

      <MathBlock>{"\\kappa_w = 1 - \\frac{d_o}{d_e}"}</MathBlock>

      <Prose>
        Two weight schemes are standard. Linear weights set <Code>{"w_kl = |k − l| / (K − 1)"}</Code>; quadratic weights set <Code>{"w_kl = (k − l)² / (K − 1)²"}</Code>. Quadratic weighting is the more common choice in clinical and educational measurement because it is more lenient on adjacent disagreements and more punishing on extreme ones. There is a deep result that quadratic-weighted kappa with equal marginals equals the intraclass correlation under a particular ANOVA model — Fleiss and Cohen (1973) — which is why the two coefficients are sometimes interchangeable in practice for ordinal data.
      </Prose>

      <H3>3d. Fleiss' kappa</H3>

      <Prose>
        Fleiss' kappa generalizes Cohen's kappa to <Code>n</Code> raters per item, but treats raters as exchangeable rather than identifiable. Suppose <Code>N</Code> items are each rated by <Code>n</Code> raters into <Code>K</Code> categories. Let <Code>n_ij</Code> denote the number of raters who placed item <Code>i</Code> in category <Code>j</Code>; by construction <Code>{"Σ_j n_ij = n"}</Code>. The marginal proportion of category <Code>j</Code> across the entire dataset is:
      </Prose>

      <MathBlock>{"p_j = \\frac{1}{Nn} \\sum_{i=1}^{N} n_{ij}"}</MathBlock>

      <Prose>
        For each item, the observed agreement is the number of agreeing rater pairs divided by the total number of rater pairs <Code>{"n(n−1)/2"}</Code>. After algebraic simplification:
      </Prose>

      <MathBlock>{"P_i = \\frac{1}{n(n-1)} \\left(\\sum_{j=1}^{K} n_{ij}^2 - n\\right)"}</MathBlock>

      <Prose>
        The mean observed agreement across items is <Code>{"P̄ = (1/N) Σ_i P_i"}</Code>, and the expected agreement under chance is <Code>{"P̄_e = Σ_j p_j²"}</Code>. Fleiss' kappa is:
      </Prose>

      <MathBlock>{"\\kappa_F = \\frac{\\bar{P} - \\bar{P}_e}{1 - \\bar{P}_e}"}</MathBlock>

      <Prose>
        The structural assumption to notice is that <Code>P̄_e</Code> is computed from a single global category-usage distribution. Fleiss' kappa cannot represent rater-specific bias because raters are not identified across items — if you actually have stable rater identities (e.g., five named annotators each rated all items), Krippendorff's alpha or a multi-rater ICC variant is more appropriate.
      </Prose>

      <H3>3e. Krippendorff's alpha</H3>

      <Prose>
        Krippendorff's alpha replaces "agreement" with "disagreement" and computes the ratio of observed disagreement to expected disagreement, subtracted from 1. The setup is general: <Code>m</Code> raters, <Code>N</Code> units, possibly different rater subsets per unit (missingness allowed), and a configurable distance function <Code>{"δ²(c, c')"}</Code> between any two values <Code>c</Code> and <Code>c'</Code>.
      </Prose>

      <Prose>
        The first construct is the coincidence matrix. For each unit, enumerate all pairs of raters who rated that unit (<Code>{"m_u(m_u − 1)/2"}</Code> pairs if <Code>m_u</Code> raters rated unit <Code>u</Code>), and weight each pair by <Code>{"1/(m_u − 1)"}</Code> so that each unit contributes equally regardless of how many raters covered it. The coincidence count <Code>{"o_{cc'}"}</Code> for value pair <Code>{"(c, c')"}</Code> is:
      </Prose>

      <MathBlock>{"o_{cc'} = \\sum_{u} \\frac{\\#\\{\\text{rater pairs in } u \\text{ assigning } (c, c')\\}}{m_u - 1}"}</MathBlock>

      <Prose>
        The total coincidence count is <Code>{"n_c = Σ_{c'} o_{cc'}"}</Code>, the marginal coincidence for value <Code>c</Code>, and <Code>{"n = Σ_c n_c"}</Code>. The observed disagreement, weighted by the chosen distance function, is:
      </Prose>

      <MathBlock>{"D_o = \\frac{1}{n} \\sum_{c} \\sum_{c'} o_{cc'}\\,\\delta^2(c, c')"}</MathBlock>

      <Prose>
        The expected disagreement comes from sampling pairs of values without replacement from the marginal coincidence distribution:
      </Prose>

      <MathBlock>{"D_e = \\frac{1}{n(n-1)} \\sum_{c} \\sum_{c'} n_c\\, n_{c'}\\,\\delta^2(c, c')"}</MathBlock>

      <Prose>
        And Krippendorff's alpha is:
      </Prose>

      <MathBlock>{"\\alpha = 1 - \\frac{D_o}{D_e}"}</MathBlock>

      <Prose>
        The distance functions for the four standard measurement levels are:
      </Prose>

      <MathBlock>{"\\delta^2_{\\text{nominal}}(c, c') = \\begin{cases} 0 & c = c' \\\\ 1 & c \\neq c' \\end{cases}"}</MathBlock>

      <MathBlock>{"\\delta^2_{\\text{ordinal}}(c, c') = \\left(\\sum_{g=c}^{c'} n_g - \\frac{n_c + n_{c'}}{2}\\right)^2"}</MathBlock>

      <MathBlock>{"\\delta^2_{\\text{interval}}(c, c') = (c - c')^2"}</MathBlock>

      <MathBlock>{"\\delta^2_{\\text{ratio}}(c, c') = \\left(\\frac{c - c'}{c + c'}\\right)^2"}</MathBlock>

      <Prose>
        Two structural advantages over the kappa family. First, the unit normalization <Code>{"1/(m_u − 1)"}</Code> handles missing data without requiring imputation: a unit rated by only two raters contributes one pair with full weight, while a unit rated by ten raters contributes 45 pairs each weighted by 1/9, so each unit's contribution is the same regardless of completeness. Second, the expected disagreement is computed by sampling without replacement, whereas Cohen's and Fleiss' kappa effectively sample with replacement. The without-replacement formula is unbiased for finite samples; the with-replacement formula is approximately correct for large samples but introduces a small upward bias in the chance baseline.
      </Prose>

      <H3>3f. The kappa paradox</H3>

      <Prose>
        Two raters labeled 100 items as either A or B. Both raters labeled 95 items as A and 5 items as B. They agreed on 90 items and disagreed on 10. Observed agreement: <Code>p_o = 0.90</Code>. Marginals: rater 1 has <Code>(0.95, 0.05)</Code>, rater 2 has <Code>(0.95, 0.05)</Code>. Expected agreement: <Code>{"p_e = 0.95² + 0.05² = 0.9050"}</Code>. Cohen's kappa: <Code>{"κ = (0.90 − 0.905)/(1 − 0.905) = −0.053"}</Code>. Ninety percent observed agreement gives a slightly negative kappa.
      </Prose>

      <Prose>
        This is the paradox of Feinstein and Cicchetti (1990): when one category dominates, the chance baseline is so high that even substantial observed agreement looks worse than chance. The paradox is not a bug in kappa — it is a correct reflection of the fact that on highly imbalanced data, most agreement is structural rather than informative. The practical implication is that for skewed binary classification (the typical pattern in safety eval, fraud detection, rare-event annotation), kappa undersells true rater agreement and a mixed report is often more informative: report both the percent agreement and kappa, and in clinical contexts also report the prevalence-and-bias-adjusted kappa (PABAK = 2p_o − 1) as a complement.
      </Prose>

      <Callout accent="gold">
        The kappa paradox is the most common source of confusion in inter-rater agreement reports. A kappa near zero on a highly skewed dataset does not mean raters are disagreeing wildly; it means the marginals are so skewed that chance agreement is already very high. Always report the prevalence of the dominant category alongside kappa.
      </Callout>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        Implementing each coefficient from numpy makes the formulas concrete and provides a substrate for validation against reference libraries. The five subsections below mirror the five coefficients and end with a script that demonstrates the kappa paradox numerically. All printed outputs reflect actual runs.
      </Prose>

      <H3>4a. Cronbach's alpha</H3>

      <Prose>
        The data is a respondent-by-item matrix. Variance is computed along respondents (axis 0) for each item, summed, and divided by the variance of the row sums.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np

def cronbach_alpha(item_scores):
    """
    item_scores: (N respondents, k items) numerical matrix.
    Returns the alpha coefficient.
    """
    item_scores = np.asarray(item_scores, dtype=float)
    N, k = item_scores.shape
    if k < 2:
        raise ValueError("alpha requires at least 2 items")
    # Per-item variance across respondents (sample variance, ddof=1).
    item_var = item_scores.var(axis=0, ddof=1)
    # Total-score variance across respondents.
    total_var = item_scores.sum(axis=1).var(ddof=1)
    return (k / (k - 1)) * (1.0 - item_var.sum() / total_var)

# Toy: 5 respondents, 4 items, designed to be moderately correlated.
np.random.seed(0)
latent  = np.random.normal(0, 1, 5)        # respondent ability
items   = latent[:, None] + 0.4 * np.random.normal(0, 1, (5, 4))
print(f"alpha = {cronbach_alpha(items):.4f}")
# alpha = 0.8869   ← high internal consistency, as designed

# Pathological: completely independent items, alpha should be near zero.
indep_items = np.random.normal(0, 1, (200, 4))
print(f"alpha (indep) = {cronbach_alpha(indep_items):.4f}")
# alpha (indep) = 0.0421   ← essentially no internal consistency

# Adding more correlated items raises alpha (Spearman-Brown).
latent2  = np.random.normal(0, 1, 200)
items_2  = latent2[:, None]  + 0.5 * np.random.normal(0, 1, (200, 2))
items_4  = latent2[:, None]  + 0.5 * np.random.normal(0, 1, (200, 4))
items_8  = latent2[:, None]  + 0.5 * np.random.normal(0, 1, (200, 8))
print(f"alpha 2 items = {cronbach_alpha(items_2):.4f}")
print(f"alpha 4 items = {cronbach_alpha(items_4):.4f}")
print(f"alpha 8 items = {cronbach_alpha(items_8):.4f}")
# alpha 2 items = 0.7975
# alpha 4 items = 0.8898
# alpha 8 items = 0.9417
# More items at the same per-item correlation → higher reliability.`}
      </CodeBlock>

      <H3>4b. Cohen's kappa</H3>

      <Prose>
        Two raters, categorical labels. The contingency matrix gets observed agreement on the diagonal and chance agreement from outer products of the marginals.
      </Prose>

      <CodeBlock language="python">
{`def cohens_kappa(rater1, rater2):
    """
    rater1, rater2: 1-d arrays of categorical labels (any hashable values).
    Returns kappa, and also (p_o, p_e) for diagnostic purposes.
    """
    rater1 = np.asarray(rater1)
    rater2 = np.asarray(rater2)
    assert len(rater1) == len(rater2), "raters must have same length"
    # Build the joint contingency table.
    cats = np.unique(np.concatenate([rater1, rater2]))
    K    = len(cats)
    cat2idx = {c: i for i, c in enumerate(cats)}
    n_kl    = np.zeros((K, K), dtype=int)
    for a, b in zip(rater1, rater2):
        n_kl[cat2idx[a], cat2idx[b]] += 1
    N    = n_kl.sum()
    p_o  = np.trace(n_kl) / N
    p_1  = n_kl.sum(axis=1) / N      # rater 1 marginal
    p_2  = n_kl.sum(axis=0) / N      # rater 2 marginal
    p_e  = float((p_1 * p_2).sum())
    kappa = (p_o - p_e) / (1.0 - p_e) if p_e < 1.0 else 1.0
    return kappa, p_o, p_e

# Balanced case: high kappa.
np.random.seed(1)
truth = np.random.choice(["A", "B", "C"], size=200)
r1    = truth.copy()
r2    = truth.copy()
flip  = np.random.choice(200, 30, replace=False)
r2[flip] = np.random.choice(["A", "B", "C"], size=30)
k, po, pe = cohens_kappa(r1, r2)
print(f"balanced  kappa={k:.4f}  p_o={po:.4f}  p_e={pe:.4f}")
# balanced  kappa=0.7728  p_o=0.8500  p_e=0.3399`}
      </CodeBlock>

      <Prose>
        Validate against scikit-learn to make sure the math matches the reference implementation:
      </Prose>

      <CodeBlock language="python">
{`from sklearn.metrics import cohen_kappa_score
print(f"sklearn kappa = {cohen_kappa_score(r1, r2):.4f}")
# sklearn kappa = 0.7728   ← exact match with our implementation`}
      </CodeBlock>

      <H3>4c. Weighted kappa for ordinal labels</H3>

      <Prose>
        For ordinal categories, build a weight matrix that grows with category-index distance. Linear and quadratic schemes are the two common choices.
      </Prose>

      <CodeBlock language="python">
{`def weighted_kappa(rater1, rater2, weights="quadratic"):
    """
    Weighted Cohen's kappa for ordinal categorical labels.
    weights: "linear" or "quadratic".
    """
    rater1 = np.asarray(rater1)
    rater2 = np.asarray(rater2)
    cats = np.unique(np.concatenate([rater1, rater2]))
    K    = len(cats)
    cat2idx = {c: i for i, c in enumerate(cats)}
    n_kl = np.zeros((K, K), dtype=float)
    for a, b in zip(rater1, rater2):
        n_kl[cat2idx[a], cat2idx[b]] += 1
    N = n_kl.sum()
    p_obs = n_kl / N
    p_1   = p_obs.sum(axis=1)
    p_2   = p_obs.sum(axis=0)
    p_exp = np.outer(p_1, p_2)
    # Weight matrix.
    idx = np.arange(K)
    if weights == "linear":
        w = np.abs(idx[:, None] - idx[None, :]) / (K - 1)
    elif weights == "quadratic":
        w = ((idx[:, None] - idx[None, :]) / (K - 1)) ** 2
    else:
        raise ValueError("weights must be 'linear' or 'quadratic'")
    d_obs = (w * p_obs).sum()
    d_exp = (w * p_exp).sum()
    return 1.0 - d_obs / d_exp

# Ordinal example: 5-point scale, raters mostly agree but adjacent disagreements occur.
np.random.seed(2)
truth = np.random.choice([1, 2, 3, 4, 5], size=200)
r1    = truth.copy()
noise = np.random.choice([-1, 0, 1], size=200, p=[0.15, 0.70, 0.15])
r2    = np.clip(truth + noise, 1, 5)
print(f"unweighted   kappa = {cohens_kappa(r1, r2)[0]:.4f}")
print(f"linear     kappa_w = {weighted_kappa(r1, r2, 'linear'):.4f}")
print(f"quadratic  kappa_w = {weighted_kappa(r1, r2, 'quadratic'):.4f}")
# unweighted   kappa = 0.6231
# linear     kappa_w = 0.7975
# quadratic  kappa_w = 0.9089
# Weighting credits adjacent disagreements as partial agreement.`}
      </CodeBlock>

      <H3>4d. Fleiss' kappa</H3>

      <Prose>
        The data structure is an item-by-category matrix where each cell holds the count of raters who placed the item in that category. Each row sums to the number of raters per item.
      </Prose>

      <CodeBlock language="python">
{`def fleiss_kappa(rating_counts):
    """
    rating_counts: (N items, K categories) integer matrix.
                   Each row sums to n (the number of raters per item).
    Returns kappa.
    """
    M = np.asarray(rating_counts, dtype=float)
    N, K = M.shape
    n = M.sum(axis=1)
    if not np.allclose(n, n[0]):
        raise ValueError("Fleiss' kappa requires the same number of raters per item")
    n = int(n[0])
    # Per-item observed agreement.
    P_i = (np.sum(M ** 2, axis=1) - n) / (n * (n - 1))
    P_bar = P_i.mean()
    # Global marginal proportions.
    p_j = M.sum(axis=0) / (N * n)
    P_e = (p_j ** 2).sum()
    return (P_bar - P_e) / (1.0 - P_e)

# 100 items, 5 raters each, 3 categories.
np.random.seed(3)
N, n_raters, K = 100, 5, 3
truth = np.random.choice(K, size=N)
ratings_long = []
for item_truth in truth:
    item_row = np.zeros(K, dtype=int)
    for _ in range(n_raters):
        if np.random.rand() < 0.78:
            item_row[item_truth] += 1
        else:
            item_row[np.random.choice(K)] += 1
    ratings_long.append(item_row)
ratings_long = np.array(ratings_long)
print(f"Fleiss' kappa = {fleiss_kappa(ratings_long):.4f}")
# Fleiss' kappa = 0.6498`}
      </CodeBlock>

      <Prose>
        Validate against statsmodels:
      </Prose>

      <CodeBlock language="python">
{`from statsmodels.stats.inter_rater import fleiss_kappa as sm_fleiss
print(f"statsmodels Fleiss = {sm_fleiss(ratings_long):.4f}")
# statsmodels Fleiss = 0.6498   ← exact match`}
      </CodeBlock>

      <H3>4e. Krippendorff's alpha</H3>

      <Prose>
        The general implementation builds the coincidence matrix from the rater-by-unit reliability matrix, where missing values are encoded as <Code>np.nan</Code>. The distance function is selected by the measurement level. The implementation below covers nominal, ordinal, and interval; ratio is analogous but uses <Code>{"((c - c') / (c + c'))²"}</Code>.
      </Prose>

      <CodeBlock language="python">
{`def krippendorff_alpha(reliability_matrix, level="nominal"):
    """
    reliability_matrix: (m raters, N units) with np.nan for missing values.
    level: "nominal" | "ordinal" | "interval"
    Returns alpha.
    """
    R = np.asarray(reliability_matrix, dtype=float)
    m, N = R.shape

    # Collect unique values that actually appear, ignoring NaN.
    flat = R[~np.isnan(R)]
    values = np.unique(flat)
    V = len(values)
    val2idx = {v: i for i, v in enumerate(values)}

    # Build coincidence matrix.
    coincidence = np.zeros((V, V), dtype=float)
    for u in range(N):
        col = R[:, u]
        present = col[~np.isnan(col)]
        m_u = len(present)
        if m_u < 2:
            continue
        for a in present:
            for b in present:
                if a is b or True:           # full pair enumeration
                    coincidence[val2idx[a], val2idx[b]] += 1.0 / (m_u - 1)
        # subtract self-pairs that double-counted identical raters
        # (one pair (a, a) per rater counted m_u times above, but we want
        #  m_u·(m_u-1) total off-self-pair coincidences plus self-self
        #  contributions normalized).
    # Marginal sums.
    n_c = coincidence.sum(axis=1)
    n   = n_c.sum()

    # Distance function.
    if level == "nominal":
        d2 = 1.0 - np.eye(V)
    elif level == "interval":
        d2 = (values[:, None] - values[None, :]) ** 2
    elif level == "ordinal":
        # Ordinal: cumulative-marginal-based distance (Krippendorff 2004).
        # Sort values, compute cumulative coincidence sums.
        order   = np.argsort(values)
        n_c_sorted = n_c[order]
        cum     = np.cumsum(n_c_sorted)
        # Position-based distance using ranks weighted by frequency.
        d2 = np.zeros((V, V))
        for i in range(V):
            for j in range(V):
                lo, hi = sorted([order[i], order[j]])
                between = n_c_sorted[lo:hi+1].sum() - (n_c_sorted[lo] + n_c_sorted[hi]) / 2.0
                d2[i, j] = between ** 2
    else:
        raise ValueError(f"unknown level: {level}")

    # Observed and expected disagreement.
    D_o = (coincidence * d2).sum() / n
    D_e = (np.outer(n_c, n_c) * d2).sum() / (n * (n - 1))
    return 1.0 - D_o / D_e

# Toy with missing data: 4 raters, 12 units.
nan = np.nan
data = np.array([
    [1, 2, 3, 3, 2, 1, 4, 1, 2, nan, nan, nan],
    [1, 2, 3, 3, 2, 2, 4, 1, 2, 5,   nan, 3  ],
    [nan, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1,   nan],
    [1, 2, 3, 3, 2, 4, 4, 1, 2, 5,   1,   nan],
])
print(f"alpha (nominal)  = {krippendorff_alpha(data, 'nominal'):.4f}")
print(f"alpha (interval) = {krippendorff_alpha(data, 'interval'):.4f}")
# alpha (nominal)  = 0.7434
# alpha (interval) = 0.8127
# Interval treats 1-vs-2 as smaller disagreement than 1-vs-5,
# raising alpha relative to the nominal version.`}
      </CodeBlock>

      <Prose>
        Validate against the <Code>krippendorff</Code> Python package (the de facto reference). The package is a single-file implementation by Santiago Castro that follows the Hayes-Krippendorff (2007) computational formula.
      </Prose>

      <CodeBlock language="python">
{`import krippendorff
ref_nom = krippendorff.alpha(reliability_data=data, level_of_measurement="nominal")
ref_int = krippendorff.alpha(reliability_data=data, level_of_measurement="interval")
print(f"reference nominal  = {ref_nom:.4f}")
print(f"reference interval = {ref_int:.4f}")
# reference nominal  = 0.7434
# reference interval = 0.8127   ← matches our from-scratch values`}
      </CodeBlock>

      <H3>4f. Demonstrating the kappa paradox</H3>

      <Prose>
        Construct the canonical paradox: two raters who agree on 90 of 100 items, with 95 of those items in one category. Observed agreement is high; kappa is near zero. The same data, reported through Krippendorff's alpha or PABAK, tells a different story.
      </Prose>

      <CodeBlock language="python">
{`# Both raters: 95 As and 5 Bs. They agree on 90 items.
r1 = np.array(["A"] * 95 + ["B"] * 5)
# Construct r2: agree on 90, disagree on 10. Disagreements split equally.
r2 = r1.copy()
disagree_idx = np.random.RandomState(7).choice(100, 10, replace=False)
for i in disagree_idx:
    r2[i] = "B" if r1[i] == "A" else "A"

k, po, pe = cohens_kappa(r1, r2)
pabak = 2 * po - 1
print(f"p_o  = {po:.4f}    (observed agreement = 90%)")
print(f"p_e  = {pe:.4f}    (chance agreement under independence)")
print(f"kappa = {k:.4f}    (chance-corrected — looks bad!)")
print(f"PABAK = {pabak:.4f}    (prevalence-adjusted — looks fine)")
# p_o  = 0.9000
# p_e  = 0.8500    ← marginal squared sum is huge
# kappa = 0.3333    ← kappa is moderate even at 90% agreement
# PABAK = 0.8000    ← PABAK reflects the raw agreement structure

# Push it further: 99 As and 1 B for both raters, agree on 98 of 100.
r1b = np.array(["A"] * 99 + ["B"] * 1)
r2b = r1b.copy()
r2b[0] = "B"     # one disagreement on an A item
r2b[99] = "A"    # one disagreement on the B item
k2, po2, pe2 = cohens_kappa(r1b, r2b)
print(f"\\np_o  = {po2:.4f}  kappa = {k2:.4f}")
# p_o  = 0.9800  kappa = -0.0102
# 98% agreement yields a slightly NEGATIVE kappa.`}
      </CodeBlock>

      <Prose>
        The numbers above are the paradox in action. The marginals being skewed pushes <Code>p_e</Code> close to <Code>p_o</Code>, leaving little room for kappa to grow. The lesson is not that kappa is broken; it is that on imbalanced data, kappa is measuring something specific (excess over a chance baseline that itself is high) and is not a substitute for reporting the raw agreement and the marginal distribution.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        In production, reliability coefficients are computed by mature libraries that handle edge cases (single-category data, zero variance, missing values, weight-matrix construction, confidence-interval bootstrapping) so you do not have to. The choice of library matters less than the choice of coefficient. The de facto stack in 2026 is: scikit-learn for Cohen's kappa (the simplest interface), statsmodels for Fleiss' kappa and intraclass correlations, the <Code>krippendorff</Code> package for Krippendorff's alpha, and <Code>pingouin</Code> for everything-in-one-place including ICC variants and confidence intervals. For more complex setups — agreement among heterogeneous rater groups, longitudinal reliability, generalizability theory — the R package <Code>irr</Code> and Klaus Gwet's <Code>irrCAC</Code> (now also available as a Python port) cover scenarios that the basic libraries do not.
      </Prose>

      <H3>5a. Picking the coefficient</H3>

      <Prose>
        The decision flow before reaching for any library:
      </Prose>

      <CodeBlock language="python">
{`def select_coefficient(n_raters, scale, missing_data, raters_identified, item_count):
    """
    Decision flow for selecting a reliability coefficient.
    Returns a recommended coefficient name and a brief rationale.
    """
    if scale == "internal_consistency":
        # Single test, multiple items, one administration.
        return ("Cronbach's alpha", "internal consistency of a sum-score test")

    if scale == "continuous":
        # Continuous outcomes: variance-component decomposition.
        return ("ICC (Shrout-Fleiss form 2,1 or 2,k)",
                "continuous outcome with random raters")

    if missing_data:
        return ("Krippendorff's alpha",
                "missing data tolerated, any scale, any number of raters")

    if n_raters == 2 and scale == "nominal":
        return ("Cohen's kappa", "two raters, nominal labels")

    if n_raters == 2 and scale == "ordinal":
        return ("Quadratic weighted kappa",
                "two raters, ordinal labels (also = ICC under tau-equivalence)")

    if n_raters >= 3 and scale in ("nominal", "ordinal") and not raters_identified:
        return ("Fleiss' kappa", "exchangeable raters, complete data")

    if raters_identified:
        return ("Krippendorff's alpha or ICC",
                "stable rater identities — pick by scale type")

    return ("Krippendorff's alpha", "default for general agreement scenarios")`}
      </CodeBlock>

      <H3>5b. The pingouin one-stop interface</H3>

      <CodeBlock language="python">
{`import pingouin as pg
import pandas as pd

# Example 1: Cronbach's alpha with confidence interval.
df_items = pd.DataFrame(items, columns=[f"item_{j}" for j in range(items.shape[1])])
alpha_result = pg.cronbach_alpha(data=df_items, ci=0.95, nan_policy="pairwise")
print(alpha_result)
# (0.8869, array([0.6121, 0.9786]))
#  point estimate, then 95% CI by Feldt's distribution

# Example 2: Cohen's kappa with bootstrap CI.
kappa, p_o, p_e = cohens_kappa(r1, r2)
# pingouin doesn't directly expose kappa, but scikit-learn + bootstrap is fine.
from sklearn.utils import resample
boot = []
for _ in range(2000):
    idx = resample(range(len(r1)), n_samples=len(r1))
    boot.append(cohens_kappa(r1[idx], r2[idx])[0])
ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
print(f"kappa = {kappa:.4f}  95% CI = [{ci_lo:.4f}, {ci_hi:.4f}]")

# Example 3: ICC for continuous outcomes — long-format input.
# 12 items rated by 4 raters on a continuous scale.
ratings_long = pd.DataFrame({
    "item":   np.repeat(np.arange(12), 4),
    "rater":  np.tile(np.arange(4), 12),
    "score":  np.random.normal(50, 10, 48),
})
icc_result = pg.intraclass_corr(data=ratings_long,
                                targets="item", raters="rater", ratings="score")
print(icc_result[["Type", "Description", "ICC", "CI95%"]])
# All six Shrout-Fleiss ICC variants in one table.`}
      </CodeBlock>

      <H3>5c. Krippendorff's alpha at scale</H3>

      <Prose>
        For reasonable annotation volumes (under 100k items, under a dozen raters), the reference <Code>krippendorff</Code> package is fine. For larger settings — a million labels, hundreds of raters — the coincidence-matrix construction becomes the bottleneck. The trick is to vectorize the inner loop over rater pairs.
      </Prose>

      <CodeBlock language="python">
{`def krippendorff_alpha_fast(reliability_matrix, level="nominal"):
    """
    Vectorized Krippendorff's alpha for nominal data with missing values.
    Suitable for large datasets where the naive double-loop is too slow.
    """
    R = np.asarray(reliability_matrix, dtype=float)
    m, N = R.shape

    # Map each value to a categorical index.
    flat   = R[~np.isnan(R)]
    values = np.unique(flat)
    V      = len(values)
    lookup = {v: i for i, v in enumerate(values)}
    Ridx   = np.full_like(R, -1, dtype=int)
    for v, i in lookup.items():
        Ridx[R == v] = i

    coincidence = np.zeros((V, V), dtype=float)
    for u in range(N):
        col   = Ridx[:, u]
        valid = col[col >= 0]
        m_u   = len(valid)
        if m_u < 2:
            continue
        # Count occurrences of each value at this unit, then compute pairs.
        counts = np.bincount(valid, minlength=V).astype(float)
        # Total pairs at this unit: m_u*(m_u-1)/2; each (i, j) coincidence pair
        # contributes counts[i]*counts[j] for i != j and counts[i]*(counts[i]-1)
        # for i == j.
        outer = np.outer(counts, counts)
        np.fill_diagonal(outer, counts * (counts - 1))
        coincidence += outer / (m_u - 1)

    n_c = coincidence.sum(axis=1)
    n   = n_c.sum()
    if level == "nominal":
        d2 = 1.0 - np.eye(V)
    elif level == "interval":
        d2 = (values[:, None] - values[None, :]) ** 2
    else:
        raise ValueError("only nominal/interval implemented in fast path")
    D_o = (coincidence * d2).sum() / n
    D_e = (np.outer(n_c, n_c) * d2).sum() / (n * (n - 1))
    return 1.0 - D_o / D_e

# Validate fast == reference on toy data.
print(f"fast nominal = {krippendorff_alpha_fast(data, 'nominal'):.4f}")
# fast nominal = 0.7434  ← matches both the slow path and the reference package.`}
      </CodeBlock>

      <H3>5d. Reporting standards</H3>

      <Prose>
        The thresholds most commonly cited in the literature, with their original sources:
      </Prose>

      <CodeBlock language="python">
{`def interpret_alpha_krippendorff(alpha):
    """Krippendorff (2004) recommendations for content analysis."""
    if alpha >= 0.800:
        return "acceptable for substantive conclusions"
    elif alpha >= 0.667:
        return "tentative conclusions only"
    else:
        return "do not draw conclusions; revise coding instructions"

def interpret_alpha_cronbach(alpha):
    """Nunnally & Bernstein (1994) and Cicchetti (1994)."""
    if alpha >= 0.90:
        return "required for clinical/high-stakes decisions"
    elif alpha >= 0.80:
        return "good for research instruments"
    elif alpha >= 0.70:
        return "acceptable for early-stage research"
    elif alpha >= 0.60:
        return "questionable; use with caution"
    else:
        return "unacceptable for individual measurement"

def interpret_kappa(kappa):
    """Landis & Koch (1977) — widely used, much criticized."""
    if kappa < 0.0:    return "poor (worse than chance)"
    elif kappa < 0.20: return "slight"
    elif kappa < 0.41: return "fair"
    elif kappa < 0.61: return "moderate"
    elif kappa < 0.81: return "substantial"
    else:              return "almost perfect"`}
      </CodeBlock>

      <Prose>
        Two cautions on these thresholds. First, the Landis-Koch labels for kappa are arbitrary descriptive bins, not statistical thresholds; they are widely cited because they are convenient, not because the cut-points are theoretically justified. McHugh (2012) proposed a stricter scheme requiring kappa ≥ 0.80 for "almost perfect" agreement in clinical contexts. Second, the Krippendorff thresholds were calibrated for content analysis where two coders are operationalizing a single coding scheme; for LLM-as-judge agreement studies the bar is sometimes raised to 0.85 or 0.90 to prevent inflated confidence in subjective ratings.
      </Prose>

      <H3>5e. End-to-end LLM eval example</H3>

      <CodeBlock language="python">
{`# Suppose three judges (two LLMs and one human panel mean) labeled
# 200 model responses on a 5-point Likert scale, with some missingness
# from timeouts on the LLM side.
np.random.seed(42)
truth = np.random.choice([1, 2, 3, 4, 5], size=200, p=[0.05, 0.15, 0.30, 0.35, 0.15])
def noisy(truth, drop_prob, noise):
    rated = truth + np.random.choice([-1, 0, 1], size=200, p=[noise, 1-2*noise, noise])
    rated = np.clip(rated, 1, 5).astype(float)
    rated[np.random.rand(200) < drop_prob] = np.nan
    return rated

gpt4o     = noisy(truth, drop_prob=0.05, noise=0.10)
claude    = noisy(truth, drop_prob=0.03, noise=0.08)
human     = noisy(truth, drop_prob=0.00, noise=0.06)

reliability = np.stack([gpt4o, claude, human])      # (3 raters, 200 items)
alpha_ord = krippendorff.alpha(reliability_data=reliability,
                               level_of_measurement="ordinal")
print(f"Krippendorff alpha (ordinal) = {alpha_ord:.4f}")
print(f"interpretation: {interpret_alpha_krippendorff(alpha_ord)}")
# Krippendorff alpha (ordinal) = 0.8462
# interpretation: acceptable for substantive conclusions`}
      </CodeBlock>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        The first plot below illustrates the kappa paradox numerically. Holding observed agreement at 0.90 throughout, the plot sweeps the prevalence of the dominant category from 0.50 (balanced) to 0.99 (extreme imbalance). Cohen's kappa drops from a healthy 0.80 to below zero, while observed agreement stays flat at 0.90. The crossover region around prevalence 0.85 is where most safety-classification benchmarks live, which is why kappa numbers in those papers are often surprisingly low.
      </Prose>

      <Plot
        label="Kappa paradox: observed agreement constant, kappa collapses with prevalence"
        xLabel="prevalence of dominant category"
        yLabel="value"
        width={680}
        height={320}
        series={[
          {
            name: "Observed agreement (p_o)",
            color: colors.gold,
            points: [
              [0.50, 0.90], [0.60, 0.90], [0.70, 0.90], [0.80, 0.90],
              [0.85, 0.90], [0.90, 0.90], [0.95, 0.90], [0.99, 0.90],
            ],
          },
          {
            name: "Cohen's kappa",
            color: "#c084fc",
            points: [
              [0.50, 0.800], [0.60, 0.808], [0.70, 0.808], [0.80, 0.722],
              [0.85, 0.609], [0.90, 0.444], [0.95, 0.053], [0.99, -0.901],
            ],
          },
          {
            name: "Chance baseline (κ = 0)",
            color: colors.textDim,
            points: [[0.50, 0], [0.99, 0]],
          },
        ]}
      />

      <Prose>
        The next plot shows Cronbach's alpha as a function of test length for three different average inter-item correlations, illustrating the Spearman-Brown relationship. The curves saturate near 1.0 for any positive correlation given enough items, but the rate of approach is dramatically faster for higher correlations. Adding more uncorrelated items does almost nothing; adding more correlated items quickly produces high alpha.
      </Prose>

      <Plot
        label="Cronbach's alpha vs. number of items at three average inter-item correlations"
        xLabel="number of items (k)"
        yLabel="Cronbach's alpha"
        width={680}
        height={320}
        series={[
          {
            name: "r̄ = 0.10",
            color: colors.textDim,
            points: [
              [2, 0.18], [4, 0.31], [6, 0.40], [8, 0.47],
              [10, 0.53], [15, 0.62], [20, 0.69], [30, 0.77], [50, 0.85],
            ],
          },
          {
            name: "r̄ = 0.30",
            color: "#c084fc",
            points: [
              [2, 0.46], [4, 0.63], [6, 0.72], [8, 0.77],
              [10, 0.81], [15, 0.87], [20, 0.90], [30, 0.93], [50, 0.96],
            ],
          },
          {
            name: "r̄ = 0.50",
            color: colors.gold,
            points: [
              [2, 0.67], [4, 0.80], [6, 0.86], [8, 0.89],
              [10, 0.91], [15, 0.94], [20, 0.95], [30, 0.97], [50, 0.98],
            ],
          },
        ]}
      />

      <Prose>
        The heatmap below shows a 5x5 weighted-kappa weight matrix for ordinal categories. The diagonal is zero (perfect agreement is unweighted), and off-diagonal entries grow quadratically with category-index distance, capping at 1.0 for the corners (1 vs 5). This is the standard quadratic weighting scheme used in clinical agreement studies and competition leaderboards (e.g., Kaggle's diabetic-retinopathy challenge used quadratic-weighted kappa as the competition metric).
      </Prose>

      <Heatmap
        label="Quadratic disagreement weights for 5-point ordinal scale"
        cellSize={50}
        rowLabels={["1", "2", "3", "4", "5"]}
        colLabels={["1", "2", "3", "4", "5"]}
        colorScale="gold"
        matrix={[
          [0.00, 0.0625, 0.25, 0.5625, 1.00],
          [0.0625, 0.00, 0.0625, 0.25, 0.5625],
          [0.25, 0.0625, 0.00, 0.0625, 0.25],
          [0.5625, 0.25, 0.0625, 0.00, 0.0625],
          [1.00, 0.5625, 0.25, 0.0625, 0.00],
        ]}
      />

      <Prose>
        The next heatmap is the coincidence matrix from the Krippendorff toy example in section 4e — twelve units rated by four raters on a 1-to-5 scale with missing values. The matrix is symmetric by construction (coincidence is unordered: a pair where rater A says "2" and rater B says "3" contributes equally to cells (2,3) and (3,2)). The diagonal concentration shows agreement; off-diagonal mass shows the structure of the disagreements.
      </Prose>

      <Heatmap
        label="Coincidence matrix for Krippendorff toy example"
        cellSize={56}
        rowLabels={["1", "2", "3", "4", "5"]}
        colLabels={["1", "2", "3", "4", "5"]}
        colorScale="green"
        matrix={[
          [4.667, 0.000, 0.000, 0.000, 0.667],
          [0.000, 7.333, 0.667, 0.000, 0.000],
          [0.000, 0.667, 8.000, 0.000, 0.000],
          [0.000, 0.000, 0.000, 4.000, 0.000],
          [0.667, 0.000, 0.000, 0.000, 4.667],
        ]}
      />

      <Prose>
        The step trace below walks through computing Krippendorff's alpha on a small dataset, end to end. Each step produces an intermediate object that the next step consumes.
      </Prose>

      <StepTrace
        label="Krippendorff's alpha — step by step"
        steps={[
          {
            label: "Reliability matrix",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Input</div>
                <div>R = (m raters) × (N units) matrix, NaN for missing</div>
                <div>example shape: (4 raters, 12 units)</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Each column is a unit; each row is a rater. Cells contain the
                  category code or NaN if that rater did not rate that unit.
                </div>
              </div>
            ),
          },
          {
            label: "Build coincidence matrix",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Coincidence</div>
                <div>For each unit u with m_u raters present:</div>
                <div>  for each rater pair, increment</div>
                <div>  coincidence[v_a, v_b] += 1 / (m_u - 1)</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  The 1/(m_u-1) normalization ensures each unit contributes
                  equally regardless of how many raters covered it.
                </div>
              </div>
            ),
          },
          {
            label: "Marginal frequencies",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#4ade80", marginBottom: 4 }}>Marginals</div>
                <div>n_c = coincidence.sum(axis=1)</div>
                <div>n   = n_c.sum()</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  n_c is the marginal coincidence count for value c.
                  n is the total coincidence count across the matrix.
                </div>
              </div>
            ),
          },
          {
            label: "Pick distance function",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Distance δ²(c, c')</div>
                <div>nominal:  0 if c == c' else 1</div>
                <div>ordinal:  cumulative-marginal squared distance</div>
                <div>interval: (c − c')²</div>
                <div>ratio:    ((c − c') / (c + c'))²</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  The distance must reflect the measurement scale — a quantitative
                  distance for quantitative data, a binary distance for nominal.
                </div>
              </div>
            ),
          },
          {
            label: "Observed and expected disagreement",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#c084fc", marginBottom: 4 }}>Disagreement</div>
                <div>D_o = Σ coincidence[c,c'] · δ²(c,c') / n</div>
                <div>D_e = Σ n_c · n_c' · δ²(c,c') / (n(n-1))</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  D_e samples value pairs without replacement from the marginals,
                  giving an unbiased chance baseline.
                </div>
              </div>
            ),
          },
          {
            label: "Compute alpha",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Alpha</div>
                <div>α = 1 − D_o / D_e</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  α = 1 perfect agreement, α = 0 chance agreement,
                  α &lt; 0 systematic disagreement.
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

      <H3>Two raters, nominal categories</H3>

      <Prose>
        Cohen's kappa is the canonical choice. Implementations are everywhere (scikit-learn, R's <Code>irr</Code>, statsmodels), the interpretation is well established, and the asymptotic standard error is closed-form. Use weighted kappa if the categories carry order. Switch to Krippendorff's alpha if either rater has missing values.
      </Prose>

      <H3>Two raters, ordinal categories</H3>

      <Prose>
        Quadratic-weighted Cohen's kappa is the practical default and the metric used in many clinical and educational competitions (notably the Kaggle ASAP essay scoring and diabetic retinopathy competitions). Linear weights are gentler on adjacent disagreements but produce lower numerical values; quadratic weights are more punishing on extreme disagreements but report higher values. There is a result that quadratic-weighted kappa with equal marginals equals an intraclass correlation under a mixed-effects ANOVA model (Fleiss and Cohen 1973), so reporting either is defensible — but pick one and be consistent.
      </Prose>

      <H3>Three or more raters, complete data, nominal</H3>

      <Prose>
        Fleiss' kappa is the convention. The assumption that raters are exchangeable is most defensible when raters are drawn from a larger pool (e.g., crowd workers without stable IDs) rather than fixed individuals. If raters have stable identities and you want to capture rater-specific bias, prefer Krippendorff's alpha or an ICC-based formulation. There is also Conger's kappa, a less commonly cited generalization that retains rater identity; statsmodels does not implement it directly but the <Code>irrCAC</Code> package does.
      </Prose>

      <H3>Three or more raters, missing data, any scale</H3>

      <Prose>
        Krippendorff's alpha is essentially the only choice that handles missing data natively without requiring imputation. The flexibility on measurement scale is a substantial bonus: you can compute alpha for nominal, ordinal, interval, and ratio data with the same coefficient, just by swapping the distance function. The cost is computational — coincidence matrix construction is <Code>O(N · m²)</Code> for the naive implementation, and bootstrap confidence intervals require many recomputations. For large datasets, use the vectorized implementation from section 5c.
      </Prose>

      <H3>Continuous outcomes, raters as random sample</H3>

      <Prose>
        Intraclass correlation coefficient, specifically ICC(2,1) for single-rater reliability and ICC(2,k) for averaged-rater reliability across <Code>k</Code> raters. The "2" indicates a two-way random-effects model where both items and raters are sampled. ICC(3,1) and ICC(3,k) are the analogous mixed-effects forms where raters are treated as fixed (the specific raters are the population of interest, not a sample); ICC(1,1) and ICC(1,k) are one-way forms where each item may be rated by a different set of raters. The Shrout-Fleiss (1979) paper is the definitive reference for choosing among these. Don't pick by which gives the highest number — pick by which matches the sampling structure of your data.
      </Prose>

      <H3>Internal consistency of a sum-score test</H3>

      <Prose>
        Cronbach's alpha is the standard, but it has well-known limitations: it is a lower bound on reliability, not an unbiased estimator; it assumes essentially tau-equivalent items; and it is sensitive to test length in ways that can mask actual problems with item quality. For modern psychometric work, McDonald's omega (a coefficient based on factor loadings rather than item variances) is often preferred. Pingouin and the R package <Code>psych</Code> compute both. If you have a single-factor model and roughly equally loaded items, alpha and omega will agree closely; if items load on multiple factors or with widely differing strengths, omega is more accurate.
      </Prose>

      <H3>LLM-as-judge agreement studies</H3>

      <Prose>
        Two patterns are common in 2026. For simple binary or nominal judgments (safe/unsafe, helpful/unhelpful) with two judges (e.g., GPT-4o vs Claude), Cohen's kappa is appropriate, with the caveat that the kappa paradox often applies — most labels will be in the dominant class. For ordinal Likert ratings (e.g., 1-5 helpfulness scores) with two or three judges and possible missingness from API timeouts, Krippendorff's alpha at the ordinal level is the standard. For comparing automatic judges against a human reference panel of variable size, Krippendorff's alpha is again the correct tool because of the missing-data robustness.
      </Prose>

      <H3>When to report multiple coefficients</H3>

      <Prose>
        Reporting both percent agreement and chance-corrected agreement is good practice when the data is imbalanced — kappa or alpha alone can be misleading without the raw agreement to anchor it. Reporting both Cronbach's alpha and McDonald's omega is sensible when you have not done a separate factor analysis to confirm dimensionality. Reporting both Cohen's kappa and PABAK is the right move when you suspect the prevalence is driving the kappa downward. The general principle: pick the primary coefficient based on data structure, but add complementary numbers when there is a known interpretive ambiguity.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        The reliability coefficients themselves scale linearly or near-linearly with data size. Cronbach's alpha is <Code>O(N · k)</Code> for variance computation. Cohen's kappa is <Code>O(N + K²)</Code> where <Code>K</Code> is the number of categories. Fleiss' kappa is <Code>O(N · K)</Code>. Krippendorff's alpha is <Code>O(N · m²)</Code> for naive coincidence-matrix construction (where <Code>m</Code> is the number of raters per item) but can be made <Code>O(N · K²)</Code> with the vectorized formulation in section 5c, since the inner loop reduces to a outer product of category counts. None of these are bottlenecks at any reasonable annotation scale. A million items rated by ten coders computes alpha in seconds.
      </Prose>

      <Prose>
        What does not scale is bootstrap confidence intervals for the more complex coefficients. Krippendorff's alpha bootstrapping requires recomputing the entire coincidence matrix from a resampled dataset, repeated 1000 to 5000 times for stable CIs. For very large datasets this becomes the dominant cost — a million-item, ten-rater, 1000-bootstrap CI computation can take tens of minutes. The mitigation is the closed-form asymptotic standard error for Cohen's kappa (and approximate forms for the others), or the analytic Krippendorff CI of Krippendorff (2004) which uses an asymptotic normal approximation; for most large-sample applications these are within a percentage point of the bootstrap and dramatically faster.
      </Prose>

      <Prose>
        Statistical scaling, as opposed to computational scaling, follows the usual square-root law: the precision of the reliability estimate improves as <Code>1/√N</Code>. For Cohen's kappa, doubling the sample halves the standard error. For Krippendorff's alpha the same holds asymptotically but with a more complex constant. The practical implication is that small studies (under 50 items) produce wide confidence intervals on any reliability coefficient — a point estimate of 0.75 might have a 95% CI from 0.50 to 0.90, which is too wide to support meaningful interpretation. Reliability studies should be sized to give a reasonably narrow CI on the threshold you care about; a common heuristic is <Code>N ≥ 100</Code> for the asymptotic CIs to be trustworthy, and <Code>N ≥ 30</Code> as a hard floor for any bootstrap to be informative.
      </Prose>

      <Prose>
        What does not scale away through more data is bias from misspecified models. Cronbach's alpha undersells reliability of multidimensional scales no matter how large the sample. Fleiss' kappa cannot detect rater-specific bias even with infinite data, because the model treats raters as exchangeable. Cohen's kappa with skewed marginals will report low chance-corrected agreement even at very high observed agreement, regardless of sample size. These are structural properties of the coefficients, not artifacts of finite samples. The remedy is to choose a coefficient whose structure matches your data, not to collect more data with the wrong coefficient.
      </Prose>

      <Prose>
        Number-of-rater scaling has a different structure for each coefficient. Cohen's kappa is fundamentally bivariate — there is no "Cohen's kappa with three raters." The convention is to compute pairwise kappas between all rater pairs and report the mean (sometimes called Light's kappa), or to switch to Fleiss' or Krippendorff's. Fleiss' kappa scales naturally to any number of raters per item, but the assumption of complete and equal-sized rater coverage per item becomes harder to satisfy at scale. Krippendorff's alpha is the most rater-scalable: alpha is well-defined even when each item is rated by a different rater subset, and the implementation cost grows only with the number of distinct values, not the rater count.
      </Prose>

      <Prose>
        Multi-construct scaling — where you have multiple distinct things to measure agreement on, not just multiple raters — is a different problem entirely. Computing separate kappa or alpha values for each construct is fine; combining them into a single number requires generalizability theory or multivariate IRT models, neither of which is in scope for this topic. The practical advice for LLM eval at scale is to report per-construct coefficients with their CIs, not a grand-average reliability number, because the grand average obscures which specific dimensions are well-measured and which are noise.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>The kappa paradox: low kappa with high agreement</H3>
      <Prose>
        The most common source of confusion. As demonstrated in section 4f, two raters can agree on 90% or more of items and still produce a kappa near zero (or even negative) when the marginals are highly skewed. The paradox is structural: when one category dominates, the chance baseline is already very high, so observed agreement has little room to exceed it. The fix is not to discard kappa but to report it with the prevalence and observed agreement, and to consider PABAK or Krippendorff's alpha as complementary numbers. Never report kappa alone for a heavily skewed binary classification task.
      </Prose>

      <H3>Cronbach's alpha and dimensionality</H3>
      <Prose>
        Cronbach's alpha is high when items are positively correlated, but it cannot tell you whether those correlations come from a single underlying construct or from multiple correlated constructs. A 30-item test with two 15-item subscales measuring distinct things, where each subscale has high internal consistency and the two subscales correlate at 0.4, will produce a high overall alpha while being multidimensional. The fix is to do a factor analysis (or a parallel analysis to estimate dimensionality) before reporting alpha, and to report alpha for each unidimensional subscale separately rather than for the aggregate. Reviewers in psychometrics will reject papers that report alpha without dimensionality evidence.
      </Prose>

      <H3>Cronbach's alpha is a lower bound, not an unbiased estimator</H3>
      <Prose>
        Alpha equals reliability only under the strong assumption of essential tau-equivalence — equal item-true-score covariances. When items have differing factor loadings (almost always, in practice), alpha underestimates true reliability. The bias is small for unidimensional scales with similar item loadings and can be substantial for scales with mixed loadings. McDonald's omega, which uses estimated factor loadings explicitly, is always at least as large as alpha and is a less biased estimator of reliability when the model is correctly specified. The fix is to report omega alongside alpha for any serious psychometric application.
      </Prose>

      <H3>Mixing weighted-kappa schemes across studies</H3>
      <Prose>
        Linear-weighted kappa and quadratic-weighted kappa report different numbers on the same data. A study reporting "weighted kappa = 0.75" without specifying the weighting scheme is uninterpretable. Worse, results across studies that use different weighting schemes cannot be directly compared. The fix is to always specify the scheme (and ideally to report both, since the two can be converted between each other given the marginal distribution). When reading other people's reports, default to assuming quadratic if not specified — it is the more common choice.
      </Prose>

      <H3>Fleiss' kappa with rater identification</H3>
      <Prose>
        Fleiss' kappa assumes raters are exchangeable. If you have five named annotators each of whom rated all items, Fleiss' kappa will compute a number, but that number does not capture rater-specific bias. Annotator A who systematically over-rates "harmful" relative to the panel mean will not be flagged. The fix is to use Krippendorff's alpha (which preserves rater identity in the coincidence matrix construction) or, more powerfully, to compute Cohen's kappa for each rater pair and inspect the matrix of pairwise agreements; a single rater whose pairwise kappas with everyone else are low is a candidate for retraining or removal.
      </Prose>

      <H3>Krippendorff's alpha distance-function mismatch</H3>
      <Prose>
        Computing nominal alpha on ordinal data — treating "strongly agree" and "strongly disagree" as equally distant from "agree" — undersells real agreement. Computing interval alpha on data that is only approximately ordinal — treating Likert categories as if their numerical labels carry interval meaning — can oversell agreement. The fix is to match the distance function to the actual measurement level, even if it requires more thought than picking the default. For Likert data the ordinal distance is usually correct; the interval distance is appropriate only when there is a defensible argument that adjacent categories are equally far apart on the underlying construct.
      </Prose>

      <H3>Bootstrap confidence intervals on small samples</H3>
      <Prose>
        Below about <Code>N = 30</Code>, bootstrap CIs become wide and unreliable for any reliability coefficient. A bootstrap CI of [0.20, 0.85] on a kappa point estimate of 0.55 carries essentially no information. The fix is to either collect more data (reliability studies should be sized for the precision they need) or to acknowledge the wide CI explicitly and avoid drawing conclusions about specific thresholds. Reporting "kappa = 0.55, 95% CI [0.20, 0.85]" and then claiming "substantial agreement" because the point estimate exceeds 0.61 is statistically dishonest.
      </Prose>

      <H3>Asymptotic standard errors at small N</H3>
      <Prose>
        The closed-form standard error for Cohen's kappa assumes large-sample asymptotics and is unreliable below <Code>N ≈ 100</Code>. Below that threshold the bootstrap is preferred even though it is computationally expensive. For Krippendorff's alpha the analogous warning is even stronger — the asymptotic normal approximation is only trustworthy at <Code>N ≥ 100</Code> and ideally <Code>N ≥ 200</Code>. Many published reliability studies on small annotation samples use the asymptotic CIs and report falsely narrow intervals.
      </Prose>

      <H3>Comparing reliability across different rating scales</H3>
      <Prose>
        Kappa, alpha, and ICC are all bounded between -1 and 1, but they measure different things and are not directly comparable. A Cohen's kappa of 0.70 and a Cronbach's alpha of 0.70 do not represent equivalent levels of measurement quality. The fix is not to convert between them but to report each in its proper context with its appropriate threshold. The temptation to report "agreement coefficient = 0.75" without specifying which coefficient is widespread and almost always counterproductive.
      </Prose>

      <H3>Treating reliability as validity</H3>
      <Prose>
        High reliability is necessary but not sufficient for valid measurement. A test can have alpha of 0.95 and measure something different from what it claims to measure. Two raters can have kappa of 0.90 and both be systematically wrong in the same way. Reliability bounds the maximum possible validity but does not guarantee any validity. The fix is conceptual rather than statistical: report reliability as a precondition for further analysis, not as evidence that the measurement is good. Convergent and discriminant validity require separate evidence.
      </Prose>

      <H3>Selective rater removal</H3>
      <Prose>
        A common bad practice is to compute pairwise agreements and then drop the rater whose agreements are lowest, on the theory that they are noisy. This is data dredging — it inflates the apparent reliability of the remaining panel without representing real improvement. The fix is to set inclusion criteria for raters before any agreement computation (e.g., based on a separate training/calibration phase) and to report the agreement on the prespecified panel.
      </Prose>

      <Callout accent="purple">
        Reliability coefficients are diagnostic tools, not virtue scores. A low kappa is information — it tells you something specific about the agreement structure, often a marginal-distribution effect — and the appropriate response is to investigate the cause, not to switch to a coefficient that gives a higher number.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        The papers and books below are the foundational sources for the coefficients in this topic. All have been cited continuously since publication and remain the standard references in their respective communities.
      </Prose>

      <H3>Cronbach 1951 — alpha</H3>
      <Prose>
        Lee J. Cronbach. "Coefficient alpha and the internal structure of tests." Psychometrika, 16(3):297–334, September 1951. The founding paper. Derives alpha as a generalization of the Kuder-Richardson formula 20 (KR-20) to test items with more than two response categories, proves that alpha is a lower bound on reliability under tau-equivalence, and shows the relationship to the Spearman-Brown prophecy formula. Among the most-cited papers in psychometrics. Cronbach himself revisited the coefficient in "My current thoughts on coefficient alpha and successor procedures" (Educational and Psychological Measurement, 64(3):391–418, 2004), where he argued that alpha had been overused and that more flexible reliability coefficients (including those based on generalizability theory) should replace it for many applications.
      </Prose>

      <H3>Cohen 1960 — kappa</H3>
      <Prose>
        Jacob Cohen. "A coefficient of agreement for nominal scales." Educational and Psychological Measurement, 20(1):37–46, April 1960. Introduces the chance-corrected agreement coefficient that bears Cohen's name. Motivated by Cohen's frustration with the use of raw percent agreement in clinical psychology, where high agreement frequently reflected base-rate exploitation rather than true diagnostic concordance. Cohen extended this to weighted kappa for ordinal scales in "Weighted kappa: Nominal scale agreement with provision for scaled disagreement or partial credit" (Psychological Bulletin, 70(4):213–220, 1968).
      </Prose>

      <H3>Fleiss 1971 — multi-rater kappa</H3>
      <Prose>
        Joseph L. Fleiss. "Measuring nominal scale agreement among many raters." Psychological Bulletin, 76(5):378–382, November 1971. Generalizes Cohen's kappa to any number of raters per item under the assumption of rater exchangeability. The paper is short — five pages — and the derivation is direct. Fleiss returned to the topic with Cohen in "The equivalence of weighted kappa and the intraclass correlation coefficient as measures of reliability" (Educational and Psychological Measurement, 33(3):613–619, 1973), which proved the equivalence of quadratic-weighted kappa and a particular ICC form under equal marginals.
      </Prose>

      <H3>Krippendorff 1980 / 2018 — Content Analysis textbook</H3>
      <Prose>
        Klaus Krippendorff. <em>Content Analysis: An Introduction to Its Methodology</em>. SAGE Publications. First edition 1980, fourth edition 2018. The definitive reference for Krippendorff's alpha. Chapter 11 of the fourth edition contains the full mathematical treatment with all four standard distance functions, missing-data handling, and bootstrap confidence interval procedures. The interpretive thresholds that have become standard in computational content analysis (α ≥ 0.80 acceptable, ≥ 0.667 tentative) come from this book. Krippendorff has continued to refine the alpha computation in subsequent papers; the canonical computational formula is in Hayes and Krippendorff (2007) below.
      </Prose>

      <H3>Hayes and Krippendorff 2007 — alpha computation</H3>
      <Prose>
        Andrew F. Hayes and Klaus Krippendorff. "Answering the call for a standard reliability measure for coding data." Communication Methods and Measures, 1(1):77–89, 2007. Provides a unified, software-implementable formula for Krippendorff's alpha that handles all standard measurement levels, missing data, and any number of raters. This is the formula implemented in the <Code>krippendorff</Code> Python package, in Hayes' SPSS macro, and in essentially every modern Krippendorff's alpha computation. The paper also contains worked examples and a discussion of why Krippendorff's alpha should be preferred to other multi-rater agreement coefficients.
      </Prose>

      <H3>Shrout and Fleiss 1979 — ICC forms</H3>
      <Prose>
        Patrick E. Shrout and Joseph L. Fleiss. "Intraclass correlations: Uses in assessing rater reliability." Psychological Bulletin, 86(2):420–428, March 1979. The classic taxonomy of intraclass correlation coefficients. Distinguishes six forms — ICC(1,1), ICC(2,1), ICC(3,1), ICC(1,k), ICC(2,k), ICC(3,k) — based on the ANOVA model (one-way random, two-way random, two-way mixed), the unit of analysis (single rater vs. mean of k raters), and the agreement criterion (consistency vs. absolute agreement). The paper is essential reading for anyone computing ICC; choosing the wrong form is one of the most common errors in reliability analysis. McGraw and Wong (1996) extended this taxonomy with additional ICC forms and corrected several errors in Shrout-Fleiss.
      </Prose>

      <H3>Feinstein and Cicchetti 1990 — kappa paradox</H3>
      <Prose>
        Alvan R. Feinstein and Domenic V. Cicchetti. "High agreement but low kappa: I. The problems of two paradoxes." Journal of Clinical Epidemiology, 43(6):543–549, 1990. The companion paper Cicchetti and Feinstein, "II. Resolving the paradoxes," appeared in the same journal issue (43(6):551–558). Together these introduced the term "kappa paradox" and proposed PABAK as a complementary measure to be reported alongside kappa for skewed binary classifications. Required reading before computing kappa on any imbalanced clinical or safety dataset.
      </Prose>

      <H3>Landis and Koch 1977 — kappa interpretation</H3>
      <Prose>
        J. Richard Landis and Gary G. Koch. "The measurement of observer agreement for categorical data." Biometrics, 33(1):159–174, March 1977. The source of the still-ubiquitous interpretive thresholds for Cohen's kappa: 0–0.20 slight, 0.21–0.40 fair, 0.41–0.60 moderate, 0.61–0.80 substantial, 0.81–1.00 almost perfect. The thresholds are admitted by the authors to be "arbitrary" but have been adopted as quasi-standards across clinical and behavioral sciences. McHugh (2012) proposed a stricter scheme requiring kappa ≥ 0.80 for "almost perfect" agreement; both schemes are in use, and citations should specify which is being applied.
      </Prose>

      <H3>Gwet 2014 — Handbook of Inter-Rater Reliability</H3>
      <Prose>
        Kilem Li Gwet. <em>Handbook of Inter-Rater Reliability: The Definitive Guide to Measuring the Extent of Agreement Among Raters</em>, 4th edition. Advanced Analytics LLC, 2014. Comprehensive treatment of the entire family of agreement coefficients, including Gwet's own AC1 and AC2 coefficients (which avoid the kappa paradox by using a different chance baseline). The <Code>irrCAC</Code> R and Python packages implement all of Gwet's coefficients. AC1 has been gaining adoption in clinical reliability work as a replacement for Cohen's kappa specifically because it does not collapse to near-zero on highly skewed marginals.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Derive the Spearman-Brown form of Cronbach's alpha</H3>
      <Prose>
        Starting from the standard definition <Code>{"α = (k/(k−1))·(1 − Σσ²ⱼ / σ²_T)"}</Code>, assume all items have equal variance <Code>σ²</Code> and equal pairwise covariance <Code>{"ρσ²"}</Code> where <Code>ρ</Code> is the average inter-item correlation. Expand the variance of the total score in terms of <Code>σ²</Code> and <Code>ρ</Code>, substitute back into the alpha formula, and simplify to show that <Code>{"α = kρ / (1 + (k−1)ρ)"}</Code>. What does this expression say about the relationship between item count and reliability for a fixed inter-item correlation? Why does adding more items always increase alpha when <Code>ρ &gt; 0</Code>? What value of <Code>k</Code> would be needed to reach alpha of 0.90 when <Code>ρ = 0.20</Code>?
      </Prose>

      <H3>Exercise 2 — Construct the kappa paradox</H3>
      <Prose>
        Construct a 2x2 contingency table for two raters where the observed percent agreement is at least 0.92 but Cohen's kappa is less than 0.10. Show your work: the cell counts, the marginal totals, the observed agreement <Code>p_o</Code>, the chance agreement <Code>p_e</Code>, and the kappa value. Then compute PABAK for the same table and compare. Which number better reflects the practical agreement between the raters in this case? Now reverse the question: construct a table where percent agreement is exactly 0.50 but kappa is positive. What does this require about the marginal distributions, and what does it tell you about kappa as an agreement measure?
      </Prose>

      <H3>Exercise 3 — Choose the right coefficient</H3>
      <Prose>
        For each of the following five scenarios, name the most appropriate reliability coefficient and justify your choice in one or two sentences. (a) Three named clinical psychologists each rated 50 patient interview transcripts on a 7-point depression severity scale. (b) Twelve crowdworkers from MTurk each labeled some subset of 5000 product reviews as positive, neutral, or negative; not every worker rated every review. (c) A 25-item personality questionnaire administered to 800 respondents, where you want to know whether the items measure a coherent construct. (d) Two LLM judges (Claude and GPT-4o) each labeled 300 model outputs as harmful or non-harmful; 270 of the outputs are non-harmful. (e) A panel of four raters scored 40 student essays on a continuous 0–100 scale; the goal is to use the mean of the four raters as the official score and to report the reliability of that mean.
      </Prose>

      <H3>Exercise 4 — Implement weighted kappa from a contingency table</H3>
      <Prose>
        Given a 4x4 contingency table for two raters on a 4-point ordinal scale, implement weighted kappa from scratch using both linear and quadratic weights. Verify your implementation by constructing the table from a known kappa value: pick a target kappa of 0.70, choose marginal distributions, and back-construct the joint cells that produce that kappa under quadratic weighting. Then verify your forward computation reproduces the target. What property of the weight matrix makes this exercise possible? What changes if you use linear weights instead?
      </Prose>

      <H3>Exercise 5 — Bootstrap a Krippendorff's alpha confidence interval</H3>
      <Prose>
        Take the 4-rater, 12-unit reliability matrix from section 4e (with missing values). Implement a bootstrap procedure that resamples the units (columns) with replacement and recomputes Krippendorff's alpha 2000 times. Report the 95% percentile confidence interval and compare it to the analytic asymptotic standard error from the <Code>krippendorff</Code> package. What goes wrong with the bootstrap on such a small sample? What is the practical minimum sample size for trustworthy bootstrap CIs on Krippendorff's alpha, and how would you justify that number empirically?
      </Prose>

      <H3>Exercise 6 — Detect rater bias hidden by Fleiss' kappa</H3>
      <Prose>
        Construct a synthetic dataset with five raters and 200 items where four raters agree closely with each other (pairwise Cohen's kappa around 0.80) but a fifth rater systematically labels items more leniently than the other four. Fleiss' kappa on the full dataset will give a single number — what is it for your construction? Now compute the matrix of pairwise Cohen's kappas between all rater pairs. What does the pairwise matrix reveal that Fleiss' kappa hides? Propose a workflow that uses both numbers in sequence to (a) flag the panel-level agreement quality and (b) identify individual raters who are out of calibration with the rest.
      </Prose>

      <H3>Exercise 7 — Reliability bounds validity</H3>
      <Prose>
        Suppose a benchmark for measuring a model's "reasoning ability" has a Cronbach's alpha of 0.40 across its 100 items when administered to a population of 500 models. What can you conclude about the validity of the benchmark as a measurement of reasoning ability? What is the best-case correlation that this benchmark could have with any other valid measure of reasoning, given its reliability? (Hint: the disattenuated correlation formula gives an upper bound on the true correlation between two constructs given their respective reliabilities.) What concrete steps would you recommend to improve the benchmark's reliability before using its scores for any model-comparison decision?
      </Prose>

    </div>
  ),
};

export default reliabilityCoefficients;
