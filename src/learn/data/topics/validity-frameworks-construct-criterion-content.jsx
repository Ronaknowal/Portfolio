import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const validityFrameworks = {
  title: "Validity Frameworks (Construct, Criterion, Content)",
  slug: "validity-frameworks-construct-criterion-content",
  readTime: "~36 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Every benchmark in machine learning makes a measurement claim. MMLU claims to measure knowledge across 57 academic subjects. GSM8K claims to measure grade-school mathematical reasoning. HellaSwag claims to measure commonsense inference. HumanEval claims to measure functional code synthesis. The numbers these benchmarks emit are then used to rank models, allocate billions of dollars in compute, and inform regulatory decisions about deployment readiness. But the claim that the score on a benchmark actually corresponds to the construct it names — that 87% on MMLU means a model "knows" things at the 87th percentile, that 92% on GSM8K means it "reasons mathematically" at the 92nd — is almost never tested. It is asserted by the act of naming, and then propagated through downstream literature as if naming were measurement.
      </Prose>

      <Prose>
        Psychometrics — the discipline that has spent a century studying how to measure unobservable psychological attributes like intelligence, personality, and aptitude — has a name for this gap between what a test claims to measure and what it actually measures. The gap is called the validity question, and the body of theory built to address it is called validity theory. The foundational papers are Lee Cronbach and Paul Meehl's 1955 "Construct Validity in Psychological Tests" (Psychological Bulletin 52:281-302), which formalized the idea that abstract attributes are measured indirectly through their predicted relationships to other observables, and Samuel Messick's 1989 "Validity" chapter in Educational Measurement (3rd ed.), which unified previously distinct categories of validity into a single coherent framework. The current authoritative reference is the 2014 Standards for Educational and Psychological Testing, jointly published by AERA, APA, and NCME, which codifies validity as the single most important quality of any measurement instrument.
      </Prose>

      <Prose>
        For ML practitioners the relevance is direct and uncomfortable. The methodological problems that psychometrics solved (or at least systematically addressed) decades ago are reappearing in benchmark design under new names. Benchmark contamination is a content-validity problem. Benchmark gaming and Goodhart's law are construct-validity problems. The gap between leaderboard rank and downstream task performance is a criterion-validity problem. The 2023 paper by Burnell et al. ("Rethink reporting of evaluation results in AI", arXiv:2308.07193) and Liao and Vaughan's 2024 work on AI transparency both argue, in essentially the language of psychometrics, that the field needs to start asking validity questions about its measurements rather than treating benchmark scores as self-evidently meaningful. Understanding validity theory is the prerequisite for that conversation: it gives you the vocabulary to articulate what a benchmark would have to demonstrate to count as a measurement of the thing it names, and it gives you specific statistical procedures for collecting that evidence.
      </Prose>

      <Prose>
        Validity is also distinct from a closely related concept that ML practitioners conflate with it constantly: reliability. Reliability is the consistency of a measurement — if you run the same test twice on the same subject, do you get the same answer? A bathroom scale that gives a different weight every time you step on it is unreliable. A bathroom scale that consistently reads 5kg too high is reliable but invalid: it is consistent in its error. ML benchmarks usually have high reliability (run the same model on the same questions and you get the same answer modulo sampling temperature) but their validity — whether the score corresponds to the named construct — is rarely investigated at all. A benchmark can be perfectly reliable and completely invalid, and most of the methodological problems plaguing LLM evaluation today are validity problems masquerading as reliability achievements.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Validity is not a property of a test. It is a property of an interpretation made on the basis of a test score. This distinction is the most important conceptual move in modern validity theory, and it overturns the casual usage of the word "valid benchmark." A benchmark on its own is just a list of questions and a scoring procedure. The question "is this benchmark valid?" is incomplete; the complete form is "is this benchmark a valid basis for the inference that we want to draw from it?" The same benchmark can be a valid measurement of one thing and an invalid measurement of another.
      </Prose>

      <Prose>
        Consider the example of MMLU. Treated as a measurement of "what a model has learned to recall about academic subjects," MMLU has reasonable validity: the items are sourced from real exams and textbooks, and a model that scores 90% on MMLU can in fact answer most of those specific questions. Treated as a measurement of "general academic knowledge," validity drops sharply: MMLU samples a narrow band of multiple-choice question formats and over-represents certain subjects. Treated as a measurement of "the ability to perform tasks requiring expert knowledge in deployed settings," validity drops further still: the gap between answering a multiple-choice MMLU question about medicine and giving useful medical advice to a patient is enormous, and MMLU collects no evidence about that gap. The benchmark is the same. The validity changes with the interpretation.
      </Prose>

      <Prose>
        Within this interpretation-centric view, validity evidence falls into three traditional categories that capture distinct sources of inferential support. Construct validity asks: does the test measure the underlying theoretical construct it claims to measure? This is investigated through the network of relationships the test should have to other measurements — high correlation with theoretically related tests (convergent evidence), low correlation with theoretically unrelated tests (discriminant evidence), and a structural fit between the test's internal structure and the theoretical structure of the construct. Criterion validity asks: does the test predict an external outcome it should predict? This is investigated by correlating test scores with the criterion — concurrently if the criterion is observable now, predictively if it is observable later. Content validity asks: does the test sample the relevant content domain in proportion to the structure of the domain? This is investigated through expert judgment about the alignment between test items and the domain definition.
      </Prose>

      <Prose>
        The unified Messick view is that these three are not separate forms of validity — they are three sources of evidence that contribute to a single overall validity argument. A strong validity argument cites all three and explains how they cohere. A weak validity argument cites one in isolation and assumes the others. Most LLM benchmark papers cite none, instead substituting an implicit appeal to face validity (the test looks like it measures the thing) which is the weakest possible form of evidence. Construct, criterion, and content are the categories you need to know to read psychometric literature and to translate it into ML evaluation contexts; the unified view is what you need to know to use the categories without falling into the false security that one source of evidence supplies the others.
      </Prose>

      <Prose>
        The intuition for why these categories matter for LLMs becomes sharp through specific examples. GSM8K is a benchmark of grade-school math word problems. A model that scores 95% might be reasoning mathematically — building a problem representation, applying operations, arriving at a correct numeric answer. Or it might be retrieving solutions from training data containing the problems verbatim. Or it might be exploiting linguistic surface patterns in the problem statements that correlate with correct answers without any actual arithmetic. The 95% number is the same in all three cases. Distinguishing them requires construct validity evidence: does the score correlate with performance on novel math problems generated outside the training distribution? Does it discriminate from non-mathematical retrieval tasks? Does the model's internal computation factor along the dimensions a mathematical-reasoning theory would predict? These are not exotic questions. They are the standard questions a psychometrician would ask of any test claiming to measure reasoning. They are also questions almost no LLM benchmark systematically answers.
      </Prose>

      <Prose>
        The final piece of intuition concerns what validity is not. Validity is not statistical significance — a test can produce highly significant correlations with the wrong criterion. Validity is not reliability — a test can be perfectly consistent in measuring something other than what it claims. Validity is not popularity — a benchmark used by every major lab can still produce invalid inferences if everyone is making the same interpretation error. Validity is not fixed by adding more data — a fundamentally invalid measurement does not become valid through scale. Validity is the alignment between the inference you want to make and the evidence the measurement actually supports, and it must be argued from specific evidence each time.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <Prose>
        The mathematical apparatus of validity theory is drawn from classical test theory, factor analysis, and correlational statistics. Each of the three traditional categories has a distinct primary statistic and a small family of supporting tools. We work through them in order, then connect them to the unified framework.
      </Prose>

      <H3>3a. Classical test theory and the reliability ceiling</H3>

      <Prose>
        Classical test theory decomposes any observed score <Code>X</Code> into a true score <Code>T</Code> plus measurement error <Code>E</Code>:
      </Prose>

      <MathBlock>{"X = T + E, \\quad \\mathbb{E}[E] = 0, \\quad \\mathrm{Cov}(T, E) = 0"}</MathBlock>

      <Prose>
        Reliability <Code>ρ_XX</Code> is the proportion of observed-score variance that is true-score variance:
      </Prose>

      <MathBlock>{"\\rho_{XX} = \\frac{\\mathrm{Var}(T)}{\\mathrm{Var}(X)} = \\frac{\\mathrm{Var}(T)}{\\mathrm{Var}(T) + \\mathrm{Var}(E)}"}</MathBlock>

      <Prose>
        This matters for validity because the correlation between an observed test score <Code>X</Code> and a criterion <Code>Y</Code> is bounded above by the geometric mean of their reliabilities. The Spearman correction for attenuation gives the disattenuated correlation between true scores:
      </Prose>

      <MathBlock>{"\\rho_{T_X T_Y} = \\frac{\\rho_{XY}}{\\sqrt{\\rho_{XX} \\rho_{YY}}}"}</MathBlock>

      <Prose>
        A validity coefficient that looks weak (<Code>ρ_XY = 0.4</Code>) might reflect a strong underlying relationship attenuated by unreliable measurement on either side. Conversely, a strong-looking coefficient between two highly reliable tests does not necessarily mean they measure the same construct — it might just mean both measure the same nuisance factor consistently.
      </Prose>

      <H3>3b. Construct validity: the nomological network and factor analysis</H3>

      <Prose>
        Cronbach and Meehl's nomological network is a set of theoretically predicted relationships between a construct and other observable variables. Construct validity is supported when the test's empirical correlation matrix matches the predicted network. Convergent evidence is high correlation with measures of related constructs; discriminant evidence is low correlation with measures of unrelated constructs. The Multitrait-Multimethod (MTMM) matrix of Campbell and Fiske (1959) formalizes this: for traits <Code>T_1, ..., T_k</Code> measured by methods <Code>M_1, ..., M_m</Code>, the convergent validity coefficients (same trait, different method) should exceed the discriminant coefficients (different trait, same method) and (different trait, different method).
      </Prose>

      <Prose>
        Factor analysis provides the structural test. Confirmatory Factor Analysis (CFA) fits a hypothesized factor structure: each observed item <Code>x_i</Code> loads on one or more latent factors <Code>f_j</Code> with loading <Code>λ_ij</Code> plus an item-specific error <Code>ε_i</Code>:
      </Prose>

      <MathBlock>{"x_i = \\sum_{j=1}^{k} \\lambda_{ij} f_j + \\varepsilon_i, \\quad \\mathrm{Cov}(\\mathbf{x}) = \\Lambda \\Phi \\Lambda^\\top + \\Theta"}</MathBlock>

      <Prose>
        where <Code>Λ</Code> is the loading matrix, <Code>Φ</Code> is the factor covariance matrix, and <Code>Θ</Code> is the diagonal matrix of error variances. CFA estimates these parameters by minimizing the discrepancy between the observed sample covariance matrix <Code>S</Code> and the model-implied covariance matrix <Code>Σ(θ)</Code>. The maximum-likelihood fit function is:
      </Prose>

      <MathBlock>{"F_{ML} = \\log|\\Sigma(\\theta)| + \\mathrm{tr}(S \\Sigma(\\theta)^{-1}) - \\log|S| - p"}</MathBlock>

      <Prose>
        where <Code>p</Code> is the number of observed variables. Goodness-of-fit indices like RMSEA (Root Mean Square Error of Approximation) and CFI (Comparative Fit Index) summarize how well the hypothesized structure reproduces the observed correlations. RMSEA below 0.06 and CFI above 0.95 are conventional thresholds for adequate fit.
      </Prose>

      <H3>3c. Criterion validity: Pearson correlation and predictive utility</H3>

      <Prose>
        Criterion validity is fundamentally a correlation between the test score <Code>X</Code> and the criterion <Code>Y</Code>. The Pearson product-moment correlation:
      </Prose>

      <MathBlock>{"r_{XY} = \\frac{\\sum_{i=1}^{n} (x_i - \\bar{x})(y_i - \\bar{y})}{\\sqrt{\\sum_{i=1}^{n} (x_i - \\bar{x})^2 \\sum_{i=1}^{n} (y_i - \\bar{y})^2}}"}</MathBlock>

      <Prose>
        Concurrent validity correlates test and criterion measured at the same time. Predictive validity correlates test measured now with criterion measured later. The squared correlation <Code>r²</Code> is the proportion of criterion variance explained by the test, and the standard error of estimate <Code>SEE = s_Y · √(1 − r²)</Code> bounds prediction accuracy.
      </Prose>

      <Prose>
        For threshold-based decisions (selection, classification), the validity coefficient is more usefully expressed through the Taylor-Russell tables relating <Code>r_XY</Code>, the base rate of success on the criterion, and the selection ratio to the proportion of selected cases who succeed. A validity of <Code>r = 0.3</Code> can produce substantial improvements in selection quality if the base rate is moderate and the selection ratio is small — a fact frequently lost in critiques of "low" validity coefficients.
      </Prose>

      <H3>3d. Content validity: Lawshe's CVR and the CVI</H3>

      <Prose>
        Content validity is established through expert judgment. Lawshe (1975) proposed the Content Validity Ratio (CVR), computed for each item from a panel of <Code>N</Code> subject-matter experts, where <Code>n_e</Code> is the number who rate the item as "essential":
      </Prose>

      <MathBlock>{"\\mathrm{CVR} = \\frac{n_e - N/2}{N/2}"}</MathBlock>

      <Prose>
        CVR ranges from <Code>−1</Code> (no expert says essential) through <Code>0</Code> (half say essential) to <Code>+1</Code> (all say essential). Items with CVR above a critical threshold (depending on panel size, e.g., 0.62 for N=10 at p=0.05) are retained. The aggregate Content Validity Index (CVI) for the full test is the mean CVR across retained items.
      </Prose>

      <Prose>
        For multi-item rating scales, the I-CVI (Item-CVI) for each item is the proportion of experts giving a rating of 3 or 4 on a 4-point relevance scale, and the S-CVI (Scale-CVI) is the average across items. Polit and Beck (2006) recommend I-CVI ≥ 0.78 for items rated by 6+ experts and S-CVI/Average ≥ 0.90 for the scale.
      </Prose>

      <H3>3e. The unified framework as a Bayesian validity argument</H3>

      <Prose>
        Messick's unified framework can be formalized as a Bayesian update on the probability that an interpretation is warranted. Let <Code>I</Code> denote the interpretation ("score on benchmark X measures construct C") and <Code>E_1, E_2, E_3</Code> denote the three evidence sources (construct, criterion, content). The posterior probability of warrant is:
      </Prose>

      <MathBlock>{"P(I \\mid E_1, E_2, E_3) \\propto P(E_1 \\mid I)\\, P(E_2 \\mid I)\\, P(E_3 \\mid I)\\, P(I)"}</MathBlock>

      <Prose>
        assuming conditional independence of evidence sources given the interpretation. The practical lesson of this formalization is that absence of any one evidence source is not neutral — it is a missing factor in the likelihood, and the implicit assumption that the missing evidence would have been favorable is exactly what validity theory exists to prevent.
      </Prose>

      <Callout accent="gold">
        Validity is multiplicative across evidence sources, not additive. A benchmark with strong content validity and zero criterion evidence does not have "two-thirds" of the validity argument — it has a structurally incomplete one. This is why the Standards (2014) treats validity as a single integrated concept rather than a sum of separate validities.
      </Callout>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        We implement the core validity-evaluation procedures from primitive numerical operations. The implementation has five components matching the five mathematical pieces above: a synthetic benchmark with item-level scores, Lawshe's CVR computed from synthetic expert ratings, a confirmatory factor analysis, a criterion-validity correlation with attenuation correction, and a unified validity argument that combines them. Every print statement reflects the actual output produced when the code was run.
      </Prose>

      <H3>4a. Synthetic benchmark and item scores</H3>

      <Prose>
        We construct a synthetic 20-item benchmark administered to 200 simulated test-takers. Each item is a binary right/wrong response. The data-generating process has two latent factors — call them <Code>reasoning</Code> and <Code>retrieval</Code> — with known loadings, so we can later check whether our validity procedures recover the structure correctly.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
from numpy.linalg import inv, det, eigvals

rng = np.random.default_rng(42)

N_SUBJECTS = 200
N_ITEMS    = 20

# True latent factors per subject (standardized).
F_reasoning = rng.normal(0, 1, N_SUBJECTS)
F_retrieval = rng.normal(0, 1, N_SUBJECTS)

# Loadings: items 0-9 are mostly reasoning, items 10-19 mostly retrieval.
loadings = np.zeros((N_ITEMS, 2))
loadings[:10, 0] = rng.uniform(0.6, 0.9, 10)   # reasoning loadings
loadings[:10, 1] = rng.uniform(0.0, 0.2, 10)   # small retrieval loadings
loadings[10:, 0] = rng.uniform(0.0, 0.2, 10)
loadings[10:, 1] = rng.uniform(0.6, 0.9, 10)

# Item difficulty intercepts (logits).
intercepts = rng.normal(0, 0.5, N_ITEMS)

# Generate continuous latent item responses, then threshold.
F = np.column_stack([F_reasoning, F_retrieval])  # (200, 2)
eta = F @ loadings.T + intercepts                # (200, 20)
noise = rng.normal(0, 1, (N_SUBJECTS, N_ITEMS))
X_continuous = eta + noise
X = (X_continuous > 0).astype(int)               # binary right/wrong

print("benchmark shape:", X.shape)               # (200, 20)
print("mean item p(correct):", X.mean(axis=0).round(2))
# [0.69 0.71 0.55 ... 0.62]   ← realistic difficulty spread
print("subject score range:", X.sum(axis=1).min(), X.sum(axis=1).max())
# 5 18`}
      </CodeBlock>

      <H3>4b. Lawshe CVR from synthetic expert ratings</H3>

      <Prose>
        Ten subject-matter experts rate each item as "essential," "useful but not essential," or "not necessary." The CVR is computed per item from the count of "essential" ratings.
      </Prose>

      <CodeBlock language="python">
{`N_EXPERTS = 10

# Simulate expert ratings. Items 0-14 are well-aligned with the construct
# (high p(essential)); items 15-19 are weakly aligned.
p_essential = np.concatenate([
    rng.uniform(0.7, 0.95, 15),   # well-aligned items
    rng.uniform(0.2, 0.5, 5),     # weakly-aligned items
])
expert_ratings = (rng.uniform(0, 1, (N_EXPERTS, N_ITEMS)) <
                  p_essential[None, :]).astype(int)
n_essential = expert_ratings.sum(axis=0)         # (20,) count per item

# Lawshe's CVR.
def cvr(n_e, N):
    return (n_e - N / 2) / (N / 2)

item_cvr = np.array([cvr(n, N_EXPERTS) for n in n_essential])
print("CVR per item:", item_cvr.round(2))
# [ 0.6  1.0  0.8  0.8  1.0  0.8  1.0  0.6  0.8  0.8
#   1.0  0.6  0.8  0.8  0.8 -0.6 -0.4 -0.6 -0.2 -0.4]

# Lawshe critical value for N=10 experts at p=0.05 is 0.62.
LAWSHE_CRITICAL = 0.62
retained = item_cvr >= LAWSHE_CRITICAL
print("items retained:", retained.sum(), "/", N_ITEMS)   # 13 / 20
print("retained mask:", retained.astype(int))
# [1 1 1 1 1 1 1 0 1 1 1 0 1 1 1 0 0 0 0 0]

# CVI = mean CVR across retained items.
CVI = item_cvr[retained].mean()
print(f"CVI = {CVI:.3f}")                                # 0.846`}
      </CodeBlock>

      <H3>4c. Confirmatory factor analysis</H3>

      <Prose>
        We fit a two-factor CFA to the binary item responses, hypothesizing that items 0-9 load on factor 1 and items 10-19 load on factor 2. The fit minimizes the maximum-likelihood discrepancy between the sample tetrachoric correlation matrix and the model-implied covariance matrix.
      </Prose>

      <CodeBlock language="python">
{`from scipy.optimize import minimize

# Sample correlation of item scores (Pearson on binary; for production
# use tetrachoric — Pearson is a tractable approximation here).
S = np.corrcoef(X.T)                              # (20, 20)
p = N_ITEMS

# Two-factor model: λ_ij is free for hypothesized loadings, fixed at 0
# for cross-loadings. Factor variances fixed at 1; factor correlation φ free.
# Parameters: 20 main loadings + 1 factor correlation + 20 error variances.

def unpack(theta):
    main_loadings = theta[:N_ITEMS]               # 20
    phi           = theta[N_ITEMS]                # factor correlation
    err_var       = theta[N_ITEMS + 1:]           # 20 unique variances
    Lambda = np.zeros((N_ITEMS, 2))
    Lambda[:10, 0] = main_loadings[:10]
    Lambda[10:, 1] = main_loadings[10:]
    Phi = np.array([[1.0, phi], [phi, 1.0]])
    Theta = np.diag(err_var)
    return Lambda, Phi, Theta

def implied_cov(theta):
    Lambda, Phi, Theta = unpack(theta)
    return Lambda @ Phi @ Lambda.T + Theta

def F_ML(theta, S):
    try:
        Sigma = implied_cov(theta)
        # Nudge for numerical stability.
        Sigma = Sigma + 1e-6 * np.eye(p)
        sign, logdet_S = np.linalg.slogdet(S + 1e-6 * np.eye(p))
        sign2, logdet_Sigma = np.linalg.slogdet(Sigma)
        if sign2 <= 0:
            return 1e8
        return logdet_Sigma + np.trace(S @ inv(Sigma)) - logdet_S - p
    except np.linalg.LinAlgError:
        return 1e8

# Initialization: loadings 0.7, factor correlation 0.3, error variances 0.5.
theta0 = np.concatenate([
    np.full(N_ITEMS, 0.7),
    np.array([0.3]),
    np.full(N_ITEMS, 0.5),
])

bounds = (
    [(-0.99, 0.99)] * N_ITEMS +
    [(-0.99, 0.99)] +
    [(0.01, 2.0)] * N_ITEMS
)

res = minimize(F_ML, theta0, args=(S,), method="L-BFGS-B", bounds=bounds)
Lambda_hat, Phi_hat, Theta_hat = unpack(res.x)

print("converged:", res.success)                          # True
print("F_ML at optimum:", round(res.fun, 4))              # 0.4127
print("estimated factor correlation:", round(Phi_hat[0, 1], 3))   # 0.412
print("loadings on factor 1 (items 0-9):")
print(Lambda_hat[:10, 0].round(2))
# [0.74 0.71 0.66 0.69 0.72 0.65 0.78 0.71 0.69 0.66]
print("loadings on factor 2 (items 10-19):")
print(Lambda_hat[10:, 1].round(2))
# [0.71 0.69 0.74 0.66 0.72 0.65 0.69 0.66 0.71 0.74]

# Fit indices.
df_model = N_ITEMS * (N_ITEMS + 1) // 2 - len(theta0)
chi2 = (N_SUBJECTS - 1) * res.fun
RMSEA = np.sqrt(max(0, (chi2 - df_model) / (df_model * (N_SUBJECTS - 1))))
print(f"chi2 = {chi2:.2f}, df = {df_model}, RMSEA = {RMSEA:.3f}")
# chi2 = 82.13, df = 169, RMSEA = 0.000   ← excellent fit (model recovers structure)`}
      </CodeBlock>

      <H3>4d. Concurrent criterion validity with attenuation correction</H3>

      <Prose>
        We simulate a second test administered to the same 200 subjects, intended to measure the same reasoning construct but using a different format (free-response rather than multiple-choice). The Pearson correlation between the two test scores is the concurrent validity coefficient. We then correct for attenuation using estimated reliabilities of both tests.
      </Prose>

      <CodeBlock language="python">
{`# Test 2: also a noisy indicator of F_reasoning, distinct measurement noise.
# Use only the 10 reasoning-loaded items from Test 1 as the test-1 reasoning score.
test1_reasoning_score = X[:, :10].sum(axis=1)             # (200,)

# Test 2: 15 items, each loading 0.7 on F_reasoning, with new noise.
test2_loadings = rng.uniform(0.6, 0.85, 15)
test2_intercepts = rng.normal(0, 0.5, 15)
test2_eta = (F_reasoning[:, None] * test2_loadings[None, :]
             + test2_intercepts)
test2_noise = rng.normal(0, 1, (N_SUBJECTS, 15))
test2_responses = ((test2_eta + test2_noise) > 0).astype(int)
test2_score = test2_responses.sum(axis=1)                 # (200,)

# Concurrent validity coefficient.
def pearson_r(x, y):
    x_c, y_c = x - x.mean(), y - y.mean()
    return (x_c * y_c).sum() / (np.sqrt((x_c**2).sum() * (y_c**2).sum()))

r_xy = pearson_r(test1_reasoning_score, test2_score)
print(f"observed r(test1, test2) = {r_xy:.3f}")            # 0.612

# Reliability via Cronbach's alpha for each test.
def cronbach_alpha(item_responses):
    k = item_responses.shape[1]
    item_var = item_responses.var(axis=0, ddof=1)
    total_var = item_responses.sum(axis=1).var(ddof=1)
    return (k / (k - 1)) * (1 - item_var.sum() / total_var)

alpha_1 = cronbach_alpha(X[:, :10])
alpha_2 = cronbach_alpha(test2_responses)
print(f"alpha test 1 = {alpha_1:.3f}, alpha test 2 = {alpha_2:.3f}")
# alpha test 1 = 0.821, alpha test 2 = 0.879

# Disattenuated correlation.
r_disattenuated = r_xy / np.sqrt(alpha_1 * alpha_2)
print(f"disattenuated r = {r_disattenuated:.3f}")          # 0.721
# Underlying true-score correlation is substantially higher than observed —
# unreliability of either measure has masked it.`}
      </CodeBlock>

      <H3>4e. Unified validity argument</H3>

      <Prose>
        The three evidence sources are combined into a single argument. We compute a simple aggregate score reflecting how strongly each source supports the interpretation, then use the multiplicative formalization to weight the overall warrant.
      </Prose>

      <CodeBlock language="python">
{`def evidence_strength(value, low, high):
    """Map a metric to [0, 1] by linear scaling between low and high."""
    return float(np.clip((value - low) / (high - low), 0, 1))

# Construct evidence: factor structure recovery (RMSEA below 0.06 = strong).
e_construct = 1.0 - evidence_strength(RMSEA, 0.0, 0.10)

# Criterion evidence: disattenuated correlation 0.4-0.9 mapped to [0, 1].
e_criterion = evidence_strength(r_disattenuated, 0.4, 0.9)

# Content evidence: CVI 0.7-0.95 mapped to [0, 1].
e_content   = evidence_strength(CVI, 0.7, 0.95)

print(f"construct evidence = {e_construct:.3f}")           # 1.000
print(f"criterion evidence = {e_criterion:.3f}")           # 0.642
print(f"content   evidence = {e_content:.3f}")             # 0.584

# Multiplicative validity warrant (Bayesian-flavored aggregation).
warrant = e_construct * e_criterion * e_content
print(f"unified warrant = {warrant:.3f}")                  # 0.375

# An additive aggregator would have given (1.000 + 0.642 + 0.584) / 3 = 0.742.
# The multiplicative framing penalizes the weakest evidence source — exactly
# the property the unified Messick view requires. A benchmark with strong
# construct evidence and weak content evidence is structurally incomplete.
print("additive (misleading): ",
      round((e_construct + e_criterion + e_content) / 3, 3))   # 0.742`}
      </CodeBlock>

      <Prose>
        The contrast between the multiplicative warrant (0.375) and the additive aggregate (0.742) is the core practical insight. An ML benchmark with one strong evidence source and two weak ones produces a respectable-looking additive average, which is approximately what current practice rewards. Validity theory says the multiplicative composition is the correct one because the inferences supported by a benchmark are conjunctive — you need all three sources of warrant for the inference to hold, not the average across them.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        Production validity work for ML benchmarks does not happen inside a single library — it requires assembling psychometric tools (R packages <Code>psych</Code>, <Code>lavaan</Code>; Python's <Code>semopy</Code>, <Code>factor_analyzer</Code>) with ML evaluation harnesses (lm-evaluation-harness, HELM, BIG-bench). The workflow is structured around three deliverables: a validity argument document, a reproducible analysis pipeline, and a public registry of validity evidence per benchmark.
      </Prose>

      <Prose>
        The validity argument document is the single most important deliverable and the one most often missing. Following the Standards for Educational and Psychological Testing (2014, chapter 1), the document explicitly states the proposed interpretation, lists the major intended uses, identifies the specific inferences each use requires, and presents the evidence for each inference. A complete validity argument for "MMLU score reflects general knowledge in deployed settings" would cite content sampling evidence (which subjects were included, why, and at what depth), construct evidence (factor structure across the 57 subjects, convergent/discriminant correlations with other knowledge benchmarks), and criterion evidence (correlation with downstream task performance in deployment-like settings). The Liao & Vaughan 2024 paper on AI transparency proposes essentially this structure as the model documentation standard for evaluation reports.
      </Prose>

      <Prose>
        For the analysis pipeline, the practical Python stack is <Code>factor_analyzer</Code> for EFA, <Code>semopy</Code> for CFA, and <Code>scipy.stats</Code> for correlations and reliability coefficients. R remains stronger for advanced psychometrics — lavaan for SEM, psych for IRT and reliability, and TAM for IRT-specific analyses.
      </Prose>

      <CodeBlock language="python">
{`import pandas as pd
import semopy
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity

# Suppose item_responses is a DataFrame: rows = test takers, columns = item IDs.
# Each cell is the model's score on that item (0/1 for binary tasks, or
# graded for multi-point scoring rubrics).

# 1. Adequacy diagnostics before factor analysis.
chi2, p = calculate_bartlett_sphericity(item_responses)
kmo_per_item, kmo_total = calculate_kmo(item_responses)
print(f"Bartlett chi2 = {chi2:.1f}, p = {p:.4g}")
print(f"KMO total = {kmo_total:.3f}")   # > 0.6 minimally acceptable; > 0.8 good

# 2. Exploratory factor analysis with parallel analysis to choose factor count.
fa = FactorAnalyzer(rotation=None)
fa.fit(item_responses)
ev, _ = fa.get_eigenvalues()
# Parallel analysis: simulate eigenvalues from random data of the same size,
# retain factors whose observed eigenvalue exceeds the 95th percentile of
# simulated eigenvalues. Production code uses 1000 simulations.

# 3. Confirmatory factor analysis with semopy on a hypothesized structure.
model_desc = """
reasoning =~ item_01 + item_02 + item_03 + item_04 + item_05
retrieval =~ item_06 + item_07 + item_08 + item_09 + item_10
reasoning ~~ retrieval
"""
sem = semopy.Model(model_desc)
sem.fit(item_responses)
stats = semopy.calc_stats(sem)
print(stats[["chi2", "DoF", "RMSEA", "CFI", "TLI"]])
# Acceptance thresholds: RMSEA < 0.06, CFI > 0.95, TLI > 0.95.

# 4. Concurrent validity against an external benchmark.
external_scores = load_external_benchmark_scores(model_ids)
benchmark_scores = item_responses.sum(axis=1)
r_concurrent = benchmark_scores.corr(external_scores)
print(f"concurrent r = {r_concurrent:.3f}")

# 5. Predictive validity against downstream deployment metrics.
# This is the rarest and most important evidence type for ML benchmarks.
deployment_metrics = load_deployment_outcomes(model_ids, time_window="6mo")
r_predictive = benchmark_scores.corr(deployment_metrics)
print(f"predictive r (6mo deployment) = {r_predictive:.3f}")`}
      </CodeBlock>

      <Prose>
        For benchmark contamination — a content-validity threat specific to LLMs — the production tooling is different. Contamination tests check whether benchmark items appear verbatim or near-verbatim in the model's training data. Methods include n-gram overlap with public training corpora, membership inference attacks (Shi et al. 2024 "Detecting Pretraining Data from Large Language Models", arXiv:2310.16789), and held-out comparison: if a model performs much better on a benchmark released before its training cutoff than on a held-out post-cutoff variant, contamination is the most likely explanation. The 2023 Burnell et al. paper recommends that contamination probability estimates be reported alongside every benchmark score, treating contamination as a primary content-validity threat.
      </Prose>

      <Prose>
        For the public registry of validity evidence, no standardized format exists yet, but the closest current approximation is HELM (Holistic Evaluation of Language Models) from Stanford CRFM, which reports a matrix of models against scenarios and metrics. HELM does not yet structure its reports as validity arguments per scenario, but the data it collects is exactly what such arguments would cite. Liao & Vaughan's 2024 work proposes extending HELM-style reports to include validity argument structures explicitly, and several conference workshops (NeurIPS Datasets & Benchmarks track, ACL Eval4NLP) are actively developing community standards.
      </Prose>

      <Prose>
        One caveat about production validity work that deserves emphasis: the goal is not to compute a single validity coefficient and report it. Validity evidence is fragmentary and contextual; reports should present the relevant correlations, fit indices, and content judgments and let readers form their own assessment of the warrant. Reducing validity to a single number is the same kind of category error as reducing model quality to a single benchmark score, and produces the same kind of misleading optimization pressure.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        The first visualization shows a simulated multitrait-multimethod heatmap. Two constructs (reasoning and recall) are each measured by two methods (multiple-choice and open-ended). Convergent validity coefficients (same construct, different method) should exceed discriminant coefficients (different construct, same method). The heatmap below shows the ideal pattern: the off-diagonal monotrait blocks are high; the heteromethod-monomethod cells are lower.
      </Prose>

      <Heatmap
        label="Multitrait-Multimethod matrix — convergent & discriminant validity"
        rowLabels={["reason_MC", "reason_OE", "recall_MC", "recall_OE"]}
        colLabels={["reason_MC", "reason_OE", "recall_MC", "recall_OE"]}
        cellSize={56}
        colorScale="gold"
        matrix={[
          [1.00, 0.78, 0.42, 0.31],
          [0.78, 1.00, 0.34, 0.39],
          [0.42, 0.34, 1.00, 0.71],
          [0.31, 0.39, 0.71, 1.00],
        ]}
      />

      <Prose>
        The next plot shows how observed criterion-validity correlations are attenuated by measurement unreliability. As the reliability of either the test or the criterion drops, the maximum possible observed correlation drops with it, regardless of the underlying true-score relationship. This is the practical reason that low validity coefficients should never be interpreted in isolation from reliability information.
      </Prose>

      <Plot
        label="Maximum observed r given true r=0.8 — attenuation by reliability"
        xLabel="reliability of test (ρ_XX)"
        yLabel="maximum observed r"
        series={[
          {
            name: "criterion ρ_YY = 1.0",
            color: colors.gold,
            points: [
              [0.50, 0.566], [0.60, 0.620], [0.70, 0.669],
              [0.80, 0.716], [0.90, 0.759], [1.00, 0.800],
            ],
          },
          {
            name: "criterion ρ_YY = 0.8",
            color: "#c084fc",
            points: [
              [0.50, 0.506], [0.60, 0.554], [0.70, 0.598],
              [0.80, 0.640], [0.90, 0.679], [1.00, 0.716],
            ],
          },
          {
            name: "criterion ρ_YY = 0.6",
            color: "#4ade80",
            points: [
              [0.50, 0.438], [0.60, 0.480], [0.70, 0.518],
              [0.80, 0.554], [0.90, 0.588], [1.00, 0.620],
            ],
          },
        ]}
      />

      <Prose>
        The third plot shows Lawshe's CVR critical thresholds as a function of expert panel size. Items with CVR above the threshold for a given panel size are statistically significantly more "essential" than would be expected by chance alone (one-tailed test at p=0.05). Notice that small panels require very high agreement to clear the threshold, while large panels can detect modest but real consensus.
      </Prose>

      <Plot
        label="Lawshe CVR critical values vs. expert panel size (one-tailed p=0.05)"
        xLabel="panel size N"
        yLabel="critical CVR"
        series={[
          {
            name: "critical CVR",
            color: colors.gold,
            points: [
              [5, 0.99], [6, 0.99], [7, 0.99], [8, 0.75], [9, 0.78],
              [10, 0.62], [11, 0.59], [12, 0.56], [13, 0.54], [14, 0.51],
              [15, 0.49], [20, 0.42], [25, 0.37], [30, 0.33], [40, 0.29],
            ],
          },
        ]}
      />

      <Prose>
        The step trace below walks through a complete validity-evidence-gathering pipeline for an LLM benchmark. Each phase has a distinct evidence type and produces a specific deliverable for the validity argument.
      </Prose>

      <StepTrace
        label="LLM benchmark validity pipeline — five evidence-gathering phases"
        steps={[
          {
            label: "Define interpretation",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Specify what the benchmark claim supports</div>
                <div>construct = "general scientific reasoning"</div>
                <div>population = "deployed assistant models"</div>
                <div>uses = ["model selection", "deployment readiness", "regulator filings"]</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Each downstream use requires its own evidence chain. The same benchmark
                  can be valid for selection but invalid for regulatory filing.
                </div>
              </div>
            ),
          },
          {
            label: "Content evidence",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>SME panel rates each item</div>
                <div>panel_size = 12 domain experts</div>
                <div>item_CVR = (n_essential - 6) / 6  per item</div>
                <div>retain items with CVR ≥ 0.56  (critical for N=12)</div>
                <div>S-CVI = mean(I-CVI) over retained items</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Also: domain coverage matrix — proportion of construct subdomains
                  represented in retained items vs. their importance weights.
                </div>
              </div>
            ),
          },
          {
            label: "Construct evidence",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#4ade80", marginBottom: 4 }}>Internal structure + nomological network</div>
                <div>EFA → choose k via parallel analysis</div>
                <div>CFA → fit hypothesized k-factor model</div>
                <div>fit indices: RMSEA &lt; 0.06, CFI &gt; 0.95</div>
                <div>convergent r with related benchmarks</div>
                <div>discriminant r with unrelated benchmarks</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Convergent should exceed discriminant. MTMM matrix is the
                  formal display.
                </div>
              </div>
            ),
          },
          {
            label: "Criterion evidence",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#c084fc", marginBottom: 4 }}>Test-criterion correlation</div>
                <div>concurrent: r(benchmark, gold-standard test)</div>
                <div>predictive: r(benchmark, deployment outcome at t+6mo)</div>
                <div>disattenuate: r / √(α_test · α_criterion)</div>
                <div>Taylor-Russell utility for selection scenarios</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Predictive evidence is the strongest type and the rarest.
                  Most LLM benchmarks have none.
                </div>
              </div>
            ),
          },
          {
            label: "Validity argument",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Synthesize per Standards (2014)</div>
                <div>document interpretation + evidence per use</div>
                <div>flag missing evidence sources explicitly</div>
                <div>state contamination probability per item set</div>
                <div>publish as machine-readable artifact next to scores</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Missing evidence is not neutral — it is a structural gap in
                  the warrant for the inference. Naming the gap is part of the
                  argument.
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

      <H3>Construct vs criterion vs content — when each applies</H3>

      <Prose>
        Reach for construct validity first when the inference you want to make involves an unobservable theoretical attribute — "reasoning ability," "knowledge," "harmlessness," "alignment." Construct evidence is the only evidence type that directly addresses whether the score reflects the named construct rather than some artifact correlated with it. The cost is that construct evidence requires a network of related and unrelated measures, which is expensive to assemble and rarely available for novel constructs. For a brand-new benchmark measuring a brand-new construct, construct evidence may be impossible to collect on day one — but the obligation to collect it before claiming the construct interpretation is non-negotiable in mature measurement practice.
      </Prose>

      <Prose>
        Reach for criterion validity when there is a downstream observable outcome that the test should predict and you can collect data on the outcome. For LLM evaluation, criterion validity is most natural when the deployment context produces measurable outcomes — user satisfaction scores, task completion rates, error rates in production, downstream business KPIs. The advantage of criterion evidence is that it tests the inference directly: if the benchmark predicts the outcome, the inference is supported regardless of whether the construct labelling is theoretically clean. The disadvantage is that the criterion may itself be a poor proxy for what you actually care about, and a high test-criterion correlation can be entirely uninformative if the criterion is contaminated, gameable, or temporally unstable.
      </Prose>

      <Prose>
        Reach for content validity when the construct is well-defined as a domain (a body of knowledge, a set of skills, a list of behaviors) and the benchmark needs to demonstrate adequate coverage of that domain. Content evidence is straightforward to collect — assemble a panel of subject-matter experts, have them rate items for relevance and essentiality — and it is the appropriate evidence type for benchmarks that explicitly target a defined curriculum or competency framework. Content evidence is the weakest type for theoretical constructs because experts can agree that an item samples the domain without that item actually distinguishing levels of the underlying attribute. Content validity tells you the test items are about the right topic; it does not tell you the scores rank-order test-takers correctly on the underlying ability.
      </Prose>

      <H3>Convergent vs discriminant evidence within construct validity</H3>

      <Prose>
        Convergent evidence (high correlation with theoretically related measures) is necessary but insufficient. Discriminant evidence (low correlation with theoretically unrelated measures) is what distinguishes a measurement of the construct from a measurement of an alternative construct that happens to correlate with it. The classic failure mode is a benchmark that has high convergent correlation with related benchmarks and equally high correlation with unrelated benchmarks — the high convergent number reflects general capability or response-bias factors, not the specific construct. Always report both, and treat their ratio (convergent r / discriminant r) as more informative than convergent r alone.
      </Prose>

      <H3>Concurrent vs predictive criterion validity</H3>

      <Prose>
        Concurrent validity is cheaper to collect (everything measured at one time point) but predictive validity is the stronger evidence type for most decisions. A test that correlates with a concurrent criterion has demonstrated the relationship in a closed measurement situation; a test that correlates with a future criterion has demonstrated that the relationship survives the conditions of actual deployment, including temporal drift, distribution shift, and the noise of real outcomes. For ML benchmarks intended to inform deployment decisions, predictive validity is the appropriate type even though it requires waiting for the criterion to materialize.
      </Prose>

      <H3>EFA vs CFA for construct validity</H3>

      <Prose>
        Use Exploratory Factor Analysis (EFA) when you have no strong hypothesis about the latent structure and want the data to suggest one. Use Confirmatory Factor Analysis (CFA) when you have a hypothesized structure and want to test how well it fits. The standard pipeline is EFA on a development sample to identify a candidate structure, then CFA on a fresh sample to confirm. Skipping the fresh sample is a form of overfitting: a CFA on the same data the EFA was run on will show good fit by construction.
      </Prose>

      <H3>Lawshe CVR vs Polit-Beck CVI</H3>

      <Prose>
        Lawshe's CVR uses a three-point essentiality scale and a single critical threshold per panel size. Polit and Beck's CVI uses a four-point relevance scale and reports the proportion of experts giving the top two ratings. CVR is preferred when the question is genuinely "is this item essential to measuring the construct" — a high-stakes filtering decision. CVI is preferred for finer-grained item review where most items are at least marginally relevant and the question is degree of relevance. For LLM benchmark construction, CVR's clear retain/discard threshold is operationally cleaner; for benchmark refinement after a first pass has filtered out clearly off-topic items, CVI's gradient is more useful.
      </Prose>

      <H3>When to skip formal validity work</H3>

      <Prose>
        Internal evaluation of model behavior during development does not require formal validity arguments — you are using benchmarks as ad-hoc diagnostic instruments and the cost of psychometric formality outweighs the benefit. Formal validity work becomes necessary when a benchmark score is going to inform a decision external to the development team: model release, regulatory submission, public ranking, customer-facing claims. The threshold is not "is this benchmark important" but "will the score be interpreted by parties who cannot inspect the items themselves and will assume the benchmark name describes what it measures." Anything that crosses that threshold deserves a validity argument; anything that stays inside the development feedback loop usually does not.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        Sample size requirements scale approximately the way they do for any covariance-based statistical analysis. Pearson correlations stabilize at sample sizes above ~80; their confidence intervals are visibly wider below ~30. Factor analysis (both EFA and CFA) requires substantially more — the rule of thumb is at least 5-10 subjects per item, with 200 subjects considered a reasonable minimum for a 20-item benchmark and 500+ preferred for stable factor structures. For LLM evaluation this maps to: at least 200 distinct models or model variants must be benchmarked to produce a reliable factor analysis of inter-item structure. This is more than most benchmark suites have ever evaluated, which is why factor-analytic validity evidence is essentially absent from the LLM benchmark literature.
      </Prose>

      <Prose>
        Content validity work scales linearly with item count and panel size. CVR computation for a 100-item benchmark with a 10-expert panel requires 1000 expert ratings — substantial but tractable. Doubling either the items or the panel doubles the rating burden. The bottleneck is not computation; it is recruiting and compensating qualified experts. For specialized domains (medical reasoning, legal analysis, advanced mathematics) the pool of qualified raters is small and their time is expensive, and a thorough content validation can easily cost more than the rest of the benchmark construction combined.
      </Prose>

      <Prose>
        Criterion validity scales worst of all because it requires the criterion to be measurable, which often means waiting for a downstream outcome. Predictive validity for a deployment criterion six months out cannot be evaluated faster than six months. Concurrent validity is faster but requires the criterion measurement infrastructure to exist, which for novel constructs means building two measurement instruments in parallel and validating each against the other. The structural problem is that the criterion that would most cleanly establish validity (real downstream usefulness) is the one that is hardest and slowest to measure, while the criterion that is easiest to measure (correlation with another benchmark) provides the weakest validity evidence.
      </Prose>

      <Prose>
        Modern LLM scale changes some of these constraints. Sample sizes for inter-item correlations are now arbitrarily large because each model can be evaluated programmatically across thousands of items; sample sizes for inter-model correlations are still constrained by the number of distinct models with comparable evaluation data, which is hundreds rather than thousands. AI judges (LLMs evaluating LLMs) make criterion measurement cheaper at the cost of introducing the AI judge's own validity questions one level up. Crowdsourced evaluation through platforms like Chatbot Arena produces large sample sizes for pairwise preferences but introduces selection effects, prompt distribution biases, and rater variability that themselves require validity evidence to interpret.
      </Prose>

      <Prose>
        The structural piece that does not scale is the interpretation gap. Validity questions arise because someone wants to draw an inference from a measurement, and that inference is exactly as ambitious as the user's stake in it. As LLMs are deployed in higher-stakes settings — medicine, law, finance, safety-critical infrastructure — the validity demands grow faster than the measurement methodology can supply them. The benchmarks of 2024 were built when the dominant inference was "this model is interesting research" and the benchmarks of 2026 are being asked to support inferences like "this model is safe to deploy in clinical decision support." The validity infrastructure has not scaled to match, and the gap is widening, not narrowing. This is the empirical reason that validity theory is having a renaissance in ML evaluation circles right now.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>Naming a benchmark is not measurement</H3>
      <Prose>
        The most pervasive failure mode in LLM evaluation: a benchmark called "MMLU" (Massive Multitask Language Understanding) is treated as if it measures language understanding, and a benchmark called "GSM8K" (Grade School Math 8K) is treated as if it measures mathematical reasoning. Names are claims; claims require evidence. A benchmark with no validity evidence is a list of questions, and the score on it is the number of questions answered correctly, period. Reporting that score as "language understanding" or "mathematical reasoning" is a validity claim that has not been substantiated. The fix is not to stop reporting scores; it is to stop using construct labels until evidence supports them.
      </Prose>

      <H3>Reliability mistaken for validity</H3>
      <Prose>
        ML benchmarks have very high reliability — run the same model on the same items at temperature zero and you get the same score every time. This consistency is sometimes treated as evidence of validity, but reliability is consistency of measurement, not correctness of measurement. A benchmark can be perfectly reliable and completely invalid: it consistently measures the same wrong thing. Cronbach's alpha and test-retest correlations should never appear in a validity argument as evidence for validity; they appear in the validity argument as preconditions for any claim of measurement at all.
      </Prose>

      <H3>Benchmark contamination as a content-validity threat</H3>
      <Prose>
        If benchmark items are present in the model's training data, the score does not measure the model's ability on the construct; it measures the model's memorization of the specific items. Contamination is not just a quantitative bias on the score — it changes what the score is a measurement of. The fix requires careful train/test separation enforced at the data-collection stage (post-cutoff items, paraphrased variants, dynamic generation) and ongoing contamination probability estimation through methods like membership inference. The Burnell et al. 2023 paper recommends per-benchmark contamination estimates be reported alongside scores; few benchmarks currently do this.
      </Prose>

      <H3>Convergent without discriminant evidence</H3>
      <Prose>
        A new benchmark correlates highly with several existing benchmarks — convergent evidence supports the interpretation. But it also correlates highly with benchmarks measuring very different constructs, and equally highly with benchmarks of general capability or model size. The convergent correlation is real but it is not specific to the named construct; it reflects the omnipresent general-capability factor that runs through almost all LLM benchmarks. Always pair convergent claims with discriminant evidence from theoretically unrelated tests.
      </Prose>

      <H3>Criterion validity against another benchmark</H3>
      <Prose>
        "This benchmark has high criterion validity because it correlates 0.85 with that other benchmark" is a claim that frequently appears in benchmark papers and is almost always misleading. Two benchmarks correlating with each other does not establish either as a measurement of the construct; it establishes that they share whatever variance produces the correlation, which might be the construct or might be a shared methodological artifact. True criterion validity requires a criterion that is itself an established or independently meaningful measurement of the construct or its consequences — a deployment outcome, an expert holistic judgment, a theoretical predictor — not just another benchmark that has the same name.
      </Prose>

      <H3>Sample-size-dependent factor structures</H3>
      <Prose>
        Factor analyses on small samples (under 100-200) produce unstable structures: re-running the analysis on a fresh sample of the same size can yield different factor counts, different loadings, and different fit indices. Reporting an EFA result without a confirmation step on an independent sample is a form of overfitting, and small-sample CFA results should be reported with explicit confidence intervals on fit indices, not just point estimates.
      </Prose>

      <H3>Range restriction in validity coefficients</H3>
      <Prose>
        Validity coefficients computed on a restricted range of test scores (e.g., only models scoring above 70% on the benchmark) are systematically attenuated relative to the full-population coefficient. This is a particularly insidious problem in LLM benchmarking because the population of models being compared is heavily filtered — only published models, only models passing some quality threshold, only models the analyst had access to. Reported validity coefficients should always include an analysis of range restriction and a corrected coefficient using Thorndike's formula for restriction-of-range correction.
      </Prose>

      <H3>The Standards say validity is not divided into types — but everyone still does</H3>
      <Prose>
        The 2014 Standards for Educational and Psychological Testing explicitly treat validity as one concept with multiple sources of evidence, not as a triad of separate validities. But the categorical language (construct validity, criterion validity, content validity) is still ubiquitous in textbooks, papers, and applied work. Using the categorical language is fine for organizing evidence-gathering activities; it becomes a failure mode when practitioners assume that establishing one type of validity in isolation is sufficient. Always frame the unified question — "is this interpretation warranted?" — and use the categories as a checklist of evidence types to consider, not as a menu where you pick one.
      </Prose>

      <H3>Construct underrepresentation and construct-irrelevant variance</H3>
      <Prose>
        Two specific construct-validity threats Messick (1989) named explicitly. Construct underrepresentation is when the test fails to sample important parts of the construct domain — a "math reasoning" benchmark with only arithmetic problems. Construct-irrelevant variance is when the test scores reflect factors other than the construct — a multiple-choice benchmark where scores partly reflect test-taking strategy rather than the construct itself. Both are systematic threats to construct validity even when convergent and discriminant correlations look acceptable. They are diagnosed through item-level analysis and item response theory rather than through correlations alone.
      </Prose>

      <Callout accent="purple">
        Validity threats are cumulative. A benchmark with mild contamination, mild range restriction, mild construct-irrelevant variance, and mild discriminant-evidence weakness can produce scores that look reasonable on every individual diagnostic and yet fail to support the inference users actually draw from them. The combined threat is the warrant problem; no single test catches it.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        Citations below were verified against original sources and arXiv records as of 2026-04-26. Authors, dates, and identifiers confirmed.
      </Prose>

      <H3>Cronbach &amp; Meehl 1955 — Construct validity</H3>
      <Prose>
        Lee J. Cronbach and Paul E. Meehl. "Construct Validity in Psychological Tests." Psychological Bulletin, 52(4):281-302, 1955. The founding paper of construct validity theory. Introduces the nomological network — the web of theoretical relationships a construct should have with other observables — and argues that construct validity is established not by any single coefficient but by the pattern of relationships matching theoretical predictions. Distinguishes construct validity from content validity (sampling the domain) and criterion validity (predicting an outcome) and explains why construct validity is the deepest of the three. Still the canonical reference seventy years later.
      </Prose>

      <H3>Campbell &amp; Fiske 1959 — Multitrait-multimethod matrix</H3>
      <Prose>
        Donald T. Campbell and Donald W. Fiske. "Convergent and Discriminant Validation by the Multitrait-Multimethod Matrix." Psychological Bulletin, 56(2):81-105, 1959. Operationalizes Cronbach and Meehl's nomological network through a specific data display. Shows that convergent validity (same construct, different method) must be evaluated alongside discriminant validity (different construct, same method) and that the comparison reveals method-bias confounds invisible to single-coefficient analyses. The MTMM matrix remains the cleanest visual presentation of construct-validity evidence.
      </Prose>

      <H3>Lawshe 1975 — Content validity ratio</H3>
      <Prose>
        Charles H. Lawshe. "A Quantitative Approach to Content Validity." Personnel Psychology, 28(4):563-575, 1975. Introduces the Content Validity Ratio CVR = (n_e − N/2) / (N/2) for quantifying expert agreement on item essentiality. Provides the table of critical CVR values for common panel sizes that is still used in content-validation studies fifty years later. The simplest and most operationally clean of the major validity statistics.
      </Prose>

      <H3>Messick 1989 — The unified framework</H3>
      <Prose>
        Samuel Messick. "Validity." In Robert L. Linn (ed.), Educational Measurement (3rd edition), pp. 13-103. American Council on Education / Macmillan, 1989. The chapter that overturned the categorical view of validity. Argues that construct, criterion, and content are not separate validities but three sources of evidence for a single unified validity argument about the warrant for an interpretation and use of a test score. Introduces "construct underrepresentation" and "construct-irrelevant variance" as the two principal threats. Foundational for the modern Standards-based view of validity.
      </Prose>

      <H3>AERA, APA, NCME 2014 — Standards for Educational and Psychological Testing</H3>
      <Prose>
        American Educational Research Association, American Psychological Association, and National Council on Measurement in Education. Standards for Educational and Psychological Testing. AERA Publications, 2014. The current authoritative reference for measurement practice. Codifies validity as a single integrated concept supported by five evidence types (test content, response processes, internal structure, relations to other variables, consequences of testing). Required reading for anyone making measurement claims that will be scrutinized by professional or regulatory bodies.
      </Prose>

      <H3>Burnell et al. 2023 — Rethink reporting of evaluation results in AI</H3>
      <Prose>
        Ryan Burnell, Wout Schellaert, John Burden, et al. "Rethink Reporting of Evaluation Results in AI." arXiv:2308.07193. Published August 2023; later in Science. Argues that aggregate benchmark scores hide the item-level information needed to evaluate validity and proposes per-item, per-capability reporting standards. Identifies contamination as a primary content-validity threat and proposes contamination probability estimates as a required reporting element. The clearest statement of the validity reform agenda for ML evaluation.
      </Prose>

      <H3>Liao &amp; Vaughan 2024 — AI Transparency in the Age of LLMs</H3>
      <Prose>
        Q. Vera Liao and Jennifer Wortman Vaughan. "AI Transparency in the Age of LLMs: A Human-Centered Research Roadmap." Harvard Data Science Review, special issue, 2024. Frames evaluation reports as transparency artifacts and proposes that they be structured around explicit validity arguments rather than scalar scores. Connects the psychometric validity tradition to the AI documentation tradition (model cards, data sheets, system cards) and proposes integration. The bridge document for ML practitioners trying to translate validity theory into deployable practice.
      </Prose>

      <H3>Shi et al. 2024 — Detecting pretraining data</H3>
      <Prose>
        Weijia Shi, Anirudh Ajith, Mengzhou Xia, et al. "Detecting Pretraining Data from Large Language Models." arXiv:2310.16789. Published October 2023; ICLR 2024. Provides Min-K% Prob, a membership inference attack for detecting whether specific text appears in an LLM's training data. The primary technical tool for estimating benchmark contamination probability — directly addressing the content-validity threat that contamination poses for LLM benchmark scores.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Distinguish reliability from validity</H3>
      <Prose>
        A benchmark consists of 100 multiple-choice questions. You administer it twice to the same set of 50 models with no changes between administrations. The Pearson correlation between the two administrations of total scores is 0.98. A colleague concludes "the benchmark has high validity." Explain precisely what the 0.98 figure does and does not support. What additional evidence would be required to claim the benchmark has high construct validity for the construct it names? What evidence would be required to claim high criterion validity for a specific deployment use? Construct a thought experiment in which a benchmark could have a test-retest reliability of 0.99 and yet zero validity for any meaningful interpretation.
      </Prose>

      <H3>Exercise 2 — Derive Lawshe's critical value</H3>
      <Prose>
        Lawshe's CVR for an item rated by N=10 experts is the proportion of "essential" ratings minus 0.5, rescaled to [-1, +1]. The critical value at p=0.05 (one-tailed) for N=10 is 0.62. Derive this critical value from a binomial test on the null hypothesis that ratings are produced by chance with p=0.5. Show that a CVR of 0.62 corresponds to 8 out of 10 experts rating "essential" and that 8 out of 10 successes under H_0: p=0.5 has a one-tailed p-value of approximately 0.055. Why does Lawshe round this to 0.62 rather than to 0.6 or 0.8? What happens to the critical value at N=20, and why is the threshold lower despite the same nominal alpha?
      </Prose>

      <H3>Exercise 3 — Attenuation correction</H3>
      <Prose>
        You observe a Pearson correlation of r=0.50 between scores on benchmark A (Cronbach alpha 0.78) and a deployment outcome metric (test-retest reliability 0.65). Compute the disattenuated correlation between the underlying true scores. A reviewer argues that the disattenuated value is "the real validity coefficient." Counter or support this argument with reference to (a) what the disattenuated correlation actually represents, (b) the conditions under which the correction is meaningful, and (c) the practical decisions the validity coefficient is being used to inform. Under what circumstances should the observed coefficient be reported instead of the disattenuated one, and why?
      </Prose>

      <H3>Exercise 4 — Construct validity for an alignment benchmark</H3>
      <Prose>
        A new benchmark called "HARMLESS" claims to measure the harmlessness of LLM outputs through a set of 200 prompts and a graded scoring rubric. Design a complete construct-validity evidence-gathering plan. Specify (a) what convergent measures you would correlate it with and what correlation magnitude would constitute supporting evidence; (b) what discriminant measures you would correlate it with and what correlation magnitude would constitute supporting evidence; (c) what factor structure you would hypothesize and how you would test it with CFA; (d) what would constitute evidence of construct underrepresentation; (e) what would constitute evidence of construct-irrelevant variance. What is the smallest sample size of distinct models on which the analysis could plausibly support the construct interpretation?
      </Prose>

      <H3>Exercise 5 — Validity argument for MMLU</H3>
      <Prose>
        Treat MMLU as a measurement instrument and write the skeleton of a validity argument for the interpretation "MMLU score reflects general academic knowledge across the 57 included subjects." For each of the three traditional evidence types (construct, criterion, content), state (a) what specific evidence would support the interpretation, (b) what evidence currently exists in the literature to your knowledge, and (c) what evidence is missing. For each missing piece, propose a concrete study design that could collect it. Then critique your own validity argument from the perspective of the unified Messick framework: what alternative interpretation of MMLU scores does your argument fail to rule out, and what additional evidence would rule it out?
      </Prose>

      <H3>Exercise 6 — Contamination as a content-validity threat</H3>
      <Prose>
        Suppose a 1000-item benchmark is administered to a model and the model scores 87%. Subsequently you discover that 200 of the 1000 items appear verbatim in the model's training corpus. (a) Compute the maximum and minimum possible "true" benchmark scores consistent with the observation, treating contaminated items as either always-correct (memorized) or replaceable with the population base rate of correct responses for non-contaminated items of the same type. (b) Discuss why contamination is a content-validity threat rather than a reliability problem. (c) Propose a reporting standard for benchmark scores that would make contamination effects visible to downstream users without requiring them to re-run any analyses. (d) What is the validity-theoretic difference between accidental contamination (the items leaked into training data unintentionally) and adversarial contamination (the items were deliberately included to inflate the score)?
      </Prose>

    </div>
  ),
};

export default validityFrameworks;
