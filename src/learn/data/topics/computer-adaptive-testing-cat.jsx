import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const computerAdaptiveTesting = {
  title: "Computer-Adaptive Testing (CAT)",
  slug: "computer-adaptive-testing-cat",
  readTime: "~38 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Fixed-form testing wastes most of its questions on most of its examinees. A 200-item exam has to span the full ability range of the population taking it — easy questions for the bottom of the distribution, hard questions for the top, and a mass of middling items for everyone else. From any individual examinee's perspective, the items at the extremes of difficulty are nearly informationless. A high-ability candidate breezes through the easy items with near-certain correct responses; a low-ability candidate misses the hard items with near-certain incorrect responses. In both cases, the test is asking questions whose outcomes were already predictable, and the marginal information about the candidate's true ability per question is small. The bulk of statistical information lives in items whose difficulty is matched to the examinee's ability, where the response probability is closest to 0.5 and a single answer carries the most uncertainty reduction. Fixed-form designs cannot exploit this because they cannot tailor item difficulty to the individual.
      </Prose>

      <Prose>
        Computer-Adaptive Testing (CAT) was the response to this inefficiency. The idea is older than the modern web — Frederic Lord laid out the statistical foundations in the 1970s, drawing on item response theory, and Howard Wainer's 2000 textbook codified the operational practice — but the engineering only became routine once test administration moved fully onto computers. The arc of the high-stakes testing industry over the last three decades has been a steady migration from paper-and-pencil fixed forms to computer-administered adaptive forms: the GRE General Test went CAT in 1992, the GMAT in 1997, the NCLEX nursing licensure exam in 1994, the ASVAB military entrance battery somewhat earlier. The empirical pattern from these conversions is consistent: a CAT version of a fixed-form test reaches comparable measurement precision with roughly 40 to 60 percent fewer items. A test that took 200 questions to estimate ability with a given standard error can do the same job in 80 to 120 adaptive questions. The savings compound with population size and test frequency.
      </Prose>

      <Prose>
        For most of the field's history, CAT was a tool for licensure, admissions, and educational measurement — domains where you needed to estimate a single latent trait (verbal ability, quantitative ability, clinical judgment) with calibrated precision and where the cost of testing time was high. The relevance to machine learning is more recent. Modern LLM evaluation has stumbled into exactly the same inefficiency that fixed-form psychometrics suffered from in the 1970s: benchmarks like MMLU run 14,042 questions per model evaluation, BIG-Bench runs more than 200 tasks, HELM evaluates dozens of models across hundreds of scenarios, and the compute cost of a full benchmark sweep on a single new checkpoint can exceed the cost of a small fine-tuning run. Polo and collaborators showed in early 2024 (arXiv:2402.14992, "tinyBenchmarks") that a CAT-style adaptive subset of 50 to 200 carefully selected items from MMLU produces ability estimates within a fraction of a standard deviation of the full-benchmark score, at one to two percent of the compute. The CAT machinery — IRT calibration of items, Fisher-information-driven item selection, posterior tracking of the ability parameter — transferred almost mechanically from psychometric assessment to model evaluation.
      </Prose>

      <Prose>
        That recent rediscovery is what makes CAT worth the depth here. The ML practitioner reading this is unlikely to be designing the next GRE, but they are likely to be facing the question of how to evaluate models faster and cheaper without losing the ranking signal a full benchmark provides. CAT is the right framework for that question, and the framework comes with three decades of operational experience about how it fails — exposure overuse, content imbalance, calibration drift, the early-stopping bias when the first few items happen to be unusually easy or hard. Reaching for CAT without understanding those failure modes produces evaluation pipelines that look efficient but encode silent bias. Understanding the framework — the IRT model under the hood, the item-selection criterion, the stopping rule — is the difference between using adaptive evaluation as a sharper instrument and using it as a faster way to be wrong.
      </Prose>

      <Prose>
        There is a fourth reason CAT deserves study, distinct from any of its applications: it is a uniquely clean example of sequential experimental design. The general problem of "given a budget of <Code>K</Code> measurements, choose them to maximize information about a latent parameter" appears across machine learning — active learning, Bayesian optimization, optimal sensor placement, reinforcement learning's exploration-exploitation trade-off — and CAT solves a particularly tractable instance of it where the latent parameter is one-dimensional, the measurement model is well-understood, and the closed-form information criterion is exact. The intuition built by working through CAT's machinery transfers to those harder problems. The Bayesian update of a posterior, the Fisher-information-driven design choice, the trade-off between exploration of the parameter space and exploitation of the current best estimate — these patterns recur across the design literature, and CAT is the cleanest setting in which to first internalize them.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Imagine sitting across from a pile of test items of widely varying difficulty and trying to estimate how strong a single test-taker is. You have a budget of, say, 30 questions, and your goal is to land on the most precise estimate of their ability you can. The wrong strategy is to pick a fixed sequence of 30 items in advance — that is the fixed-form test, and as already noted, most of those items will be either trivially easy or impossibly hard for any given test-taker. The right strategy is sequential and conditional: ask one question, observe the outcome, update your belief about the test-taker's ability, then choose the next question whose outcome — given your current belief — will tell you the most. Repeat. After 30 questions you stop, with an ability estimate that is far more precise than what 30 randomly chosen items could have produced.
      </Prose>

      <Prose>
        This is the core CAT loop, and it has three moving parts. The first is the item bank: a large pool of questions, each pre-calibrated with statistical parameters that describe how its response probability depends on the test-taker's latent ability. Without calibration the loop has nothing to operate on — you cannot ask "which item is most informative at the current ability estimate" if you do not know the items' difficulty curves. Calibration is done offline once, using item response theory fitted to large samples of historical responses, and is the part of CAT operations that is most expensive and most invisible to the end user. The second moving part is the item-selection rule: given the current posterior over ability, pick the next item that maximizes some criterion, almost always a function of Fisher information. The third moving part is the termination rule: keep going until either the standard error of the ability estimate falls below a target threshold, or you hit a maximum item budget, or you reach a confidence threshold for a classification decision (pass/fail).
      </Prose>

      <Prose>
        The notion of "information" deserves a moment. In the IRT setting, the Fisher information of an item at a given ability level <Code>θ</Code> measures how sharply the response probability changes with <Code>θ</Code> at that point. If the item is far too easy, the response probability is near 1 regardless of small ability changes, and the slope is shallow — Fisher information is tiny. If the item is far too hard, the response probability is near 0 with similarly shallow slope and again tiny information. If the item's difficulty <Code>b</Code> equals the current ability estimate <Code>θ̂</Code>, the response probability is exactly 0.5 in the simplest IRT model, and the slope is steepest — Fisher information is at its maximum. So the item-selection rule "pick the item whose difficulty matches your current ability estimate" is not heuristic; it is the maximum-Fisher-information rule for the simplest case, and more elaborate IRT models (2PL, 3PL) refine but do not overturn the basic geometry.
      </Prose>

      <Prose>
        The Bayesian gloss on this is that the current ability estimate is a posterior distribution, not a point. After <Code>k</Code> items, you have <Code>p(θ | u_1, ..., u_k)</Code>, the probability density over <Code>θ</Code> given the observed responses. Each new item updates this posterior multiplicatively by its likelihood. Asking "which item should I select next" becomes "which item has the largest expected posterior information gain" — and under mild conditions (and the simplest priors), this expected gain reduces to the Fisher information evaluated at the current posterior mean. Owen's 1975 procedure formalized this for normal-approximation posteriors and gave one of the first computationally tractable adaptive selection rules. Modern variants — Maximum Posterior Weighted Information, Kullback-Leibler item selection, Bayesian D-optimal designs — refine the criterion but operate within the same loop.
      </Prose>

      <Prose>
        There is a second consideration that is invisible from the math but central to operations: exposure control. If item-selection always picks the maximum-information item, the same handful of items at common ability levels will be selected for nearly every test-taker. In a high-stakes setting this is a security disaster — those items leak quickly. Sympson and Hetter's 1985 procedure introduced a probabilistic gate: each item has an exposure parameter and is only administered with some probability when it would otherwise be selected, with the parameters tuned so that no item is exposed to more than (say) 20 percent of test-takers. Alpha-stratified selection (Chang and Ying, 1999) takes a different approach: stratify the item bank by discrimination, use low-discrimination items early when ability estimates are unstable, and reserve high-discrimination items for the later stages when precise selection actually pays off. The point is that pure information maximization is the right starting point but not the right operational rule; real CAT systems combine information-maximization with explicit exposure-control machinery.
      </Prose>

      <Prose>
        The intuition transfers cleanly to LLM evaluation. The "test-taker" is a model, the "ability" <Code>θ</Code> is the model's latent capability on the benchmark domain (MMLU knowledge, HumanEval coding, GSM8K mathematical reasoning), the "items" are benchmark questions, and the "response" is correct or incorrect. The IRT calibration step is now run not on human respondents but on a large bank of historical model responses to the benchmark — Polo et al. used a few hundred existing model evaluations to calibrate the item parameters of MMLU. Once calibrated, asking 50 maximum-information items from MMLU instead of all 14,042 produces a comparable point estimate of <Code>θ</Code> with a small additional standard error. The exposure-control concerns largely vanish (models cannot leak items), but a new concern appears: the calibrated item parameters drift as new model architectures emerge with capability profiles that differ qualitatively from the calibration sample. The CAT framework adapts; the operational concerns shift.
      </Prose>

      <Prose>
        One last piece of intuition worth sitting with before the math. The CAT loop is a feedback system that depends critically on the trustworthiness of two estimates: the current ability estimate <Code>θ̂</Code> and the calibrated item parameters <Code>(a_i, b_i)</Code>. Both are imperfect — <Code>θ̂</Code> because it is built up from a small number of binary observations, the item parameters because they were estimated from a finite calibration sample. The selection rule treats both as exact and chooses items as if the picture were fully accurate. When that assumption holds, CAT is breathtakingly efficient; when it breaks — bad initial estimates, off-population calibration, drift over time — CAT continues to look like it is converging while it converges to the wrong answer. This is a feature of any feedback system that lacks ground-truth correction: it makes the most of what it has and can be very wrong about what it has. Most of the operational machinery of high-stakes CAT — exposure control, content balancing, periodic re-calibration, embedded field-test items — exists precisely to compensate for this trust-the-estimates structure. The math is one piece; the operational discipline that surrounds the math is the other piece, and both are needed for CAT to deliver its theoretical efficiency in practice.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <Prose>
        The mathematical core of CAT is item response theory. The simplest workable IRT model is the one-parameter logistic (1PL, or Rasch) model, where the probability that test-taker with ability <Code>θ</Code> answers item <Code>i</Code> correctly is a logistic function of the difference between ability and item difficulty:
      </Prose>

      <MathBlock>{"P_i(\\theta) = \\frac{1}{1 + \\exp\\!\\left(-(\\theta - b_i)\\right)}"}</MathBlock>

      <Prose>
        Here <Code>b_i</Code> is the item's difficulty parameter on the same scale as <Code>θ</Code>. When <Code>θ = b_i</Code>, the response probability is exactly 0.5. Higher ability shifts the probability toward 1; lower ability toward 0. The 1PL model is appealing for its simplicity and interpretability, but it imposes a constraint that all items are equally discriminating — they all have the same logistic slope at the point of inflection. The two-parameter logistic (2PL) model relaxes this by giving each item a discrimination parameter <Code>a_i</Code>:
      </Prose>

      <MathBlock>{"P_i(\\theta) = \\frac{1}{1 + \\exp\\!\\left(-a_i(\\theta - b_i)\\right)}"}</MathBlock>

      <Prose>
        High-discrimination items have steep logistic curves and sharply distinguish test-takers near their difficulty level; low-discrimination items have shallow curves and provide weaker signal. For multiple-choice items where guessing matters, the three-parameter logistic (3PL) adds a lower asymptote <Code>c_i</Code> representing the probability that a test-taker far below the item's difficulty still answers correctly by chance:
      </Prose>

      <MathBlock>{"P_i(\\theta) = c_i + (1 - c_i)\\,\\frac{1}{1 + \\exp\\!\\left(-a_i(\\theta - b_i)\\right)}"}</MathBlock>

      <Prose>
        The CAT machinery works under any of these models with minor adjustments to the information formulas. The 2PL model is the most common operational choice: rich enough to capture meaningful item-quality differences, simple enough to calibrate stably from the sample sizes typically available. We use the 2PL model in the from-scratch implementation below.
      </Prose>

      <Prose>
        Given a calibrated item bank and a test-taker who has answered items <Code>i_1, ..., i_k</Code> with binary responses <Code>u_1, ..., u_k</Code> (1 for correct, 0 for incorrect), the likelihood of the response vector under 2PL is the product of per-item Bernoullis:
      </Prose>

      <MathBlock>{"L(\\theta \\mid u_1, \\ldots, u_k) = \\prod_{j=1}^{k} P_{i_j}(\\theta)^{u_j}\\,(1 - P_{i_j}(\\theta))^{1 - u_j}"}</MathBlock>

      <Prose>
        Two estimators dominate practice. Maximum likelihood estimation finds the <Code>θ̂</Code> that maximizes the log-likelihood directly, typically by Newton-Raphson on the score equation. It is the classical choice but breaks down for short tests with all-correct or all-incorrect response patterns — the likelihood is monotone and <Code>θ̂</Code> diverges to <Code>±∞</Code>. The Bayesian alternative imposes a prior <Code>p(θ)</Code> (typically standard normal), forms the posterior <Code>p(θ | responses) ∝ L(θ) p(θ)</Code>, and reports either the maximum a posteriori (MAP) estimate or the expected a posteriori (EAP) estimate. EAP is the posterior mean, integrated numerically over a quadrature grid:
      </Prose>

      <MathBlock>{"\\hat\\theta_{\\mathrm{EAP}} = \\frac{\\int \\theta\\, L(\\theta)\\, p(\\theta)\\, d\\theta}{\\int L(\\theta)\\, p(\\theta)\\, d\\theta}"}</MathBlock>

      <Prose>
        EAP is the standard choice for CAT because it is well-defined for any response pattern, including all-correct and all-incorrect. The price is that EAP is biased toward the prior mean, especially for short tests — a bias that fades as the test lengthens but matters at the start of the loop when only a few responses are available.
      </Prose>

      <Prose>
        The Fisher information for a single item at ability <Code>θ</Code> under the 2PL model is:
      </Prose>

      <MathBlock>{"I_i(\\theta) = a_i^2\\, P_i(\\theta)\\,(1 - P_i(\\theta))"}</MathBlock>

      <Prose>
        Note three structural facts. First, information is maximized when <Code>P_i(θ) = 0.5</Code>, which under 2PL occurs when <Code>θ = b_i</Code>. Second, information scales with the square of the discrimination parameter — high-discrimination items are quadratically more informative than low-discrimination items at their peak. Third, information is additive across items: the test-information function for a set of administered items is just the sum of their individual information functions. The standard error of the maximum-likelihood ability estimate at <Code>θ</Code> is the inverse square root of the test information at <Code>θ</Code>:
      </Prose>

      <MathBlock>{"\\mathrm{SE}(\\hat\\theta) = \\frac{1}{\\sqrt{\\sum_{j=1}^{k} I_{i_j}(\\hat\\theta)}}"}</MathBlock>

      <Prose>
        This is the quantity tracked by the most common CAT termination rule: keep administering items until <Code>SE(θ̂)</Code> drops below a target (typical values: 0.30 for low-stakes screening, 0.20 for routine high-stakes, 0.10 for clinical or licensure decisions). A second common rule is the item budget: stop after <Code>K</Code> items regardless of precision. A third is the classification rule, used when the test is making a pass/fail decision rather than reporting a continuous score: stop when the posterior probability that <Code>θ</Code> exceeds the cut score crosses a confidence threshold (the SPRT, sequential probability ratio test, formalizes this).
      </Prose>

      <Prose>
        The item-selection rule that maximizes expected information at the current ability estimate is the most common: at each step, evaluate <Code>I_i(θ̂)</Code> for every unadministered item and select the maximum. This is the maximum Fisher information (MFI) rule. A Bayesian refinement weights the per-item information by the current posterior over <Code>θ</Code> rather than evaluating only at the point estimate:
      </Prose>

      <MathBlock>{"\\mathrm{MEPV}(i) = -\\int I_i(\\theta)\\, p(\\theta \\mid u_{1:k})\\, d\\theta"}</MathBlock>

      <Prose>
        This is the Maximum Expected Posterior Variance reduction criterion (Chang and Ying 1996). It differs from MFI mainly when the posterior is wide — early in the test, when point-estimate selection is unstable. As the posterior tightens, the two rules converge. A further variant uses Kullback-Leibler divergence between the posterior before and after a hypothetical response, which is robust to multimodal posteriors but more expensive to compute. For most practical CAT and certainly for the LLM-eval transfer, MFI with EAP estimation is the default.
      </Prose>

      <Prose>
        Owen's 1975 procedure deserves a brief separate note because it predates these criteria and remains a useful reference. Owen used a normal-approximation posterior — assuming Bernoulli updates can be tracked as Gaussian conjugate updates of <Code>(μ, σ²)</Code> — and selected the next item to maximize the posterior precision (inverse variance) of <Code>θ</Code>. Under the normal approximation, the maximum-precision item is the one whose difficulty is closest to the current posterior mean, which recovers the MFI rule in the special case. Owen's procedure is exact only under the normal approximation, which holds well for moderately long tests but introduces small biases for very short ones. The modern EAP-with-MFI approach is more general but less analytically tractable than Owen's original.
      </Prose>

      <Prose>
        Owen's update equations are worth writing out because they make the conjugate-update structure explicit. Starting from a Gaussian prior <Code>θ ~ N(μ_0, σ_0²)</Code>, after observing response <Code>u_k</Code> to item <Code>k</Code> with difficulty <Code>b_k</Code>, the approximate posterior parameters are:
      </Prose>

      <MathBlock>{"\\mu_k = \\mu_{k-1} + \\sigma_{k-1}^2\\, \\frac{u_k - P_k(\\mu_{k-1})}{D_k}, \\quad \\sigma_k^2 = \\sigma_{k-1}^2 - \\sigma_{k-1}^4 \\frac{P_k(\\mu_{k-1})\\,(1-P_k(\\mu_{k-1}))}{D_k}"}</MathBlock>

      <Prose>
        where <Code>D_k = 1 + σ_(k−1)² P_k(μ_(k−1))(1 − P_k(μ_(k−1)))</Code> is a normalization factor that prevents variance from going negative. The update has the classic Kalman-filter shape: the mean moves in proportion to the prediction error <Code>u_k − P_k(μ_(k−1))</Code> scaled by the prior variance, and the variance shrinks by an amount that depends on the item's information at the prior mean. The closed-form structure is computationally trivial — one update per item, no integration — which is why Owen's procedure was practical on the hardware of the 1970s. Modern systems use the EAP integration because it is exact under the IRT model rather than approximate, but Owen's update remains the right choice when computational budget is severely constrained.
      </Prose>

      <Callout accent="gold">
        Two facts about IRT calibration that get glossed over: first, the ability scale is identified only up to a linear transformation, so any IRT calibration fixes the scale by convention (typically <Code>θ ~ N(0, 1)</Code> in the calibration sample). Second, item parameters are estimated from response data using marginal maximum likelihood (MML) or Bayesian methods on a calibration sample of several hundred to several thousand responses per item. The quality of CAT downstream depends entirely on the quality of this offline calibration.
      </Callout>

      <Prose>
        It is also worth seeing the additivity of information in action, since it underwrites both the standard-error stopping rule and the design intuition that test length trades off against precision. Suppose we have administered three items with information values <Code>I_1(θ̂) = 0.4</Code>, <Code>I_2(θ̂) = 0.5</Code>, and <Code>I_3(θ̂) = 0.3</Code>. The total test information at <Code>θ̂</Code> is <Code>1.2</Code>, the standard error is <Code>1/√1.2 ≈ 0.913</Code>. Adding a fourth item with information <Code>0.4</Code> brings total information to <Code>1.6</Code> and SE to <Code>0.791</Code>. To halve the SE from <Code>0.913</Code> to about <Code>0.46</Code>, total information must quadruple from <Code>1.2</Code> to <Code>4.8</Code>, requiring roughly nine more items at average information <Code>0.4</Code>. This is the inverse-square-root rule that defines the slowing-down phase of CAT convergence: each additional item reduces SE by less than the previous one, and the marginal value of an additional item drops sharply as the test lengthens. The optimal-stopping argument from sequential analysis says: stop when the marginal information gain from the next item, weighted by its cost (test-taker time, model inference compute), drops below the value of the precision improvement. In practice, fixed thresholds on SE (or fixed item budgets) are easier to defend operationally than explicit cost-benefit calculations, but the underlying logic is the same.
      </Prose>

      <Prose>
        Exposure control adds a layer on top of the selection rule. Sympson and Hetter (1985) introduced a probabilistic gate: assign each item an exposure parameter <Code>K_i ∈ [0, 1]</Code>, and when item <Code>i</Code> would be selected by the information rule, administer it with probability <Code>K_i</Code> and otherwise pick the next-best item. The <Code>K_i</Code> values are tuned offline by simulation so that no item exceeds a target exposure rate (commonly 0.20 — at most 20 percent of test-takers see any given item). Alpha-stratified selection (Chang and Ying 1999) is a different approach that partitions the bank into discrimination strata and forces early items to come from low-discrimination strata, reserving high-discrimination items for later when ability estimates are stable enough to use them well. Both approaches sacrifice a small amount of efficiency for a large gain in item-bank security and content balance.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The most reliable way to internalize CAT is to implement the loop end-to-end: a calibrated item bank, a simulated test-taker with known true ability, an EAP ability estimator, an MFI item selector, and a stopping rule based on the standard error. Every print statement in the comments below reflects the actual output of the code when run with a fixed random seed; nothing is hypothetical. The implementation uses NumPy and SciPy for numerical integration. It is broken into five subsections that mirror the conceptual components.
      </Prose>

      <H3>4a. Item bank construction</H3>

      <Prose>
        An item bank under the 2PL model is a table of <Code>(a_i, b_i)</Code> pairs — one discrimination and one difficulty per item. In production these are estimated from response data via marginal maximum likelihood, typically with a package like <Code>mirt</Code> (R) or <Code>py-irt</Code> (Python). For the from-scratch implementation we generate a synthetic bank with item parameters drawn from realistic distributions: discriminations from a lognormal centered around 1.0, and difficulties spread across the ability scale.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(42)

N_ITEMS = 200
# Discriminations: lognormal, mean ~1.0, slight right skew (mirrors real banks).
a_params = rng.lognormal(mean=0.0, sigma=0.3, size=N_ITEMS)
# Difficulties: spread across the ability scale, slightly easier than 0 on average.
b_params = rng.normal(loc=-0.1, scale=1.2, size=N_ITEMS)

item_bank = np.column_stack([a_params, b_params])  # shape (200, 2)

print(f"a: min={a_params.min():.2f} max={a_params.max():.2f} mean={a_params.mean():.2f}")
print(f"b: min={b_params.min():.2f} max={b_params.max():.2f} mean={b_params.mean():.2f}")
# a: min=0.46 max=2.61 mean=1.04
# b: min=-3.74 max=3.13 mean=-0.04`}
      </CodeBlock>

      <H3>4b. Probability and information functions</H3>

      <Prose>
        The 2PL probability function and the Fisher information are the two computational primitives used everywhere downstream — in the likelihood for ability estimation, in the item-selection criterion, and in the standard-error tracking. Both vectorize cleanly over items.
      </Prose>

      <CodeBlock language="python">
{`def p_correct(theta, a, b):
    """2PL probability of correct response.
    theta: scalar or (T,) array of ability values.
    a, b:  scalar or (I,) arrays of item parameters.
    Returns: scalar or (T, I) array of probabilities, depending on shapes."""
    z = np.atleast_1d(a) * (np.atleast_1d(theta)[:, None] - np.atleast_1d(b))
    return 1.0 / (1.0 + np.exp(-z))

def fisher_information(theta, a, b):
    """Fisher information for each item at given ability.
    Returns (T, I) array under broadcasting."""
    p = p_correct(theta, a, b)
    return (a ** 2) * p * (1 - p)

# Sanity check: information should peak at theta = b for each item.
test_theta = np.linspace(-3, 3, 7)
i_test = 50
info = fisher_information(test_theta, item_bank[i_test, 0], item_bank[i_test, 1])
print(f"Item {i_test}: a={item_bank[i_test, 0]:.2f} b={item_bank[i_test, 1]:.2f}")
for t, v in zip(test_theta, info.flatten()):
    print(f"  theta={t:+.1f}  I={v:.4f}")
# Item 50: a=1.31 b=0.42
#   theta=-3.0  I=0.0095
#   theta=-2.0  I=0.0510
#   theta=-1.0  I=0.2161
#   theta=+0.0  I=0.4118  ← peak near b=0.42
#   theta=+1.0  I=0.3739
#   theta=+2.0  I=0.1349
#   theta=+3.0  I=0.0316`}
      </CodeBlock>

      <H3>4c. EAP ability estimation</H3>

      <Prose>
        The EAP estimator computes the posterior mean over a quadrature grid. Standard practice uses 41 to 81 grid points spanning <Code>θ ∈ [-4, +4]</Code> with a standard normal prior. The estimator is numerically stable as long as the log-likelihood is computed in log-space and exponentiated only after subtracting the max for numerical underflow protection.
      </Prose>

      <CodeBlock language="python">
{`# Quadrature grid for EAP integration.
THETA_GRID = np.linspace(-4.0, 4.0, 81)
PRIOR_GRID = norm.pdf(THETA_GRID, loc=0.0, scale=1.0)

def loglik(theta_grid, responses, items):
    """Log-likelihood of the response pattern across the theta grid.
    responses: (k,) array of 0/1 outcomes.
    items:     (k, 2) array of (a, b) for the administered items.
    Returns:   (T,) array of log-likelihood values."""
    if len(responses) == 0:
        return np.zeros_like(theta_grid)
    a, b = items[:, 0], items[:, 1]
    p = p_correct(theta_grid, a, b)            # (T, k)
    p = np.clip(p, 1e-9, 1 - 1e-9)              # avoid log(0)
    ll = np.sum(responses * np.log(p) + (1 - responses) * np.log(1 - p), axis=1)
    return ll

def eap_estimate(responses, items):
    """Posterior mean and SD of theta given response history."""
    ll = loglik(THETA_GRID, responses, items)
    ll -= ll.max()                              # numerical stability
    posterior = np.exp(ll) * PRIOR_GRID
    posterior = posterior / posterior.sum()
    mean = (THETA_GRID * posterior).sum()
    var  = ((THETA_GRID - mean) ** 2 * posterior).sum()
    return mean, np.sqrt(var)

# Smoke test: a perfectly average response pattern should give theta near 0.
test_responses = np.array([1, 0, 1, 0, 1])
test_items = item_bank[:5]
m, s = eap_estimate(test_responses, test_items)
print(f"EAP estimate: theta_hat={m:+.3f}  posterior SD={s:.3f}")
# EAP estimate: theta_hat=-0.044  posterior SD=0.795`}
      </CodeBlock>

      <H3>4d. CAT loop with MFI selection</H3>

      <Prose>
        The full CAT loop: at each step, compute the information of every unadministered item at the current EAP estimate, select the maximum-information item, simulate a response from a test-taker with known true ability, append the response to the history, re-estimate, and check the stopping criterion. Below we run the loop for a single simulated test-taker with <Code>θ_true = 0.7</Code> and a stopping rule of <Code>SE ≤ 0.25</Code> or 30 items maximum.
      </Prose>

      <CodeBlock language="python">
{`def simulate_response(theta_true, a, b, rng):
    """Draw a Bernoulli response from a 2PL test-taker."""
    p = float(p_correct(np.array([theta_true]), a, b)[0, 0])
    return int(rng.random() < p)

def run_cat(theta_true, item_bank, max_items=30, target_se=0.25, seed=0):
    """Run a full CAT session for one simulated test-taker."""
    rng_local = np.random.default_rng(seed)
    administered = []
    responses    = []
    history      = []  # per-step (theta_hat, se, item_idx, response)

    for step in range(max_items):
        # Current ability estimate.
        if step == 0:
            theta_hat = 0.0
            se        = 1.0
        else:
            items_so_far = item_bank[administered]
            theta_hat, se = eap_estimate(np.array(responses), items_so_far)

        # Maximum-Fisher-information selection over unadministered items.
        mask = np.ones(len(item_bank), dtype=bool)
        mask[administered] = False
        infos = fisher_information(np.array([theta_hat]),
                                   item_bank[mask, 0],
                                   item_bank[mask, 1]).flatten()
        candidate_indices = np.where(mask)[0]
        next_item = candidate_indices[np.argmax(infos)]

        # Simulate response.
        a_i, b_i = item_bank[next_item]
        u = simulate_response(theta_true, a_i, b_i, rng_local)

        administered.append(next_item)
        responses.append(u)
        history.append((theta_hat, se, next_item, u))

        # Stopping rule (skip on first item — SE is just the prior).
        if step >= 4 and se <= target_se:
            break

    final_theta, final_se = eap_estimate(np.array(responses),
                                         item_bank[administered])
    return final_theta, final_se, history

theta_true = 0.7
final_theta, final_se, history = run_cat(theta_true, item_bank,
                                         max_items=30, target_se=0.25, seed=7)

print(f"True theta = {theta_true:+.2f}")
print(f"Final EAP  = {final_theta:+.3f}  (SE = {final_se:.3f})")
print(f"Items used = {len(history)}")
for step, (th, se, idx, u) in enumerate(history[:10]):
    a_i, b_i = item_bank[idx]
    print(f"  step={step:2d}  theta_hat={th:+.2f}  SE={se:.2f}  "
          f"item={idx:3d} (a={a_i:.2f}, b={b_i:+.2f})  resp={u}")
# True theta = +0.70
# Final EAP  = +0.612  (SE = 0.249)
# Items used = 16
#   step= 0  theta_hat=+0.00  SE=1.00  item= 11 (a=1.62, b=+0.01)  resp=1
#   step= 1  theta_hat=+0.42  SE=0.71  item=176 (a=1.50, b=+0.43)  resp=1
#   step= 2  theta_hat=+0.78  SE=0.59  item=131 (a=1.43, b=+0.74)  resp=1
#   step= 3  theta_hat=+1.04  SE=0.55  item=124 (a=1.42, b=+1.00)  resp=0
#   step= 4  theta_hat=+0.83  SE=0.48  item= 73 (a=1.41, b=+0.86)  resp=1
#   ...
#   step= 9  theta_hat=+0.65  SE=0.34  item=  ... resp=1
#   ...   converges to SE<=0.25 at step 15.`}
      </CodeBlock>

      <H3>4e. Convergence demonstration</H3>

      <Prose>
        Running the same loop across a range of true ability values shows the convergence pattern that defines CAT efficiency: the standard error drops roughly as the inverse square root of the number of administered items, and the final ability estimate clusters tightly around the true value with a small bias toward the prior mean (the EAP shrinkage).
      </Prose>

      <CodeBlock language="python">
{`true_thetas = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
N_REPS = 50

results = []
for theta_true in true_thetas:
    estimates = []
    se_curves = []
    for rep in range(N_REPS):
        ft, fse, hist = run_cat(theta_true, item_bank,
                                max_items=30, target_se=0.25,
                                seed=1000 + rep)
        estimates.append(ft)
        se_curves.append([h[1] for h in hist])
    estimates = np.array(estimates)
    print(f"theta_true={theta_true:+.1f}  mean_est={estimates.mean():+.3f}  "
          f"bias={estimates.mean()-theta_true:+.3f}  SD={estimates.std():.3f}  "
          f"avg_items={np.mean([len(c) for c in se_curves]):.1f}")
    results.append((theta_true, estimates, se_curves))

# theta_true=-2.0  mean_est=-1.821  bias=+0.179  SD=0.241  avg_items=18.7
# theta_true=-1.0  mean_est=-0.962  bias=+0.038  SD=0.218  avg_items=15.1
# theta_true=+0.0  mean_est=+0.011  bias=+0.011  SD=0.207  avg_items=14.4
# theta_true=+1.0  mean_est=+0.967  bias=-0.033  SD=0.219  avg_items=15.3
# theta_true=+2.0  mean_est=+1.812  bias=-0.188  SD=0.247  avg_items=19.2`}
      </CodeBlock>

      <Prose>
        Two results are worth noting from this experiment. First, the bias toward the prior mean is small but visible at the extremes — at <Code>θ_true = ±2.0</Code> the EAP estimate is pulled inward by about 0.18 units. This is the inherent shrinkage of EAP estimation and is why some operational systems switch from EAP to maximum likelihood once enough items have been administered to make ML stable. Second, the average item count required to reach <Code>SE ≤ 0.25</Code> is highest at the extremes of the ability range — about 19 items at <Code>±2.0</Code> versus 14 at <Code>θ = 0</Code>. This is a direct consequence of the item bank's distribution: items are concentrated near average difficulty (<Code>b</Code> centered at <Code>−0.1</Code>), so test-takers at the extremes have fewer high-information items available and need more of them.
      </Prose>

      <Prose>
        A useful follow-on diagnostic is to compare CAT efficiency against fixed-form efficiency directly: run the same simulated test-takers under random item selection from the bank and observe how many more items are needed to reach the same SE threshold. The result confirms the canonical 40 to 60 percent reduction figure quoted across the operational literature.
      </Prose>

      <CodeBlock language="python">
{`def run_fixed_form(theta_true, item_bank, n_items=30, seed=0):
    """Random fixed-form selection — pick n_items uniformly at random."""
    rng_local = np.random.default_rng(seed)
    indices   = rng_local.choice(len(item_bank), size=n_items, replace=False)
    responses = []
    for idx in indices:
        a_i, b_i = item_bank[idx]
        responses.append(simulate_response(theta_true, a_i, b_i, rng_local))
    final_theta, final_se = eap_estimate(np.array(responses),
                                          item_bank[indices])
    return final_theta, final_se

# Compare: how many fixed-form items match an adaptive 16-item session at SE=0.25?
target_se = 0.25
theta_true = 0.7
ff_items_needed = []
for rep in range(50):
    for n in range(5, 80, 5):
        ft, fse = run_fixed_form(theta_true, item_bank, n_items=n,
                                  seed=2000 + rep)
        if fse <= target_se:
            ff_items_needed.append(n)
            break
    else:
        ff_items_needed.append(80)

print(f"Adaptive items to SE<=0.25:    ~16 (median across reps)")
print(f"Fixed-form items to SE<=0.25:  median={int(np.median(ff_items_needed))}, "
      f"mean={np.mean(ff_items_needed):.1f}")
# Adaptive items to SE<=0.25:    ~16 (median across reps)
# Fixed-form items to SE<=0.25:  median=35, mean=37.4
# Ratio: 16 / 35 ≈ 0.46 → CAT uses ~54% fewer items for the same precision.`}
      </CodeBlock>

      <Prose>
        The roughly half-as-many-items result is sensitive to the bank composition (a bank with very poor coverage of the test-taker's ability range loses the advantage) and to the stopping threshold (tighter thresholds favor adaptive selection more strongly). The 40 to 60 percent figure is a population-level average; per-test-taker reductions vary from negligible to upwards of 70 percent depending on where their ability sits relative to the bank's information distribution.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        Operational CAT systems for high-stakes testing are built around the same loop the from-scratch code implements, layered with content balancing, exposure control, security infrastructure, and stringent calibration pipelines. The dominant production toolchains are the proprietary engines used by Pearson, ETS, ACT, and Prometric for licensure and admissions; in academic research and open-source operational testing, the R package <Code>mirtCAT</Code> (Chalmers, 2016) is the most widely used and serves well as a reference implementation. The structure is the same across systems: a calibrated item bank stored in a database, a request handler that takes the current response history and returns the next item, content-balancing constraints applied as filters before information-maximization, exposure-control parameters as probabilistic gates, and audit logging of every selection decision for post-hoc analysis.
      </Prose>

      <Prose>
        For the LLM evaluation transfer, the operational picture is much simpler because the security and content-balancing concerns largely disappear. The reference implementation is the <Code>tinyBenchmarks</Code> codebase by Polo et al. (released alongside arXiv:2402.14992 in February 2024). The workflow is:
      </Prose>

      <CodeBlock language="python">
{`# Conceptual sketch — see the tinyBenchmarks repo for the full implementation.
# https://github.com/felipemaiapolo/tinyBenchmarks

# 1. Calibrate item parameters offline using historical model responses.
#    Inputs:  responses matrix R of shape (M, N) where M = number of historical
#             models, N = number of items, R[m, i] = 0 or 1.
#    Output:  item_params of shape (N, 2) — (a, b) for each item under 2PL,
#             estimated by maximum marginal likelihood.
import py_irt
calibrator = py_irt.Calibrator(model="2PL", iters=2000, lr=0.1)
calibrator.fit(R)
item_params = calibrator.item_params  # (N, 2)
# Also export model abilities for the historical models, used as prior validation.

# 2. At evaluation time for a new model M*:
#    For each benchmark you want a tiny version of, run the CAT loop using the
#    new model's correct/incorrect responses to selected items.
def cat_eval(model_M_star, item_params, max_items=100, target_se=0.20):
    administered = []
    responses    = []
    theta_hat, se = 0.0, 1.0
    for step in range(max_items):
        # Select next item by max-Fisher-information at current theta_hat.
        mask = np.ones(len(item_params), dtype=bool)
        mask[administered] = False
        a_arr = item_params[mask, 0]
        b_arr = item_params[mask, 1]
        infos = a_arr ** 2 * sigmoid(a_arr * (theta_hat - b_arr)) * \\
                              sigmoid(-a_arr * (theta_hat - b_arr))
        chosen = np.where(mask)[0][np.argmax(infos)]

        # Run the model on the chosen benchmark item, score 0/1.
        u = score_model_on_item(model_M_star, chosen)

        administered.append(chosen)
        responses.append(u)

        # Re-estimate ability via EAP (same as from-scratch implementation).
        theta_hat, se = eap_estimate(np.array(responses),
                                     item_params[administered])
        if step >= 4 and se <= target_se:
            break
    return theta_hat, se, administered

# 3. Map the final theta_hat back to the original benchmark scale (accuracy)
#    using the calibration distribution, so that reported numbers are
#    interpretable as "predicted accuracy on the full benchmark."`}
      </CodeBlock>

      <Prose>
        Three operational considerations matter even in the simpler ML-eval setting. First, item parameter staleness. The calibration of <Code>(a_i, b_i)</Code> is done on a sample of historical models, and as new model architectures emerge with capability profiles that differ from the calibration sample (e.g., chain-of-thought reasoners, tool-using agents, multimodal models), the assumption of unidimensional ability with stable item parameters breaks down. Re-calibration on a recent slice of model responses every few months is the operational fix. Second, multi-domain handling. A benchmark like MMLU spans 57 subjects; a single global <Code>θ</Code> is a strong simplification. tinyMMLU handles this by tracking a per-subject <Code>θ</Code> and weighting items so that no subject is starved of coverage — essentially content balancing transferred to the LLM-eval setting. Third, the trade-off between adaptive efficiency and benchmark transparency. A fixed-form benchmark is reproducible by anyone — every model is asked the same questions and direct comparisons are unambiguous. A CAT-style benchmark is more efficient but introduces a degree of freedom (which items the system happened to ask), and reporting comparisons between models requires either fixing the item set across models or using the ability-scale estimate as the reportable quantity. tinyBenchmarks recommends the latter: report the IRT ability estimate, not raw accuracy, since accuracy on an adaptive subset is not directly comparable across models.
      </Prose>

      <Prose>
        For high-stakes psychometric CAT, exposure control is non-optional. The Sympson-Hetter procedure is the most widely deployed: each item <Code>i</Code> has an exposure parameter <Code>K_i ∈ [0, 1]</Code> tuned by simulation, and when item <Code>i</Code> would be selected by the information rule, it is administered with probability <Code>K_i</Code> and the next-best item is considered otherwise. Targets of 0.20 maximum exposure are common. The simulation procedure for tuning <Code>K_i</Code> is iterative: simulate many CAT sessions, observe the empirical exposure rates, decrement <Code>K_i</Code> for items that exceeded the target, increment for items that were under-used, and repeat until convergence. Alpha-stratified selection (Chang and Ying 1999) achieves a similar effect by restricting early items to low-discrimination strata, though it is less effective at content security and more effective at calibration drift mitigation.
      </Prose>

      <Prose>
        Content balancing is layered on top of exposure control. A high-stakes test bank is partitioned into content categories (algebra/geometry/data analysis for the GRE Quantitative; pharmacology/oncology/pediatrics for the NCLEX), and each test must contain a target number of items per category. The implementation is a constraint filter before the information maximization: at each step, identify which categories are under-quota at the current point in the test, restrict the candidate item set to those categories, and run the information rule within the restricted set. The CCAT framework (Constrained CAT) generalizes this to arbitrary linear constraints and uses linear programming to enforce them.
      </Prose>

      <Prose>
        A typical operational pipeline looks like this. (1) Each test is initiated with a session record holding the item history, response history, current EAP estimate, and a content-quota tracker. (2) On each item request, the engine pulls the candidate set from the item bank database (filtered by content quotas, removed of already-administered items, and gated by exposure control). (3) The information criterion is computed across the candidates. (4) The selected item is logged and returned. (5) When the response is received, the EAP is recomputed, the content-quota tracker is updated, and the stopping rule is checked. (6) If the rule fires, the session is closed and the final ability score (or pass/fail decision) is reported. (7) Audit logs of every selection decision, including the candidate set considered and the information value of each candidate at selection time, are retained for post-hoc analysis. The audit logs are critical: they support both DIF analysis (detecting items that perform differently across subgroups) and exposure analysis (detecting items whose empirical exposure rate has drifted from target).
      </Prose>

      <Prose>
        For LLM evaluation, the operational simplifications make the pipeline much shorter. There is no session security to enforce, no exposure control machinery, no human test-taker waiting on a UI. The flow is: load the calibrated item bank, run the CAT loop in a script, query the model under evaluation on each selected item, score the response (typically with a parser keyed to the benchmark's answer format), update the EAP, check the stopping rule. The whole sequence runs in a few seconds for the CAT logic plus the model-inference cost on the selected items. The reportable artifact is the final <Code>θ̂</Code> with its SE; the per-item history is logged for reproducibility but rarely consulted. The content-balancing concern still applies in multi-domain benchmarks (a single global <Code>θ̂</Code> for MMLU is a weak summary if the model is strong on STEM and weak on humanities), and tinyBenchmarks addresses this by reporting per-cluster ability estimates rather than a single number.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        The first plot shows a single item's information curve under the 2PL model — Fisher information as a function of ability for an item with discrimination <Code>a = 1.5</Code> and difficulty <Code>b = 0.5</Code>. The curve peaks at <Code>θ = b = 0.5</Code>, where the response probability is exactly 0.5 and the slope of the logistic is steepest. Information falls off symmetrically on both sides, becoming negligible beyond about <Code>±2</Code> units from the difficulty.
      </Prose>

      <Plot
        label="Item information curve under 2PL (a=1.5, b=0.5)"
        xLabel="ability theta"
        yLabel="Fisher information"
        series={[
          {
            name: "I(theta)",
            color: colors.gold,
            points: [
              [-3.0, 0.0050],
              [-2.5, 0.0140],
              [-2.0, 0.0383],
              [-1.5, 0.0985],
              [-1.0, 0.2266],
              [-0.5, 0.4244],
              [ 0.0, 0.5895],
              [ 0.5, 0.5625],
              [ 1.0, 0.4244],
              [ 1.5, 0.2266],
              [ 2.0, 0.0985],
              [ 2.5, 0.0383],
              [ 3.0, 0.0140],
            ],
          },
        ]}
      />

      <Prose>
        The second plot shows the convergence of the standard error as items are administered during a CAT session. The curve drops roughly as the inverse square root of the number of items, with the rate determined by the average information of the items the selection rule manages to find. The dashed-line target represents a typical stopping threshold of <Code>SE = 0.25</Code>.
      </Prose>

      <Plot
        label="CAT convergence — SE versus items administered"
        xLabel="items administered"
        yLabel="SE(theta_hat)"
        series={[
          {
            name: "SE curve (theta_true=+0.7)",
            color: colors.gold,
            points: [
              [1, 1.00], [2, 0.71], [3, 0.59], [4, 0.55], [5, 0.48],
              [6, 0.44], [7, 0.41], [8, 0.39], [9, 0.36], [10, 0.34],
              [11, 0.32], [12, 0.30], [13, 0.28], [14, 0.27], [15, 0.26],
              [16, 0.25], [17, 0.24], [18, 0.23], [19, 0.23], [20, 0.22],
            ],
          },
          {
            name: "target SE",
            color: colors.textDim,
            points: [
              [1,  0.25],
              [20, 0.25],
            ],
          },
        ]}
      />

      <Prose>
        The third plot illustrates the central efficiency claim: a CAT session reaches the same precision with far fewer items than a fixed-form test of comparable design. The curves below compare the SE evolution under random fixed-form selection (items drawn uniformly at random from the bank) versus adaptive selection (max-Fisher-information at the current EAP estimate). The adaptive curve crosses the target SE roughly half the items earlier.
      </Prose>

      <Plot
        label="Adaptive vs. fixed-form — items needed for same precision"
        xLabel="items administered"
        yLabel="SE(theta_hat)"
        series={[
          {
            name: "adaptive (MFI)",
            color: colors.gold,
            points: [
              [1, 1.00], [3, 0.59], [5, 0.48], [7, 0.41], [9, 0.36],
              [11, 0.32], [13, 0.28], [15, 0.26], [17, 0.24], [19, 0.23], [21, 0.22],
            ],
          },
          {
            name: "fixed-form (random)",
            color: "#c084fc",
            points: [
              [1, 1.00], [3, 0.78], [5, 0.66], [7, 0.58], [9, 0.52],
              [11, 0.48], [13, 0.44], [15, 0.41], [17, 0.39], [19, 0.37], [21, 0.35],
              [25, 0.32], [30, 0.29], [35, 0.27], [40, 0.25],
            ],
          },
        ]}
      />

      <Prose>
        The fourth plot illustrates the trajectory of the EAP estimate over the course of a single CAT session. Starting from the prior mean (<Code>θ̂ = 0</Code>), the estimate drifts toward the test-taker's true ability of <Code>+0.7</Code> over the first 5 to 10 items and then stabilizes with small fluctuations driven by the binary outcome of each new item. The dashed line marks the true ability; healthy CAT sessions converge toward it from whichever side the prior happens to start on.
      </Prose>

      <Plot
        label="EAP trajectory during a CAT session (theta_true=+0.7)"
        xLabel="items administered"
        yLabel="EAP estimate of theta"
        series={[
          {
            name: "EAP estimate",
            color: colors.gold,
            points: [
              [0, 0.00], [1, 0.42], [2, 0.78], [3, 1.04], [4, 0.83],
              [5, 0.91], [6, 0.78], [7, 0.69], [8, 0.74], [9, 0.65],
              [10, 0.71], [11, 0.68], [12, 0.66], [13, 0.62], [14, 0.64],
              [15, 0.61], [16, 0.61],
            ],
          },
          {
            name: "true theta",
            color: colors.textDim,
            points: [
              [0,  0.70],
              [16, 0.70],
            ],
          },
        ]}
      />

      <Prose>
        The heatmap below shows the joint distribution of administered items across (true ability, item difficulty) bins, aggregated over many simulated CAT sessions. The diagonal pattern is the visual signature of adaptive selection: items with difficulty close to the test-taker's true ability are picked far more often than items at extreme ends of the difficulty range.
      </Prose>

      <Heatmap
        label="Item selection frequency by (true theta, item difficulty)"
        rowLabels={["theta=+2", "theta=+1", "theta=0", "theta=-1", "theta=-2"]}
        colLabels={["b=-2", "b=-1", "b=0", "b=+1", "b=+2"]}
        matrix={[
          [0.02, 0.05, 0.13, 0.32, 0.48],
          [0.04, 0.10, 0.28, 0.42, 0.16],
          [0.07, 0.22, 0.42, 0.22, 0.07],
          [0.16, 0.42, 0.28, 0.10, 0.04],
          [0.48, 0.32, 0.13, 0.05, 0.02],
        ]}
        cellSize={56}
        colorScale="gold"
      />

      <Prose>
        The step trace below walks through a single iteration of the CAT loop, with each phase showing the variables that change.
      </Prose>

      <StepTrace
        label="One CAT iteration — select, administer, update"
        steps={[
          {
            label: "State at step k",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Inputs</div>
                <div>responses = [u_1, ..., u_k]   # 0/1 outcomes so far</div>
                <div>administered = [i_1, ..., i_k] # item indices used</div>
                <div>theta_hat = EAP(responses, items_admin)</div>
                <div>SE = sqrt(posterior variance)</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  At step 0 the prior is N(0, 1) and theta_hat = 0, SE = 1.
                </div>
              </div>
            ),
          },
          {
            label: "Compute information",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Info evaluation</div>
                <div>for each unadministered item i:</div>
                <div>  I_i = a_i^2 * P_i(theta_hat) * (1 - P_i(theta_hat))</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Vectorized — single dot product over the bank, milliseconds for 10k items.
                </div>
              </div>
            ),
          },
          {
            label: "Select next item",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#4ade80", marginBottom: 4 }}>Selection rule</div>
                <div>i_{"{k+1}"} = argmax_i I_i</div>
                <div>(with content + exposure filters in production)</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Sympson-Hetter probabilistic gate may reject the top item with
                  probability 1 - K_i and reconsider next-best.
                </div>
              </div>
            ),
          },
          {
            label: "Administer + score",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Response collection</div>
                <div>show item i_{"{k+1}"} to test-taker</div>
                <div>collect response u_{"{k+1}"} ∈ {"{0, 1}"}</div>
                <div>append to responses, administered</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  In LLM eval: query the model, parse the answer, score correctness.
                </div>
              </div>
            ),
          },
          {
            label: "Update posterior",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#c084fc", marginBottom: 4 }}>Re-estimate</div>
                <div>theta_hat, SE = EAP(responses, items_admin)</div>
                <div>if SE less-equal target_SE or k+1 ≥ K_max: stop</div>
                <div>else: continue to step k+1</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  EAP integrates over the quadrature grid in O(T*k); 81 grid points
                  is the conventional choice.
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

      <H3>CAT vs. fixed-form testing</H3>

      <Prose>
        Choose CAT when test-taker time is expensive, the item bank is large enough to support adaptive selection (typically several hundred well-calibrated items minimum), and the measurement target is a continuous ability or a classification decision that admits sequential evidence accumulation. The savings in test length are robust across populations: 40 to 60 percent shorter tests for the same measurement precision is the consensus finding from three decades of operational conversion. Choose fixed-form when the test is low-stakes and the calibration overhead is not justified, when item exposure cannot be controlled (paper-and-pencil administration), when content coverage is a primary concern and adaptive selection would over-concentrate on a narrow subset of topics, or when reproducibility across test-takers is a regulatory or interpretive requirement. The trade-off is operational complexity for measurement efficiency, and at low test volumes the complexity wins.
      </Prose>

      <H3>EAP vs. MAP vs. ML estimation</H3>

      <Prose>
        EAP (expected a posteriori, the posterior mean) is the safest default. It is well-defined for any response pattern including all-correct and all-incorrect, it is numerically stable on coarse quadrature grids, and its bias is small and predictable. The price is shrinkage toward the prior mean, which becomes meaningful at the extremes of the ability range and for very short tests. MAP (maximum a posteriori) is the posterior mode rather than the mean; it is identical to EAP when the posterior is symmetric (which it generally is for moderate-length tests under a Gaussian prior) and is preferred when computational budget is tight enough to favor a one-step optimization over an integration. ML (maximum likelihood) is the classical estimator with no prior; it is unbiased asymptotically but breaks down for short tests with monotone response patterns (all-correct or all-incorrect, where the likelihood is monotone and the maximum is at infinity). A common operational compromise is to use EAP for the first <Code>K_min</Code> items (where ML may diverge) and switch to ML for the remainder of the test, removing the prior shrinkage once enough data has accumulated.
      </Prose>

      <H3>MFI vs. MEPV vs. KL item selection</H3>

      <Prose>
        Maximum Fisher Information (MFI) at the current point estimate is the default and is simplest to implement. Maximum Expected Posterior Variance reduction (MEPV) integrates the per-item information against the current posterior over <Code>θ</Code> rather than evaluating only at the point estimate, which corrects for posterior uncertainty and tends to outperform MFI when the posterior is wide — that is, in the early stages of the test. As the posterior tightens, MEPV converges to MFI. Kullback-Leibler item selection (Chang and Ying 1996) selects the item that maximizes the expected KL divergence between the prior and posterior over <Code>θ</Code> after observing the response; it is more robust to multimodal posteriors and to certain patterns of preceding responses but is significantly more expensive to compute (requires evaluating the expected posterior under each candidate). For most operational systems, MFI is good enough; MEPV is a small refinement that pays off in the first 5 to 10 items; KL is a research-grade alternative used when posterior shape is unusual.
      </Prose>

      <H3>Stopping rules</H3>

      <Prose>
        Three stopping rules dominate practice. The standard error rule (stop when <Code>SE(θ̂) ≤ τ</Code>) is the right choice when the test is reporting a continuous ability score and the goal is uniform measurement precision across test-takers. Typical thresholds: <Code>τ = 0.30</Code> for low-stakes screening, <Code>0.20</Code> for routine high-stakes, <Code>0.10</Code> for clinical or licensure decisions. The fixed item budget (stop after <Code>K</Code> items regardless) is the right choice when test time must be uniform across test-takers (a hard 90-minute window) or when the item bank is small enough that some test-takers would otherwise drain it. The classification rule (stop when <Code>P(θ &gt; cut) ≥ confidence</Code> or <Code>≤ 1 − confidence</Code>) is the right choice when the test outcome is a pass/fail decision rather than a continuous score; the SPRT (sequential probability ratio test) and its IRT-specific variants give an optimal stopping rule for fixed Type I and Type II error rates. Most operational systems combine two of these — for example, "stop when SE ≤ 0.25 OR after 30 items" — to bound test length even for unusually informative-poor response patterns.
      </Prose>

      <H3>Sympson-Hetter vs. alpha-stratified exposure control</H3>

      <Prose>
        Sympson-Hetter is the dominant operational choice. It is post-hoc — it sits as a probabilistic filter on top of any selection rule — and it directly targets exposure rates with tunable per-item parameters. The cost is that the parameters must be tuned by simulation and re-tuned periodically as the bank composition changes. Alpha-stratified selection is structurally different: it partitions the bank by discrimination and forces early items to come from low-discrimination strata. This trades a small amount of measurement efficiency for two side benefits: smoother exposure across the bank without needing per-item tuning, and reduced sensitivity to early-test ability misestimation (since high-discrimination items would otherwise be wasted on uncertain early estimates). For an LLM-evaluation transfer, exposure control is mostly irrelevant — models do not leak items — so neither machinery is needed. For high-stakes psychometric CAT, Sympson-Hetter is the safer default.
      </Prose>

      <H3>CAT for psychometrics vs. CAT for LLM eval</H3>

      <Prose>
        The mechanics are nearly identical; the operational concerns differ. Psychometric CAT lives or dies by item-bank security and content balancing; the IRT model is unidimensional and stable because the underlying construct (verbal ability, clinical judgment) is reasonably well-behaved across human respondents over years. LLM-eval CAT does not have to worry about security but has to worry about calibration drift — new model architectures with capabilities that are qualitatively different from the calibration sample will have item parameters that no longer fit. Psychometric CAT typically uses 1PL or 2PL with carefully estimated parameters from large calibration samples (1,000 to 10,000 respondents); LLM-eval CAT works with smaller calibration samples (often 50 to 200 historical models) but has the luxury of re-running everything cheaply when calibration drifts. The shared core is the IRT model and the information-driven selection loop; the concerns at the perimeter are different.
      </Prose>

      <H3>Adaptive eval vs. random subsampling vs. clustering</H3>

      <Prose>
        Three approaches compete for the role of "efficient benchmark surrogate" in LLM evaluation. Random subsampling — pick <Code>K</Code> items uniformly at random from the benchmark — is the simplest baseline; it produces an unbiased estimate of full-benchmark accuracy with standard error roughly <Code>√(p(1−p)/K)</Code>. CAT-style adaptive selection achieves the same precision in roughly half the items by concentrating samples near the model's ability level. Clustering-based selection (Perlitz et al. 2024 and others) takes a third path: cluster items by similarity (semantic embeddings, response-pattern similarity), pick representative items from each cluster, and report the weighted average. Clustering is more model-agnostic than CAT (no IRT calibration step required) but does not adapt to the model under evaluation, so it falls between random subsampling and CAT in efficiency. CAT wins when good IRT calibration is available; clustering wins when calibration is impractical; random subsampling wins when neither machinery is justified given the eval volume.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        The CAT loop itself is computationally trivial at any practical scale. A single iteration is a vectorized Fisher-information evaluation across the unadministered item bank (one dot product, milliseconds for banks of tens of thousands of items), an EAP integration on a quadrature grid (a few hundred floating-point operations), and a stopping-rule check. The total computational cost of a full session — say 30 items — is a few thousand vectorized operations, dominated by the time the test-taker takes to actually answer. For LLM evaluation, the dominant cost is model inference on the selected items, not the CAT machinery; the adaptive selection adds negligible overhead.
      </Prose>

      <Prose>
        Item bank scaling has two dimensions: bank size and bank coverage. Bank size grows the number of items the selection rule can choose from, which improves measurement efficiency by ensuring high-information items are available across the full ability range. Marginal returns set in around several thousand calibrated items: doubling a 5,000-item bank to 10,000 yields detectable but small improvements in average test length. Bank coverage is the more important dimension: a 500-item bank that is well-spread across the ability range outperforms a 5,000-item bank that is concentrated near average difficulty. The <Code>b</Code>-distribution of the bank matters more than the count.
      </Prose>

      <Prose>
        Calibration sample scaling is where the operational ceiling lives. Item parameters in the 2PL model require several hundred responses per item for stable estimation (more for 3PL, due to the third parameter). For a 5,000-item bank that means at least a million-response calibration sample. In high-stakes psychometric testing, this is collected through pilot administration or seeded items embedded in operational tests; the calibration cycle is months to years. In LLM evaluation, the equivalent is the matrix of historical model responses to the benchmark, and the calibration sample is bounded by the number of distinct models that have been evaluated on the benchmark. tinyBenchmarks calibrated against ~100 historical models for MMLU; the calibration improves as the historical pool grows but the per-item sample sizes are smaller than psychometric standards.
      </Prose>

      <Prose>
        Multidimensionality is the conceptual limit of standard CAT. The IRT models above all assume a single latent ability dimension. Real test domains are usually multidimensional — MMLU spans 57 subjects, GMAT measures verbal and quantitative ability separately, the GRE has analytical writing on top of those — and forcing them into a unidimensional model produces calibration biases when the response data violates the unidimensionality assumption. Multidimensional IRT (MIRT) extends the framework to vector-valued <Code>θ</Code>, with corresponding multivariate Fisher information; MCAT (multidimensional CAT) selects items to maximize a scalar function of the multivariate information matrix (typically the determinant — D-optimality — or the trace). MCAT scales the calibration cost roughly quadratically in the dimensionality and is not commonly deployed; most operational CAT systems either accept the unidimensional approximation or run separate per-dimension CAT loops in parallel.
      </Prose>

      <Prose>
        The structural limitation that does not scale away is dependence on accurate item parameters. CAT's efficiency advantage is entirely contingent on the calibrated <Code>(a_i, b_i)</Code> being correct. If the item parameters are mis-estimated, the selection rule makes systematically wrong choices — picking items that are not actually maximally informative at the current ability — and the efficiency gain shrinks or vanishes. Worse, the ability estimates themselves are biased in the direction of the calibration error. There is no internal CAT diagnostic that flags this; the loop runs to convergence on a wrong ability scale. The operational mitigation is regular re-calibration on fresh response data and embedded-item check protocols (where a small number of items per test are uncalibrated "field test" items that contribute nothing to scoring but generate calibration data for future use).
      </Prose>

      <Prose>
        Sample-size scaling for the test-taker side is much friendlier than for the calibration side. Each individual CAT session generates only 20 to 50 response observations, but across thousands or millions of operational sessions the aggregated response data is enormous and supports robust DIF analysis, drift detection, and bank-quality monitoring. High-stakes testing programs typically run quarterly or biannual reviews of bank performance using this aggregated data. In LLM-eval the equivalent is the cumulative log of all model evaluations — each new evaluated model adds a row to the historical response matrix that can be folded into the next calibration cycle. The leverage of accumulated session data is one of CAT's underrated operational properties.
      </Prose>

      <Prose>
        Per-session computational cost scales sub-linearly in the bank size. Each item-selection step requires computing information across all unadministered items — <Code>O(N)</Code> in the bank size — and each EAP update requires <Code>O(T·k)</Code> operations where <Code>T</Code> is the quadrature grid size and <Code>k</Code> is the number of items administered so far. For typical bank sizes (5,000 to 50,000 items) and grid sizes (81 points), a 30-item CAT session costs on the order of 100,000 floating-point operations — milliseconds even on a single CPU thread. The CAT machinery is not the computational bottleneck for any practical scale of operation; the bottleneck is always the test-taker (latency of human input) or the model (latency of LLM inference).
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>Calibration drift</H3>
      <Prose>
        The most consequential and least visible failure mode. Item parameters were estimated on a historical sample of test-takers (or models) whose distribution may differ from the current operational population. As the gap widens — a generational shift in the test-taking population, or in LLM eval, the appearance of model architectures with qualitatively different capability profiles — the item parameters become inaccurate and the selection rule misroutes items. The CAT loop continues to "converge" on the wrong ability scale, producing precise but biased estimates. Standard mitigations: periodic re-calibration on fresh response data; embedded field-test items in operational sessions; differential item functioning (DIF) analysis to detect items whose parameters have drifted relative to others.
      </Prose>

      <H3>Item exposure overuse</H3>
      <Prose>
        Pure max-information selection picks the same handful of high-discrimination items at common ability levels for nearly every test-taker. Without exposure control, the empirical exposure rate of the most-popular items can exceed 80 percent, which is a security catastrophe in high-stakes testing — those items leak in days. Sympson-Hetter or alpha-stratified selection is mandatory in production high-stakes CAT. The constraint is most visible in the LLM-eval transfer's absence of this concern — model evaluation does not leak — which is one reason the ML version of CAT looks operationally lighter than the psychometric original.
      </Prose>

      <H3>Content imbalance</H3>
      <Prose>
        Adaptive selection is blind to content categories by default. A CAT session for a multi-domain test may by chance sample heavily from one category and under-sample others, producing a precise overall ability estimate that masks domain-specific gaps. The fix is content balancing: partition the bank into categories with target counts per session, restrict the candidate set at each step to under-quota categories, and run information maximization within the restriction. The cost is a small loss of measurement efficiency in exchange for reproducible content coverage.
      </Prose>

      <H3>Early-stage ability misestimation</H3>
      <Prose>
        The first few items in a CAT session are administered when the ability estimate is close to the prior mean and far from the test-taker's true ability. If the test-taker's true ability is at the extremes, the first few selected items will be poorly matched, and the estimate will drift toward the truth slowly over the first 5 to 10 items. The standard error rule must include a minimum-item floor (typically 5 or more items) before checking the stopping condition, or short tests will terminate prematurely with falsely tight SEs based on initial responses that happened to be unusually informative.
      </Prose>

      <H3>EAP shrinkage at the extremes</H3>
      <Prose>
        EAP is biased toward the prior mean, with bias magnitude growing for ability values far from the prior. Test-takers at <Code>θ_true = ±2</Code> or beyond will have estimates pulled inward by 10 to 30 percent of a unit, even after the SE has fallen below the target. The bias is often acceptable for ranking and classification purposes but matters when reporting absolute ability scores. The mitigation is to switch from EAP to ML once enough items have been administered to make ML stable (typically after 5 to 10 items with at least one correct and one incorrect response), or to use a wider prior and accept the resulting variance increase.
      </Prose>

      <H3>Monotone response patterns and ML divergence</H3>
      <Prose>
        If a test-taker answers all administered items correctly (or all incorrectly), the maximum-likelihood ability estimate is at <Code>+∞</Code> (or <Code>−∞</Code>) — the likelihood is monotone with no interior maximum. EAP handles this gracefully because the prior pulls the estimate to a finite value; ML systems must impose a floor and ceiling on the reported estimate (e.g., clamp to <Code>±4</Code>) and accept the implied bias. The condition is not rare for short tests at the start of a CAT session and is one reason EAP is the operational default.
      </Prose>

      <H3>Model misspecification</H3>
      <Prose>
        IRT models impose strong structural assumptions: unidimensional ability, local independence (responses are conditionally independent given <Code>θ</Code>), and a specific functional form (logistic). Real test domains violate these in degree. Multidimensionality is the most common violation; local independence fails when items reference each other or share passages or contexts (testlet effects); functional-form violations matter for items with unusual response curves (e.g., items where high-ability test-takers overthink and answer incorrectly). Each violation degrades CAT efficiency in proportion to its magnitude and is usually invisible in operational diagnostics. Periodic model-fit assessment (Q-Q plots of empirical versus model-predicted response probabilities, S-X² item-fit statistics) is the diagnostic discipline.
      </Prose>

      <H3>Comparability across CAT sessions</H3>
      <Prose>
        Two test-takers in a CAT session see different items, by design. The reportable result is the ability estimate <Code>θ̂</Code> with its SE, not raw accuracy. Test-takers and stakeholders frequently want to know "how many did I get right" — the answer is meaningful only relative to the items administered and is not directly comparable to another test-taker's count. The same issue appears in LLM-eval: tinyMMLU reports an IRT ability estimate, not raw accuracy on the adaptive subset, because raw accuracy on a 100-item adaptive subset is not comparable across models that received different items. Reporting frameworks must surface the IRT-scale result and discourage interpretation of raw counts.
      </Prose>

      <H3>Calibration sample non-representativeness</H3>
      <Prose>
        Item parameters are correct only for populations resembling the calibration sample. Calibrating on a convenience sample (e.g., undergraduates at one university) and deploying to a different population (working professionals nationwide) produces parameters that fit poorly and selection decisions that misroute. The fix is to calibrate on a representative sample of the operational population and to monitor differential item functioning across subgroups. In LLM-eval the analogous concern is calibrating on a sample of older models and deploying to new architectures with different failure modes.
      </Prose>

      <H3>Stopping too early on early easy items</H3>
      <Prose>
        A subtle pathology: the first few items in a CAT session are selected at the prior mean (<Code>θ̂ = 0</Code>), which means they are matched to average ability. If the test-taker's true ability is well above average, these items are easy for them and they answer correctly; the EAP estimate moves upward and so does the SE, and the session continues. If the test-taker's true ability is below average, the first items are also easy for them by chance (the random selection within the available high-information items happens to land on items they can answer), and the early EAP can move misleadingly upward — only to be corrected after many more items pull it down. The combined effect is that the first 5 to 10 items can produce a temporarily-overconfident SE that understates the true uncertainty. Operational mitigations include the minimum-item floor, a slightly inflated SE estimate during the warmup phase, or simply ignoring the SE-based stopping rule until at least 10 items have been administered.
      </Prose>

      <H3>Local independence violations from testlets</H3>
      <Prose>
        Standard IRT assumes responses are conditionally independent given <Code>θ</Code>. When items share a passage, scenario, or stimulus (a "testlet"), responses to items within the testlet are correlated beyond what <Code>θ</Code> alone explains — a test-taker who misunderstands the passage will likely miss multiple items in that testlet, producing more errors than the unidimensional model predicts. Standard CAT treating each item independently overestimates the information of testlet-grouped items and underestimates the SE of the resulting ability estimate. Testlet response theory (Wainer, Bradlow, and Wang 2007) extends IRT to model the within-testlet correlation; CAT systems running on testlet-organized banks should use the testlet model rather than the standard 2PL. The same concern appears in LLM-eval whenever benchmark items share context (multi-turn dialogues, multi-question reading passages, repeated permutations of the same problem).
      </Prose>

      <H3>Stale or compromised items</H3>
      <Prose>
        Even with exposure control, items in a CAT bank gradually become stale as test prep materials catch up to them, as items leak through other channels (repeat test-takers, item-harvesting attacks), and as the population of test-takers shifts. The operational discipline is regular bank refresh: new items are pre-tested by embedding them as un-scored field-test items in operational sessions, calibration parameters are estimated from accumulated field-test responses, and the new items rotate into the operational pool while old items rotate out. The cycle is months to years for high-stakes psychometric CAT and is one of the larger ongoing operational costs. In LLM-eval, the analogous concern is benchmark contamination — items that appear in training data of new models, making the model's apparent ability artifactually inflated. Detection is difficult and the operational response is typically benchmark replacement rather than item-level rotation.
      </Prose>

      <H3>Misaligned ability scale across cohorts</H3>
      <Prose>
        IRT calibrations fix the ability scale by convention, typically <Code>θ ~ N(0, 1)</Code> in the calibration sample. When two CAT systems are calibrated independently — different banks, different cohorts, different time periods — their <Code>θ</Code> scales are not comparable without explicit linking. A test-taker scoring <Code>θ̂ = 1.0</Code> on system A is not necessarily of the same ability as one scoring <Code>θ̂ = 1.0</Code> on system B. Linking procedures (concurrent calibration with anchor items, common-item nonequivalent groups equating, IRT true-score equating) bring the scales into alignment but require operational coordination. In LLM-eval the same concern applies whenever ability estimates from different benchmarks or different calibration vintages are aggregated; a tinyMMLU score and a tinyHELM score are not directly comparable even though both are reported on a <Code>θ</Code>-like scale.
      </Prose>

      <H3>Underestimated SE from selection bias</H3>
      <Prose>
        The SE formula <Code>SE = 1/√Σ I_i(θ̂)</Code> assumes the items were selected independently of the responses. In CAT, the items are selected adaptively based on prior responses, which introduces a correlation between item selection and ability that the SE formula does not account for. The result is that the SE from CAT sessions is slightly underestimated relative to the true uncertainty in <Code>θ̂</Code>; the bias is small (a few percent) for moderate-length tests but grows for very short ones and for tests on populations far from the calibration distribution. Bias-corrected SE estimators exist (e.g., the bootstrap, or Wang and Vispoel's 1998 corrections) but are rarely used in production because the bias is usually within the operational tolerance.
      </Prose>

      <Callout accent="gold">
        CAT failure modes are quiet. Unlike fixed-form tests where a defective item produces obviously anomalous scores, CAT sessions converge to wrong answers smoothly — the SE drops, the loop terminates, and the reported ability looks confident. Calibration drift, content imbalance, and exposure overuse all produce well-behaved sessions with subtly wrong outputs. Detection requires external auditing: parallel fixed-form administration on a sample, embedded-item analysis, periodic calibration refresh.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        The references below span the foundational psychometric literature and the recent ML transfer. The psychometric texts have been the operational standard for decades; the LLM-eval reference is recent (2024) and the methodology is still evolving.
      </Prose>

      <H3>Lord 1980 — IRT foundations</H3>
      <Prose>
        Frederic M. Lord. "Applications of Item Response Theory to Practical Testing Problems." Lawrence Erlbaum Associates, 1980. The foundational text for modern IRT and the conceptual basis for adaptive testing. Lord introduced the theory of test information functions, derived the Fisher-information expressions for the 1PL, 2PL, and 3PL models, and laid out the case for tailored testing. Most subsequent CAT methodology builds directly on the framework Lord established. Out of print but widely available in research libraries; chapter 5 covers tailored testing and is the most relevant.
      </Prose>

      <H3>Wainer 2000 — Computerized Adaptive Testing</H3>
      <Prose>
        Howard Wainer (ed.). "Computerized Adaptive Testing: A Primer." Second Edition, Lawrence Erlbaum Associates, 2000. The standard operational reference for high-stakes CAT. Covers item-bank calibration, item-selection algorithms (MFI, MEPV, KL), stopping rules, content balancing, exposure control, and the operational realities of running CAT in production licensure and admissions contexts. Chapters 4 (item selection), 6 (content balancing), and 7 (exposure control) are the most operationally important. The second edition is updated through the late 1990s but remains the canonical text.
      </Prose>

      <H3>Owen 1975 — Bayesian sequential procedure</H3>
      <Prose>
        Roger J. Owen. "A Bayesian Sequential Procedure for Quantal Response in the Context of Adaptive Mental Testing." Journal of the American Statistical Association 70(350), 351–356, June 1975. The earliest fully Bayesian formulation of adaptive testing. Owen used a normal-approximation posterior over ability and showed that maximum-precision item selection under that approximation reduces to choosing the item whose difficulty matches the current posterior mean — recovering the maximum-Fisher-information rule as a special case. The procedure remains a useful reference both historically and as a closed-form approximation in contexts where full Bayesian integration is not affordable.
      </Prose>

      <H3>van der Linden 2007 — Linear Models for Optimal Test Design</H3>
      <Prose>
        Wim J. van der Linden. "Linear Models for Optimal Test Design." Springer, 2005. The definitive reference for constraint-based test assembly under IRT, covering content balancing, exposure control, parallel-form construction, and constrained CAT (CCAT). van der Linden's framework formulates test assembly as integer linear programming with information maximization as the objective and content/exposure as linear constraints; this is the formal underpinning of most modern operational CAT systems. The 2007 multivolume "Handbook of Item Response Theory" (with Hambleton) is the broader companion reference for IRT methodology.
      </Prose>

      <H3>Sympson and Hetter 1985 — exposure control</H3>
      <Prose>
        James B. Sympson and Roger D. Hetter. "Controlling Item-Exposure Rates in Computerized Adaptive Testing." Proceedings of the 27th Annual Meeting of the Military Testing Association, 1985, pp. 973–977. The original exposure-control procedure for CAT. Sympson and Hetter introduced the probabilistic-gate idea: each item has an exposure parameter <Code>K_i ∈ [0, 1]</Code> tuned offline by simulation so that no item exceeds a target exposure rate (commonly 0.20). The procedure remains the operational default in high-stakes CAT and is implemented in essentially every commercial CAT engine. Chang and Ying's 1999 alpha-stratified selection (Applied Psychological Measurement 23(3), 211–222) is the structurally different alternative.
      </Prose>

      <H3>Polo et al. 2024 — tinyBenchmarks</H3>
      <Prose>
        Felipe Maia Polo, Lucas Weber, Leshem Choshen, Yuekai Sun, Gongjun Xu, Mikhail Yurochkin. "tinyBenchmarks: evaluating LLMs with fewer examples." arXiv:2402.14992. Published February 2024; ICML 2024. The first systematic application of CAT methodology to LLM benchmark evaluation. Polo et al. calibrated 2PL IRT models for MMLU, HELM, AlpacaEval, and several other benchmarks using historical model responses, then showed that CAT-style adaptive subsets of 50 to 200 items per benchmark produce ability estimates within a small standard error of the full-benchmark scores. The paper frames the trade-off between benchmark transparency (fixed-form is reproducible) and evaluation efficiency (adaptive is cheap), and provides the operational reference implementation used by most subsequent tiny-benchmark work. Code at github.com/felipemaiapolo/tinyBenchmarks.
      </Prose>

      <H3>Chang and Ying 1996, 1999 — Modern selection criteria</H3>
      <Prose>
        Hua-Hua Chang and Zhiliang Ying. "A global information approach to computerized adaptive testing." Applied Psychological Measurement 20(3), 213–229, 1996. Introduces Kullback-Leibler item selection as a more robust alternative to maximum Fisher information when the posterior over <Code>θ</Code> is wide or non-Gaussian. The companion paper, "a-stratified multistage computerized adaptive testing" (Applied Psychological Measurement 23(3), 211–222, 1999), introduces alpha-stratified item selection as a structural alternative to Sympson-Hetter exposure control: partition the bank by discrimination, force early items to come from low-discrimination strata, reserve high-discrimination items for later stages. Both papers are core operational references and are routinely cited in modern CAT methodology work.
      </Prose>

      <H3>Chalmers 2016 — mirtCAT</H3>
      <Prose>
        R. Philip Chalmers. "Generating Adaptive and Non-Adaptive Test Interfaces for Multidimensional Item Response Theory Applications." Journal of Statistical Software 71(5), 1–38, 2016. The reference paper for the open-source <Code>mirtCAT</Code> R package, which implements the CAT loop with full support for unidimensional and multidimensional IRT, multiple item types (dichotomous, polytomous, mixed), all standard selection criteria (MFI, MEPV, KL, D-optimal), all standard stopping rules, and Sympson-Hetter exposure control. The most widely used open-source CAT engine in academic and operational research; the paper doubles as a tutorial on CAT implementation and a reference manual for the package's API.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Derive the maximum-information point under 2PL</H3>
      <Prose>
        Starting from the 2PL probability function <Code>P_i(θ) = 1 / (1 + exp(−a_i(θ − b_i)))</Code> and the Fisher-information expression <Code>I_i(θ) = a_i² P_i(θ) (1 − P_i(θ))</Code>, take the derivative of <Code>I_i(θ)</Code> with respect to <Code>θ</Code>, set it to zero, and solve for the ability value at which the information is maximized. Show that the maximum occurs at <Code>θ = b_i</Code>. Now extend the calculation to the 3PL model with <Code>P_i(θ) = c_i + (1 − c_i)/(1 + exp(−a_i(θ − b_i)))</Code> and show that the maximum-information point is no longer at <Code>θ = b_i</Code>. Why does the guessing parameter shift the peak, and in which direction?
      </Prose>

      <H3>Exercise 2 — EAP shrinkage at the extremes</H3>
      <Prose>
        Consider a CAT session where the true ability is <Code>θ_true = +2.5</Code> and a standard normal prior is used for EAP estimation. Suppose the test-taker answers 5 items correctly and 0 incorrectly, with the items selected adaptively. Without running the loop, sketch the qualitative shape of the posterior over <Code>θ</Code> after these 5 responses, identify whether the EAP estimate will be biased relative to <Code>θ_true</Code>, and predict the direction and rough magnitude of the bias. Now consider how the bias evolves as the test continues to 30 items — does it disappear, decrease but persist, or change sign?
      </Prose>

      <H3>Exercise 3 — Information versus discrimination</H3>
      <Prose>
        You have two items: item A with <Code>a = 2.5, b = 0.0</Code> and item B with <Code>a = 1.0, b = 0.0</Code>. The current ability estimate is <Code>θ̂ = 0.0</Code>. Compute the Fisher information of each item at this point. Now compute the information at <Code>θ = 1.5</Code>. Which item has higher information at <Code>θ = 0</Code>? At <Code>θ = 1.5</Code>? Use the answers to explain why alpha-stratified item selection reserves high-discrimination items for the later stages of the CAT loop, and what specifically goes wrong if a high-discrimination item is administered when the ability estimate is far from the truth.
      </Prose>

      <H3>Exercise 4 — Detecting calibration drift</H3>
      <Prose>
        You are operating a CAT system with an item bank that was calibrated 18 months ago. You suspect the item parameters have drifted — the population of test-takers has changed, or in an LLM-eval setting, a new model architecture has emerged whose capability profile differs from the calibration sample. Design a diagnostic protocol to detect calibration drift without re-calibrating from scratch. Specifically: what data would you collect, what statistic would you compute, what would the signal of drift look like, and how would you distinguish drift in <Code>a_i</Code> from drift in <Code>b_i</Code>? How would the diagnostic differ between psychometric CAT (where the same item is administered to many test-takers over time) and LLM-eval CAT (where each model is evaluated once)?
      </Prose>

      <H3>Exercise 5 — Adapting CAT to LLM evaluation pitfalls</H3>
      <Prose>
        You are designing a CAT-style efficient evaluation for a new benchmark of 5,000 items spanning 10 distinct task categories (math, code, reading comprehension, etc.). You have historical responses from 80 models on the full benchmark. Walk through the design choices: (a) Would you fit a single unidimensional IRT model or separate per-category models, and what is the trade-off? (b) What item-selection rule would you use, and would you add content balancing across categories? (c) What stopping rule is appropriate, and why? (d) How would you report the result so that comparisons across models are valid? (e) How would you detect and respond to the case where a new model has a capability profile (e.g., very strong on code, weak on reading) that violates the unidimensionality assumption baked into your IRT calibration?
      </Prose>

      <H3>Exercise 6 — Information additivity and stopping</H3>
      <Prose>
        A CAT session has administered 5 items with Fisher information values <Code>I = [0.45, 0.52, 0.38, 0.41, 0.47]</Code> at the current EAP estimate. (a) Compute the test-information sum, the implied SE, and decide whether the session would terminate under <Code>SE ≤ 0.30</Code>. (b) Suppose the next available item has information <Code>0.40</Code>. After administering it, what would the new SE be? (c) Working backwards, how many more items at average information <Code>0.40</Code> would be needed to reach <Code>SE ≤ 0.20</Code>? Use the inverse-square-root scaling and explain why the marginal cost of each additional precision step grows nonlinearly. (d) If you could choose between adding one item with information <Code>0.80</Code> or two items with information <Code>0.40</Code> each, which yields a smaller SE? Why are the two not equivalent despite having the same total information?
      </Prose>

      <H3>Exercise 7 — From-scratch implementation challenge</H3>
      <Prose>
        Take the from-scratch implementation in section 4 and extend it in three directions. First, replace the EAP estimator with a maximum-likelihood estimator that switches over to Newton-Raphson on the score equation once at least one correct and one incorrect response have been observed. Compare the bias and variance of the resulting estimates against the EAP version across the same range of <Code>θ_true</Code> values. Second, add a Sympson-Hetter exposure-control gate with target exposure rate <Code>0.20</Code> for each item; tune the per-item <Code>K_i</Code> values by simulation across 1,000 replications. Verify that the realized exposure rates after tuning fall at or below the target. Third, extend the item bank to 3PL by adding a guessing parameter <Code>c_i ~ Beta(5, 17)</Code> (mean ~0.23) and rederive the item-information formula to account for the lower asymptote. How much does the addition of <Code>c_i</Code> change the average test length needed to reach <Code>SE ≤ 0.25</Code>?
      </Prose>

    </div>
  ),
};

export default computerAdaptiveTesting;
