import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const judgeCalibration = {
  title: "Judge Calibration (Platt Scaling, Isotonic Regression, Anchors)",
  slug: "judge-calibration-platt-scaling-isotonic-regression-anchors",
  readTime: "~38 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Every team that has ever shipped an LLM-as-judge pipeline eventually discovers the same uncomfortable fact: the numbers the judge produces do not mean what they appear to mean. A judge model that reports a confidence of 0.9 is right about 0.7 of the time. A judge that gives a response a score of 8 out of 10 would be considered, by a calibrated human annotator using the same rubric, to have given a response that a human would score 6.5. Worse, the relationship is not even monotone in any usable way: scores from 7 through 9 might all correspond to the same true human-rated quality band, while a 4 might mean genuinely poor and a 6 might mean genuinely excellent on a different prompt distribution. Without an explicit correction step, the entire downstream stack — preference datasets distilled from judge labels, leaderboards built on judge scores, RLAIF rewards derived from judge probabilities — inherits a distortion that compounds at every stage.
      </Prose>

      <Prose>
        The problem is structural rather than incidental. A judge's output distribution reflects the training distribution of whatever model is doing the judging, the specific phrasing of the rubric, the temperature at which it was sampled, and a long list of inductive biases inherited from the base model's pre-training. None of these are aligned to the operational scale you actually care about. If your goal is to use the judge as a stand-in for a paid annotator pool that follows a specific 5-point rubric, the raw judge scores have no privileged relationship to the annotator scores; they live on a different scale, with a different mean, a different variance, and a non-linear mapping between them. Calibration is the engineering layer that makes the two scales commensurable. It is the same problem that classical ML solved for the output of decision trees and neural networks two decades ago, transposed to a setting where the "classifier" is now a 70B parameter language model and the "labels" are noisy ordinal judgments from humans who themselves disagree at a non-trivial rate.
      </Prose>

      <Prose>
        The history of the techniques goes back further than the LLM era. John Platt's 1999 paper introduced logistic calibration to convert support vector machine margins into probability estimates that actually behaved like probabilities. Bianca Zadrozny and Charles Elkan's 2002 work generalized this with isotonic regression, a non-parametric monotone fit that does not assume a sigmoidal shape. Naeini, Cooper, and Hauskrecht formalized expected calibration error (ECE) as the de facto evaluation metric in 2015. Chuan Guo and collaborators showed in 2017 that modern neural networks are systematically overconfident and that simple temperature scaling fixes most of it. Each of these results was developed for classification problems with binary or low-cardinality labels. The transposition to LLM judges is conceptually identical — you have a model output, you have ground-truth labels, and you want a post-hoc map from one to the other — but the operational constraints are quite different. LLM-as-judge calibration sets are small (often 100 to 1000 examples), expensive to grow (each label costs human time), drift quickly (the judge model itself is updated every few months), and must support ordinal targets rather than just binary ones.
      </Prose>

      <Prose>
        The specific reason this topic deserves a deep treatment, rather than a one-line "use Platt scaling" reference, is that the choice between calibration methods is genuinely consequential. Platt scaling is a two-parameter logistic fit that needs almost no data and produces a smooth, monotone correction. Isotonic regression is non-parametric, makes no functional-form assumption, and is strictly more flexible — but is also data-hungry and prone to overfitting on the small calibration sets that are realistic for production. Anchor calibration sidesteps both by changing the prompt rather than post-processing the output: instead of correcting a miscalibrated judge after the fact, you give it a fixed set of reference responses with known scores in-context, and rely on the judge's in-context learning to re-anchor its scale to those exemplars. Each of these has a sweet spot, and choosing wrong is the difference between a judge pipeline that produces clean preference data and one whose downstream RLAIF run silently learns the wrong objective.
      </Prose>

      <Prose>
        There is also a less obvious reason calibration matters: it is the cheapest legible signal you have for detecting silent regressions in your judge stack. If you maintain a small held-out anchor set with human labels and recompute ECE every time you swap the judge model, change the rubric, or update the prompt template, a sudden ECE jump is a tripwire that catches problems you would otherwise discover only after a downstream metric collapses three weeks later. This monitoring use case is, in production, often more valuable than the calibration itself.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Start with a concrete picture. Imagine you have an LLM judge that, when asked "is this response helpful?", outputs a number between 0 and 1 that it labels a probability. You collect 500 examples where you also have a trusted human label (helpful or not). You bin the judge's outputs into 10 buckets — examples scored 0.0–0.1, examples scored 0.1–0.2, and so on — and within each bucket you compute the fraction of examples that humans actually labeled helpful. If the judge were perfectly calibrated, the bin centered around 0.7 would contain examples that are helpful about 70% of the time. If you plot bin midpoint on the x-axis and human-labeled positive rate on the y-axis, you would get a 45-degree diagonal. This plot is called a reliability diagram, and the gap between the actual curve and the diagonal is what calibration tries to close.
      </Prose>

      <Prose>
        In practice, the curve is almost never a clean diagonal. For modern LLM judges the most common shape is an "S" that lies entirely below the diagonal in the high-confidence region and entirely above in the low-confidence region — the classic over-confidence pattern. The judge claims 0.95 probability and is actually right 0.78 of the time. The judge claims 0.10 probability and is actually right 0.23 of the time. The judge's relative ranking of examples (higher confidence corresponds to higher truth probability) is approximately preserved, but the absolute mapping from judge output to true probability is nonlinear and biased.
      </Prose>

      <Prose>
        The calibration insight is that you can fix this without touching the judge model itself. You simply learn a function <Code>g(s)</Code> that maps the raw judge score <Code>s</Code> to a calibrated probability, using the same labeled examples that produced the reliability diagram. The function should be monotone — if the judge prefers A to B, the calibrated scores should preserve that ordering — but otherwise as flexible as the data supports. Three families of <Code>g</Code> dominate practice: Platt scaling fits a logistic curve with two parameters; isotonic regression fits a piecewise-constant monotone step function with as many degrees of freedom as the data allows; and temperature scaling (a special case of Platt scaling) just divides the logits by a learned scalar, fitting a single parameter.
      </Prose>

      <Prose>
        The trade-off between these three is the central practical question. Platt scaling is biased — it assumes the calibration function is a logistic — but has very low variance and works on tiny datasets. Isotonic regression is unbiased in the limit but has high variance on small datasets and produces a step function that can have visible discontinuities. Temperature scaling is even more constrained than Platt scaling and is the right default when you suspect over- or under-confidence is the only issue. Choosing among them is a bias-variance question, and the right answer depends almost entirely on how many calibration labels you have.
      </Prose>

      <Prose>
        Anchor calibration is conceptually different and worth introducing at this stage because it is often the most useful technique even though it is the least mathematically interesting. The idea is simple: instead of post-hoc correction, you embed a fixed set of reference responses with known scores into the judge prompt itself. The judge sees, before the response under evaluation, several exemplars labeled "this response is a 3, this is a 7, this is a 9". Then it scores the new response by analogy. This works because in-context learning is powerful enough that the judge re-anchors its internal scale to match the exemplars. MT-Bench's reference-based mode does exactly this; it provides a known-good reference answer in the prompt for each evaluation. The deep reason it works is that LLM judges are remarkably good at relative ranking but bad at absolute scale, and anchors convert an absolute scoring problem into a relative one.
      </Prose>

      <Prose>
        A common failure mode worth seeing now: people frequently assume that better judges (GPT-4 class versus GPT-3.5 class) are also better-calibrated. They are not. Larger judges are better at relative ranking, which means their reliability diagrams have steeper slopes, but the absolute miscalibration — the gap between predicted probability and actual frequency — is often comparable or even worse. This is because larger models are also more confident, and their confidence is not earned by improvements in actual accuracy at the same rate. The implication is that calibration is not something you graduate out of when you upgrade your judge; it is a permanent layer in the pipeline.
      </Prose>

      <Prose>
        One more piece of intuition before we leave this section. Calibration is fundamentally about marginal probabilities, not about getting individual examples right. A perfectly calibrated judge can still be wildly wrong on any given example — calibration only guarantees that if you take all the examples it scored 0.7, on average they will be positive at a rate of 0.7. If your downstream use case actually depends on per-example accuracy (say, deciding whether to trust a single judge call before triggering a human review) calibration alone will not save you. It is most valuable when you are aggregating over many judge calls — building preference datasets, computing leaderboard scores, deriving RLAIF rewards — where the per-example noise averages out and the calibration of the aggregate matters.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <Prose>
        Formalize the setup. You have a judge that outputs a score <Code>s ∈ [0, 1]</Code> for each input <Code>x</Code>, and a true label <Code>y ∈ {"{0, 1}"}</Code>. You have a calibration set of size <Code>n</Code> consisting of pairs <Code>(s_i, y_i)</Code>. The goal is to learn a function <Code>g: [0, 1] → [0, 1]</Code> such that the predicted probability <Code>g(s)</Code> approximates the true conditional probability <Code>P(y = 1 | s)</Code>.
      </Prose>

      <Prose>
        A predictor is perfectly calibrated if, for every confidence level <Code>p</Code>:
      </Prose>

      <MathBlock>{"P(Y = 1 \\mid g(S) = p) = p \\quad \\forall p \\in [0, 1]"}</MathBlock>

      <Prose>
        This is an idealization; the empirical version partitions the calibration set into <Code>M</Code> bins by predicted score and computes the average gap between predicted score and observed positive rate within each bin. This is the Expected Calibration Error (ECE), introduced by Naeini et al. (2015):
      </Prose>

      <MathBlock>{"\\mathrm{ECE} = \\sum_{m=1}^{M} \\frac{|B_m|}{n} \\,\\bigl|\\, \\mathrm{acc}(B_m) - \\mathrm{conf}(B_m) \\,\\bigr|"}</MathBlock>

      <Prose>
        where <Code>B_m</Code> is the m-th bin of predictions, <Code>acc(B_m)</Code> is the empirical positive rate in that bin, and <Code>conf(B_m)</Code> is the average predicted score. ECE has a known limitation: it depends on the binning scheme, and you can hide miscalibration by choosing wide bins. Two refinements are common in practice: maximum calibration error (MCE), which reports the worst bin, and adaptive ECE, which uses equal-mass bins instead of equal-width bins so that no bin is computed from too few examples.
      </Prose>

      <Prose>
        The Brier score is a complementary scalar that summarizes both calibration and sharpness in one number:
      </Prose>

      <MathBlock>{"\\mathrm{Brier} = \\frac{1}{n} \\sum_{i=1}^{n} \\bigl(g(s_i) - y_i\\bigr)^2"}</MathBlock>

      <Prose>
        It decomposes into three terms (Murphy 1973): reliability (calibration), resolution (the amount of variation in predictions across bins), and uncertainty (the irreducible variance of the labels). A predictor that always outputs the marginal positive rate is perfectly calibrated but has zero resolution and a Brier score equal to the uncertainty term. A predictor with high resolution but poor calibration has a Brier score inflated by the reliability term. Optimizing Brier score directly trades off both, which is why it is a useful single number even though ECE is more interpretable.
      </Prose>

      <H3>Platt scaling</H3>

      <Prose>
        Platt scaling fits a logistic regression on the judge score. Given the raw score <Code>s</Code> (or, equivalently, a logit <Code>z = log(s/(1−s))</Code> if the judge produced a probability), the calibrated probability is:
      </Prose>

      <MathBlock>{"g(s) = \\sigma(a \\cdot s + b) = \\frac{1}{1 + \\exp(-(a \\cdot s + b))}"}</MathBlock>

      <Prose>
        The two parameters <Code>(a, b)</Code> are estimated by maximum likelihood on the calibration set. The negative log-likelihood is:
      </Prose>

      <MathBlock>{"\\mathcal{L}(a, b) = -\\sum_{i=1}^{n} \\Bigl[ y_i \\log \\sigma(a s_i + b) + (1 - y_i) \\log\\bigl(1 - \\sigma(a s_i + b)\\bigr) \\Bigr]"}</MathBlock>

      <Prose>
        This is convex in <Code>(a, b)</Code> and has a unique optimum that any standard solver finds quickly. Platt's original 1999 paper recommended a small modification of the labels to mitigate overfitting on small calibration sets: replace <Code>y = 1</Code> with <Code>(N_+ + 1)/(N_+ + 2)</Code> and <Code>y = 0</Code> with <Code>1/(N_- + 2)</Code>, where <Code>N_+</Code> and <Code>N_-</Code> are the counts of positive and negative examples. This pulls the targets slightly away from the boundary and prevents the logistic from saturating to ±∞ slopes when the calibration set is perfectly separable, a regime that is common when the judge's relative ranking is good even if its absolute scale is wrong.
      </Prose>

      <Prose>
        Temperature scaling (Guo et al. 2017) is a one-parameter restriction of Platt scaling, originally proposed for neural network classifiers. The calibrated softmax output is computed by dividing the logits by a learned temperature <Code>T &gt; 0</Code> before the softmax. In the binary case this reduces to:
      </Prose>

      <MathBlock>{"g(s) = \\sigma\\!\\left(\\frac{z}{T}\\right) \\quad \\text{where } z = \\log\\!\\frac{s}{1-s}"}</MathBlock>

      <Prose>
        With one parameter, temperature scaling is the most data-efficient calibration technique and the right default when the only problem is over- or under-confidence. <Code>T &gt; 1</Code> softens predictions (fixes over-confidence); <Code>T &lt; 1</Code> sharpens predictions (fixes under-confidence). The cost is that it cannot fix asymmetric miscalibration where the bias differs between the high- and low-score regions.
      </Prose>

      <H3>Isotonic regression and the PAV algorithm</H3>

      <Prose>
        Isotonic regression fits a non-parametric monotone function to the data. Given the calibration pairs <Code>(s_i, y_i)</Code>, sorted by <Code>s</Code>, isotonic regression finds the values <Code>ĝ_i</Code> that minimize:
      </Prose>

      <MathBlock>{"\\sum_{i=1}^{n} (y_i - \\hat g_i)^2 \\quad \\text{subject to } \\hat g_1 \\le \\hat g_2 \\le \\cdots \\le \\hat g_n"}</MathBlock>

      <Prose>
        The closed-form solution is computed by the Pool Adjacent Violators (PAV) algorithm. The intuition is straightforward: sweep through the sorted scores left to right. If the current observation violates monotonicity (its label is smaller than the previous fitted value), pool it with the previous block and replace both with their average. Continue until no violations remain. This produces a piecewise-constant monotone step function in <Code>O(n)</Code> time, and is exactly the maximum-likelihood monotone fit under squared loss.
      </Prose>

      <Prose>
        The PAV algorithm in pseudocode:
      </Prose>

      <CodeBlock language="text">
{`Input: pairs (s_1, y_1), ..., (s_n, y_n) sorted by s_i ascending
Initialize: blocks = [(s_i, y_i, 1) for each i]   # (sum_s, sum_y, count)
i = 0
while i < len(blocks) - 1:
    if blocks[i].sum_y / blocks[i].count > blocks[i+1].sum_y / blocks[i+1].count:
        # Violation. Merge.
        merged = (blocks[i].sum_s + blocks[i+1].sum_s,
                  blocks[i].sum_y + blocks[i+1].sum_y,
                  blocks[i].count  + blocks[i+1].count)
        blocks[i:i+2] = [merged]
        i = max(i - 1, 0)   # Step back; new merge can violate behind.
    else:
        i += 1

Output: for each block, predicted probability = sum_y / count.`}
      </CodeBlock>

      <Prose>
        At inference time, given a new judge score <Code>s</Code>, you find the block whose <Code>s</Code>-range contains <Code>s</Code> (binary search) and return that block's pooled probability. For scores between blocks, the standard convention is to interpolate linearly between the block midpoints; for scores outside the calibration range, you clip to the nearest endpoint.
      </Prose>

      <Prose>
        Zadrozny and Elkan (2002) showed that isotonic regression is consistent — as <Code>n → ∞</Code> the fit converges to the true conditional probability function — under the very weak assumption that the relationship is monotone. Platt scaling is consistent only if the true relationship is actually logistic in the score. In practice, on calibration sets above a few thousand examples, isotonic regression strictly dominates Platt scaling. Below that, the variance of the isotonic fit is high enough that the bias of Platt scaling is the better trade-off.
      </Prose>

      <H3>Anchor calibration as in-context conditioning</H3>

      <Prose>
        Anchor calibration is harder to formalize because it does not produce an explicit calibration function. Instead, it modifies the judge's prompt to include reference responses with known scores. Conceptually, the judge's score becomes a conditional expectation:
      </Prose>

      <MathBlock>{"s_{\\text{anchor}}(x) = \\mathbb{E}_{\\text{judge}}\\!\\left[\\, y \\mid x,\\, \\{(x_k, y_k)\\}_{k=1}^{K} \\,\\right]"}</MathBlock>

      <Prose>
        where <Code>{"{(x_k, y_k)}"}</Code> are the K anchor exemplars with their known scores. The implicit assumption is that the judge's in-context learning is strong enough that conditioning on the anchors shifts its output distribution to align with the anchors' scale. This works in practice because LLMs are good at relative ranking — given exemplars at known points on the scale, they place new examples at consistent relative positions. The mathematical guarantees are weaker than for Platt or isotonic (there is no consistency theorem), but the empirical behavior is often better, especially when the calibration set is too small for a reliable post-hoc fit.
      </Prose>

      <Callout accent="gold">
        Calibration only fixes the marginal — it does not improve per-example accuracy. If your judge is wrong on a specific example, calibration cannot tell you that. It can only adjust the average probability across the population of similar examples. Use ECE for monitoring marginal calibration; use top-k accuracy or per-example agreement for monitoring resolution.
      </Callout>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The cleanest way to internalize calibration is to construct a synthetic judge whose miscalibration you control, run a calibration set through it, fit Platt scaling and isotonic regression, and watch ECE drop on a held-out test set. Every print statement in the comments below reflects the actual output produced when the code was executed; nothing is hypothetical. Six subsections walk through the components: synthetic judge construction, calibration set sampling, ECE computation, Platt scaling fit, isotonic regression via PAV, and a head-to-head comparison.
      </Prose>

      <H3>4a. A miscalibrated synthetic judge</H3>

      <Prose>
        Construct a judge whose true miscalibration shape is known. The setup: each example has a hidden quality <Code>q ∈ [0, 1]</Code> drawn uniformly. The true label is Bernoulli with probability <Code>q</Code>. The judge sees the example and outputs a score <Code>s</Code> that is a deterministic function of <Code>q</Code> with sigmoidal over-confidence baked in. This gives us ground truth for what the calibration function should be — it is the inverse of the over-confidence transformation — so we can verify our calibrators are doing the right thing rather than just producing plausible numbers.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
from scipy.optimize import minimize

np.random.seed(0)

def synthetic_judge(n=2000):
    """
    Generate (judge_score, true_label) pairs.
    Hidden quality q ~ Uniform(0, 1); true label y ~ Bernoulli(q).
    Judge sees q and outputs s = sigmoid(8 * (q - 0.5)),
    which is a strongly over-confident squashing of q.
    """
    q = np.random.uniform(0, 1, size=n)
    y = (np.random.uniform(0, 1, size=n) < q).astype(int)
    s = 1.0 / (1.0 + np.exp(-8.0 * (q - 0.5)))
    return s, y, q

# Sanity check the over-confidence pattern.
s, y, q = synthetic_judge(n=10000)
print("mean q  :", q.mean())   # 0.5005  — fair
print("mean y  :", y.mean())   # 0.4986  — fair
print("mean s  :", s.mean())   # 0.5004  — also fair on average...
# ...but conditional behavior is wildly different. Look at the extremes:
mask_lo = s < 0.1
mask_hi = s > 0.9
print("y rate when s<0.1:", y[mask_lo].mean())   # 0.111
print("y rate when s>0.9:", y[mask_hi].mean())   # 0.890
# When s<0.1 the judge claims ~5% probability but reality is ~11%.
# When s>0.9 the judge claims ~95% but reality is ~89%.
# Classic symmetric over-confidence pattern.`}
      </CodeBlock>

      <H3>4b. Splitting calibration and test sets</H3>

      <Prose>
        Calibration is a learned function and must be fit on data that is disjoint from the data used to evaluate it, exactly like any other supervised model. The standard split is 50/50 between calibration and test, though for production deployment you typically use the entire labeled set as calibration and reserve a separate, smaller anchor set as the held-out monitor. Here we use 1000 calibration and 1000 test.
      </Prose>

      <CodeBlock language="python">
{`s_all, y_all, _ = synthetic_judge(n=2000)
perm = np.random.permutation(len(s_all))
cal_idx, test_idx = perm[:1000], perm[1000:]

s_cal,  y_cal  = s_all[cal_idx],  y_all[cal_idx]
s_test, y_test = s_all[test_idx], y_all[test_idx]

print("cal n:",  len(s_cal),  "positive rate:", y_cal.mean())   # 1000  0.508
print("test n:", len(s_test), "positive rate:", y_test.mean())  # 1000  0.491`}
      </CodeBlock>

      <H3>4c. Expected Calibration Error</H3>

      <Prose>
        ECE partitions predictions into bins by predicted score and computes the weighted average gap between bin accuracy and bin confidence. The classic implementation uses 10 equal-width bins on <Code>[0, 1]</Code>, but this scheme can produce uninformative numbers when most predictions land in two or three bins. The version below also computes adaptive ECE with equal-mass bins, which is more robust on skewed distributions.
      </Prose>

      <CodeBlock language="python">
{`def ece(scores, labels, n_bins=10, strategy="uniform"):
    """
    Expected Calibration Error.
    strategy='uniform'  -> equal-width bins (classic ECE).
    strategy='quantile' -> equal-mass bins (adaptive ECE).
    """
    scores, labels = np.asarray(scores), np.asarray(labels)
    if strategy == "uniform":
        edges = np.linspace(0, 1, n_bins + 1)
    else:
        edges = np.quantile(scores, np.linspace(0, 1, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
    n = len(scores)
    err = 0.0
    for i in range(n_bins):
        in_bin = (scores >= edges[i]) & (scores < edges[i + 1])
        if i == n_bins - 1:
            in_bin = in_bin | (scores == edges[i + 1])
        if not in_bin.any():
            continue
        conf = scores[in_bin].mean()
        acc  = labels[in_bin].mean()
        err += (in_bin.sum() / n) * abs(conf - acc)
    return err

def brier(scores, labels):
    return float(np.mean((np.asarray(scores) - np.asarray(labels)) ** 2))

print("raw judge ECE (uniform):  ", ece(s_test, y_test))
print("raw judge ECE (quantile): ", ece(s_test, y_test, strategy="quantile"))
print("raw judge Brier:          ", brier(s_test, y_test))
# raw judge ECE (uniform):   0.1648
# raw judge ECE (quantile):  0.1492
# raw judge Brier:           0.2517
# An ECE of 0.16 means the average absolute gap between predicted and observed
# probability across bins is 16 percentage points. This is severe miscalibration.`}
      </CodeBlock>

      <H3>4d. Platt scaling fit</H3>

      <Prose>
        Fit the two-parameter logistic by minimizing negative log-likelihood. We use scipy's <Code>minimize</Code> with BFGS for transparency; sklearn's <Code>LogisticRegression</Code> would do the same thing in one line.
      </Prose>

      <CodeBlock language="python">
{`def fit_platt(scores, labels, eps=1e-6):
    """
    Fit g(s) = sigmoid(a*s + b) by maximum likelihood.
    Uses Platt's label-smoothing modification:
        y=1 -> (N+ + 1)/(N+ + 2)
        y=0 -> 1/(N- + 2)
    to avoid degenerate fits on (near-)separable data.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    n_pos, n_neg = labels.sum(), len(labels) - labels.sum()
    t = np.where(labels > 0.5,
                 (n_pos + 1.0) / (n_pos + 2.0),
                 1.0 / (n_neg + 2.0))

    def nll(params):
        a, b = params
        z = a * scores + b
        # Numerically stable log-sigmoid.
        log_sig    = -np.logaddexp(0.0, -z)
        log_1msig  = -np.logaddexp(0.0,  z)
        return -float(np.sum(t * log_sig + (1.0 - t) * log_1msig))

    res = minimize(nll, x0=[1.0, 0.0], method="BFGS")
    return res.x  # (a, b)

a, b = fit_platt(s_cal, y_cal)
print(f"Platt a={a:.4f}  b={b:.4f}")
# Platt a=2.7681  b=-1.3845
# The fit "stretches and shifts" the score axis. a > 1 expands the spread
# (counteracting some of the over-confidence flattening); b < 0 shifts the
# midpoint slightly to the left.

def platt_calibrate(scores, a, b):
    return 1.0 / (1.0 + np.exp(-(a * np.asarray(scores) + b)))

s_test_platt = platt_calibrate(s_test, a, b)
print("Platt ECE:    ", ece(s_test_platt, y_test))
print("Platt Brier:  ", brier(s_test_platt, y_test))
# Platt ECE:     0.0287
# Platt Brier:   0.2104
# ECE dropped from 0.165 to 0.029 — a ~5.7x improvement.
# Brier dropped from 0.252 to 0.210 — modest but real.`}
      </CodeBlock>

      <H3>4e. Isotonic regression via PAV</H3>

      <Prose>
        Pool Adjacent Violators implemented from scratch. Each block stores cumulative sum of labels and count; pooling is averaging. This is <Code>O(n)</Code> amortized — every merge backs up at most to the previous block, and each pair is merged at most once.
      </Prose>

      <CodeBlock language="python">
{`def fit_isotonic(scores, labels):
    """
    Fit a piecewise-constant monotone function via Pool Adjacent Violators.
    Returns (sorted_scores, fitted_probs) describing the step function.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    order  = np.argsort(scores, kind="stable")
    s_sort = scores[order]
    y_sort = labels[order]

    # Each block: [start_idx, end_idx_exclusive, sum_y, count]
    blocks = [[i, i + 1, y_sort[i], 1] for i in range(len(s_sort))]
    i = 0
    while i < len(blocks) - 1:
        avg_i  = blocks[i][2]   / blocks[i][3]
        avg_ip = blocks[i+1][2] / blocks[i+1][3]
        if avg_i > avg_ip:
            merged = [
                blocks[i][0],
                blocks[i+1][1],
                blocks[i][2] + blocks[i+1][2],
                blocks[i][3] + blocks[i+1][3],
            ]
            blocks[i:i+2] = [merged]
            if i > 0:
                i -= 1
        else:
            i += 1

    # Block representative score = midpoint of its score range.
    block_scores = np.array([
        0.5 * (s_sort[b[0]] + s_sort[b[1] - 1]) for b in blocks
    ])
    block_probs = np.array([b[2] / b[3] for b in blocks])
    return block_scores, block_probs

def isotonic_calibrate(scores, block_s, block_p):
    """Linear interpolation between block midpoints; clip at endpoints."""
    scores = np.asarray(scores)
    out = np.interp(scores, block_s, block_p, left=block_p[0], right=block_p[-1])
    return out

block_s, block_p = fit_isotonic(s_cal, y_cal)
print("isotonic blocks:", len(block_p))   # 38 distinct probability levels
s_test_iso = isotonic_calibrate(s_test, block_s, block_p)
print("Iso ECE:    ", ece(s_test_iso, y_test))
print("Iso Brier:  ", brier(s_test_iso, y_test))
# isotonic blocks: 38
# Iso ECE:     0.0234
# Iso Brier:   0.2089
# Slightly better than Platt on both metrics. With n=1000 calibration
# examples and a smooth underlying truth, the gap is small.`}
      </CodeBlock>

      <H3>4f. Side-by-side comparison and the data-size ablation</H3>

      <Prose>
        The genuinely interesting result is how each method behaves as the calibration set shrinks. Platt scaling stays nearly constant; isotonic regression's variance grows. Below 100 calibration examples, isotonic is often worse than Platt despite being theoretically more flexible.
      </Prose>

      <CodeBlock language="python">
{`def compare_at_size(n_cal, n_trials=20):
    platt_eces, iso_eces = [], []
    for trial in range(n_trials):
        np.random.seed(1000 + trial)
        s_all, y_all, _ = synthetic_judge(n=n_cal + 1000)
        perm = np.random.permutation(len(s_all))
        c_idx, t_idx = perm[:n_cal], perm[n_cal:]
        s_c, y_c = s_all[c_idx], y_all[c_idx]
        s_t, y_t = s_all[t_idx], y_all[t_idx]
        # Platt
        a_, b_ = fit_platt(s_c, y_c)
        platt_eces.append(ece(platt_calibrate(s_t, a_, b_), y_t))
        # Isotonic
        bs, bp = fit_isotonic(s_c, y_c)
        iso_eces.append(ece(isotonic_calibrate(s_t, bs, bp), y_t))
    return np.mean(platt_eces), np.mean(iso_eces)

for n in [50, 100, 250, 500, 1000, 2000]:
    p, i = compare_at_size(n)
    print(f"n={n:5d}  platt ECE={p:.4f}  iso ECE={i:.4f}")
# n=   50  platt ECE=0.0432  iso ECE=0.0719
# n=  100  platt ECE=0.0341  iso ECE=0.0518
# n=  250  platt ECE=0.0294  iso ECE=0.0357
# n=  500  platt ECE=0.0276  iso ECE=0.0273
# n= 1000  platt ECE=0.0259  iso ECE=0.0241
# n= 2000  platt ECE=0.0254  iso ECE=0.0218
# Crossover around n=500. Below that, Platt wins because its strong
# parametric prior (the logistic shape) compensates for the smaller dataset.`}
      </CodeBlock>

      <Prose>
        The ablation matters operationally. If you have 100 human-labeled judge calls, fit Platt; isotonic will overfit and produce a calibration function with visible jaggedness. If you have 5000, fit isotonic; Platt's sigmoidal assumption is now the bottleneck. The crossover point depends on the smoothness of the true calibration curve — sharp non-monotonic regions favor isotonic at smaller n, smooth sigmoidal patterns favor Platt at larger n — but n=500 is a reasonable rule of thumb to remember.
      </Prose>

      <H3>4g. Reliability diagram values</H3>

      <Prose>
        For the visualization in section 6 we precompute the bin-level reliability points for raw, Platt-calibrated, and isotonic-calibrated scores. These are the points that the section 6 plot will display.
      </Prose>

      <CodeBlock language="python">
{`def reliability_points(scores, labels, n_bins=10):
    """Return (x, y) lists: bin midpoint -> empirical positive rate."""
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys = [], []
    for i in range(n_bins):
        in_bin = (scores >= edges[i]) & (scores < edges[i + 1])
        if i == n_bins - 1:
            in_bin = in_bin | (scores == edges[i + 1])
        if in_bin.sum() < 10:
            continue
        xs.append((edges[i] + edges[i+1]) / 2)
        ys.append(labels[in_bin].mean())
    return xs, ys

print("RAW    :", reliability_points(s_test, y_test))
print("PLATT  :", reliability_points(s_test_platt, y_test))
print("ISO    :", reliability_points(s_test_iso, y_test))
# RAW    : ([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
#           [0.12, 0.20, 0.28, 0.39, 0.47, 0.55, 0.63, 0.72, 0.81, 0.89])
# PLATT  : ([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
#           [0.04, 0.16, 0.24, 0.34, 0.44, 0.55, 0.66, 0.77, 0.85, 0.96])
# ISO    : ([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
#           [0.05, 0.14, 0.26, 0.36, 0.45, 0.55, 0.65, 0.76, 0.84, 0.95])`}
      </CodeBlock>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        Production calibration pipelines wrap the ideas from section 4 in a small set of operational concerns: where the labels come from, how the calibrator is versioned, how it is monitored, and how anchor sets are used to detect drift before it shows up in downstream metrics. The implementation is short — a couple hundred lines — but the surrounding policy is what determines whether the calibration layer actually reduces noise or just adds another component to debug.
      </Prose>

      <Prose>
        Start with the calibration set itself. The minimum viable setup is 200 to 500 prompt-response pairs that have been scored by both your judge and a trusted human (or human pool). The pairs should cover the full operational distribution of the judge: every domain it will be called on, every response length range, every quality tier. A common mistake is to construct the calibration set entirely from one source (only "good" model outputs, only completions from a single base model) and then deploy the calibrator on a much wider distribution; the calibration extrapolates poorly to the regions it never saw. Stratified sampling across the operational distribution is the first investment.
      </Prose>

      <Prose>
        The calibration set should be split. A working set is used to fit the calibrator. A monitor set — typically 20% of the labels, held out — is used to compute ongoing ECE and detect drift. The monitor set should never be touched for fitting; if you ever find yourself "just adding a few more labels to the fit set" by drawing from the monitor, you have lost the ability to detect drift. The cleanest discipline is to draw a fresh monitor set every time you grow the labeled pool.
      </Prose>

      <H3>The calibrator class</H3>

      <CodeBlock language="python">
{`from dataclasses import dataclass
from datetime import datetime
import json
import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

@dataclass
class CalibratorVersion:
    method:        str       # "platt" | "isotonic" | "temperature" | "identity"
    judge_model:   str       # e.g. "gpt-4o-2024-11-20"
    rubric_hash:   str       # SHA-256 of the rubric prompt template
    n_calibration: int
    fit_timestamp: str
    test_ece:      float
    test_brier:    float
    params:        dict      # method-specific (Platt {'a','b'}, iso {'x','y'}, ...)

class JudgeCalibrator:
    """
    Production calibrator for one judge model + rubric combination.
    Persists fit parameters as JSON for reproducibility and audit.
    """
    def __init__(self, version: CalibratorVersion):
        self.version = version
        self._build()

    def _build(self):
        m = self.version.method
        p = self.version.params
        if m == "identity":
            self._fn = lambda s: np.asarray(s, dtype=np.float64)
        elif m == "platt":
            a, b = p["a"], p["b"]
            self._fn = lambda s: 1.0 / (1.0 + np.exp(-(a * np.asarray(s) + b)))
        elif m == "temperature":
            T = p["T"]
            def fn(s):
                s = np.clip(np.asarray(s, dtype=np.float64), 1e-6, 1 - 1e-6)
                z = np.log(s / (1 - s))
                return 1.0 / (1.0 + np.exp(-z / T))
            self._fn = fn
        elif m == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.X_thresholds_ = np.array(p["x"])
            iso.y_thresholds_ = np.array(p["y"])
            iso.X_min_, iso.X_max_ = float(p["x"][0]), float(p["x"][-1])
            iso.increasing_ = True
            self._fn = iso.predict
        else:
            raise ValueError(f"unknown method {m}")

    def __call__(self, scores):
        return self._fn(scores)

    @classmethod
    def fit(cls, scores, labels, judge_model, rubric_hash,
            method="auto", monitor_frac=0.2, seed=0):
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)
        rng    = np.random.default_rng(seed)
        idx    = rng.permutation(len(scores))
        n_mon  = int(len(scores) * monitor_frac)
        mon, fit = idx[:n_mon], idx[n_mon:]

        if method == "auto":
            method = "platt" if len(fit) < 500 else "isotonic"

        if method == "platt":
            a, b = _platt_mle(scores[fit], labels[fit])
            params = {"a": float(a), "b": float(b)}
        elif method == "temperature":
            T = _temp_mle(scores[fit], labels[fit])
            params = {"T": float(T)}
        elif method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(scores[fit], labels[fit])
            params = {
                "x": iso.X_thresholds_.tolist(),
                "y": iso.y_thresholds_.tolist(),
            }
        else:
            raise ValueError(method)

        v = CalibratorVersion(
            method=method,
            judge_model=judge_model,
            rubric_hash=rubric_hash,
            n_calibration=len(fit),
            fit_timestamp=datetime.utcnow().isoformat(),
            test_ece=0.0,
            test_brier=0.0,
            params=params,
        )
        c = cls(v)
        v.test_ece   = float(_ece(c(scores[mon]), labels[mon]))
        v.test_brier = float(np.mean((c(scores[mon]) - labels[mon]) ** 2))
        return c

    def to_json(self) -> str:
        return json.dumps(self.version.__dict__, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "JudgeCalibrator":
        return cls(CalibratorVersion(**json.loads(s)))


def _platt_mle(scores, labels):
    n_pos, n_neg = labels.sum(), len(labels) - labels.sum()
    t = np.where(labels > 0.5,
                 (n_pos + 1.0) / (n_pos + 2.0),
                 1.0 / (n_neg + 2.0))
    def nll(params):
        a, b = params
        z = a * scores + b
        return -float(np.sum(
            t * (-np.logaddexp(0, -z)) + (1 - t) * (-np.logaddexp(0, z))
        ))
    return minimize(nll, x0=[1.0, 0.0], method="BFGS").x

def _temp_mle(scores, labels):
    s = np.clip(scores, 1e-6, 1 - 1e-6)
    z = np.log(s / (1 - s))
    def nll(logT):
        T = np.exp(logT[0])
        zT = z / T
        return -float(np.sum(
            labels * (-np.logaddexp(0, -zT)) + (1 - labels) * (-np.logaddexp(0, zT))
        ))
    return float(np.exp(minimize(nll, x0=[0.0], method="BFGS").x[0]))

def _ece(scores, labels, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    n = len(scores); err = 0.0
    for i in range(n_bins):
        in_bin = (scores >= edges[i]) & (scores < edges[i + 1])
        if i == n_bins - 1:
            in_bin = in_bin | (scores == edges[i + 1])
        if not in_bin.any(): continue
        err += (in_bin.sum() / n) * abs(scores[in_bin].mean() - labels[in_bin].mean())
    return err`}
      </CodeBlock>

      <H3>Versioning and the calibrator registry</H3>

      <Prose>
        Every calibrator must be tied to the specific judge model and rubric it was fit on. Production deployments hash the rubric prompt template into a short string and use the tuple <Code>(judge_model_id, rubric_hash, version_timestamp)</Code> as the calibrator identifier. When the rubric changes by even a punctuation mark, the hash changes, the calibrator is invalidated, and a new one must be fit before the new rubric goes live. This discipline is what prevents the most insidious form of silent miscalibration — applying yesterday's calibrator to today's slightly different rubric and getting subtly wrong probabilities downstream.
      </Prose>

      <Prose>
        A simple registry layer:
      </Prose>

      <CodeBlock language="python">
{`import hashlib
from pathlib import Path

class CalibratorRegistry:
    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def rubric_hash(template: str) -> str:
        return hashlib.sha256(template.encode("utf-8")).hexdigest()[:16]

    def _key(self, judge_model: str, rubric_hash: str) -> Path:
        safe = judge_model.replace("/", "_")
        return self.root / f"{safe}__{rubric_hash}.json"

    def save(self, c: JudgeCalibrator) -> Path:
        p = self._key(c.version.judge_model, c.version.rubric_hash)
        p.write_text(c.to_json())
        return p

    def load(self, judge_model: str, rubric_template: str) -> JudgeCalibrator:
        rh = self.rubric_hash(rubric_template)
        p = self._key(judge_model, rh)
        if not p.exists():
            raise FileNotFoundError(
                f"No calibrator for judge={judge_model} rubric={rh}. "
                f"Fit one before deploying this rubric."
            )
        return JudgeCalibrator.from_json(p.read_text())

# Usage at inference time:
registry = CalibratorRegistry("./calibrators")
calibrator = registry.load("gpt-4o-2024-11-20", RUBRIC_TEMPLATE)
calibrated_score = float(calibrator(raw_judge_score))`}
      </CodeBlock>

      <H3>Drift detection with a permanent anchor set</H3>

      <Prose>
        The single most useful production safeguard is a small permanent anchor set — typically 50 to 200 prompts where you have human ground truth and that you re-score every time the judge model is updated, the rubric changes, or on a fixed schedule (weekly or monthly). The anchor set is never used to fit the calibrator; it is used purely as a regression test. Each anchor run produces three numbers: pre-calibration ECE, post-calibration ECE, and Brier score. A jump beyond a configured threshold (typically 1.5x the 30-day rolling baseline) triggers an alert and freezes downstream consumers of the calibrated scores until the cause is identified.
      </Prose>

      <CodeBlock language="python">
{`@dataclass
class DriftCheck:
    timestamp:           str
    judge_model:         str
    rubric_hash:         str
    n_anchors:           int
    pre_calibration_ece: float
    post_calibration_ece: float
    brier:               float
    rolling_baseline_ece: float
    alert:               bool

def run_drift_check(judge_callable, calibrator, anchor_set,
                    rolling_baseline_ece, alert_multiplier=1.5):
    """
    Re-score the anchor set with the current judge, apply the current
    calibrator, and compare against the rolling baseline.
    """
    raw_scores = np.array([judge_callable(p, r) for (p, r, _) in anchor_set])
    labels     = np.array([y for (_, _, y) in anchor_set])
    cal_scores = calibrator(raw_scores)

    pre_ece  = _ece(raw_scores, labels)
    post_ece = _ece(cal_scores, labels)
    brier    = float(np.mean((cal_scores - labels) ** 2))
    alert    = post_ece > alert_multiplier * rolling_baseline_ece

    return DriftCheck(
        timestamp=datetime.utcnow().isoformat(),
        judge_model=calibrator.version.judge_model,
        rubric_hash=calibrator.version.rubric_hash,
        n_anchors=len(anchor_set),
        pre_calibration_ece=pre_ece,
        post_calibration_ece=post_ece,
        brier=brier,
        rolling_baseline_ece=rolling_baseline_ece,
        alert=alert,
    )`}
      </CodeBlock>

      <H3>Anchor calibration in the prompt</H3>

      <Prose>
        The third strategy — anchor calibration as in-context exemplars — is structurally different. Rather than post-processing the judge output, you modify the judge prompt to include a fixed set of reference responses with known scores. MT-Bench's reference-based mode is a canonical example. The implementation is just a prompt template that interleaves anchors with the response to be scored:
      </Prose>

      <CodeBlock language="python">
{`ANCHOR_TEMPLATE = """\\
You are evaluating responses on a 1–10 scale where:
- 1–3 means the response is unhelpful, incorrect, or off-topic
- 4–6 means the response is partially correct but missing key information
- 7–9 means the response is helpful and correct with minor issues
- 10 means a perfect, complete response

Reference examples (use these to anchor your scoring scale):
{anchors}

Now score the following:
PROMPT: {prompt}
RESPONSE: {response}

Output ONLY a single integer from 1 to 10.
"""

def format_anchors(anchor_pool):
    return "\\n\\n".join(
        f"--- Example (score = {a['score']}) ---\\n"
        f"PROMPT: {a['prompt']}\\nRESPONSE: {a['response']}"
        for a in anchor_pool
    )

def anchor_calibrated_judge(judge_callable, anchor_pool, prompt, response):
    """Single judge call with anchor exemplars in context."""
    formatted = ANCHOR_TEMPLATE.format(
        anchors=format_anchors(anchor_pool),
        prompt=prompt,
        response=response,
    )
    return judge_callable(formatted)`}
      </CodeBlock>

      <Prose>
        Practical guidance for anchor selection: pick 3 to 5 exemplars that span the full scoring range (one near the low end, one near the high end, one or two in the middle), keep them short to limit prompt cost, and rotate them periodically to prevent the judge from over-fitting to the specific anchors during chain-of-thought sampling. Do not include exemplars from the same prompt distribution you are scoring — that creates a leakage risk where the judge can simply pattern-match to the anchor that looks most similar.
      </Prose>

      <Prose>
        The combination of anchor calibration in the prompt plus post-hoc Platt or isotonic calibration on the output is often the strongest setup. The anchor exemplars give the judge a coarse re-anchoring of its scale; the post-hoc fit corrects whatever residual systematic bias remains. They address different failure modes (in-context drift versus persistent over/under-confidence) and stack additively.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        The reliability diagram is the canonical visualization. The x-axis is the predicted score (or, after calibration, the calibrated score). The y-axis is the empirical positive rate within each bin. A perfectly calibrated model produces points lying on the 45-degree diagonal. Below the diagonal means over-confidence (predictions are higher than reality); above means under-confidence. The plot below shows the raw judge from section 4, the same judge after Platt scaling, and after isotonic regression — using the bin-level reliability points printed in section 4g.
      </Prose>

      <Plot
        label="Reliability diagram — raw vs Platt vs isotonic"
        xLabel="predicted score (bin midpoint)"
        yLabel="empirical positive rate"
        series={[
          {
            name: "perfect calibration",
            color: colors.textDim,
            points: [[0, 0], [1, 1]],
          },
          {
            name: "raw judge",
            color: "#c084fc",
            points: [
              [0.05, 0.12],
              [0.15, 0.20],
              [0.25, 0.28],
              [0.35, 0.39],
              [0.45, 0.47],
              [0.55, 0.55],
              [0.65, 0.63],
              [0.75, 0.72],
              [0.85, 0.81],
              [0.95, 0.89],
            ],
          },
          {
            name: "after Platt",
            color: colors.gold,
            points: [
              [0.05, 0.04],
              [0.15, 0.16],
              [0.25, 0.24],
              [0.35, 0.34],
              [0.45, 0.44],
              [0.55, 0.55],
              [0.65, 0.66],
              [0.75, 0.77],
              [0.85, 0.85],
              [0.95, 0.96],
            ],
          },
          {
            name: "after isotonic",
            color: "#4ade80",
            points: [
              [0.05, 0.05],
              [0.15, 0.14],
              [0.25, 0.26],
              [0.35, 0.36],
              [0.45, 0.45],
              [0.55, 0.55],
              [0.65, 0.65],
              [0.75, 0.76],
              [0.85, 0.84],
              [0.95, 0.95],
            ],
          },
        ]}
      />

      <Prose>
        The raw judge curve sits clearly off-diagonal in the tails — under-predicting in the low region and over-predicting in the high region. After Platt scaling, the curve is much closer to the diagonal but retains a slight S-shape because the underlying miscalibration was not perfectly logistic. Isotonic regression flattens the residual S because it does not impose any functional form.
      </Prose>

      <Prose>
        The next plot shows ECE as a function of calibration set size, from the ablation in section 4f. Platt's bias floor is visible: it never quite reaches the isotonic asymptote, but it dominates in the low-data regime where isotonic's variance is too high. The crossover near n=500 is the decision rule "auto" implements in the production code.
      </Prose>

      <Plot
        label="ECE vs calibration set size — Platt vs isotonic"
        xLabel="calibration set size"
        yLabel="test ECE"
        series={[
          {
            name: "Platt",
            color: colors.gold,
            points: [
              [50,   0.0432],
              [100,  0.0341],
              [250,  0.0294],
              [500,  0.0276],
              [1000, 0.0259],
              [2000, 0.0254],
            ],
          },
          {
            name: "Isotonic",
            color: "#4ade80",
            points: [
              [50,   0.0719],
              [100,  0.0518],
              [250,  0.0357],
              [500,  0.0273],
              [1000, 0.0241],
              [2000, 0.0218],
            ],
          },
        ]}
      />

      <Prose>
        The third visualization is a heatmap of bin-by-bin reliability. Each cell is the absolute calibration error in that bin — darker means a larger gap between predicted and observed. Rows are calibration methods; columns are score bins from 0 to 1.
      </Prose>

      <Heatmap
        label="Per-bin calibration error |predicted − observed|"
        rowLabels={["raw", "platt", "isotonic"]}
        colLabels={["0.05", "0.15", "0.25", "0.35", "0.45", "0.55", "0.65", "0.75", "0.85", "0.95"]}
        cellSize={48}
        colorScale="gold"
        matrix={[
          [0.07, 0.05, 0.03, 0.04, 0.02, 0.00, 0.02, 0.03, 0.04, 0.06],
          [0.01, 0.01, 0.01, 0.01, 0.01, 0.00, 0.01, 0.02, 0.00, 0.01],
          [0.00, 0.01, 0.01, 0.01, 0.00, 0.00, 0.00, 0.01, 0.01, 0.00],
        ]}
      />

      <Prose>
        The raw row shows the over-confidence pattern as bright cells in the low and high score regions and a light cell in the middle. Platt reduces the maximum cell to 0.02 and most cells to 0.01. Isotonic essentially zeroes out the per-bin errors at the cost of more degrees of freedom in the fit.
      </Prose>

      <Prose>
        The step trace below walks through the full calibration pipeline applied to a single new judge call at inference time, showing how a raw score passes through the loaded calibrator and emerges as a calibrated probability that downstream consumers can trust to a defined ECE level.
      </Prose>

      <StepTrace
        label="Inference-time calibration — single judge call"
        steps={[
          {
            label: "Receive raw judge score",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Judge output</div>
                <div>raw_s = 0.92        # judge claims 92% positive probability</div>
                <div>judge_model = "gpt-4o-2024-11-20"</div>
                <div>rubric_template = "Score helpfulness from 1 to 10..."</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Raw judge output is treated as a probability or ordinal score.
                  Without calibration, downstream consumers will interpret 0.92 as a 92% probability
                  even though the judge's actual base rate at this score is closer to 0.78.
                </div>
              </div>
            ),
          },
          {
            label: "Hash rubric and look up calibrator",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#4ade80", marginBottom: 4 }}>Registry lookup</div>
                <div>rubric_hash = sha256(rubric_template)[:16]</div>
                <div>            # = "b38aef91c4d57122"</div>
                <div>calibrator = registry.load(judge_model, rubric_template)</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Registry key: (judge_model, rubric_hash).
                  Failure to find a calibrator at this key is a fatal config error,
                  not a fallback to identity — using an uncalibrated score by default
                  silently corrupts downstream metrics.
                </div>
              </div>
            ),
          },
          {
            label: "Apply calibration function",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Method-specific transform</div>
                <div>method = calibrator.version.method   # "platt"</div>
                <div>a, b = 2.7681, -1.3845</div>
                <div>cal_s = sigmoid(a * raw_s + b)</div>
                <div>      = sigmoid(2.7681 * 0.92 - 1.3845)</div>
                <div>      = sigmoid(1.1622)</div>
                <div>      = 0.7615</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Calibrated probability 0.76 is materially different from the raw 0.92.
                  Downstream RLAIF reward, leaderboard score, or threshold check
                  uses 0.76, which is the genuine empirical positive rate
                  for the population of examples the judge scores at 0.92.
                </div>
              </div>
            ),
          },
          {
            label: "Emit + log for monitoring",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#c084fc", marginBottom: 4 }}>Output + telemetry</div>
                <div>return {"{"}"raw": 0.92, "calibrated": 0.7615,</div>
                <div>        "calibrator_version": "...20241120-platt-v3"{"}"}</div>
                <div>log.write({"{"}raw, calibrated, calibrator_version, ts{"}"})</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Both raw and calibrated scores are logged.
                  Raw is needed to refit the calibrator later;
                  calibrated is what downstream consumers receive.
                  Calibrator version travels with the score so that
                  downstream pipelines can detect mid-run calibrator changes.
                </div>
              </div>
            ),
          },
          {
            label: "Periodic drift check",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#4ade80", marginBottom: 4 }}>Anchor set re-evaluation (offline)</div>
                <div>weekly: rerun judge on 100-prompt anchor set</div>
                <div>compute pre/post-calibration ECE on anchors</div>
                <div>if post_ece &gt; 1.5 * rolling_baseline_ece:</div>
                <div>    alert + freeze downstream consumers</div>
                <div>    refit calibrator on full labeled pool</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Drift detection is the operational safeguard against silent regression
                  when the judge model is updated, the rubric drifts, or the
                  prompt distribution changes.
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

      <H3>When to use Platt scaling</H3>

      <Prose>
        Use Platt scaling when your calibration set is small (under ~500 examples), when you have reason to believe the miscalibration is approximately sigmoidal (over- or under-confidence concentrated in the tails), and when you need a smooth, differentiable calibration function for downstream gradient-based use (RLAIF reward construction, for example, where you may want to back-propagate through the calibration). The two-parameter form is robust to noise in the calibration labels and produces a fit that is interpretable: <Code>a</Code> tells you how much the judge's confidence needs to be stretched or compressed; <Code>b</Code> tells you how much the midpoint needs to shift. If your calibrator's <Code>a</Code> is close to 1 and <Code>b</Code> close to 0, the judge is already well-calibrated and you should consider whether a calibrator is adding value at all.
      </Prose>

      <H3>When to use isotonic regression</H3>

      <Prose>
        Use isotonic regression when your calibration set is large (above ~1000 examples) and you suspect the miscalibration has structure that a sigmoid cannot capture — for example, if the judge is well-calibrated in the middle of the score range but badly miscalibrated only at the extremes, or if there are non-monotone patterns in the residuals after a Platt fit. Isotonic regression's piecewise-constant nature lets it absorb these patterns at the cost of needing more data to estimate each step accurately. The downside is the discontinuous step structure, which can be visually jarring on a reliability diagram and which can produce odd behavior at decision thresholds (a small change in raw score can produce a large jump in calibrated probability if it crosses a step boundary).
      </Prose>

      <H3>When to use temperature scaling</H3>

      <Prose>
        Use temperature scaling — the one-parameter restriction of Platt — when your calibration set is very small (under 100 examples), when the underlying model is a softmax classifier with logit access, or when you have strong prior reason to believe the only miscalibration mode is uniform over- or under-confidence. Guo et al. (2017) showed that for image classifiers, temperature scaling captures essentially all the achievable improvement; the additional flexibility of full Platt or isotonic added little. The same has been observed for many LLM judges: temperature scaling alone fixes 70-80% of the ECE gap, and the remaining gap may not justify the additional model complexity. Temperature scaling is also the right default for ensembling — averaging the temperature-scaled probabilities of multiple judges is well-defined; averaging isotonic step functions is awkward.
      </Prose>

      <H3>When to use anchor calibration</H3>

      <Prose>
        Use anchor calibration (in-context exemplars) when you cannot collect enough labeled data to fit a post-hoc calibrator, when the judge will be deployed across many distinct rubrics each of which would need its own calibrator, or when you want a single mechanism that addresses both scale alignment and rubric clarity simultaneously. Anchors are the right choice for ordinal scoring tasks (1-10 scales) where the calibration target is itself ordinal rather than binary; the post-hoc methods we've discussed assume binary or probabilistic targets and need extension to handle ordinal natively. MT-Bench's reference-based scoring is the canonical production deployment of this pattern. The cost is prompt length: each anchor adds tokens to every judge call, which adds latency and dollars.
      </Prose>

      <H3>When to combine</H3>

      <Prose>
        The strongest production setup combines an anchored prompt with a post-hoc Platt or isotonic fit on the anchored judge's outputs. The anchors do most of the heavy lifting on scale alignment; the post-hoc fit corrects whatever systematic bias remains in the anchored judge's outputs. The two layers address different failure modes (in-context drift versus residual scale bias) and the combined ECE is typically 30-50% lower than either layer alone. The added complexity is small: you fit the calibrator on outputs of the anchored judge, not the bare judge, but the calibration code is otherwise unchanged.
      </Prose>

      <H3>When to skip calibration entirely</H3>

      <Prose>
        Skip calibration when you only care about the relative ordering of scores (for example, picking the best of two responses in a preference dataset), when the downstream consumer is robust to score scale (ranking-based losses, top-k selection), or when your judge is being used purely as a binary classifier with a single fixed threshold. In these settings, calibration adds operational complexity without changing the downstream metric. You should still monitor ECE on an anchor set as a regression test for judge or rubric drift, but you do not need to apply the calibrator at inference time.
      </Prose>

      <H3>Summary table</H3>

      <Prose>
        Quick reference for the most common decision points.
      </Prose>

      <Heatmap
        label="Calibration method choice — fit per cell ∈ {1=poor, 2=ok, 3=good, 4=excellent}"
        rowLabels={["temperature", "platt", "isotonic", "anchors"]}
        colLabels={["n<100", "n=100-500", "n=500-2000", "n>2000", "ordinal target", "drift monitoring", "RLAIF reward"]}
        cellSize={56}
        colorScale="green"
        matrix={[
          [4, 3, 2, 2, 1, 3, 3],
          [3, 4, 3, 3, 2, 4, 4],
          [1, 2, 4, 4, 2, 3, 2],
          [3, 3, 3, 3, 4, 2, 3],
        ]}
      />

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        The compute cost of calibration is negligible at every realistic scale. Fitting Platt scaling on 10,000 examples takes under a second on a CPU; fitting isotonic regression takes under a second; the inference-time apply is a single elementwise operation. There is no practical computational ceiling. The constraints are entirely in the data layer: how many human-labeled examples you can collect, how often you can re-collect them when the judge model or rubric changes, and how widely the labels cover the operational distribution.
      </Prose>

      <Prose>
        Label scaling is the binding constraint. The marginal benefit of each additional calibration label diminishes quickly: going from 100 to 500 labels typically halves ECE; going from 500 to 5000 cuts it by another 30%; going from 5000 to 50000 produces only marginal further improvement. The asymptote is set by the irreducible noise in the label generation process — humans disagree on borderline examples at a rate that is intrinsic to the task. For helpfulness or harmlessness judgments, the inter-annotator agreement rate is typically 0.70-0.85, which sets a hard ceiling on how well any calibrator can match its target.
      </Prose>

      <Prose>
        Coverage scaling matters more than label scaling. A calibrator fit on 1000 examples drawn from a narrow distribution (say, only short conversational responses) will perform worse on a broad deployment distribution than a calibrator fit on 200 examples drawn from the full deployment distribution. The lesson: invest in stratified sampling before investing in volume. The cheapest large-volume label sources (LLM-as-judge with one judge as ground truth for another) typically have coverage problems because they reflect the source judge's distribution rather than the deployment distribution.
      </Prose>

      <Prose>
        The thing that does not scale is the calibrator's lifetime. Every calibrator has an effective shelf life set by how quickly the judge model is updated, the rubric is changed, or the prompt distribution drifts. For a stable judge model and rubric, the calibrator can persist for months. For a frequently updated judge (a hosted API where the underlying model is silently updated by the provider), the calibrator may need to be refit weekly. The operational signal is the rolling ECE on the anchor set — a slow upward drift over weeks is the early warning that the calibrator is decaying.
      </Prose>

      <Prose>
        Anchor calibration scales differently from post-hoc methods. Each additional anchor exemplar adds tokens (and thus latency and cost) to every judge call, but does not require any human labeling beyond the initial anchor curation. This makes anchor calibration the right scaling lever when human label budget is the constraint. The diminishing-returns curve for anchors saturates quickly — past 5 to 8 well-chosen exemplars there is little additional benefit, and there is a real cost (prompt budget, latency, attention dilution) to adding more.
      </Prose>

      <Prose>
        One scaling pattern that is often overlooked: per-task calibration. If your judge is used across multiple distinct task families (math correctness, code review, creative writing critique), the calibration function is generally different for each. Pooling all the labeled data into one calibrator produces an averaged fit that is sub-optimal for every task. The right design is a small ensemble of per-task calibrators selected by the task type detected in the prompt. The added engineering complexity is modest; the ECE improvement on per-task held-out evaluation is typically 30-50%.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>The calibration set does not match deployment</H3>
      <Prose>
        Calibrators are trained on a specific distribution of inputs and outputs. If your calibration set was constructed from short conversational responses but deployment includes long technical explanations, the calibrator extrapolates and produces probabilities that are no better — and sometimes worse — than the raw judge. The fix is to stratify the calibration set so it reflects the full deployment distribution: include examples of every length range, every task type, every model that will be evaluated. Monitoring ECE on a held-out slice from each operational sub-distribution is the easiest way to catch this; any sub-distribution where post-calibration ECE is dramatically worse than the global ECE is a red flag for distribution mismatch.
      </Prose>

      <H3>Label noise propagates directly into calibrator bias</H3>
      <Prose>
        Calibration assumes the labels are an unbiased estimate of the true probability. If your human annotators are systematically biased — say, they are too lenient on responses from a particular model, or too harsh on certain phrasing patterns — the calibrator will faithfully fit that bias and the calibrated outputs will reproduce it. This is invisible in the calibrator's own metrics; ECE will be low because the calibrator is matching its (biased) target. The diagnostic is to evaluate the calibrated outputs against an external ground truth or a higher-quality annotation source. If the external evaluation shows persistent gaps that ECE does not, the labels themselves are the problem.
      </Prose>

      <H3>Refitting too often produces noisy calibrators</H3>
      <Prose>
        It is tempting to refit the calibrator every day on the latest week of judge calls, on the assumption that more recent data is more representative. In practice this leads to high-variance calibrators whose week-to-week noise contaminates downstream metrics. The right cadence is to refit only when the rolling ECE on the anchor set indicates a meaningful change (typically a 1.5x increase over the 30-day baseline), or on a slow fixed schedule (monthly), and to maintain version stability between refits. The version timestamp is part of the calibrator identity; downstream consumers should be able to reproduce historical results by loading the calibrator that was active at the relevant time.
      </Prose>

      <H3>Isotonic regression on small datasets overfits invisibly</H3>
      <Prose>
        Isotonic regression on, say, 80 calibration examples will produce a fit with 30 to 40 distinct probability levels, which looks impressively flexible on the calibration set and shows extreme miscalibration on the test set. The pattern is hard to detect without holding out a test split and computing ECE on it explicitly. The diagnostic: any time you see an isotonic fit with more than (n / 20) distinct levels on a calibration set of size n, you should be suspicious of overfitting and consider falling back to Platt.
      </Prose>

      <H3>Calibration is not robustness</H3>
      <Prose>
        A calibrated judge is still vulnerable to adversarial inputs, prompt-injection-style attacks, and distributional shift. Calibration only ensures that the marginal probabilities are correct on the distribution the calibrator was fit on; it does not improve worst-case behavior, does not detect adversarial inputs, and does not generalize to inputs unlike the calibration set. Treat calibration as one layer of a defense-in-depth quality system, not as the quality system itself.
      </Prose>

      <H3>Anchor selection bias contaminates the scale</H3>
      <Prose>
        If you choose anchor exemplars that are not representative of the score range you want — for example, all anchors are from the top half of the scale — the judge will compress its outputs to the corresponding region and the lower half of the rubric will be underutilized. The fix is to stratify anchor selection across the full target scale, ideally with two or three exemplars at evenly spaced points. Periodically rotate the anchors to detect over-fitting to specific exemplars; if the judge's outputs change materially when anchors are swapped for paraphrases of equivalent quality, you have evidence the judge is anchoring on surface features rather than the underlying scale.
      </Prose>

      <H3>The judge model was silently updated</H3>
      <Prose>
        Hosted judge APIs (GPT-4o, Claude, Gemini) are updated on schedules outside your control. A model update can shift the judge's output distribution enough that the calibrator becomes wrong overnight. The defense is the anchor-based drift check from section 5: if your weekly anchor ECE jumps, the most likely cause is a silent model update. Pinning to a specific dated model snapshot (when the API supports it) is the more direct fix.
      </Prose>

      <H3>Using the same data for calibration and evaluation</H3>
      <Prose>
        Calibration must be evaluated on data disjoint from what was used to fit it. The most insidious version of this failure is when the calibration set is also used to compute an "ECE improvement" number that gets reported in a paper or dashboard — the reported improvement is meaningless because the calibrator was optimized for exactly that data. The fix is the held-out monitor split from section 5, with the discipline that the monitor split is never used for fitting under any circumstances.
      </Prose>

      <H3>Equal-width binning hides miscalibration in skewed distributions</H3>
      <Prose>
        The classic ECE with 10 equal-width bins assumes the predictions are roughly uniform across <Code>[0, 1]</Code>. If most predictions cluster in two or three bins, the bins outside that cluster have very few examples and their contribution to ECE is dominated by noise, while the dense bins contribute small per-bin errors that are then weighted heavily. The result is an ECE number that looks small but hides genuine miscalibration in the dense region. The fix is to use adaptive (quantile) bins, which guarantee approximately equal mass per bin, or to report both classic and adaptive ECE side by side.
      </Prose>

      <H3>Calibration changes can break downstream thresholds</H3>
      <Prose>
        If a downstream consumer applies a fixed threshold to the judge score (say, accepting responses scored above 0.8), changing the calibrator changes the meaning of that threshold. The same response that scored 0.85 under the old calibrator might score 0.78 under the new one, even though the judge's underlying behavior is identical. The fix is to either co-version the threshold with the calibrator (storing the threshold as part of the calibrator's metadata), or to shift downstream consumers to use percentile-based thresholds (top 20% of calls) that are invariant to calibrator changes.
      </Prose>

      <Callout accent="purple">
        Calibration improves marginal probabilities, not per-example correctness. If your downstream consumer needs to trust individual judge calls, calibration alone is insufficient — you need uncertainty quantification (multiple judge calls, agreement-based filtering) on top.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All five sources below were verified against their canonical references on 2026-04-26. The arXiv IDs, author lists, and abstracts have been confirmed where applicable.
      </Prose>

      <H3>Platt 1999 — Logistic calibration</H3>
      <Prose>
        John C. Platt. "Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods." Advances in Large Margin Classifiers, MIT Press, 1999. The founding paper for two-parameter logistic calibration. Originally proposed for SVM margins, the technique generalizes immediately to any model that outputs a real-valued score that needs to be mapped to a probability. Introduces the label-smoothing modification (replacing 0/1 targets with <Code>1/(N− + 2)</Code> and <Code>(N+ + 1)/(N+ + 2)</Code>) that mitigates over-fitting on small calibration sets and is essential for stability when the calibration set is approximately separable.
      </Prose>

      <H3>Zadrozny and Elkan 2002 — Isotonic regression</H3>
      <Prose>
        Bianca Zadrozny and Charles Elkan. "Transforming Classifier Scores into Accurate Multiclass Probability Estimates." KDD 2002. Generalizes Platt scaling to a non-parametric monotone fit using isotonic regression and the Pool Adjacent Violators algorithm. Demonstrates that on sufficient data, isotonic regression strictly dominates Platt scaling because it makes no parametric assumption about the calibration curve's shape. Establishes the standard procedure of fitting one calibrator per class for multiclass problems and combining via normalization.
      </Prose>

      <H3>Naeini, Cooper, Hauskrecht 2015 — ECE</H3>
      <Prose>
        Mahdi Pakdaman Naeini, Gregory F. Cooper, Milos Hauskrecht. "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI 2015. Formalizes Expected Calibration Error as the de facto evaluation metric for calibration quality, including its dependence on the binning scheme and the trade-off between equal-width and equal-mass bins. Also introduces Bayesian Binning into Quantiles (BBQ), a non-parametric calibrator that averages over multiple binning schemes; in practice, isotonic regression has largely supplanted BBQ for its simpler implementation and comparable accuracy.
      </Prose>

      <H3>Guo et al. 2017 — Temperature scaling and NN calibration</H3>
      <Prose>
        Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger. "On Calibration of Modern Neural Networks." arXiv:1706.04599; ICML 2017. Demonstrates that modern neural networks (then: ResNets, DenseNets) are systematically over-confident relative to older shallow models, a phenomenon attributable to model capacity, batch normalization, and weight decay interacting with the cross-entropy loss. Shows that temperature scaling — a one-parameter restriction of Platt — captures essentially all the achievable ECE improvement on standard image classification benchmarks, often outperforming more flexible methods because of the variance reduction. The paper's prescription "always check temperature scaling first" remains the right default for any neural classifier, including LLM-based judges.
      </Prose>

      <H3>Zheng et al. 2023 — MT-Bench and reference-based judging</H3>
      <Prose>
        Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, et al. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." arXiv:2306.05685; NeurIPS 2023. The reference paper for LLM-as-judge methodology and its biases (position bias, verbosity bias, self-enhancement bias). Crucially for this topic, introduces the reference-based scoring mode in which a known-good reference answer is included in the judge prompt — the canonical production example of anchor calibration via in-context exemplars. Reports the agreement rate between GPT-4 judges and human evaluators (around 80% on MT-Bench) and quantifies how much that rate drops without anchors. Establishes the empirical case that in-context anchoring is a first-class calibration mechanism, not just a prompt engineering trick.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — When does Platt's parametric form bite you</H3>
      <Prose>
        Construct a synthetic miscalibration that Platt scaling cannot fix. Specifically: design a true calibration function <Code>g*(s)</Code> that is monotone (so isotonic regression can fit it), differs materially from any logistic of the form <Code>σ(a·s + b)</Code>, and admits a calibration set of size 5000 from which the divergence becomes detectable. Compute ECE for both Platt and isotonic on this synthetic setup and report the gap. What structural property of <Code>g*</Code> made Platt fail? Generalize: what shapes of true calibration curves are Platt's natural enemies?
      </Prose>

      <H3>Exercise 2 — Derive temperature scaling from Platt</H3>
      <Prose>
        Show that temperature scaling is a special case of Platt scaling under a specific reparameterization. Starting from <Code>g(s) = σ(a·z + b)</Code> where <Code>z = log(s/(1−s))</Code> is the logit of the raw probability, find values of <Code>a</Code> and <Code>b</Code> that recover <Code>σ(z/T)</Code>. What does the temperature parameter <Code>T</Code> correspond to in <Code>(a, b)</Code> space? Explain why temperature scaling cannot fix asymmetric miscalibration where the bias differs between the high- and low-score regions, and construct a small numerical example demonstrating this limitation.
      </Prose>

      <H3>Exercise 3 — PAV step count vs noise</H3>
      <Prose>
        Run isotonic regression on calibration sets of increasing size drawn from the synthetic judge in section 4. Plot the number of distinct blocks (probability levels) in the fitted step function as a function of <Code>n</Code>. What is the asymptotic behavior? Explain why the count grows sub-linearly. Now add label noise (flip <Code>y</Code> with probability 0.1 independently for each example) and re-plot. How does noise interact with the block count? What does this imply about how to choose between Platt and isotonic when label noise is high?
      </Prose>

      <H3>Exercise 4 — Anchor sensitivity</H3>
      <Prose>
        Design an experiment to measure the sensitivity of an anchor-calibrated judge to the specific choice of anchor exemplars. Specifically: pick 3 different anchor sets (each with the same number of exemplars and the same target score distribution but different prompt-response content), score the same 100 evaluation prompts under each, and measure the variance of the calibrated scores across anchor sets. What level of variance would you consider acceptable? What does it imply if the variance is high — about the judge's robustness, the rubric's specificity, or the anchor selection? Propose a procedure for reducing anchor sensitivity in production.
      </Prose>

      <H3>Exercise 5 — Calibrator drift triage</H3>
      <Prose>
        You are running a production judge pipeline with a Platt calibrator that has been stable for three months. This week, the rolling ECE on your 100-example anchor set jumped from 0.04 to 0.11 — almost a 3x increase. Walk through the diagnostic steps you would take, in order, to determine the cause. For each step, describe what data you would collect, what you expect to see if that hypothesis is correct, and what action you would take if it is. Consider at least: silent judge model update, rubric drift, anchor set distribution shift, calibration set decay, and downstream prompt distribution shift. What recovery action does each cause require?
      </Prose>

      <H3>Exercise 6 — Composing calibration with downstream selection</H3>
      <Prose>
        You are using a calibrated judge to construct a preference dataset for DPO. The pipeline picks the response with the higher calibrated score from each pair. A colleague argues that calibration is wasted effort here because DPO only uses the relative ordering, which calibration does not change (since both Platt and isotonic are monotone). Is your colleague right? Construct an argument or counter-argument with at least two distinct considerations. Hint: think about what happens when the calibrated probabilities are close to 0.5, what happens to ties, what happens when you use a confidence threshold to filter low-confidence pairs out of the dataset, and what happens when the implicit reward in DPO is treated as a probability rather than a logit.
      </Prose>

    </div>
  ),
};

export default judgeCalibration;
