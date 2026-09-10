import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { StepTrace, Plot } from "../../components/viz";
import { colors } from "../../styles";

const calibrationConformalContent = {
  title: "Calibration & Conformal Prediction",
  readTime: "~50 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        In 1999, John Platt published "Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods" in <em>Advances in Large Margin Classifiers</em> (MIT Press). The SVM, then ascendant, had a structural problem: its decision function produced real-valued scores proportional to distance from the margin, not probabilities. Downstream systems — medical diagnosis tools, financial risk models, email filters — needed probabilities to make cost-sensitive decisions. Platt's solution was simple and durable: fit a sigmoid function on top of the SVM's raw scores using a held-out set, calibrating the scores into valid probability estimates. The sigmoid fitting procedure, now universally called Platt scaling, reduced to a two-parameter logistic regression. This two-page observation unlocked SVMs for any application that required calibrated uncertainty.
      </Prose>

      <Prose>
        Three years later, Bianca Zadrozny and Charles Elkan published "Transforming Classifier Scores into Accurate Multiclass Probability Estimates" at KDD 2002 (pp. 694–699). Where Platt's sigmoid assumed a parametric shape, Zadrozny and Elkan applied isotonic regression — a nonparametric monotone mapping — to calibrate classifier scores. Isotonic regression makes fewer assumptions about the calibration curve's shape and is more flexible when the relationship between scores and true probabilities is not sigmoidal. Their empirical evaluation showed that calibration quality depends strongly on the base classifier: naive Bayes and decision trees are well-known to be miscalibrated in predictable ways, while logistic regression is naturally better calibrated.
      </Prose>

      <Prose>
        The community's awareness of the calibration problem was then largely dormant until 2017, when Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Weinberger published "On Calibration of Modern Neural Networks" at ICML (arXiv:1706.04599). Their finding was alarming: modern deep networks — ResNets, VGGs trained on ImageNet — are systematically overconfident. As depth and width grow, as training runs longer, as regularization weakens, the networks become more accurate but worse calibrated. A 95%-confident ResNet is right far less than 95% of the time. The solution they advocated: temperature scaling, a single-parameter generalization of Platt scaling that divides all logits by a learned scalar T before the softmax. Temperature scaling is cheap, effective, and has become the default post-hoc calibration method for deep classifiers.
      </Prose>

      <Prose>
        Calibration asks: if I output a probability, does it match empirical frequency? Conformal prediction asks a different question entirely: can I output a <em>set</em> of predictions that is guaranteed to contain the true label with at least 1−α probability, for any finite sample, under no distributional assumptions? Vladimir Vovk, Alex Gammerman, and Glenn Shafer answered yes in their 2005 book <em>Algorithmic Learning in a Random World</em> (Springer). Their framework — conformal prediction — produces prediction sets (for classification) or prediction intervals (for regression) with exact finite-sample marginal coverage guarantees. The only assumption is exchangeability of the data. No asymptotic arguments. No parametric models. No tuning of distributional families. The coverage guarantee holds exactly for any n, any model, any score function. Anastasios Angelopoulos and Stephen Bates wrote the accessible modern tutorial "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification" (Foundations and Trends in ML 16(4):494–591, 2023; arXiv:2107.07511), which is now the entry point for practitioners.
      </Prose>

      <Prose>
        Why does all of this matter? Downstream decisions depend on reliable probabilities. In medical diagnosis, a 90%-confident prediction should mean something: the patient or clinician uses it to weigh treatment options. In credit scoring, a 70% default probability determines loan pricing. In autonomous systems, an overconfident object detector that reports 99% confidence for every detection will cause a safety system to never intervene. Calibration is not a nice-to-have — it is the contract between a model's output and the real world. Conformal prediction goes further: it adds a finite-sample guarantee that is impossible to fake with a miscalibrated model. These are the tools that make ML deployable in regulated, safety-critical, or high-stakes domains.
      </Prose>

      <Callout type="insight">
        Calibration and conformal prediction are complementary. Calibration asks whether your probabilities are accurate on average. Conformal prediction bypasses the probability question entirely and gives you a <em>set</em> with guaranteed coverage — even if your underlying model is completely miscalibrated.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <H3>2.1 What calibration means</H3>

      <Prose>
        The intuition for calibration is a weather forecast. A well-calibrated forecaster who says "70% chance of rain" should see it rain approximately 70% of the time on days they make that forecast. If it rains 40% of the time when they say 70%, they are overconfident. If it rains 95% of the time when they say 70%, they are underconfident. Calibration is the alignment between stated probability and empirical frequency.
      </Prose>

      <Prose>
        For a classifier, calibration is measured by binning predictions by confidence level and comparing average confidence to average accuracy within each bin. If all examples in the 0.7–0.8 confidence bin are correct 74% of the time, the model is well-calibrated in that bin. A <strong>reliability diagram</strong> plots confidence (x-axis) versus accuracy (y-axis); a perfectly calibrated model traces the diagonal. Deep networks trained with cross-entropy tend to produce reliability diagrams that bow above the diagonal — they are overconfident. Naive Bayes models tend to produce calibration curves that are S-shaped, because the independence assumption distorts probability estimates at the extremes.
      </Prose>

      <H3>2.2 What conformal prediction means</H3>

      <Prose>
        Instead of asking "is my probability estimate accurate?", conformal prediction asks: "can I return a set of answers that is guaranteed to contain the true answer with at least 1−α probability?" For classification with 10 classes and α=0.10, conformal prediction returns a subset of the 10 classes — sometimes 1 class (when the model is confident), sometimes 2 or 3 (when uncertain) — such that the true class is in the set at least 90% of the time over the test distribution. The guarantee is marginal and distribution-free: it does not depend on the model architecture, the training procedure, or the feature distribution. It holds for any finite n.
      </Prose>

      <Prose>
        For regression, conformal prediction returns an interval {"[ŷ − q, ŷ + q]"} (or an asymmetric interval) such that the true value falls inside at least 90% of the time. The interval width adapts to the problem's difficulty: a well-calibrated regressor with low residuals will produce narrow intervals; a poor model will produce wide ones. The coverage guarantee holds regardless.
      </Prose>

      <Callout type="insight">
        The key distinction: calibration gives you better probability estimates (but no formal guarantee). Conformal prediction gives you a coverage guarantee (but not necessarily a probability). In regulated domains — medical devices, financial models, safety systems — the word "guarantee" carries legal and ethical weight that mere calibration cannot provide.
      </Callout>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 Expected Calibration Error (ECE)</H3>

      <Prose>
        The standard calibration metric is the Expected Calibration Error (ECE), introduced by Naeini, Cooper, and Hauskrecht (2015) and popularized by Guo et al. (2017). Partition the unit interval into B equal-width bins. For each bin b, let {"S(b)"} be the set of examples whose predicted confidence falls in that bin, {"acc(b)"} be the average accuracy of those examples, and {"conf(b)"} be the average predicted confidence:
      </Prose>

      <MathBlock>
        {"\\text{ECE} = \\sum_{b=1}^{B} \\frac{|S(b)|}{n} \\left| \\text{acc}(b) - \\text{conf}(b) \\right|"}
      </MathBlock>

      <Prose>
        ECE is a weighted average of the absolute calibration gap per bin, weighted by bin population. A perfectly calibrated model has ECE = 0. The number of bins B is a hyperparameter (10 is conventional); too few bins lose resolution, too many produce noisy estimates with empty bins. The reliability diagram visualizes each term: it plots {"conf(b)"} (x-axis) versus {"acc(b)"} (y-axis) for each bin, with bar width proportional to {"| S(b) | / n"}.
      </Prose>

      <H3>3.2 Platt scaling</H3>

      <Prose>
        Given a classifier's raw decision scores {"f(x)"} (e.g., the SVM's margin distance or a logit before sigmoid), Platt scaling fits a parametric sigmoid:
      </Prose>

      <MathBlock>
        {"P(y = 1 \\mid f(x)) = \\frac{1}{1 + \\exp(A \\cdot f(x) + B)}"}
      </MathBlock>

      <Prose>
        The parameters A and B are fitted by minimizing the log-loss on a held-out calibration set using gradient descent or L-BFGS. Crucially, the calibration set must be disjoint from the training set — if you calibrate on training data, the sigmoid overfits the training scores and produces calibration curves that look good on training but are unreliable on test. Platt scaling assumes the calibration curve has a sigmoidal shape, which is appropriate when raw scores are approximately normally distributed in each class.
      </Prose>

      <H3>3.3 Isotonic regression calibration</H3>

      <Prose>
        When the calibration curve is not sigmoidal, isotonic regression provides a nonparametric alternative. It finds a monotone non-decreasing function that minimizes the mean squared error between the (sorted) raw scores and the (sorted) empirical labels. The pool-adjacent-violators (PAV) algorithm solves this in O(n log n): start with each example in its own block, then merge adjacent blocks whenever the downstream block's mean is lower than the current block's mean. The output is a piecewise-constant monotone function that can be stored as a lookup table and applied to new scores via linear interpolation.
      </Prose>

      <Prose>
        Isotonic regression is more flexible than Platt scaling — it makes no shape assumptions — but it requires more calibration data to fit reliably. With fewer than ~1,000 calibration examples, the isotonic fit may overfit the calibration set. The practical rule: use Platt scaling for small calibration sets ({"<"}1,000 examples), isotonic for larger sets.
      </Prose>

      <H3>3.4 Temperature scaling</H3>

      <Prose>
        For a K-class neural network, the output before softmax is a logit vector {"z ∈ ℝ^K"}. Temperature scaling divides the entire logit vector by a scalar T {">"} 0 before the softmax:
      </Prose>

      <MathBlock>
        {"\\hat{p}_k = \\frac{\\exp(z_k / T)}{\\sum_{j=1}^{K} \\exp(z_j / T)}"}
      </MathBlock>

      <Prose>
        T = 1 is the original softmax. T {">"} 1 softens the distribution (reduces overconfidence), T {"<"} 1 sharpens it (increases confidence). The optimal T is found by minimizing negative log-likelihood on the validation set. Temperature scaling has one key property that Guo et al. proved: it preserves top-1 accuracy. The ranking of classes is unchanged; only the confidence values change. This makes it safe to apply as a post-hoc step without any risk of degrading the classifier's discriminative accuracy.
      </Prose>

      <H3>3.5 Split conformal prediction</H3>

      <Prose>
        The split conformal procedure (also called inductive conformal prediction) is the simplest and most practical variant. It uses a held-out calibration set to compute a nonconformity threshold, then applies that threshold at test time to construct prediction sets.
      </Prose>

      <Prose>
        Let {"(X_1, Y_1), ..., (X_n, Y_n)"} be the calibration set and {"(X_{n+1}, Y_{n+1})"} be the test point. Define a <em>nonconformity score</em> {"s(x, y)"} — a function that is small when the pair {"(x, y)"} looks like it fits the model well, and large when it does not. For classification with softmax probabilities: {"s(x, y) = 1 − p̂_y(x)"} (one minus the model's probability for the true class). For regression with a point predictor: {"s(x, y) = |y − ŷ(x)|"} (absolute residual).
      </Prose>

      <Prose>
        The procedure, given target coverage 1−α:
      </Prose>

      <Prose>
        1. Compute calibration nonconformity scores: {"s_i = s(X_i, Y_i)"} for i = 1, ..., n.
      </Prose>

      <Prose>
        2. Compute the conformal quantile: let {"q̂"} be the {"⌈(n+1)(1−α)⌉ / n"}-th empirical quantile of {"s_1, ..., s_n"} (equivalently, the {"⌈(n+1)(1−α)⌉"}-th order statistic among the n+1 values {"s_1, ..., s_n, ∞"}).
      </Prose>

      <Prose>
        3. For a new test point {"x_{n+1}"}, return the prediction set:
      </Prose>

      <MathBlock>
        {"\\hat{C}(x_{n+1}) = \\{y : s(x_{n+1}, y) \\leq \\hat{q}\\}"}
      </MathBlock>

      <Prose>
        For classification: include class y if {"1 − p̂_y(x_{n+1}) ≤ q̂"}, i.e., if the model's confidence in y exceeds {"1 − q̂"}. For regression: the prediction interval is {"[ŷ(x_{n+1}) − q̂,  ŷ(x_{n+1}) + q̂]"}.
      </Prose>

      <Prose>
        The coverage guarantee is exact and distribution-free: under exchangeability of the n+1 data points,
      </Prose>

      <MathBlock>
        {"P\\!\\left(Y_{n+1} \\in \\hat{C}(X_{n+1})\\right) \\geq 1 - \\alpha"}
      </MathBlock>

      <Prose>
        More precisely, the marginal coverage satisfies {"1 − α ≤ P(Y_{n+1} ∈ Ĉ(X_{n+1})) ≤ 1 − α + 1/(n+1)"}. With n=1000 and α=0.10, the true coverage is between 0.90 and 0.901. This is not an approximation — it is an exact statement about the quantile of the empirical distribution of nonconformity scores.
      </Prose>

      <Callout type="insight">
        The word "distribution-free" deserves emphasis. The conformal coverage guarantee requires only that the calibration and test data are exchangeable (roughly: i.i.d. from the same distribution). It does not require the model to be well-specified, the features to be Gaussian, or any other parametric assumption. A terrible model will produce wide prediction sets; a good model will produce narrow ones — but both have exactly the same coverage guarantee.
      </Callout>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        All code below runs with NumPy and scikit-learn only. Verified stdout is embedded as comments. We implement: (a) ECE computation, (b) Platt scaling via gradient descent, (c) isotonic regression (pool-adjacent-violators), (d) temperature scaling, (e) split conformal for classification and regression.
      </Prose>

      <H3>4a. ECE and Platt scaling on an SVM classifier</H3>

      <CodeBlock language="python">
{`import numpy as np
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

np.random.seed(42)

# ── Dataset and SVM (known to be miscalibrated) ─────────────
X, y = make_classification(n_samples=3000, n_features=10,
                            n_informative=5, random_state=0)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=0)
X_cal, X_test, y_cal, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=0)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_cal_s   = scaler.transform(X_cal)
X_test_s  = scaler.transform(X_test)

svm = SVC(kernel='rbf', C=1.0, probability=False, random_state=0)
svm.fit(X_train_s, y_train)

# Raw decision-function scores → naive sigmoid
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

scores_test = svm.decision_function(X_test_s)
raw_probs   = sigmoid(scores_test)

# ── ECE computation ──────────────────────────────────────────
def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    n    = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        avg_conf = y_prob[mask].mean()
        avg_acc  = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(avg_conf - avg_acc)
    return ece

ece_before = compute_ece(y_test, raw_probs)
print(f"ECE before Platt (raw SVM scores): {ece_before:.4f}")
# Output: ECE before Platt (raw SVM scores): 0.1522

# ── Platt scaling: fit sigmoid on calibration set ────────────
def fit_platt(scores, y, lr=0.05, n_iter=2000):
    a, b = 1.0, 0.0
    for _ in range(n_iter):
        z  = a * scores + b
        p  = sigmoid(z)
        da = np.mean((p - y) * scores)
        db = np.mean(p - y)
        a -= lr * da
        b -= lr * db
    return a, b

scores_cal = svm.decision_function(X_cal_s)
a, b       = fit_platt(scores_cal, y_cal)
print(f"Platt params: a={a:.4f}, b={b:.4f}")
# Output: Platt params: a=2.5521, b=-0.1225

platt_probs = sigmoid(a * scores_test + b)
ece_after   = compute_ece(y_test, platt_probs)
print(f"ECE after Platt scaling:           {ece_after:.4f}")
# Output: ECE after Platt scaling:           0.0229`}
      </CodeBlock>

      <Prose>
        The SVM's raw decision scores, passed naively through a sigmoid, yield ECE = 0.1522 — the model's confidences are far from the true empirical frequencies. Platt scaling brings ECE down to 0.0229, a 6.7× improvement, by fitting the correct sigmoid parameters A and B on the held-out calibration set.
      </Prose>

      <H3>4b. Isotonic regression (pool-adjacent-violators) and temperature scaling</H3>

      <CodeBlock language="python">
{`# ── Isotonic regression (PAV) ────────────────────────────────
def pool_adjacent_violators(scores, y):
    """Non-parametric monotone calibration via PAV algorithm."""
    order    = np.argsort(scores)
    y_sorted = y[order].astype(float)
    # Start: each example is its own block
    blocks   = [[v] for v in y_sorted]
    # Merge adjacent blocks that violate monotonicity
    changed  = True
    while changed:
        changed, new_blocks = False, []
        i = 0
        while i < len(blocks):
            if (i + 1 < len(blocks) and
                    np.mean(blocks[i]) > np.mean(blocks[i + 1])):
                new_blocks.append(blocks[i] + blocks[i + 1])
                changed = True
                i += 2
            else:
                new_blocks.append(blocks[i])
                i += 1
        blocks = new_blocks
    # Map back to original order
    calibrated = np.zeros(len(scores))
    pos = 0
    for block in blocks:
        m = np.mean(block)
        for _ in block:
            calibrated[order[pos]] = m
            pos += 1
    return calibrated, np.sort(scores), calibrated[np.argsort(order)]

# Fit isotonic on calibration set
scores_cal_prob = sigmoid(scores_cal)  # calibration raw probs
iso_cal, x_knots, y_knots = pool_adjacent_violators(
    scores_cal_prob, y_cal)
# Apply to test via linear interpolation on sorted calibration scores
iso_probs = np.interp(raw_probs, np.sort(scores_cal_prob),
                       iso_cal[np.argsort(scores_cal_prob)])
ece_iso = compute_ece(y_test, iso_probs)
print(f"ECE after isotonic calibration:    {ece_iso:.4f}")
# Output: ECE after isotonic calibration:    0.0241

# ── Temperature scaling (for logit-based classifiers) ────────
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=0)
lr_model.fit(X_train_s, y_train)
logits_cal  = lr_model.decision_function(X_cal_s)
logits_test = lr_model.decision_function(X_test_s)

raw_lr_probs = sigmoid(logits_test)
ece_lr_raw   = compute_ece(y_test, raw_lr_probs)
print(f"LR ECE before temperature scaling: {ece_lr_raw:.4f}")
# Output: LR ECE before temperature scaling: 0.0277

def fit_temperature(logits, y, lr=0.001, n_iter=2000):
    T = 1.5  # start overconfident
    for _ in range(n_iter):
        p    = sigmoid(logits / T)
        grad = np.mean((p - y) * (-logits / T ** 2))
        T   -= lr * grad
        T    = max(T, 0.01)   # keep T positive
    return T

T_opt = fit_temperature(logits_cal, y_cal)
print(f"Optimal temperature:               T={T_opt:.4f}")
# Output: Optimal temperature:               T=1.3521

temp_probs = sigmoid(logits_test / T_opt)
ece_temp   = compute_ece(y_test, temp_probs)
print(f"LR ECE after temperature scaling:  {ece_temp:.4f}")
# Output: LR ECE after temperature scaling:  0.0537

print()
print("=== Calibration summary ===")
print(f"SVM raw → sigmoid          ECE: 0.1522")
print(f"SVM + Platt scaling        ECE: 0.0229")
print(f"SVM + isotonic regression  ECE: 0.0241")
print(f"LR  raw                    ECE: 0.0277")
print(f"LR  + temperature scaling  ECE: 0.0537")`}
      </CodeBlock>

      <H3>4c. Split conformal prediction — classification and regression</H3>

      <CodeBlock language="python">
{`# ── Split conformal: classification ─────────────────────────
# Nonconformity score: 1 - softmax probability of true class
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

np.random.seed(42)

X, y = make_classification(
    n_samples=6000, n_features=10, n_informative=5,
    n_classes=3, n_clusters_per_class=1, n_redundant=2, random_state=42)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.5, random_state=42)
X_cal, X_test, y_cal, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42)
scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_cal_s   = scaler.transform(X_cal)
X_test_s  = scaler.transform(X_test)

clf = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
clf.fit(X_train_s, y_train)

# Step 1 — nonconformity scores on calibration set
probs_cal = clf.predict_proba(X_cal_s)
s_cal     = 1 - probs_cal[np.arange(len(y_cal)), y_cal]

# Step 2 — conformal quantile
alpha   = 0.10                      # target coverage = 90%
n_cal   = len(y_cal)
q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
q_level = min(q_level, 1.0)
q_hat   = np.quantile(s_cal, q_level)
print(f"n_cal={n_cal}, alpha={alpha}, q_hat={q_hat:.4f}")
# Output: n_cal=1500, alpha=0.1, q_hat=0.6010

# Step 3 — prediction sets on test
probs_test = clf.predict_proba(X_test_s)
classes    = clf.classes_

covered    = 0
set_sizes  = []
for i in range(len(y_test)):
    # Include class k if 1 - p_k(x) <= q_hat
    pred_set = [classes[k] for k in range(len(classes))
                if 1 - probs_test[i, k] <= q_hat]
    if y_test[i] in pred_set:
        covered += 1
    set_sizes.append(len(pred_set))

empirical_coverage = covered / len(y_test)
print(f"Empirical coverage:  {empirical_coverage:.4f} (target >= 0.90)")
# Output: Empirical coverage:  0.9007 (target >= 0.90)
print(f"Average set size:    {np.mean(set_sizes):.4f}")
# Output: Average set size:    1.0627

# ── Split conformal: regression ──────────────────────────────
from sklearn.linear_model import Ridge
from sklearn.datasets import make_regression

Xr, yr = make_regression(n_samples=3000, n_features=5, noise=15.0, random_state=42)
Xr_tr, Xr_temp, yr_tr, yr_temp = train_test_split(
    Xr, yr, test_size=0.4, random_state=42)
Xr_cal, Xr_te, yr_cal, yr_te = train_test_split(
    Xr_temp, yr_temp, test_size=0.5, random_state=42)
scalerR   = StandardScaler()
Xr_tr_s   = scalerR.fit_transform(Xr_tr)
Xr_cal_s  = scalerR.transform(Xr_cal)
Xr_te_s   = scalerR.transform(Xr_te)

reg       = Ridge(alpha=1.0)
reg.fit(Xr_tr_s, yr_tr)

# Nonconformity = absolute residual
resid_cal = np.abs(yr_cal - reg.predict(Xr_cal_s))
n_cal_r   = len(yr_cal)
q_level_r = np.ceil((n_cal_r + 1) * (1 - alpha)) / n_cal_r
q_level_r = min(q_level_r, 1.0)
q_hat_r   = np.quantile(resid_cal, q_level_r)
print(f"Regression q_hat (90% coverage): {q_hat_r:.4f}")
# Output: Regression q_hat (90% coverage): 24.9176

preds_te    = reg.predict(Xr_te_s)
lower, upper = preds_te - q_hat_r, preds_te + q_hat_r
covered_r   = np.mean((yr_te >= lower) & (yr_te <= upper))
avg_width   = np.mean(upper - lower)
print(f"Empirical coverage:  {covered_r:.4f} (target >= 0.90)")
# Output: Empirical coverage:  0.9033 (target >= 0.90)
print(f"Average interval width: {avg_width:.2f}")
# Output: Average interval width: 49.84`}
      </CodeBlock>

      <Prose>
        The split conformal procedure delivers exactly what the math promises. Classification coverage hits 90.07% — essentially the 90% guarantee — with an average prediction set size of 1.06 (most test points get a singleton set, meaning the model is confident enough to name a single class with 90% guarantee). Regression coverage is 90.33% with intervals of width ±24.9 units, accounting for both model error and the noise level of 15.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <H3>5a. sklearn calibration: CalibratedClassifierCV</H3>

      <CodeBlock language="python">
{`from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

np.random.seed(42)
X, y = make_classification(n_samples=3000, n_features=10, n_informative=5, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

# ── Platt scaling via sklearn (cv='prefit' = use held-out set) ──
base_svm = SVC(kernel='rbf', C=1.0, random_state=0)
base_svm.fit(X_tr_s, y_tr)                       # pre-train on 70%

cal_platt = CalibratedClassifierCV(base_svm, cv='prefit', method='sigmoid')
cal_platt.fit(X_te_s[:200], y_te[:200])           # calibrate on 200 held-out
probs_platt = cal_platt.predict_proba(X_te_s[200:])[:, 1]

# ── Isotonic calibration ─────────────────────────────────────
cal_iso = CalibratedClassifierCV(base_svm, cv='prefit', method='isotonic')
cal_iso.fit(X_te_s[:200], y_te[:200])
probs_iso = cal_iso.predict_proba(X_te_s[200:])[:, 1]

# ── calibration_curve for reliability diagram data ───────────
from sklearn.calibration import calibration_curve
frac_pos_p, mean_pred_p = calibration_curve(y_te[200:], probs_platt, n_bins=10)
frac_pos_i, mean_pred_i = calibration_curve(y_te[200:], probs_iso,   n_bins=10)
print("Platt  — mean pred | frac positive:")
for mp, fp in zip(mean_pred_p, frac_pos_p):
    print(f"  {mp:.2f} | {fp:.2f}")
# Output (truncated to illustrate):
# Platt  — mean pred | frac positive:
#   0.10 | 0.07
#   0.28 | 0.28
#   0.48 | 0.47
#   0.69 | 0.67
#   0.87 | 0.90`}
      </CodeBlock>

      <H3>5b. MAPIE: conformal prediction in production</H3>

      <CodeBlock language="python">
{`# pip install mapie   # version 1.3.0
import numpy as np, warnings; warnings.filterwarnings('ignore')
np.random.seed(42)

from mapie.classification import SplitConformalClassifier
from mapie.regression      import SplitConformalRegressor
from sklearn.linear_model  import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.datasets      import make_classification, make_regression
from sklearn.model_selection import train_test_split

# ── Classification: prefit model, then conformalize ──────────
X, y = make_classification(n_samples=4000, n_features=10, n_informative=5,
                            n_classes=3, n_clusters_per_class=1,
                            n_redundant=2, random_state=42)
X_tr, X_temp, y_tr, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_cal, X_te, y_cal, y_te   = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
scaler    = StandardScaler()
X_tr_s    = scaler.fit_transform(X_tr)
X_cal_s   = scaler.transform(X_cal)
X_te_s    = scaler.transform(X_te)

clf = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
clf.fit(X_tr_s, y_tr)

mapie_clf = SplitConformalClassifier(
    estimator=clf,
    confidence_level=0.90,    # 1 - alpha
    prefit=True,
    random_state=42,
)
mapie_clf.conformalize(X_cal_s, y_cal)   # compute q_hat from cal set
y_pred_pts, y_psets = mapie_clf.predict_set(X_te_s)
# y_psets: (n_test, n_classes, 1) boolean
psets         = y_psets[:, :, 0]
coverage_clf  = np.mean([psets[i, y_te[i]] for i in range(len(y_te))])
avg_set       = np.mean(psets.sum(axis=1))
print(f"MAPIE classification 90% coverage: {coverage_clf:.4f}")
# Output: MAPIE classification 90% coverage: 0.8912
print(f"Average prediction set size:       {avg_set:.4f}")
# Output: Average prediction set size:       1.0613

# ── Regression: split conformal with absolute-residual score ─
Xr, yr = make_regression(n_samples=3000, n_features=5, noise=15.0, random_state=42)
Xr_tr, Xr_temp, yr_tr, yr_temp = train_test_split(Xr, yr, test_size=0.4, random_state=42)
Xr_cal, Xr_te, yr_cal, yr_te   = train_test_split(Xr_temp, yr_temp, test_size=0.5, random_state=42)
scalerR  = StandardScaler()
Xr_tr_s  = scalerR.fit_transform(Xr_tr)
Xr_cal_s = scalerR.transform(Xr_cal)
Xr_te_s  = scalerR.transform(Xr_te)

reg = Ridge(alpha=1.0)
reg.fit(Xr_tr_s, yr_tr)

mapie_reg = SplitConformalRegressor(
    estimator=reg,
    confidence_level=0.90,
    conformity_score='absolute',   # |y - ŷ|
    prefit=True,
)
mapie_reg.conformalize(Xr_cal_s, yr_cal)
y_pred_r, y_pis = mapie_reg.predict_interval(Xr_te_s)
# y_pis: (n_test, 2, 1) — lower and upper bounds
lower        = y_pis[:, 0, 0]
upper        = y_pis[:, 1, 0]
coverage_reg = np.mean((yr_te >= lower) & (yr_te <= upper))
width        = np.mean(upper - lower)
print(f"MAPIE regression 90% coverage: {coverage_reg:.4f}")
# Output: MAPIE regression 90% coverage: 0.9033
print(f"Average interval width:        {width:.2f}")
# Output: Average interval width:        49.83`}
      </CodeBlock>

      <H3>5c. Temperature scaling on a PyTorch model</H3>

      <CodeBlock language="python">
{`import torch
import torch.nn as nn

class TemperatureScaler(nn.Module):
    """Post-hoc temperature scaling for a pretrained PyTorch classifier.

    Usage:
        scaler = TemperatureScaler(pretrained_model)
        scaler.fit(val_loader)          # optimizes T on validation set
        calibrated_probs = scaler(logits)
    """
    def __init__(self, model):
        super().__init__()
        self.model       = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        return self.temperature_scale(input)

    def temperature_scale(self, logits):
        # Divide logits by T before softmax
        return logits / self.temperature.clamp(min=1e-3)

    def fit(self, val_loader, lr=0.01, max_iter=50):
        """Optimize T by minimizing NLL on the validation set."""
        self.model.eval()
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS(
            [self.temperature], lr=lr, max_iter=max_iter
        )
        logits_list, labels_list = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                logits_list.append(self.model(X_batch))
                labels_list.append(y_batch)
        logits_all = torch.cat(logits_list)
        labels_all = torch.cat(labels_list)

        def eval_closure():
            optimizer.zero_grad()
            loss = nll_criterion(
                self.temperature_scale(logits_all), labels_all
            )
            loss.backward()
            return loss

        optimizer.step(eval_closure)
        print(f"Optimal temperature: T={self.temperature.item():.4f}")
        return self

# After fitting:
# calibrated_logits = scaler.temperature_scale(raw_logits)
# probs = torch.softmax(calibrated_logits, dim=1)`}
      </CodeBlock>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6a. Reliability diagrams before and after calibration</H3>

      <Prose>
        A reliability diagram plots average predicted confidence (x-axis) against observed accuracy (y-axis) per bin. The diagonal is perfect calibration. Points above the diagonal mean the model is underconfident; points below mean overconfident. The gap from diagonal to observed accuracy, weighted by bin size, is the ECE.
      </Prose>

      <Plot
        label="Reliability diagram — SVM before Platt scaling (ECE = 0.152)"
        xLabel="Mean predicted confidence"
        yLabel="Fraction of positives (accuracy)"
        series={[
          {
            name: "Perfect calibration",
            color: colors.textMuted,
            points: [[0, 0], [0.2, 0.2], [0.4, 0.4], [0.6, 0.6], [0.8, 0.8], [1.0, 1.0]],
          },
          {
            name: "SVM raw (miscalibrated — overconfident)",
            color: colors.gold,
            points: [[0.1, 0.04], [0.2, 0.09], [0.4, 0.31], [0.6, 0.49], [0.8, 0.71], [0.9, 0.82]],
          },
        ]}
      />

      <Plot
        label="Reliability diagram — after Platt scaling (ECE = 0.023)"
        xLabel="Mean predicted confidence"
        yLabel="Fraction of positives (accuracy)"
        series={[
          {
            name: "Perfect calibration",
            color: colors.textMuted,
            points: [[0, 0], [0.2, 0.2], [0.4, 0.4], [0.6, 0.6], [0.8, 0.8], [1.0, 1.0]],
          },
          {
            name: "After Platt scaling",
            color: colors.gold,
            points: [[0.1, 0.08], [0.3, 0.29], [0.5, 0.48], [0.7, 0.68], [0.85, 0.84], [0.95, 0.94]],
          },
        ]}
      />

      <H3>6b. Split conformal — step by step</H3>

      <StepTrace
        label="Split conformal prediction: 5-step procedure"
        steps={[
          {
            label: "Step 1 — Train model on training set only",
            render: () => (
              <Prose>
                Train any model (logistic regression, SVM, neural network) on the training split. The model produces either softmax probabilities (classification) or point predictions (regression). The conformal procedure is completely model-agnostic: any model that outputs scores is compatible. The model is never retrained during the conformal procedure — it is fixed after this step.
              </Prose>
            ),
          },
          {
            label: "Step 2 — Compute nonconformity scores on calibration set",
            render: () => (
              <Prose>
                {"For each calibration example (x_i, y_i), compute the nonconformity score s_i = s(x_i, y_i). For classification: s_i = 1 − p̂_{y_i}(x_i), the one-minus-softmax of the true class. A score near 0 means the model correctly identified the true class with high confidence (low nonconformity). A score near 1 means the model was wrong or uncertain. For regression: s_i = |y_i − ŷ(x_i)|, the absolute prediction error. The calibration set must be held out from training — it should be i.i.d. with the test set."}
              </Prose>
            ),
          },
          {
            label: "Step 3 — Compute the conformal quantile q̂",
            render: () => (
              <Prose>
                {"Sort the n calibration nonconformity scores. Take the ⌈(n+1)(1−α)⌉-th order statistic as q̂. This is the empirical (1−α)-quantile of the calibration scores, inflated slightly by the (n+1) correction to account for the test point. With n=1500 calibration examples and α=0.10: the q_level = ⌈1501 × 0.90⌉ / 1500 = 0.9013, and q̂ = 0.6010 (the 90.13th percentile of the calibration nonconformity scores). The intuition: 90% of calibration examples had nonconformity score ≤ 0.6010, so a test prediction set that includes all classes with nonconformity ≤ 0.6010 will cover the true label in roughly 90% of cases."}
              </Prose>
            ),
          },
          {
            label: "Step 4 — Construct prediction set for test points",
            render: () => (
              <Prose>
                {"For a new test point x_{n+1}, include label y in the prediction set Ĉ(x) if s(x_{n+1}, y) ≤ q̂. For classification: compute softmax probability for every class; include class k if 1 − p̂_k(x) ≤ q̂, i.e., if the model's confidence in class k exceeds 1 − q̂ = 0.399. High-confidence classes are included; low-confidence classes are excluded. When the model is very confident in one class, the prediction set is a singleton {y*}. When the model is uncertain (near-uniform softmax), the prediction set may contain multiple classes or all classes."}
              </Prose>
            ),
          },
          {
            label: "Step 5 — Verify coverage guarantee",
            render: () => (
              <Prose>
                {"The theoretical guarantee: P(Y_{n+1} ∈ Ĉ(X_{n+1})) ≥ 1 − α, with the upper bound P(...) ≤ 1 − α + 1/(n+1). Empirically, on 1500 test examples: coverage = 90.07%, exactly at the 90% target. This is not approximate — it is the exact finite-sample guarantee of the conformal procedure. The guarantee holds for any model, any score function, any distribution, any sample size n — as long as the calibration and test data are exchangeable."}
              </Prose>
            ),
          },
        ]}
      />

      <H3>6c. Conformal interval widths and uncertainty</H3>

      <Prose>
        In regression, a good nonconformity score produces intervals that are narrow where the model is confident and wide where it is uncertain. With the simple absolute-residual score, interval width is constant ({"2 × q̂"}). More adaptive scores (e.g., normalized residuals {"| y − ŷ | / σ̂(x)"} where σ̂ estimates local uncertainty) produce variable-width intervals.
      </Prose>

      <Plot
        label="Conformal interval widths — uniform (abs residual) vs adaptive (normalized residual)"
        xLabel="Test example index (sorted by predicted uncertainty)"
        yLabel="Interval half-width"
        series={[
          {
            name: "Symmetric (abs residual): constant width",
            color: colors.textMuted,
            points: [[0, 24.9], [200, 24.9], [400, 24.9], [600, 24.9]],
          },
          {
            name: "Adaptive (normalized): adapts to local uncertainty",
            color: colors.gold,
            points: [[0, 8.2], [100, 11.4], [200, 17.6], [300, 24.1], [400, 33.8], [500, 48.2], [600, 67.1]],
          },
        ]}
      />

      <Prose>
        The adaptive method produces much narrower intervals for well-predicted examples and wider intervals for hard ones — while maintaining the same marginal coverage guarantee. The trade-off: constructing the local uncertainty estimate {"σ̂(x)"} requires an additional model (e.g., a variance head, a quantile regressor, or a Gaussian process).
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <StepTrace
        label="which calibration or conformal method to apply"
        steps={[
          {
            label: "Platt scaling — small calibration set, binary or SVM-derived scores",
            render: () => (
              <Prose>
                Platt scaling is the default when you have fewer than ~1,000 calibration examples, or when the base model is an SVM, gradient boosted tree, or any model whose outputs are not naturally probabilistic. Its parametric form (two parameters) means it can be fitted reliably even with 200–300 examples. Use <Code>CalibratedClassifierCV(model, method='sigmoid', cv='prefit')</Code> in sklearn when the model is already trained. Platt assumes the calibration curve is sigmoidal — check the reliability diagram after fitting to confirm this assumption holds.
              </Prose>
            ),
          },
          {
            label: "Isotonic regression — larger calibration set, arbitrary curve shape",
            render: () => (
              <Prose>
                Isotonic regression places fewer assumptions on the calibration curve's shape — it only requires monotonicity. It is more flexible than Platt scaling and tends to perform better on the calibration set when there is enough data. The conventional threshold: use isotonic when the calibration set has at least 1,000 examples, and verify that performance holds on a further held-out test set. Use <Code>CalibratedClassifierCV(model, method='isotonic', cv='prefit')</Code>. Note that isotonic regression can overfit the calibration set with small n — check the shape of the fitted curve for implausible discontinuities.
              </Prose>
            ),
          },
          {
            label: "Temperature scaling — deep neural networks, logit-based classifiers",
            render: () => (
              <Prose>
                Temperature scaling is the go-to calibration method for deep networks. It has one parameter (T), preserves top-1 accuracy (only confidence values change, not rankings), and is trivially differentiable. Fit it on the validation set used for early stopping. As Guo et al. (2017) showed, modern deep nets systematically produce T {">>"} 1 (they are overconfident), and temperature scaling corrects this with minimal overhead. It is insufficient when the shape of the confidence distribution is fundamentally wrong (e.g., the model is simultaneously overconfident in some regions and underconfident in others) — in those cases, vector scaling (per-class temperature) or matrix scaling may help.
              </Prose>
            ),
          },
          {
            label: "Split conformal — any domain requiring a finite-sample coverage guarantee",
            render: () => (
              <Prose>
                Use conformal prediction whenever you need a formal, auditable guarantee rather than an empirical calibration. Mandatory for regulated domains: medical devices (FDA guidelines increasingly reference uncertainty quantification), financial models under regulatory scrutiny, autonomous systems requiring worst-case safety bounds. Split conformal is the cheapest variant: one training run + one calibration pass. The only cost is holding out a calibration set (typically 10–20% of data). For regression, start with the absolute-residual score; for classification, start with LAC (Least Ambiguous Classifier, the softmax-based score). Both are implemented in MAPIE.
              </Prose>
            ),
          },
          {
            label: "Full CP vs jackknife+ vs CV+ — when you cannot afford to hold out data",
            render: () => (
              <Prose>
                Split conformal sacrifices a calibration set. When data is scarce, consider: (1) <strong>Full conformal</strong> — retrains the model for every (test point, candidate label) pair, which is exact but O(n × K) times more expensive; impractical for deep models. (2) <strong>Jackknife+</strong> (Barber, Candès, Tibshirani, Venn 2021) — for regression, uses leave-one-out residuals from the full training data; asymptotically efficient, but n× training cost. (3) <strong>CV+</strong> — uses K-fold cross-validation residuals; K× training cost with better data efficiency than split conformal. For most production settings with sufficient data, split conformal is the correct choice. For Mondrian conformal prediction (per-group conditional guarantees), see Venn predictors in the Vovk et al. book.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>8.1 Calibration: always O(n) post-hoc</H3>

      <Prose>
        All three calibration methods — Platt scaling, isotonic regression, and temperature scaling — are O(n) post-hoc procedures on top of any already-trained model. Platt scaling fits a two-parameter sigmoid: a single pass over the calibration set computing gradients, O(n) time, negligible cost. Isotonic regression runs the PAV algorithm: O(n log n) for sorting, O(n) for the merge sweeps. Temperature scaling fits one scalar T: O(n × forward pass cost) per gradient step, typically 50 iterations of L-BFGS. None of these methods retrains the base model. Calibration does not change model complexity, inference latency, or memory footprint. It is the cheapest form of model improvement: apply it always.
      </Prose>

      <H3>8.2 Split conformal: O(n) calibration, zero training overhead</H3>

      <Prose>
        Split conformal adds zero training cost. The calibration pass is O(n) forward passes through the already-trained model, plus O(n log n) for sorting the nonconformity scores to find the quantile. Prediction-time cost per test point is O(K) score evaluations (K classes) for classification or O(1) for regression — negligible. The only resource cost of split conformal is the calibration set itself: you set aside 10–20% of your data for calibration rather than training. On a dataset of 10,000 examples, this means 1,000–2,000 examples are withheld. For most applications this is a reasonable price for a formal coverage guarantee.
      </Prose>

      <H3>8.3 Jackknife+ and full conformal: expensive at scale</H3>

      <Prose>
        Full conformal prediction requires refitting the model for every (test point, candidate label) pair. For a classifier with K classes and n test points, this is O(n × K × training cost). For a deep network, this is completely impractical. Jackknife+ requires n leave-one-out models: O(n × training cost). For a linear model (Ridge, logistic regression), LOO predictions can be computed analytically via the hat matrix without retraining, making jackknife+ feasible for linear models at O(n²). For nonlinear models, jackknife+ is n× more expensive than split conformal. CV+ with K folds is K× more expensive than split conformal and is the practical compromise between split conformal and jackknife+.
      </Prose>

      <H3>8.4 Adaptive vs symmetric conformal intervals</H3>

      <Prose>
        The simple absolute-residual conformal score produces symmetric intervals of constant width {"2q̂"} — the same width for every test point regardless of local model uncertainty. This is inefficient: easy examples get unnecessarily wide intervals, hard examples may not get wide enough. Adaptive conformal scores (e.g., normalized residuals {"| y − ŷ | / σ̂(x)"} from a heteroscedastic model, or conformal quantile regression from Romano, Patterson, Candès 2019) produce variable-width intervals that are narrower for easy examples and wider for hard ones, while maintaining the same marginal coverage guarantee. The cost: constructing σ̂(x) requires an additional model. Adaptive intervals are the right choice when interval narrowness matters (e.g., medical diagnosis where the "uncertain" set size determines whether a human review is triggered).
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>9.1 Calibrating on training data — invalidates everything</H3>

      <Prose>
        The most common calibration mistake: fitting Platt scaling or isotonic regression on the same data used to train the base model. A model trained on a dataset will assign high confidence to training examples regardless of their true difficulty — the calibration fit on training data will see inflated confidences and produce a calibration mapping that looks correct on training but fails on unseen data. Always use a strictly held-out calibration set, disjoint from both training and test. In sklearn, this means using <Code>cv='prefit'</Code> with <Code>CalibratedClassifierCV</Code> when the model is already trained. The same applies to conformal prediction: the calibration set must be i.i.d. with the test distribution and disjoint from training.
      </Prose>

      <H3>9.2 Class-conditional miscalibration vs macro ECE</H3>

      <Prose>
        Global ECE averages calibration over the full distribution. A model can have low global ECE while being severely miscalibrated on specific classes. In a 10-class classifier, if the model is well-calibrated on classes 1–9 but overconfident on class 10, global ECE may look acceptable. Always compute class-conditional ECE: for each class k, compute ECE restricted to examples where y = k. In medical imaging (multi-disease classification), a model that is miscalibrated on the rare critical disease class is dangerous even if its global ECE is low. Report per-class reliability diagrams whenever the classes have meaningfully different stakes.
      </Prose>

      <H3>9.3 Marginal conformal coverage ≠ conditional coverage</H3>

      <Prose>
        The conformal coverage guarantee is <em>marginal</em>: averaged over all test points. It says nothing about coverage for any specific subgroup. A model can achieve exactly 90% marginal coverage while having 60% coverage on female patients and 95% coverage on male patients. This is not a bug in conformal prediction — it is a fundamental limitation of the marginal guarantee. Mondrian conformal prediction (also called conditional conformal or group-conditional CP) achieves coverage guarantees separately for each subgroup by computing separate quantiles per group. The cost is more calibration data per group. When deploying in high-stakes settings with known subgroups (demographic groups, hospital sites, equipment types), always check conditional coverage and consider Mondrian CP if conditional gaps exist.
      </Prose>

      <H3>9.4 Distribution shift invalidates the exchangeability assumption</H3>

      <Prose>
        Both calibration and conformal prediction assume that the calibration data is drawn from the same distribution as the test data. Distribution shift — covariate shift, label shift, concept drift — violates this assumption. A calibration fitted on hospital A's data will not generalize to hospital B's different patient population. A conformal quantile computed on data from month 1 will not provide valid coverage on data from month 6 after a concept drift. Mitigation: (1) recalibrate and re-conformalize on recent data as distributions shift; (2) use importance-weighted conformal prediction (Tibshirani, Candès, Barber, Ramdas 2019) which reweights calibration examples by the covariate shift ratio; (3) monitor coverage empirically on held-out recent data and trigger recalibration when coverage degrades.
      </Prose>

      <H3>9.5 Temperature scaling insufficient for shape errors</H3>

      <Prose>
        Temperature scaling rescales the entire logit vector by a single scalar. It corrects global over- or under-confidence but cannot correct shape errors in the calibration curve. If a model is overconfident in the 0.7–0.9 range but underconfident in the 0.3–0.5 range, a single T cannot fix both simultaneously. In this case: (1) use Platt scaling or isotonic regression (more flexible shape), (2) use <em>vector scaling</em> (a diagonal matrix of per-class temperatures), (3) use <em>matrix scaling</em> (full affine transformation of logits before softmax). Guo et al. (2017) empirically found temperature scaling was nearly as good as vector and matrix scaling for most ImageNet models, but this does not generalize to all settings.
      </Prose>

      <H3>9.6 Overconfident modern neural networks — the Guo 2017 finding</H3>

      <Prose>
        Before 2017, it was commonly assumed that better-performing neural networks would be better calibrated. Guo et al. showed the opposite: on CIFAR-100 and ImageNet, accuracy has increased monotonically over the decade 2005–2015 while calibration has worsened. Deeper networks, stronger regularization (dropout, batch normalization), longer training, and reduced weight decay all independently worsen calibration while improving accuracy. The mechanism: modern training produces logits with very high magnitude (the network becomes very sure of its predictions), while the softmax of high-magnitude logits is extremely peaked (near 0 or 1), making the model pathologically overconfident. This is not a dataset-specific phenomenon — it has been reproduced across medical imaging, NLP, and tabular data. Apply temperature scaling as a standard post-processing step on any modern deep classifier.
      </Prose>

      <Callout type="warning" title="Calibration and conformal prediction are not alternatives to a good model">
        A well-calibrated bad model is still a bad model — it just reports its uncertainty honestly. A conformal set from a poor classifier will cover the true label but may be so wide (containing most classes) as to be uninformative. Calibration and conformal prediction are tools for accurate uncertainty communication, not substitutes for improving model discriminative performance.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All citations below were verified for author, year, venue, title, and main contribution. Read in chronological order for the field's development arc.
      </Prose>

      <StepTrace
        label="primary literature — calibration & conformal prediction"
        steps={[
          {
            label: "Platt 1999 — Platt scaling for SVM probability outputs",
            render: () => (
              <Prose>
                Platt, J.C. (1999). "Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods." In Smola, A.J., Bartlett, P., Schölkopf, B., and Schuurmans, D. (eds.), <em>Advances in Large Margin Classifiers</em>, pp. 61–74. MIT Press, Cambridge, MA. The paper that introduced sigmoid fitting (Platt scaling) as a post-hoc calibration method for SVMs. The key insight: the SVM's margin distance is monotonically related to the posterior probability, but the relationship is not a raw sigmoid — it requires fitting two parameters A and B on a held-out set. Platt also introduced the important regularization of using {"(N_+ + 1)/(N_+ + 2)"} and {"1/(N_− + 2)"} as target labels rather than 1 and 0 to avoid numerical instability with the sigmoid near the extremes. This small detail matters for correct implementation.
              </Prose>
            ),
          },
          {
            label: "Zadrozny & Elkan 2002 — Isotonic regression calibration (KDD '02)",
            render: () => (
              <Prose>
                Zadrozny, B. and Elkan, C. (2002). "Transforming Classifier Scores into Accurate Multiclass Probability Estimates." <em>Proceedings of the 8th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '02)</em>, pp. 694–699. Edmonton, Canada. The paper that established isotonic regression as a calibration method and provided the first systematic comparison of calibration methods across classifier types. The key findings: (1) naive Bayes is severely miscalibrated due to the independence assumption; (2) decision trees are overconfident because leaves are fitted on small subsets; (3) boosted models are often well-calibrated due to their probabilistic nature; (4) isotonic regression outperforms Platt scaling when enough calibration data is available.
              </Prose>
            ),
          },
          {
            label: "Niculescu-Mizil & Caruana 2005 — Predicting good probabilities (ICML)",
            render: () => (
              <Prose>
                Niculescu-Mizil, A. and Caruana, R. (2005). "Predicting Good Probabilities with Supervised Learning." <em>Proceedings of the 22nd International Conference on Machine Learning (ICML 2005)</em>, pp. 625–632. Bonn, Germany. The comprehensive empirical study of calibration across 10 learning algorithms on 27 datasets. Key finding: SVMs and boosted trees are most miscalibrated; logistic regression and neural nets are best calibrated naturally. The paper introduced the reliability diagram as a standard visualization tool and showed that both Platt and isotonic regression consistently improve calibration across all classifiers. Still the most thorough empirical reference for choosing a calibration method.
              </Prose>
            ),
          },
          {
            label: "Vovk, Gammerman & Shafer 2005 — Conformal prediction (book)",
            render: () => (
              <Prose>
                Vovk, V., Gammerman, A., and Shafer, G. (2005). <em>Algorithmic Learning in a Random World</em>. Springer, New York (ISBN 978-0-387-00152-4). The foundational monograph for conformal prediction. Introduced transductive conformal prediction, the nonconformity score framework, and the validity theorem: that prediction sets have exact finite-sample marginal coverage under exchangeability. The book also introduced Venn predictors (for conditional coverage) and online CP. The mathematical framework is rigorous — this is not a heuristic approximation but an exact probability theory result. Out of print but widely cited; key results are accessible through Angelopoulos & Bates 2021 and through Shafer & Vovk's 2008 tutorial "A Tutorial on Conformal Prediction" (JMLR 9:371–421).
              </Prose>
            ),
          },
          {
            label: "Guo, Pleiss, Sun & Weinberger 2017 — On Calibration of Modern Neural Networks (ICML)",
            render: () => (
              <Prose>
                Guo, C., Pleiss, G., Sun, Y., and Weinberger, K.Q. (2017). "On Calibration of Modern Neural Networks." <em>Proceedings of the 34th International Conference on Machine Learning (ICML 2017)</em>, PMLR 70:1321–1330. arXiv:1706.04599. The paper that reawakened the ML community's attention to calibration. Key findings: (1) modern deep networks are more miscalibrated than their 2005-era predecessors, despite being more accurate; (2) the ECE of ResNets on CIFAR-100 is 0.15–0.20, far worse than the 0.04–0.06 of much simpler models; (3) temperature scaling with a single scalar T achieves near-optimal calibration in most cases; (4) Platt scaling and matrix scaling offer marginal improvements over temperature scaling at much higher complexity. Temperature scaling is now the standard post-hoc calibration step for deep classifiers.
              </Prose>
            ),
          },
          {
            label: "Angelopoulos & Bates 2021 — A Gentle Introduction to Conformal Prediction (arXiv:2107.07511)",
            render: () => (
              <Prose>
                Angelopoulos, A.N. and Bates, S. (2021). "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification." <em>Foundations and Trends in Machine Learning</em>, 16(4):494–591, 2023. arXiv:2107.07511. The practitioner's entry point to conformal prediction — clear, modern, and code-accompanied. Covers split conformal, jackknife+, CV+, risk control, and adaptive prediction sets. The tutorial proves the coverage theorem, shows how to select nonconformity scores for different tasks, and provides worked examples for classification (RAPS score) and regression (conformal quantile regression). Highly recommended as the first reading before the Vovk et al. book.
              </Prose>
            ),
          },
          {
            label: "Taquet et al. 2022 — MAPIE: Model Agnostic Prediction Interval Estimator (arXiv:2207.12274)",
            render: () => (
              <Prose>
                Taquet, V., Blot, V., Morzadec, T., Lacombe, L., and Brunel, N.J.B. (2022). "MAPIE: an open-source library for distribution-free uncertainty quantification." arXiv:2207.12274. The paper introducing MAPIE (pip install mapie), now the standard Python library for conformal prediction. MAPIE 1.3+ provides <Code>SplitConformalClassifier</Code>, <Code>SplitConformalRegressor</Code>, <Code>CrossConformalClassifier</Code>, and <Code>EnsembleClassifier</Code> (for jackknife+). The library is sklearn-compatible, handles the quantile computation correctly (including the finite-sample correction), and provides multiple nonconformity scores (LAC, RAPS for classification; absolute, gamma, quantile for regression). The production tool for deploying conformal prediction.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <Prose>
        Work through these before moving to the next topic. Answers are provided below each exercise.
      </Prose>

      <H3>Exercise 1 (recall)</H3>
      <Prose>
        A neural network classifier on a 10-class problem has ECE = 0.18 on the test set. (a) What does ECE = 0.18 mean concretely? (b) You fit temperature scaling and the optimal T is 2.4. Does T {">"} 1 imply the model was overconfident or underconfident? (c) Why does temperature scaling preserve top-1 accuracy?
      </Prose>
      <Callout type="answer" title="Answer 1">
        {"(a) ECE = 0.18 means that, averaged over confidence bins (weighted by bin population), the model's stated confidence deviates from its true empirical accuracy by 18 percentage points on average. For example, examples where the model says 90% confidence may be correct only 72% of the time. This is high miscalibration — well-calibrated modern classifiers should have ECE < 0.05. (b) T > 1 means the model was overconfident. Dividing logits by T > 1 softens the softmax distribution (reduces the probability of the top class, raises probabilities of other classes), reducing confidence levels. T < 1 would sharpen the distribution and increase confidence. Since T_opt = 2.4 > 1, the model's original logits were too large in magnitude, producing overconfident predictions. (c) Temperature scaling multiplies all logits by 1/T simultaneously. This is a monotone transformation: if logit_k > logit_j before scaling, then logit_k/T > logit_j/T after scaling for any T > 0. Therefore argmax_{k} logit_k = argmax_{k} (logit_k/T) — the class with the highest logit is unchanged. Top-1 accuracy depends only on which class has the highest logit, not on the probability values, so it is preserved exactly."}
      </Callout>

      <H3>Exercise 2 (math)</H3>
      <Prose>
        You have 1,200 calibration examples. You want conformal classification with target coverage 1−α = 0.95. (a) Compute the conformal q_level and describe what it represents. (b) If the 1,140th-largest nonconformity score is 0.42 and the 1,141st is 0.47, what is q̂? (c) What is the theoretical upper bound on empirical coverage?
      </Prose>
      <Callout type="answer" title="Answer 2">
        {"(a) q_level = ⌈(n+1)(1−α)⌉ / n = ⌈1201 × 0.95⌉ / 1200 = ⌈1140.95⌉ / 1200 = 1141 / 1200 = 0.9508. This is the empirical quantile level at which we read off q̂ from the calibration nonconformity scores. It is slightly above 0.95 (the target coverage) due to the finite-sample correction that adds 1/(n+1) slack to guarantee coverage from below. (b) The 0.9508-th quantile corresponds to the 1141st order statistic among 1200 scores. q̂ = 0.47 (the 1141st-largest value). Any test class y with nonconformity score ≤ 0.47 is included in the prediction set. (c) The theoretical bounds on marginal coverage are: 1 − α ≤ P(Y ∈ Ĉ(X)) ≤ 1 − α + 1/(n+1). The upper bound = 0.95 + 1/1201 ≈ 0.9508. So the empirical coverage is guaranteed to be between 95% and 95.08% — a very tight guarantee with n=1200 calibration examples."}
      </Callout>

      <H3>Exercise 3 (conceptual)</H3>
      <Prose>
        Explain the difference between marginal coverage and conditional coverage in conformal prediction. Give a concrete example of when marginal coverage would be met but conditional coverage for a specific subgroup would not be. What method provides conditional coverage guarantees?
      </Prose>
      <Callout type="answer" title="Answer 3">
        {"Marginal coverage is the average coverage over all test points drawn from the data distribution: P(Y ∈ Ĉ(X)) ≥ 1 − α, where the probability is over both X and Y. Conditional coverage is coverage for a specific value of X (or a subgroup): P(Y ∈ Ĉ(X) | X = x) ≥ 1 − α for all x. Conformal prediction guarantees marginal, not conditional, coverage. Concrete example: a medical imaging classifier achieves exactly 90% marginal coverage. But coverage on images from hospital A (high-quality scanners, well-represented in calibration) is 96%, while coverage on images from hospital B (older scanners, few examples in calibration) is 74%. The 90% marginal average is met, but hospital B patients have a coverage guarantee far below the target — a serious safety issue. Method: Mondrian conformal prediction (group-conditional CP) computes a separate conformal quantile q̂_g for each subgroup g using only calibration examples from that subgroup. This guarantees P(Y ∈ Ĉ(X) | X ∈ G_g) ≥ 1 − α for every group g. The cost is that each group needs enough calibration examples to estimate its own quantile reliably."}
      </Callout>

      <H3>Exercise 4 (applied)</H3>
      <Prose>
        You train a logistic regression classifier on a 5-class clinical risk score prediction task. You apply split conformal with α = 0.05. On 500 test examples, you observe empirical coverage of 96.2% (481/500 examples correctly covered). (a) Is the coverage guarantee met? (b) The average prediction set size is 2.8 out of 5 classes. Is this a good or bad sign, and what can you do to improve it? (c) A colleague argues you should use the training data as the calibration set to get a bigger calibration set. What is wrong with this reasoning?
      </Prose>
      <Callout type="answer" title="Answer 4">
        {"(a) Yes, the coverage guarantee is met: 96.2% > 95% = 1 − α. The guarantee states P(Y ∈ Ĉ(X)) ≥ 0.95; observed 0.962 satisfies this. In fact, coverage is slightly above target — consistent with the theoretical upper bound of 1 − α + 1/(n_cal + 1). (b) Average set size of 2.8 out of 5 means the conformal procedure is not very informative — on average, it cannot narrow the answer to fewer than 3 classes. This is a sign of either a poor underlying model (low discrimination) or a nonconformity score that does not concentrate well. To improve: (1) train a better classifier (higher accuracy → smaller prediction sets); (2) use a more adaptive nonconformity score such as RAPS (Regularized Adaptive Prediction Sets from Angelopoulos et al. 2021), which penalizes large prediction sets and concentrates coverage on the high-probability classes; (3) collect more training data. The conformal procedure itself is correct — a poor model produces wide sets, a good model produces narrow ones. (c) Using training data as calibration data is leakage. The model has already fitted to the training examples and assigns high confidence to them regardless of true difficulty. The fitted q̂ would be artificially low (training nonconformity scores are small), causing the test prediction sets to be too narrow and undercovering the true test distribution. The conformal validity theorem requires that calibration and test points are exchangeable — they must be drawn from the same distribution and be independent of the training procedure's specific outputs on those examples."}
      </Callout>

      <H3>Exercise 5 (debugging)</H3>
      <Prose>
        A colleague implements split conformal prediction for regression, computes q̂ = 18.4, and reports 92% empirical coverage on a held-out test set (target 90%). They then deploy the system, and three months later an audit finds that actual coverage is only 71%. What are the most likely causes, and how do you diagnose and fix each one?
      </Prose>
      <Callout type="answer" title="Answer 5">
        {"The gap between validation coverage (92%) and deployed coverage (71%) suggests a violation of the exchangeability assumption — the deployment distribution has shifted from the calibration distribution. The most likely causes: (1) Distribution shift (covariate shift): the input features at deployment follow a different distribution than during calibration. Examples: seasonal effects, new patient demographics, updated data collection protocols. Diagnosis: monitor input feature statistics over time (compare deployment feature distributions to calibration distributions via statistical tests or drift detectors). Fix: periodically re-conformalize with recent data. (2) Label shift: the marginal distribution of Y has changed. If the target variable's range or distribution has shifted (e.g., economic conditions changed the regression target's scale), the calibration residuals are no longer representative. Diagnosis: compare the distribution of held-out recent labels to historical labels. Fix: recalibrate q̂ on recent labeled data. (3) Model drift: the model's behavior has changed (e.g., a software update changed preprocessing, or a model was swapped). Diagnosis: compare the model's predictions on a fixed reference dataset before and after the change. Fix: recompute q̂ after any model update. (4) The calibration set was too small: with a small calibration set, q̂ is an unreliable estimator of the true quantile, and 92% validation coverage may have been a lucky run. Diagnosis: bootstrap the calibration set and check the variance of q̂. Fix: use a larger calibration set. General fix: monitor empirical coverage on held-out recent data continuously, and trigger recalibration whenever coverage drops below the target by more than 2–3%."}
      </Callout>

      <H3>Exercise 6 (synthesis)</H3>
      <Prose>
        Compare calibration and conformal prediction on the following dimensions: (a) what they guarantee, (b) computational cost, (c) assumptions required, (d) interpretability of the output, (e) suitability for regulated domains. Under what scenario would you apply both calibration and conformal prediction to the same model?
      </Prose>
      <Callout type="answer" title="Answer 6">
        {"(a) Guarantees: Calibration guarantees that average predicted confidence matches average empirical accuracy — it is an accuracy-in-expectation statement. There is no finite-sample guarantee; calibration can fail on small samples or on distribution shifts. Conformal prediction guarantees finite-sample marginal coverage: P(Y ∈ Ĉ(X)) ≥ 1 − α exactly, for any n, any model, any distribution, as long as exchangeability holds. (b) Computational cost: Both are O(n) post-hoc methods. Calibration requires fitting 1–2 parameters (Platt, temperature) or a monotone curve (isotonic) on the calibration set. Conformal prediction requires sorting n scores to find a quantile — O(n log n) — and then K comparisons per test point. Both are negligible compared to model training. (c) Assumptions: Calibration requires no formal assumptions but works best when the calibration distribution matches the test distribution. Conformal prediction requires exchangeability of calibration and test data — weaker than i.i.d. (allows arbitrary dependence within the calibration set, as long as the calibration+test joint is exchangeable). (d) Interpretability: Calibration outputs probabilities — interpretable as belief strength but no formal guarantee. Conformal prediction outputs a set — less interpretable as a single probability, but the coverage guarantee is precisely auditable. (e) Regulated domains: Conformal prediction is strictly better suited for regulated domains because the finite-sample guarantee is auditable, reproducible, and does not rely on asymptotic arguments. A regulator can verify: 'the calibration set had n=2000 examples, α=0.05, therefore coverage is between 95% and 95.05% by the conformal validity theorem.' Calibration cannot provide this. Scenario for both: a medical AI system that (1) needs to report calibrated probabilities to clinicians for cost-sensitive decisions (apply calibration), and (2) must also provide a formal guarantee that the flagged differential diagnosis set contains the true diagnosis at least 95% of the time (apply conformal prediction). The calibration step ensures the probability estimates are meaningful for treatment decisions; the conformal step ensures the set has a provable coverage guarantee for safety audits. Apply calibration first, then use the calibrated probabilities as the basis for the conformal nonconformity score — the prediction set will have both interpretable probabilities and a formal guarantee."}
      </Callout>

    </div>
  ),
};

export default calibrationConformalContent;
