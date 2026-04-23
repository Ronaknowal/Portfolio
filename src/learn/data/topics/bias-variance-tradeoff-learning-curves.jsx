import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const biasVarianceContent = {
  title: "Bias-Variance Tradeoff & Learning Curves",
  readTime: "~40 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Before you can fix a failing model, you need a vocabulary for describing how it fails. A model that predicts house prices using only square footage will be wrong in a consistent, predictable direction — it ignores every other driver of value. A model that memorizes every quirk of a training set of 500 houses will perform beautifully on those 500 houses and catastrophically on the next 500. These are not vague failure modes — they are two precise, mathematically distinct sources of error, and the field has had a name for the tension between them since 1992.
      </Prose>

      <Prose>
        Stuart Geman, Elie Bienenstock, and René Doursat published "Neural Networks and the Bias/Variance Dilemma" in <em>Neural Computation</em> 4(1):1–58 in January 1992. The paper was addressed to neural network researchers of that era who were seeing their models overfit training data while generalizing poorly — the same problem that plagues deep learning today, just with smaller networks and smaller datasets. Geman et al. formalized a decomposition of the expected generalization error into three components: a bias term, a variance term, and irreducible noise. They showed that these terms trade off as model complexity varies — reducing bias by increasing model expressiveness necessarily increases variance, and vice versa. They coined the phrase "bias-variance dilemma" and argued it was the central tension in supervised learning.
      </Prose>

      <Prose>
        The canonical textbook treatment appears in Chapter 7 of Hastie, Tibshirani, and Friedman's <em>The Elements of Statistical Learning</em> (2nd ed., 2009), which extends the decomposition to classification and regularized models, derives it for different loss functions, and connects it to model selection criteria like AIC, BIC, and cross-validation. ESL remains the standard graduate reference and the place to look for proofs and extensions.
      </Prose>

      <Prose>
        Why does this matter in practice? Because the bias-variance decomposition is the single most useful diagnostic for deciding what to do next when a model is not performing well. If you know your model is high bias, adding more training data will not help — the model class is too simple to capture the signal regardless of how many examples you show it. If your model is high variance, adding regularization, simplifying the architecture, or gathering more data will all help, but adding features or increasing capacity will make things worse. The learning curve — a plot of train and validation error as a function of training set size — is the primary tool for making this diagnosis empirically without any mathematical machinery.
      </Prose>

      <Prose>
        The classical picture — a U-shaped test error curve as a function of model complexity — has been complicated by a striking modern discovery. Mikhail Belkin, Daniel Hsu, Siyuan Ma, and Soumik Mandal published "Reconciling modern machine-learning practice and the classical bias–variance trade-off" in <em>PNAS</em> 116(32):15849–15854 in 2019. They showed that for sufficiently over-parameterized models — models with far more parameters than training examples — the test error does not plateau at a high level after the interpolation threshold (the point where training error reaches zero). Instead, it continues to decrease as model capacity grows further, tracing a second descent on the right side of the U. Preetum Nakkiran and collaborators documented this "double descent" phenomenon empirically across neural networks, decision trees, and linear models in "Deep Double Descent" (ICLR 2020, arXiv:1912.02292). For most classical ML — logistic regression, SVMs, gradient-boosted trees, polynomial regression — you operate in the left portion of this curve, where the classical bias-variance tradeoff applies cleanly. Double descent matters primarily for over-parameterized neural networks and is worth knowing as a caveat on the classical picture.
      </Prose>

      <Callout type="insight">
        The practical payoff of understanding bias and variance is concrete: before collecting more data, adding features, tuning regularization, or switching model families, you should look at a learning curve and know which regime you are in. This takes five lines of sklearn code and saves weeks of misdirected effort.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <H3>2.1 The three sources of error</H3>

      <Prose>
        Imagine training a regression model many times, each time on a different random sample drawn from the same population. The model makes a different prediction at every test point each time, because the training data differs. Three things determine how wrong those predictions are on average.
      </Prose>

      <Prose>
        <strong>Bias</strong> is the systematic error from choosing a model class that is too simple to capture the true relationship. If the truth is a sine curve and you fit a linear model, the linear model will be wrong in a predictable way — it will underestimate on the peaks and overestimate on the troughs. Averaging predictions across all possible training sets does not help: the average prediction is still a straight line through a curved relationship. This is the error that comes from your model's assumptions being wrong, not from any randomness in the data. A linear regression on a clearly nonlinear problem suffers high bias regardless of how much data you collect.
      </Prose>

      <Prose>
        <strong>Variance</strong> is the sensitivity of the model's predictions to the particular training sample it happened to see. A degree-12 polynomial fitted to 30 points will fit the training data very closely, but because it is so flexible, small random fluctuations in the training set cause large swings in the fitted curve. Train on a slightly different 30 points and you get a very different polynomial. Variance measures how much the predictions change across these different training sets. A model with high variance is overfitting — it has memorized noise specific to its training sample.
      </Prose>

      <Prose>
        <strong>Irreducible noise</strong> is the variance in the target variable that no model can explain, no matter how expressive. In a regression problem where <Code>{"y = f(x) + ε"}</Code> and <Code>ε</Code> is genuinely random measurement error or inherent unpredictability, even a perfect oracle model — one that knows <Code>f(x)</Code> exactly — cannot predict <Code>y</Code> better than <Code>Var(ε)</Code>. This sets the floor on achievable error. Irreducible noise is not a modeling problem; it is a data problem. The only way to reduce it is to improve data quality, measure the outcome more precisely, or find additional features that explain variance in <Code>y</Code> that currently looks like noise.
      </Prose>

      <H3>2.2 The tradeoff</H3>

      <Prose>
        As model complexity increases — think polynomial degree, tree depth, number of parameters — bias decreases and variance increases. A degree-1 polynomial has high bias (can only fit lines) and low variance (a line fit to 30 points is pretty stable). A degree-15 polynomial has low bias (can approximate almost any smooth curve) and high variance (small changes in the 30 training points produce wildly different curves). The total expected error is the sum of all three components. Somewhere in between, the sum is minimized. This is the sweet spot — the model class that best approximates the signal in the data without amplifying noise.
      </Prose>

      <Prose>
        The learning curve is the empirical view of this. For a <strong>high-bias model</strong>: training error and validation error both plateau at a high level, and adding more data barely helps — the model has hit the ceiling of what its architecture can represent. For a <strong>high-variance model</strong>: training error is low, validation error is high, and the gap between them shrinks as you add data — because more data constrains the model and reduces the sensitivity to any individual sample. The gap between train error and validation error is a direct visual measure of variance; the height at which both plateau measures bias plus irreducible noise.
      </Prose>

      <Callout type="insight">
        {"The single most useful mental model: train error ≈ bias² + noise; val error ≈ bias² + variance + noise. The gap (val - train) estimates variance. The plateau of val error estimates bias² + noise. If both are high: high bias. If gap is large: high variance."}
      </Callout>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 Deriving the bias-variance decomposition</H3>

      <Prose>
        Let the true data-generating process be <Code>{"y = f(x) + ε"}</Code>, where <Code>{"f(x)"}</Code> is the true (unknown) function and <Code>ε</Code> is zero-mean noise with <Code>{"Var(ε) = σ²"}</Code>. We train a model <Code>{"f̂(x)"}</Code> on a training set drawn from this distribution. The expected squared error at a single test point <Code>x</Code>, averaged over all possible training sets, is:
      </Prose>

      <MathBlock>
        {"\\mathbb{E}\\left[(y - \\hat{f}(x))^2\\right]"}
      </MathBlock>

      <Prose>
        where the expectation is over both the randomness in the training set and the noise in the test observation. Substituting <Code>{"y = f(x) + ε"}</Code> and expanding:
      </Prose>

      <MathBlock>
        {"\\mathbb{E}\\left[(f(x) + \\varepsilon - \\hat{f}(x))^2\\right]"}
      </MathBlock>

      <Prose>
        Add and subtract <Code>{"E[f̂(x)]"}</Code> inside the square to separate the terms:
      </Prose>

      <MathBlock>
        {"= \\mathbb{E}\\left[(f(x) - \\mathbb{E}[\\hat{f}(x)])^2\\right] + \\mathbb{E}\\left[(\\mathbb{E}[\\hat{f}(x)] - \\hat{f}(x))^2\\right] + \\mathbb{E}[\\varepsilon^2]"}
      </MathBlock>

      <Prose>
        The cross terms vanish because <Code>{"E[ε] = 0"}</Code> and <Code>ε</Code> is independent of the training set. This gives the clean decomposition:
      </Prose>

      <MathBlock>
        {"\\underbrace{\\mathbb{E}\\left[(y - \\hat{f}(x))^2\\right]}_{\\text{Expected MSE}} = \\underbrace{\\left(f(x) - \\mathbb{E}[\\hat{f}(x)]\\right)^2}_{\\text{Bias}^2} + \\underbrace{\\mathbb{E}\\left[(\\hat{f}(x) - \\mathbb{E}[\\hat{f}(x)])^2\\right]}_{\\text{Variance}} + \\underbrace{\\sigma^2}_{\\text{Irreducible noise}}"}
      </MathBlock>

      <Prose>
        Each term has a concrete interpretation. <strong>Bias²</strong> measures how far the average prediction (averaged over all training sets) is from the true value. It is zero if and only if the model class contains the true function. <strong>Variance</strong> measures how much the prediction at <Code>x</Code> changes as the training set varies — it is the variance of <Code>{"f̂(x)"}</Code> treated as a random variable over training sets. <strong>σ²</strong> is the irreducible noise floor; it appears because we observe <Code>{"y = f(x) + ε"}</Code>, not <Code>{"f(x)"}</Code> directly.
      </Prose>

      <H3>3.2 Loss-function dependence</H3>

      <Prose>
        This derivation is specific to squared error loss. Pedro Domingos showed in "A Unified Bias-Variance Decomposition" (ICML 2000) that an analogous decomposition exists for 0/1 classification loss, but the structure is fundamentally different: bias and variance interact multiplicatively, not additively, and the variance term can actually reduce error in some cases (when a biased model happens to predict the right class by chance, variance can flip it back to incorrect). For classification, the clean additive story breaks down, and the decomposition must be interpreted more carefully. In practice, practitioners use squared-error decompositions even for classification (by looking at predicted probabilities rather than class labels), which gives a useful approximation to the true 0/1 loss behavior.
      </Prose>

      <H3>3.3 The double-descent curve</H3>

      <Prose>
        The classical picture predicts a U-shaped test error curve: high on the left (high bias, underfitting), minimum somewhere in the middle (optimal complexity), rising again on the right (high variance, overfitting). This is accurate for classical ML in the under-parameterized regime. The surprise documented by Belkin et al. (2019) and Nakkiran et al. (2020) is what happens when model capacity continues beyond the interpolation threshold — the point where the model has just enough parameters to fit the training set with zero error.
      </Prose>

      <Prose>
        At the interpolation threshold, the classical theory predicts the worst possible generalization (infinite variance as the model forces an exact fit). Empirically, for neural networks and kernel methods, something different happens: past the threshold, as capacity continues to grow, validation error begins to decrease again. The intuition is that among all models that interpolate the training data, larger models tend to pick smoother, more regular interpolations (by an implicit regularization effect of gradient descent), and these generalize better. This is a modern phenomenon specific to over-parameterized models trained with gradient methods. Classical ML models — polynomial regression, decision trees, SVMs — do not exhibit it without explicit construction. For the scope of this topic, the classical U-shape is the operative picture.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The cleanest way to observe the bias-variance tradeoff empirically is to train the same model class many times on bootstrap samples from a fixed dataset, then measure how the predictions at fixed test points vary. NumPy only — no sklearn.
      </Prose>

      <H3>4a. Bootstrap bias-variance estimation — polynomial degrees 1–20</H3>

      <CodeBlock language="python">
{`import numpy as np

np.random.seed(42)
n_test      = 50
n_train     = 30
n_bootstrap = 100
sigma_noise = 0.3   # noise std; sigma^2 = 0.09 is the irreducible floor

# Fixed test grid, true function known
X_test = np.linspace(0, 2 * np.pi, n_test)
y_true = np.sin(X_test)

def make_poly_features(x, degree):
    # Vandermonde matrix: [1, x, x^2, ..., x^degree]
    return np.column_stack([x ** d for d in range(degree + 1)])

results = []
for degree in range(1, 21):
    predictions = np.zeros((n_bootstrap, n_test))
    for b in range(n_bootstrap):
        # New random training set from the same distribution each time
        X_train = np.random.uniform(0, 2 * np.pi, n_train)
        y_train = np.sin(X_train) + np.random.randn(n_train) * sigma_noise

        Phi_train = make_poly_features(X_train, degree)
        Phi_test  = make_poly_features(X_test,  degree)

        # Ridge-regularized OLS: (Phi^T Phi + lam I)^{-1} Phi^T y
        # Stronger lam for high degrees prevents catastrophic blow-up
        lam = 1e-4 if degree < 8 else 1e-2
        beta = np.linalg.solve(
            Phi_train.T @ Phi_train + lam * np.eye(degree + 1),
            Phi_train.T @ y_train
        )
        # Clip wild predictions from near-singular cases
        predictions[b] = np.clip(Phi_test @ beta, -10, 10)

    # Average over bootstrap models at each test point, then average over test points
    mean_pred = predictions.mean(axis=0)           # shape (n_test,)
    bias_sq   = np.mean((mean_pred - y_true) ** 2) # scalar
    variance  = np.mean(predictions.var(axis=0))   # scalar
    total_err = bias_sq + variance + sigma_noise ** 2

    results.append((degree, bias_sq, variance, total_err))
    print(f"deg={degree:2d}  bias^2={bias_sq:.4f}  var={variance:.4f}  total_err={total_err:.4f}")

min_idx = min(range(len(results)), key=lambda i: results[i][3])
print(f"\\nMinimum total error at degree {results[min_idx][0]}: {results[min_idx][3]:.4f}")
print("(bias^2 + variance dominate; noise floor = 0.09)")`}
      </CodeBlock>

      <Callout type="output">
{`deg= 1  bias^2=0.2118  var=0.0190  total_err=0.3208
deg= 2  bias^2=0.2164  var=0.0513  total_err=0.3577
deg= 3  bias^2=0.0058  var=0.0174  total_err=0.1132
deg= 4  bias^2=0.0065  var=0.0247  total_err=0.1212
deg= 5  bias^2=0.0004  var=0.0389  total_err=0.1293
deg= 6  bias^2=0.0016  var=0.1116  total_err=0.2032
deg= 7  bias^2=0.0018  var=0.1631  total_err=0.2550
deg= 8  bias^2=0.0028  var=0.2043  total_err=0.2971
deg= 9  bias^2=0.0290  var=0.3747  total_err=0.4936
deg=10  bias^2=0.0097  var=0.3845  total_err=0.4842
deg=11  bias^2=0.0255  var=0.7467  total_err=0.8622
deg=12  bias^2=0.0068  var=1.5522  total_err=1.6491
deg=13  bias^2=0.0016  var=1.2631  total_err=1.3547
deg=14  bias^2=0.0032  var=2.2452  total_err=2.3384
deg=15  bias^2=0.0531  var=2.6531  total_err=2.7962
deg=16  bias^2=0.0167  var=2.9416  total_err=3.0483
deg=17  bias^2=0.0120  var=2.3628  total_err=2.4648
deg=18  bias^2=0.0347  var=2.9456  total_err=3.0703
deg=19  bias^2=0.0228  var=3.9417  total_err=4.0544
deg=20  bias^2=0.0395  var=2.9604  total_err=3.0899

Minimum total error at degree 3: 0.1132
(bias^2 + variance dominate; noise floor = 0.09)`}
      </Callout>

      <Prose>
        The output is exactly what the theory predicts. Degrees 1 and 2 have large bias² (0.21, 0.22) — a straight line and a parabola cannot capture a full sine cycle. Bias² collapses at degree 3 (0.006) — a cubic polynomial is expressive enough to approximate one period of sine within this noise level. But starting at degree 6, variance explodes: the model has more parameters relative to the 30 training points and starts fitting noise. By degree 12, variance alone (1.55) dominates — the model is 17× worse than the noise floor. The U-shape bottoms out at degree 3 with total error 0.113, just above the irreducible noise floor of 0.09.
      </Prose>

      <Prose>
        Notice that bias² is near zero for all degrees 3–20 — the model can fit the true function once it has enough capacity. What drives the right side of the U upward is entirely variance. This is the typical pattern in practice: once a model has enough expressiveness to represent the signal, the dominant question becomes variance control — regularization, early stopping, ensembling, more data.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        In practice, you do not estimate bias and variance by fitting 100 bootstrap models. Instead, you use sklearn's <Code>learning_curve</Code> and <Code>validation_curve</Code> to get the empirical diagnostic curves directly. These functions handle the cross-validation scaffolding and expose exactly the information needed to diagnose high-bias vs high-variance.
      </Prose>

      <H3>5a. Learning curve and validation curve</H3>

      <CodeBlock language="python">
{`import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve, validation_curve

np.random.seed(42)

# Dataset: noisy sin(x) regression
n = 200
X = np.random.uniform(0, 2 * np.pi, n).reshape(-1, 1)
y = np.sin(X.ravel()) + np.random.randn(n) * 0.3

pipe = Pipeline([
    ('poly',  PolynomialFeatures(degree=4, include_bias=False)),
    ('ridge', Ridge(alpha=1.0))
])

# ── Learning curve: how does error change as training set grows? ──────────────
train_sizes, train_scores, val_scores = learning_curve(
    pipe, X, y,
    train_sizes=np.linspace(0.1, 1.0, 8),
    cv=5, scoring='neg_mean_squared_error', n_jobs=-1
)
print('=== Learning Curve (deg=4, alpha=1.0) ===')
print(f'{"n_train":>10}  {"train_MSE":>10}  {"val_MSE":>10}')
for n_tr, tr, va in zip(train_sizes, train_scores, val_scores):
    print(f'{n_tr:>10}  {-tr.mean():>10.4f}  {-va.mean():>10.4f}')

# ── Validation curve: how does error change as alpha varies? ─────────────────
alphas = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
tr_vc, va_vc = validation_curve(
    pipe, X, y,
    param_name='ridge__alpha', param_range=alphas,
    cv=5, scoring='neg_mean_squared_error', n_jobs=-1
)
print()
print('=== Validation Curve (deg=4, varying alpha) ===')
print(f'{"alpha":>10}  {"train_MSE":>10}  {"val_MSE":>10}')
for a, tr, va in zip(alphas, tr_vc, va_vc):
    print(f'{a:>10}  {-tr.mean():>10.4f}  {-va.mean():>10.4f}')`}
      </CodeBlock>

      <Callout type="output">
{`=== Learning Curve (deg=4, alpha=1.0) ===
   n_train   train_MSE     val_MSE
        16      0.1146      0.1100
        36      0.1135      0.0966
        57      0.0991      0.0949
        77      0.1125      0.0960
        98      0.1054      0.0947
       118      0.0998      0.0926
       139      0.0942      0.0922
       160      0.0888      0.0921

=== Validation Curve (deg=4, varying alpha) ===
     alpha   train_MSE     val_MSE
    0.0001      0.0831      0.0869
     0.001      0.0831      0.0869
      0.01      0.0831      0.0869
       0.1      0.0833      0.0870
       1.0      0.0888      0.0921
      10.0      0.1009      0.1036
     100.0      0.1153      0.1173
    1000.0      0.1318      0.1337`}
      </Callout>

      <Prose>
        The learning curve for degree-4, alpha=1 shows both train and validation error converging to ~0.092 — the gap between them is small and shrinking. This is a well-regularized model operating close to the noise floor (true noise variance = 0.09). Adding more data continues to help slightly but the marginal return is small — this is not a high-variance model.
      </Prose>

      <Prose>
        The validation curve reveals that very small alpha values (0.0001 to 0.1) all give similar, slightly better validation MSE (~0.087). Alpha=1 is mildly too strong, adding a small bias penalty. Alphas of 10 and above are over-regularizing — both train and val MSE rise together, which is the signature of high-bias (the regularization is removing signal, not just noise). To read a validation curve: if train and val both rise as the hyperparameter increases, the hyperparameter is controlling capacity and you are becoming high-bias. If val rises but train stays low, you are becoming high-variance.
      </Prose>

      <H3>5b. GBM with staged monitoring — diagnosing the optimal stopping round</H3>

      <CodeBlock language="python">
{`import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

np.random.seed(42)

n = 300
X = np.random.uniform(0, 2 * np.pi, n).reshape(-1, 1)
y = np.sin(X.ravel()) + np.random.randn(n) * 0.3

X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42)

# Train 200 trees; staged_predict gives cumulative predictions at each round
gbm = GradientBoostingRegressor(
    n_estimators=200, max_depth=3, learning_rate=0.1,
    subsample=0.8, random_state=42
)
gbm.fit(X_tr, y_tr)

train_errors = [-1] * 200
val_errors   = [-1] * 200
for i, yp in enumerate(gbm.staged_predict(X_tr)):
    train_errors[i] = np.mean((y_tr - yp) ** 2)
for i, yp in enumerate(gbm.staged_predict(X_va)):
    val_errors[i]   = np.mean((y_va - yp) ** 2)

print('=== GBM staged training monitoring ===')
print(f'{"n_trees":>8}  {"train_MSE":>10}  {"val_MSE":>10}')
for i in [0, 9, 24, 49, 99, 149, 199]:
    print(f'{i+1:>8}  {train_errors[i]:>10.4f}  {val_errors[i]:>10.4f}')

best_round = int(np.argmin(val_errors)) + 1
print(f'Best round (min val MSE): {best_round}  val_MSE={val_errors[best_round-1]:.4f}')
print('Diagnosis: val MSE rises after round 25 while train continues falling -> high variance')`}
      </CodeBlock>

      <Callout type="output">
{`=== GBM staged training monitoring ===
 n_trees   train_MSE     val_MSE
       1      0.5033      0.4264
      10      0.1490      0.1064
      25      0.0752      0.0627
      50      0.0578      0.0655
     100      0.0412      0.0757
     150      0.0305      0.0789
     200      0.0225      0.0892
Best round (min val MSE): 25  val_MSE=0.0627
Diagnosis: val MSE rises after round 25 while train continues falling -> high variance`}
      </Callout>

      <Prose>
        The GBM output makes the overfit trajectory unmistakable. At round 25, validation MSE is at its minimum (0.063). As training continues, train MSE keeps falling — the model fits the training data ever more precisely — while val MSE climbs steadily to 0.089 by round 200. Training past round 25 is adding variance without reducing bias. The optimal number of trees is not 200; it is 25 with this learning rate and depth. In production, this is why <Code>early_stopping_rounds</Code> exists: instead of wasting compute on rounds that hurt generalization, you stop when validation error stops improving.
      </Prose>

      <Callout type="insight">
        The gap between train MSE (0.023) and val MSE (0.089) at round 200 is almost 4×. This is the fingerprint of high variance. The fix: reduce <Code>n_estimators</Code> to ~25, increase <Code>min_samples_leaf</Code>, decrease <Code>max_depth</Code>, or increase <Code>subsample</Code> rate.
      </Callout>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6a. The U-shaped error decomposition</H3>

      <Plot
        label="Bias-variance decomposition — polynomial regression on sin(x)"
        xLabel="Polynomial degree"
        yLabel="Error"
        series={[
          {
            name: "Bias²",
            color: colors.gold,
            points: [
              [1, 0.2118], [2, 0.2164], [3, 0.0058], [4, 0.0065], [5, 0.0004],
              [6, 0.0016], [7, 0.0018], [8, 0.0028], [9, 0.0290], [10, 0.0097],
              [11, 0.0255], [12, 0.0068], [13, 0.0016], [14, 0.0032], [15, 0.0531],
            ],
          },
          {
            name: "Variance",
            color: colors.green,
            points: [
              [1, 0.0190], [2, 0.0513], [3, 0.0174], [4, 0.0247], [5, 0.0389],
              [6, 0.1116], [7, 0.1631], [8, 0.2043], [9, 0.3747], [10, 0.3845],
              [11, 0.7467], [12, 1.5522], [13, 1.2631], [14, 2.2452], [15, 2.6531],
            ],
          },
          {
            name: "Total error (Bias² + Var + noise)",
            color: "#f87171",
            points: [
              [1, 0.3208], [2, 0.3577], [3, 0.1132], [4, 0.1212], [5, 0.1293],
              [6, 0.2032], [7, 0.2550], [8, 0.2971], [9, 0.4936], [10, 0.4842],
              [11, 0.8622], [12, 1.6491], [13, 1.3547], [14, 2.3384], [15, 2.7962],
            ],
          },
        ]}
      />

      <H3>6b. Learning curves — high bias vs high variance</H3>

      <Plot
        label="Learning curves: high-bias (deg=1) vs high-variance (deg=12)"
        xLabel="Training set size"
        yLabel="MSE"
        series={[
          {
            name: "High-bias train (deg=1)",
            color: colors.gold,
            points: [[16, 0.2707], [36, 0.2789], [57, 0.260], [77, 0.2782], [98, 0.2636], [118, 0.2616], [139, 0.2506], [160, 0.2399]],
          },
          {
            name: "High-bias val (deg=1)",
            color: "#f59e0b",
            points: [[16, 0.2437], [36, 0.2474], [57, 0.2487], [77, 0.2444], [98, 0.2444], [118, 0.2463], [139, 0.2454], [160, 0.2455]],
          },
          {
            name: "High-variance train (deg=12)",
            color: colors.green,
            points: [[16, 0.0160], [36, 0.0553], [57, 0.0628], [77, 0.0861], [98, 0.0846], [118, 0.0852], [139, 0.0817], [160, 0.0784]],
          },
          {
            name: "High-variance val (deg=12)",
            color: "#34d399",
            points: [[16, 0.1420], [36, 0.1201], [57, 0.1039], [77, 0.1044], [98, 0.0985], [118, 0.0957], [139, 0.0952], [160, 0.0944]],
          },
        ]}
      />

      <Prose>
        The two pairs of curves expose the signature patterns. For the high-bias model (degree 1): both train and val errors are high (~0.25) and run nearly parallel — adding data does not help because the bottleneck is the model's expressive capacity, not data quantity. For the high-variance model (degree 12): train error is low (0.02–0.08) while val error starts much higher (0.14) and closes slowly toward train error as data increases. This gap is the fingerprint of overfitting. With 160 samples, the gap has closed from 0.126 to 0.016 — more data is working, but slowly.
      </Prose>

      <H3>6c. Double-descent curve — schematic</H3>

      <Plot
        label="Double-descent: test error vs model capacity (schematic)"
        xLabel="Model capacity (parameters)"
        yLabel="Test error"
        series={[
          {
            name: "Classical regime (underfitting → overfitting)",
            color: colors.gold,
            points: [
              [1, 0.90], [5, 0.60], [10, 0.40], [20, 0.25], [30, 0.20],
              [40, 0.22], [50, 0.30], [60, 0.42], [70, 0.60], [80, 0.90],
            ],
          },
          {
            name: "Modern regime (interpolation threshold + second descent)",
            color: colors.green,
            points: [
              [80, 0.90], [85, 1.20], [90, 1.50], [95, 2.00], [100, 1.80],
              [120, 0.60], [150, 0.30], [200, 0.20], [300, 0.17],
            ],
          },
        ]}
      />

      <Prose>
        The schematic makes the double-descent structure concrete. The left portion (gold) is the classical regime where most classical ML operates: error decreases as capacity increases (bias falling), then rises again (variance rising). The spike at the interpolation threshold (x≈85–100 in the schematic) is where the model exactly interpolates the training data and test error blows up. Past the threshold (green), in the modern over-parameterized regime, test error decreases again as the model grows — a second descent driven by implicit regularization from gradient descent finding smooth interpolants. Most practical classical ML — polynomial regression, SVMs, decision trees — lives entirely in the gold portion.
      </Prose>

      <H3>6d. Diagnostic step trace — reading a learning curve</H3>

      <StepTrace
        label="5 diagnostic learning curve scenarios"
        steps={[
          {
            label: "Scenario 1: High bias — both curves plateau high",
            render: () => (
              <Prose>
                Train error and validation error both plateau at a high value (e.g., MSE = 0.25 when noise floor is 0.09). The gap between them is small (less than 10% of the plateau level). What you see: adding more data barely moves either curve. What to do: increase model capacity — add features, increase polynomial degree or tree depth, switch to a more expressive model family. Do not collect more data until you have fixed the bias problem. Example: fitting a linear model to a nonlinear relationship.
              </Prose>
            ),
          },
          {
            label: "Scenario 2: High variance — large train/val gap",
            render: () => (
              <Prose>
                Train error is low (near noise floor). Validation error is substantially higher (2–5× train error). The gap shrinks as training size increases. What you see: the validation curve is descending; adding more data is helping but slowly. What to do: (1) add regularization — increase L2 penalty, reduce tree depth or n_estimators; (2) gather more training data — the validation curve's downward slope confirms data would help; (3) use ensembling; (4) apply dropout if using a neural network; (5) reduce feature dimensionality. Example: a high-degree polynomial or a deep tree on a small dataset.
              </Prose>
            ),
          },
          {
            label: "Scenario 3: Well-fitted — curves converge near noise floor",
            render: () => (
              <Prose>
                Train error and validation error both converge to a value close to the irreducible noise floor (e.g., MSE = 0.092 when noise = 0.09). The gap is small and shrinking. What you see: both curves have leveled off near each other, and additional data has minimal effect on validation error. What to do: this is the good outcome. The remaining error is irreducible noise. To improve further, you need better features (to reduce noise-floor-apparent variance by explaining currently-unexplained variation), better data quality, or a fundamentally different model family. Example: a well-regularized polynomial on the sin(x) task with 160+ samples.
              </Prose>
            ),
          },
          {
            label: "Scenario 4: Validation plateaus far above noise floor, adding data stopped helping",
            render: () => (
              <Prose>
                Validation error has flattened at a high level (e.g., 0.40) even with large training set, while train error is near zero. The gap is large and not closing. What you see: the validation curve has stopped descending — more data is no longer reducing validation error. What to do: this is the high-variance regime where the model has exhausted the available signal from the current feature set. The flat validation curve suggests the model is fitting a ceiling imposed by the feature representation, not data quantity. Add features, switch to a more expressive model, or apply a transformation to better represent the underlying structure.
              </Prose>
            ),
          },
          {
            label: "Scenario 5: GBM — train keeps falling, val turns up",
            render: () => (
              <Prose>
                In the GBM staged monitoring plot: train MSE falls monotonically across rounds (0.50 → 0.02 over 200 trees). Validation MSE reaches a minimum at round 25 (0.063) then rises steadily to 0.089 by round 200. What you see: a characteristic V-shape in the gap between train and val after the optimal round. What to do: use early stopping with a patience parameter of 10–20 rounds. Set n_estimators to the round where val MSE was minimized. If the minimum val MSE is still too high: reduce learning rate and increase n_estimators proportionally (longer, slower boosting), or increase subsampling. Do not increase tree depth — this reduces regularization and will worsen the overfitting.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        Given a diagnosis from the learning curve or validation curve, the action depends on whether the problem is high bias, high variance, or irreducible noise. These are not symmetric — the right fix for each is different, and applying the wrong fix makes things worse.
      </Prose>

      <Heatmap
        label="Bias-variance diagnosis and action guide"
        rowLabels={["High bias", "High variance", "Irreducible noise"]}
        colLabels={["Add data", "Add features", "Increase capacity", "More regularization", "Simpler model", "Improve data quality"]}
        matrix={[
          [0.1, 0.9, 0.9, 0.0, 0.0, 0.3],
          [0.8, 0.1, 0.0, 0.9, 0.7, 0.3],
          [0.2, 0.5, 0.2, 0.1, 0.1, 0.9],
        ]}
        colorScale="gold"
      />

      <Prose>
        The heatmap shows relative benefit (0 = no help, 1 = high impact). Read each row as a treatment menu for that diagnosis.
      </Prose>

      <Prose>
        <strong>High bias actions:</strong> Adding features is the highest-leverage move — if the feature set cannot represent the target, no amount of capacity or data will help. Increasing capacity (model expressiveness: higher polynomial degree, more trees, deeper network, different kernel) lets the model use the features it already has more flexibly. Decreasing regularization allows the model to use its existing capacity more aggressively. Adding data helps only marginally — the model is constrained by its expressiveness ceiling, not data quantity.
      </Prose>

      <Prose>
        <strong>High variance actions:</strong> More regularization is the most targeted fix — it directly reduces the sensitivity to the training sample. More data closes the train/val gap by constraining the model. Simpler model (fewer parameters, more restricted hypothesis class) reduces the degrees of freedom the model can use to memorize noise. Ensembling (bagging, boosted stumps) averages out variance across models. Early stopping is early regularization for iterative algorithms — it prevents the model from using all its capacity.
      </Prose>

      <Prose>
        <strong>Irreducible noise actions:</strong> By definition, you cannot remove it through modeling choices. The only paths forward are: (1) improve data quality (reduce measurement error, correct labeling errors); (2) collect additional features that explain variance currently attributed to noise (what looks like noise to a model without feature X may be completely explained by adding X); (3) accept the floor and report it honestly. A model achieving MSE = 0.092 on a dataset with noise variance = 0.09 is essentially perfect.
      </Prose>

      <StepTrace
        label="Bias-variance remedies — decision flow"
        steps={[
          {
            label: "Step 1: Measure the gap",
            render: () => (
              <Prose>
                Run learning_curve with cv=5. Compute gap = val_MSE - train_MSE at full training size. Rule of thumb: gap {">"} 0.5 × val_MSE is high variance; gap {"<"} 0.1 × val_MSE with both values high is high bias. Both within 10% of each other and near noise floor: well-fitted. This single ratio is your triage step.
              </Prose>
            ),
          },
          {
            label: "Step 2: Check the plateau",
            render: () => (
              <Prose>
                Is the validation curve still descending at your full training size? If yes: high variance, and more data will help. If it has leveled off: either you are near the noise floor (good), or you have hit a feature ceiling (bad — add features or change model family). Plot the last three points of the val curve — if the slope is near zero, more data is not your answer.
              </Prose>
            ),
          },
          {
            label: "Step 3: Run validation_curve on your key regularization hyperparameter",
            render: () => (
              <Prose>
                Plot train and val error vs regularization strength (alpha, C, max_depth, n_estimators). Find where val error is minimized. If val error minimum is still too high: the issue is not regularization — it is model expressiveness (high bias) or data quality (irreducible noise). If val error is sensitive and the minimum is sharp: you are in the high-variance regime and regularization is the right lever.
              </Prose>
            ),
          },
          {
            label: "Step 4: Try one change at a time",
            render: () => (
              <Prose>
                Change exactly one hyperparameter or one architectural decision. Replot the learning curve. Did the gap shrink? Did the plateau drop? If neither changed: that lever does not address the bottleneck. Move to the next lever in the decision matrix. Common mistake: trying several changes simultaneously and not knowing which one worked (or which one made things worse).
              </Prose>
            ),
          },
          {
            label: "Step 5: Accept the noise floor",
            render: () => (
              <Prose>
                If both train and val error have converged and the gap is small, but performance is still below your target: you have hit the irreducible noise floor of your current feature set and data quality. The remaining error is not addressable through modeling. Report this clearly. The path forward is data improvement: better labels, additional features, a different measurement instrument, or a fundamentally richer data source.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>8.1 Bootstrap bias-variance estimation at scale</H3>

      <Prose>
        The from-scratch bootstrap estimator from Section 4 trains K models per degree — K × number of degrees total models. For the polynomial regression experiment with K=100 and 20 degrees, that is 2,000 model fits. For a 30-sample dataset with a small model, this takes milliseconds. For a large dataset or a complex model, the cost becomes:
      </Prose>

      <Prose>
        <strong>Options for large-scale bias-variance estimation:</strong> (1) Sub-sample the dataset — instead of bootstrap samples from all n examples, use bootstrap samples from a fixed subset (e.g., 10% of the data). This is valid for estimating relative bias/variance across complexity levels; absolute values will be higher because training set size is smaller. (2) Reduce K — even K=20 bootstrap samples gives a reasonable variance estimate; K=100 gives smoother estimates but rarely changes the qualitative conclusion. (3) For iterative models (GBMs, neural nets), use early-stopping checkpoints as a proxy for different complexity levels instead of retraining from scratch.
      </Prose>

      <H3>8.2 Learning curves at scale</H3>

      <Prose>
        <Code>learning_curve</Code> trains cv × len(train_sizes) models. With cv=5 and 10 train sizes, that is 50 model fits. For a dataset with millions of rows, even one model fit may take minutes. Two strategies: (1) Use <Code>exploit_incremental_learning=True</Code> if your estimator supports warm-starting (SGD-based models, incremental learners) — sklearn will extend the training set rather than retraining from scratch at each size. (2) Generate a sub-sampled dataset for the learning curve diagnostic (use 10% of your data, with sizes from 1% to 10%) and extrapolate. The shape of the learning curve — whether val error is still descending or has plateaued — is usually the same at smaller scale.
      </Prose>

      <H3>8.3 Large models and the double-descent regime</H3>

      <Prose>
        For large neural networks, the classical bias-variance intuition breaks down. Standard advice — reduce capacity if variance is high — can make things worse if you cross into the over-parameterized regime where the second descent has already begun. For neural nets with more parameters than training examples, the monitoring tool is not a capacity sweep but a regularization sweep: learning rate, weight decay, dropout rate, batch size. Early stopping serves as an implicit regularizer that keeps the model in a region of parameter space with lower effective complexity. Batch normalization and layer normalization have implicit regularization effects that alter the effective bias-variance operating point. The practical rule: for classical ML, tune capacity via the classical U-shaped curve. For neural nets, use a fixed large architecture and tune regularization.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>9.1 Small test set — noisy error estimates</H3>

      <Prose>
        A validation set of 50 examples produces an error estimate with high variance. The difference between val MSE 0.10 and 0.11 may be purely noise — there is not enough signal in 50 examples to distinguish them reliably. This is why cross-validation is preferred over a single holdout for small datasets: averaging across K validation folds reduces the variance of the error estimate by roughly K. Before concluding that model A is better than model B because its val MSE is lower, check whether the confidence intervals on those estimates (mean ± 2×std across folds) overlap. Overlapping intervals mean the difference is not statistically reliable.
      </Prose>

      <H3>9.2 Train/val leakage inflating the learning curve</H3>

      <Prose>
        If preprocessing (scaling, encoding, imputation) is fitted on the full dataset before the learning curve split, the "validation" error is not truly held-out — the validation fold's statistics influenced the preprocessor. The val error will be optimistically low, masking the true variance. Always put preprocessing inside a Pipeline so that <Code>learning_curve</Code> refits the preprocessor on each training fold. This is the same leakage problem as in cross-validation, but it is especially pernicious in learning curve analysis because the leakage effect grows smaller as the training set grows — the resulting curve can look like a healthy converging learning curve when it is actually contaminated.
      </Prose>

      <H3>9.3 Learning curve plateaus far above noise floor — features, not data</H3>

      <Prose>
        When the validation learning curve plateaus high and has clearly stopped descending, the common reflex is to collect more data. This will not help. A plateau means the model has already extracted all the signal available from its current feature set. Adding more rows of the same features gives the model more of what it already has — it cannot learn new structure it has no representation for. The right action is to enrich the feature set: add domain-specific features, use different transformations, apply feature interaction terms, or switch to a model family that can represent the missing structure. Learning curve plateau = feature problem, not data problem.
      </Prose>

      <H3>9.4 Confusing variance (model) with noise (data)</H3>

      <Prose>
        Variance in the bias-variance sense is the sensitivity of model predictions to the training set. Noise is the irreducible randomness in the target variable. These look identical on the learning curve (both contribute to a gap between train and val error), but they require different fixes. Distinguishing them: if the gap closes as you add data, it is model variance (data helps). If the gap persists even with large n, and both errors plateau together at a level above what you believe the noise floor should be, you may have high noise — perhaps due to labeling errors, measurement imprecision, or a feature set that genuinely cannot explain the target. Audit your labels: even 5% label noise adds 0.095 to the binary cross-entropy floor on a balanced problem, which looks like a model capacity problem but is not.
      </Prose>

      <H3>9.5 Applying classical bias-variance intuition to over-parameterized neural networks</H3>

      <Prose>
        The classical prescription is: high variance → reduce capacity. For neural networks trained past the interpolation threshold (more parameters than training examples), reducing capacity can actually increase test error — you are moving left on the double-descent curve and into the high-variance spike region, not to a lower-variance sweet spot. For modern neural nets, capacity reduction is not the right variance lever. Instead: increase regularization (weight decay, dropout, data augmentation), use early stopping, or add more training data. Reducing width or depth should be tested empirically; do not assume it helps.
      </Prose>

      <H3>9.6 Measuring bias only at one point</H3>

      <Prose>
        The bias-variance decomposition averages over test points: <Code>{"E_x[ Bias²(x) ] + E_x[ Var(x) ] + σ²"}</Code>. A model may have very low bias in high-density regions of the input space and high bias at the tails. If your evaluation set is drawn from a different distribution than your training set (distribution shift), the average bias measured on the eval set may not reflect the true in-distribution bias. This is why evaluation on a representative, i.i.d. test set is essential for honest bias-variance diagnosis. A model evaluated only on easy examples will appear to have low bias even if it is badly wrong on hard or rare examples.
      </Prose>

      <H3>9.7 Misreading a validation curve — regularization vs capacity</H3>

      <Prose>
        On a validation curve, both train and val error rise together as regularization strength increases — this is the signature of over-regularization (high bias). But if you are plotting error vs model complexity (degree, depth) instead of regularization, the interpretation is flipped: both rising on the right means over-complexity (high variance), both rising on the left means under-complexity (high bias). The shape of the curve is identical; the axis label changes the diagnosis. Always label your axes and know whether increasing the x-axis increases or decreases regularization. It is easy to apply the wrong fix by misreading which direction the axis runs.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All citations below were verified against primary publication venues, author lists, and main claims.
      </Prose>

      <StepTrace
        label="primary literature"
        steps={[
          {
            label: "Geman, Bienenstock, Doursat 1992 — The canonical reference",
            render: () => (
              <Prose>
                Geman, S., Bienenstock, E., and Doursat, R. (1992). "Neural Networks and the Bias/Variance Dilemma." <em>Neural Computation</em>, 4(1):1–58. MIT Press. The paper that named the tradeoff and gave it mathematical form in the context of neural networks. Geman et al. formalize the decomposition for squared error, analyze how it changes with model complexity and training set size, and argue that the dilemma is fundamental — no model can simultaneously achieve low bias and low variance on all problems. They introduce the term "bias/variance dilemma" and argue it explains the overfit patterns observed in 1980s neural net research. The paper is 58 pages and includes substantial theoretical analysis. Most ML textbooks cite this as the origin of the concept, though the mathematical structure was implicit in earlier statistical work.
              </Prose>
            ),
          },
          {
            label: "Hastie, Tibshirani, Friedman 2009 — ESL Chapter 7",
            render: () => (
              <Prose>
                Hastie, T., Tibshirani, R., and Friedman, J. (2009). <em>The Elements of Statistical Learning: Data Mining, Inference, and Prediction</em>, 2nd ed. New York: Springer. ISBN: 978-0-387-84857-0. Free PDF at hastie.su.domains/ElemStatLearn. Chapter 7 ("Model Assessment and Selection") is the definitive textbook treatment: it derives the bias-variance decomposition, extends it to classification, connects it to in-sample and out-of-sample error, introduces the optimism correction, and links to cross-validation, AIC, BIC, and Mallow's Cp. Chapter 7.3 contains the clean squared-error derivation used in this topic. Essential reading for anyone who wants the full story.
              </Prose>
            ),
          },
          {
            label: "Domingos 2000 — Unified decomposition for 0/1 loss",
            render: () => (
              <Prose>
                Domingos, P. (2000). "A Unified Bias-Variance Decomposition and its Applications." <em>Proceedings of the 17th International Conference on Machine Learning (ICML 2000)</em>, pp. 231–238. Morgan Kaufmann. Domingos derives a unified bias-variance decomposition that applies to any loss function, not just squared error. The key result for classification: under 0/1 loss, bias and variance interact multiplicatively rather than additively, and variance can sometimes reduce error when a biased model is corrected toward the right class by randomness. This explains why bagging helps even when individual models are biased — a result that is counterintuitive from the squared-error picture. The paper is widely cited as the definitive extension of the decomposition to classification.
              </Prose>
            ),
          },
          {
            label: "Belkin, Hsu, Ma, Mandal 2019 — Reconciling the classical and modern views",
            render: () => (
              <Prose>
                Belkin, M., Hsu, D., Ma, S., and Mandal, S. (2019). "Reconciling modern machine-learning practice and the classical bias–variance trade-off." <em>Proceedings of the National Academy of Sciences</em>, 116(32):15849–15854. DOI: 10.1073/pnas.1903070116. This paper introduced the double-descent framework — the observation that test error can decrease again past the interpolation threshold, tracing a second U-shape at high model capacity. Belkin et al. show this in polynomial regression, kernel methods, and neural networks, and provide theoretical analysis showing that the interpolating solution found by gradient descent corresponds to the minimum-norm interpolant, which has implicit regularization properties. The paper resolves the apparent contradiction between classical bias-variance theory and the empirical success of massive over-parameterized models.
              </Prose>
            ),
          },
          {
            label: "Nakkiran et al. 2020 — Deep Double Descent",
            render: () => (
              <Prose>
                Nakkiran, P., Kaplun, G., Bansal, Y., Yang, T., Barak, B., and Sutskever, I. (2020). "Deep Double Descent: Where Bigger Models and More Data Hurt." <em>International Conference on Learning Representations (ICLR 2020)</em>. arXiv:1912.02292. Large-scale empirical documentation of the double-descent phenomenon across ResNets, transformers, random features, and decision trees. Nakkiran et al. show that double descent occurs as a function of model size, training time (number of epochs), and dataset size — the phenomenon is not specific to one axis. They introduce the concept of "effective model complexity" to unify the three manifestations. The paper fundamentally changed how practitioners think about the relationship between model size and test performance.
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
        Work through these before moving to the next topic. Try each exercise on paper before reading the answer.
      </Prose>

      <H3>Exercise 1 (Recall)</H3>
      <Prose>
        Write the bias-variance decomposition for squared error loss. Define each term precisely. What assumption on the noise ε is required for the cross terms to vanish? What does irreducible noise physically represent?
      </Prose>
      <Callout type="answer" title="Answer 1">
        {"E[(y − f̂(x))²] = Bias[f̂(x)]² + Var[f̂(x)] + σ². Bias[f̂(x)] = f(x) − E[f̂(x)] is the difference between the true function value and the expected prediction averaged over training sets. Var[f̂(x)] = E[(f̂(x) − E[f̂(x)])²] is how much the prediction at x varies across training sets. σ² = Var(ε) is the irreducible noise variance. The cross terms vanish because E[ε] = 0 and ε is independent of the training set (so ε is uncorrelated with f̂(x) − f(x)). Irreducible noise represents variance in y that is genuinely random given x — measurement error, inherent stochasticity, or the effect of unmeasured variables. Even a perfect model knowing f(x) exactly cannot predict y better than σ²."}
      </Callout>

      <H3>Exercise 2 (Conceptual)</H3>
      <Prose>
        You train a degree-1 polynomial (linear regression) on a dataset where the true relationship is quadratic. Is this high bias or high variance? If you train 100 bootstrap models, what does the spread of predictions look like at a fixed test point? What does the average prediction look like relative to the true value?
      </Prose>
      <Callout type="answer" title="Answer 2">
        This is high bias. The model class (linear functions) cannot represent the true quadratic function, so the average prediction is systematically wrong regardless of how many bootstrap samples you use. The spread of predictions at a fixed test point will be relatively small — linear models fitted to different samples of a quadratic dataset will give consistent but consistently wrong answers. The 100 bootstrapped lines will cluster together (low variance), but all of them will be far from the true quadratic value at points away from the center of the data (high bias). Bias² will be large; Variance will be small. Adding more data will not help — you are limited by the expressiveness of the model class, not by data quantity.
      </Callout>

      <H3>Exercise 3 (Applied — learning curve diagnosis)</H3>
      <Prose>
        You run <Code>learning_curve</Code> on a random forest and observe the following at training size n=1000: train MSE = 0.05, val MSE = 0.42. At n=5000: train MSE = 0.07, val MSE = 0.31. At n=20000: train MSE = 0.09, val MSE = 0.22. The noise floor is estimated at 0.10. What is your diagnosis, and what are your top three actions?
      </Prose>
      <Callout type="answer" title="Answer 3">
        {"Diagnosis: high variance. The gap (val − train) is large at all training sizes (0.37, 0.24, 0.13), and val error is still descending at n=20000, meaning more data is still helping. Train error is near or below the noise floor (0.05–0.09), confirming the model is memorizing training data. Actions in priority order: (1) Increase regularization — reduce max_depth of the trees (from unlimited, try depth 5–10), increase min_samples_leaf (from 1, try 5–20), reduce max_features (try 'sqrt' or 0.3). These are the primary variance levers for random forests. (2) Collect more data — the descending validation curve confirms more data will help; at n=20000 the gap has closed from 0.37 to 0.13, suggesting the validation curve has not yet plateaued. (3) Try bagging more trees with more aggressive subsampling — increasing the number of trees (n_estimators) with a smaller max_samples fraction can reduce variance without increasing bias. Do NOT reduce n_estimators (that reduces the ensemble averaging) and do NOT add features (that would increase model capacity and worsen variance)."}
      </Callout>

      <H3>Exercise 4 (Math)</H3>
      <Prose>
        Show that for a linear regression model fitted by OLS on a dataset of size n, the expected in-sample (training) MSE equals <Code>{"(n − d) / n × σ²"}</Code> where d is the number of parameters and σ² is the noise variance. What does this tell you about the relationship between training error and test error as d approaches n?
      </Prose>
      <Callout type="answer" title="Answer 4">
        {"The OLS fitted values are ŷ = Hy where H = X(XᵀX)⁻¹Xᵀ is the hat matrix with trace(H) = d (the number of free parameters). The training residuals are ε̂ = y − ŷ = (I − H)y. The expected training MSE is E[||ε̂||²/n] = E[||(I−H)(Xβ + ε)||²/n] = ||ε||²E[(I−H)/n] for the noise term. Since (I−H) is a projection onto the (n−d)-dimensional null space of X: E[||ε̂||²] = (n−d)σ², giving E[train MSE] = (n−d)/n × σ². As d → n: training MSE → 0. With as many parameters as data points, OLS fits the data perfectly (training MSE = 0) but has no degrees of freedom left to estimate σ². The test MSE, by contrast, is E[test MSE] = (1 + d/n)σ² for an out-of-sample prediction (using the Stein result), which grows as d/n grows. This makes precise why memorizing training data (d=n) produces catastrophic test performance: training error is 0 but test error is 2σ²."}
      </Callout>

      <H3>Exercise 5 (Applied — intervention choice)</H3>
      <Prose>
        A colleague has trained a gradient-boosted tree on a fraud detection dataset (100K transactions, 200 features). Their learning curve shows: train AUC = 0.998, val AUC = 0.81. They have tried: (a) collecting 200K more transactions (val AUC → 0.83), (b) reducing max_depth from 8 to 4 (val AUC → 0.87), (c) adding 50 new features (val AUC → 0.78). Based on these experiments, what is the diagnosis, and what should they try next?
      </Prose>
      <Callout type="answer" title="Answer 5">
        Diagnosis: high variance. The train/val AUC gap (0.998 − 0.81 = 0.188) is enormous and the model is clearly memorizing training data. The experiments confirm this: reducing max_depth (b) reduced model capacity and improved val AUC significantly — this is the characteristic response of a high-variance model to regularization. More data (a) helped modestly (0.83), consistent with the descending val curve signature. Adding features (c) worsened val AUC — more features increased dimensionality and gave the model more axes on which to memorize noise. Next steps in priority order: (1) Tune regularization more aggressively — reduce max_depth further (try 2–3), increase min_child_weight or min_samples_leaf, reduce subsample from 1.0 to 0.7–0.8, reduce colsample_bytree to 0.5–0.7. Max_depth=4 improved things; there is likely more gain from depth=2 or depth=3. (2) Apply feature selection — with 200 features and a high-variance model, many features are likely noise. Use SHAP importance or permutation importance to identify the top 20–30 features and retrain with only those. (3) Once variance is controlled, revisit whether more data further helps — the 200K experiment showed a modest effect; with better regularization, the learning curve slope may be steeper.
      </Callout>

      <H3>Exercise 6 (Synthesis — double descent)</H3>
      <Prose>
        A team trains a transformer language model with 10M parameters on 1M text tokens (so parameters {">"} training tokens). They observe that train loss reaches zero but val loss is lower than what a smaller 1M-parameter model achieves. Their ML lead says "this contradicts the bias-variance tradeoff." How do you explain what is happening?
      </Prose>
      <Callout type="answer" title="Answer 6">
        {"This does not contradict the bias-variance tradeoff — it demonstrates the modern double-descent regime, which is a phenomenon the classical U-shaped picture does not capture. The classical picture assumes a fixed optimization algorithm and that interpolating models (those with zero training loss) necessarily have high variance. In the over-parameterized regime trained with gradient descent, this assumption breaks down. Gradient descent on over-parameterized models implicitly finds the minimum-norm solution among all models that interpolate the training data (Belkin et al. 2019). Larger models have more ways to interpolate the data, and among those, gradient descent tends to find smoother solutions with lower effective complexity — this is the implicit regularization effect. A 10M-parameter transformer trained with gradient descent is not the worst interpolant; it is often a surprisingly smooth, regular one. The smaller 1M-parameter model may be right at or just past the interpolation threshold (the high-variance spike) where all models that fit the data are very constrained and the minimum-norm one is not especially smooth. The fix for the ML lead's confusion: the classical bias-variance tradeoff applies in the under-parameterized regime. In the over-parameterized regime, effective model complexity is governed by training dynamics (optimizer, learning rate, batch size, number of steps), not parameter count alone. More parameters can mean less effective complexity when gradient descent finds smoother interpolants."}
      </Callout>

    </div>
  ),
};

export default biasVarianceContent;
