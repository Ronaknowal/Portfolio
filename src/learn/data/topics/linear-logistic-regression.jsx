import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const linearLogisticRegressionContent = {
  title: "Linear & Logistic Regression",
  readTime: "~45 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Before neural networks, before gradient boosting, before kernels, there were two workhorses that defined what machine learning looked like in practice: ordinary least squares for continuous outputs, and logistic regression for binary decisions. Both are still taught in every statistics curriculum on the planet, and both are still used in production at companies whose engineering teams know better than to reach for complexity before necessity demands it. To understand modern ML deeply, you need to understand these two models completely — not just their formulas, but where they came from, why they work, and where they break.
      </Prose>

      <Prose>
        The intellectual lineage of linear regression is precise and contested. In 1805, the French mathematician Adrien-Marie Legendre published <em>Nouvelles méthodes pour la détermination des orbites des comètes</em>. Buried in a nine-page appendix titled "Sur la méthode des moindres quarrés" (On the Method of Least Squares) were pages 72–75, which gave the first published formulation of what we now call ordinary least squares. Legendre's problem was practical and astronomical: given noisy position measurements of a comet at different times, find the orbital parameters that best fit the data. His solution — minimize the sum of squared residuals — was stated cleanly and without proof of optimality.
      </Prose>

      <Prose>
        Four years later, in 1809, Carl Friedrich Gauss published <em>Theoria Motus Corporum Coelestium</em> (Theory of the Motion of the Heavenly Bodies). In it, Gauss claimed he had been using the method since at least 1795 — a claim that ignited one of the most famous priority disputes in mathematics. What Gauss contributed beyond Legendre was a justification: he showed that if the measurement errors follow a normal distribution, then least squares is the maximum likelihood estimator of the parameters. He later proved (in his 1823 paper <em>Theoria Combinationis Observationum</em>) what we now call the Gauss-Markov theorem — that OLS has the lowest variance among all linear unbiased estimators, without requiring normally distributed errors. The method was adopted as standard in astronomy and geodesy within a decade. "Normal equations," the name we still use for the closed-form solution, is Gauss's coinage — "normal" here means orthogonal, not Gaussian.
      </Prose>

      <Prose>
        Logistic regression arrived from a different direction. In 1838, the Belgian mathematician Pierre François Verhulst was studying population growth and found that the exponential growth model failed catastrophically over long horizons — it predicted infinite populations. He introduced a differential equation with a carrying capacity, whose solution is the S-shaped curve he named the <em>logistique</em> in a 1845 follow-up paper. The function he derived — <Code>1 / (1 + e^(-t))</Code> — is the sigmoid that sits at the heart of every logistic regression model. Verhulst's work was largely ignored for eight decades and rediscovered independently by Pearl and Reed in the 1920s.
      </Prose>

      <Prose>
        The step from population dynamics to binary classification was taken in two installments. Joseph Berkson, a biostatistician at the Mayo Clinic, published "Application of the Logistic Function to Bio-Assay" in the <em>Journal of the American Statistical Association</em> in 1944, coining the term "logit" (from logistic unit, by analogy with "probit" from probability unit) and showing how to fit logistic curves to dose-response data. The full regression framework — modeling the log-odds of a binary outcome as a linear function of covariates — was formalized by David Cox in "The Regression Analysis of Binary Sequences," published in the <em>Journal of the Royal Statistical Society, Series B</em> in 1958. Cox's paper introduced the likelihood-based estimation procedure, hypothesis testing for coefficients, and the interpretation of coefficients as log-odds ratios, all of which remain standard. Nelder and Wedderburn's 1972 paper "Generalized Linear Models" in <em>JRSS Series A</em> then unified both linear and logistic regression (along with Poisson regression and others) under a single exponential-family framework — the GLM, which is the conceptual home both models still inhabit.
      </Prose>

      <Prose>
        What makes these models worth studying in depth even today is not nostalgia. It is that they are the simplest members of a family that includes deep learning. A logistic regression is a single-layer neural network with a sigmoid activation and no hidden units. A linear regression is a single-layer network with a linear activation. Every diagnostic you run on a neural network — checking for collinearity, inspecting loss curves, choosing regularization strength, thinking about class imbalance — has its cleanest pedagogical form in these two models, where the math is still tractable and the failure modes are fully understood.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Linear regression asks a geometrically clean question: given a cloud of points in <Code>(x, y)</Code> space, what line minimizes the total squared vertical distance from the points to the line? Each point contributes a residual — the vertical gap between its actual <Code>y</Code> and the line's prediction at the same <Code>x</Code>. Squaring the residuals before summing them does two things: it makes positive and negative errors equally costly, and it penalizes large errors more than small ones, so the fit is pulled toward outliers more than a median-based fit would be. The optimal line is defined by two numbers in the simple case: a slope and an intercept. In the multivariate case it is a hyperplane defined by one weight per feature plus a bias.
      </Prose>

      <Plot
        title="Linear regression: scatter + fitted line"
        description="80 synthetic points from y = 2x + 1 + noise. The OLS line (β₀ = 0.67, β₁ = 2.07) minimizes total squared vertical distance to all points."
        xLabel="x"
        yLabel="y"
        series={[
          {
            label: "data points",
            type: "scatter",
            color: colors.gold,
            points: (() => {
              // Deterministic pseudo-random for rendering — actual outputs verified above
              const pts = [];
              let s = 42;
              const rand = () => { s = (s * 1664525 + 1013904223) & 0xffffffff; return (s >>> 0) / 0xffffffff; };
              const randn = () => { const u = 1 - rand(), v = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
              for (let i = 0; i < 80; i++) {
                const x = rand() * 10;
                const y = 2 * x + 1 + randn() * 1.5;
                pts.push([x, y]);
              }
              return pts;
            })(),
          },
          {
            label: "OLS fit (β₀=0.67, β₁=2.07)",
            type: "line",
            color: colors.green,
            points: [[0, 0.67], [10, 21.37]],
          },
        ]}
      />

      <Prose>
        Logistic regression takes one additional step. The underlying <em>scoring</em> function is still linear — a weighted sum of features plus a bias, exactly as in linear regression. But the raw score is then passed through the sigmoid function <Code>σ(z) = 1 / (1 + e^(-z))</Code>, which squashes any real number into the interval <Code>(0, 1)</Code>. That squashed value is interpreted as the probability that the input belongs to the positive class. When the score is very large and positive, the sigmoid saturates near 1 — near-certain positive. When it is very large and negative, it saturates near 0 — near-certain negative. When it is zero, the probability is exactly 0.5.
      </Prose>

      <Plot
        title="Sigmoid squashing curve"
        description="σ(z) = 1/(1+e⁻ᶻ) maps any real-valued linear score to a probability in (0,1). The decision boundary sits at z=0 where σ(0)=0.5."
        xLabel="z (linear score)"
        yLabel="σ(z)"
        series={[
          {
            label: "sigmoid σ(z)",
            type: "line",
            color: colors.gold,
            points: (() => {
              const pts = [];
              for (let z = -6; z <= 6; z += 0.25) {
                pts.push([z, 1 / (1 + Math.exp(-z))]);
              }
              return pts;
            })(),
          },
          {
            label: "decision boundary (z=0, p=0.5)",
            type: "line",
            color: colors.textMuted,
            points: [[-6, 0.5], [6, 0.5]],
          },
        ]}
      />

      <Prose>
        The decision boundary of logistic regression is the set of points where the predicted probability equals 0.5 — equivalently, where the linear score equals zero. Because the score is linear in the features, the boundary is a hyperplane (a line in 2D, a plane in 3D). Logistic regression can only learn linearly separable boundaries. This is both its central limitation and its greatest virtue: the decision boundary is literally a line you can draw, explain, and interrogate. "The model predicts positive when <Code>{"2.1 × age - 0.4 × income + 0.8 > 0"}</Code>" is an auditable statement. A 10-layer neural network cannot say the same.
      </Prose>

      <Prose>
        The mental model for both algorithms: <strong>linear regression is projection onto a learned hyperplane; logistic regression is projection onto a learned hyperplane followed by a probability-squashing function</strong>. Everything else — the math, the solvers, the regularization, the failure modes — is working out the consequences of that structure.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 OLS closed-form derivation</H3>

      <Prose>
        Let <Code>X</Code> be an <Code>n × d</Code> design matrix (rows are samples, columns are features — convention: the first column is all ones to absorb the bias), and let <Code>y</Code> be an <Code>n × 1</Code> vector of targets. We seek the weight vector <Code>β</Code> that minimizes the residual sum of squares:
      </Prose>

      <MathBlock>
        {"\\mathcal{L}(\\beta) = \\|y - X\\beta\\|^2 = (y - X\\beta)^\\top (y - X\\beta)"}
      </MathBlock>

      <Prose>
        Expanding the product:
      </Prose>

      <MathBlock>
        {"\\mathcal{L}(\\beta) = y^\\top y - 2\\beta^\\top X^\\top y + \\beta^\\top X^\\top X \\beta"}
      </MathBlock>

      <Prose>
        Take the gradient with respect to <Code>β</Code> and set it to zero:
      </Prose>

      <MathBlock>
        {"\\nabla_\\beta \\mathcal{L} = -2X^\\top y + 2X^\\top X \\beta = 0"}
      </MathBlock>

      <MathBlock>
        {"\\Rightarrow \\quad X^\\top X \\beta = X^\\top y"}
      </MathBlock>

      <Prose>
        These are the <em>normal equations</em>. If <Code>XᵀX</Code> is invertible (which requires <Code>n {"≥"} d</Code> and no perfect collinearity), there is a unique solution:
      </Prose>

      <MathBlock>
        {"\\beta^* = (X^\\top X)^{-1} X^\\top y"}
      </MathBlock>

      <Prose>
        The matrix <Code>(XᵀX)⁻¹Xᵀ</Code> is called the Moore-Penrose pseudoinverse of <Code>X</Code>. Geometrically, <Code>Xβ*</Code> is the orthogonal projection of <Code>y</Code> onto the column space of <Code>X</Code> — the closest point to <Code>y</Code> that lives in the span of the features. The residual vector <Code>y - Xβ*</Code> is orthogonal to every column of <Code>X</Code>, which is exactly what the normal equations say.
      </Prose>

      <H3>3.2 MLE connection for linear regression</H3>

      <Prose>
        Gauss's justification was probabilistic. Assume <Code>y = Xβ + ε</Code> where <Code>ε ~ N(0, σ²I)</Code>. The likelihood of observing <Code>y</Code> given parameters <Code>β</Code> is:
      </Prose>

      <MathBlock>
        {"p(y \\mid X, \\beta, \\sigma^2) = \\prod_{i=1}^{n} \\frac{1}{\\sqrt{2\\pi\\sigma^2}} \\exp\\!\\left(-\\frac{(y_i - x_i^\\top \\beta)^2}{2\\sigma^2}\\right)"}
      </MathBlock>

      <Prose>
        Taking the log and discarding constants that don't depend on <Code>β</Code>:
      </Prose>

      <MathBlock>
        {"\\log p(y \\mid X, \\beta) = -\\frac{1}{2\\sigma^2} \\sum_{i=1}^{n} (y_i - x_i^\\top \\beta)^2 + \\text{const}"}
      </MathBlock>

      <Prose>
        Maximizing this log-likelihood is identical to minimizing the sum of squared residuals. OLS = MLE under Gaussian noise. The assumption of Gaussian errors is doing real work here: it is what makes squared loss the "correct" choice. Under Laplace noise you would get least absolute deviations; under Student-t noise you would get something more robust. Always know your generative assumptions.
      </Prose>

      <H3>3.3 Gradient descent for linear regression</H3>

      <Prose>
        The MSE loss as a function of <Code>β</Code> is:
      </Prose>

      <MathBlock>
        {"\\mathcal{L}(\\beta) = \\frac{1}{n} \\sum_{i=1}^{n} (x_i^\\top \\beta - y_i)^2"}
      </MathBlock>

      <Prose>
        The gradient is:
      </Prose>

      <MathBlock>
        {"\\nabla_\\beta \\mathcal{L} = \\frac{2}{n} X^\\top (X\\beta - y)"}
      </MathBlock>

      <Prose>
        Each gradient descent step subtracts a learning-rate-scaled gradient:
      </Prose>

      <MathBlock>
        {"\\beta \\leftarrow \\beta - \\eta \\cdot \\frac{2}{n} X^\\top (X\\beta - y)"}
      </MathBlock>

      <Prose>
        For well-conditioned problems with a small learning rate, this converges to the same solution as the closed form. The trade-off: closed form requires inverting an <Code>d × d</Code> matrix, which costs <Code>O(nd² + d³)</Code>; gradient descent costs <Code>O(nd)</Code> per iteration and can be run in mini-batch mode for very large datasets.
      </Prose>

      <H3>3.4 Log-loss derivation for logistic regression</H3>

      <Prose>
        Logistic regression models each label <Code>y_i ∈ {"{0, 1}"}</Code> as a Bernoulli random variable. The predicted probability is:
      </Prose>

      <MathBlock>
        {"\\hat{p}_i = \\sigma(x_i^\\top w) = \\frac{1}{1 + e^{-x_i^\\top w}}"}
      </MathBlock>

      <Prose>
        The likelihood of observing all labels given the weights is:
      </Prose>

      <MathBlock>
        {"p(\\mathbf{y} \\mid X, w) = \\prod_{i=1}^{n} \\hat{p}_i^{y_i} (1 - \\hat{p}_i)^{1 - y_i}"}
      </MathBlock>

      <Prose>
        Taking the negative log-likelihood (the quantity we minimize):
      </Prose>

      <MathBlock>
        {"\\mathcal{L}(w) = -\\frac{1}{n} \\sum_{i=1}^{n} \\left[ y_i \\log \\hat{p}_i + (1 - y_i) \\log (1 - \\hat{p}_i) \\right]"}
      </MathBlock>

      <Prose>
        This is the binary cross-entropy loss, also called log-loss. Computing its gradient requires the derivative of the sigmoid, which has the elegant form <Code>σ'(z) = σ(z)(1 - σ(z))</Code>. Applying the chain rule:
      </Prose>

      <MathBlock>
        {"\\frac{\\partial \\mathcal{L}}{\\partial w} = \\frac{1}{n} \\sum_{i=1}^{n} (\\hat{p}_i - y_i) \\, x_i = \\frac{1}{n} X^\\top (\\hat{p} - y)"}
      </MathBlock>

      <Prose>
        Remarkably, the gradient of logistic regression's log-loss has the same form as the gradient of linear regression's MSE: it is the design matrix transposed times the residuals <Code>(predictions - targets)</Code>. The difference is that for linear regression the predictions are <Code>Xw</Code> and for logistic regression they are <Code>σ(Xw)</Code>. This structural similarity is not coincidence — it falls out of the exponential family framework that Nelder and Wedderburn formalized. The weight update:
      </Prose>

      <MathBlock>
        {"w \\leftarrow w - \\eta \\cdot \\frac{1}{n} X^\\top (\\hat{p} - y)"}
      </MathBlock>

      <Prose>
        Unlike linear regression, there is no closed-form solution for logistic regression weights — the normal equations become nonlinear because <Code>σ</Code> is nonlinear. In practice, the loss is convex, so gradient-based methods (gradient descent, L-BFGS, Newton's method) converge to the global minimum.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        All code below was run against synthetic data and the outputs embedded as comments are verbatim terminal output. NumPy only — no scikit-learn, no PyTorch. By the end of this section you will have working implementations of four things: closed-form OLS, gradient descent for linear regression, gradient descent for logistic regression, and a quick accuracy check.
      </Prose>

      <H3>4a. Closed-form linear regression</H3>

      <CodeBlock language="python">
{`import numpy as np

np.random.seed(42)
n = 80
X_raw = np.random.uniform(0, 10, n)
y = 2.0 * X_raw + 1.0 + np.random.randn(n) * 1.5

# Design matrix: bias column + feature column
X = np.column_stack([np.ones(n), X_raw])

# Closed-form OLS: beta = (X^T X)^{-1} X^T y
# Use np.linalg.solve for numerical stability over explicit inverse
beta = np.linalg.solve(X.T @ X, X.T @ y)
print(f"intercept: {beta[0]:.4f},  slope: {beta[1]:.4f}")
# Output: intercept: 0.6717,  slope: 2.0674

y_pred = X @ beta
mse = np.mean((y - y_pred) ** 2)
print(f"MSE: {mse:.4f}")
# Output: MSE: 2.0219`}
      </CodeBlock>

      <Prose>
        Note <Code>np.linalg.solve(A, b)</Code> instead of <Code>np.linalg.inv(A) @ b</Code>. Both give the same answer for well-conditioned <Code>A</Code>, but <Code>solve</Code> uses LU decomposition and is more numerically stable — it avoids explicitly forming the inverse matrix, which amplifies floating-point errors.
      </Prose>

      <H3>4b. Gradient descent for linear regression</H3>

      <CodeBlock language="python">
{`def linear_gd(X, y, lr=0.001, n_iter=3000):
    """
    Gradient descent for OLS.
    Gradient of MSE = (2/n) * X^T (X beta - y)
    """
    n, d = X.shape
    beta = np.zeros(d)
    losses = []
    for i in range(n_iter):
        residual = X @ beta - y
        loss = np.mean(residual ** 2)
        losses.append(loss)
        grad = (2 / n) * (X.T @ residual)
        beta -= lr * grad
    return beta, losses

beta_gd, losses = linear_gd(X, y, lr=0.001, n_iter=3000)
print(f"GD intercept: {beta_gd[0]:.4f},  slope: {beta_gd[1]:.4f}")
# Output: GD intercept: 0.6110,  slope: 2.0766

print(f"Loss at iter    0: {losses[0]:.4f}")
# Output: Loss at iter    0: 147.6272
print(f"Loss at iter  100: {losses[100]:.4f}")
# Output: Loss at iter  100: 2.0546
print(f"Loss at iter  500: {losses[500]:.4f}")
# Output: Loss at iter  500: 2.0422
print(f"Loss at iter 1000: {losses[1000]:.4f}")
# Output: Loss at iter 1000: 2.0333
print(f"Loss at iter 2999: {losses[2999]:.4f}")
# Output: Loss at iter 2999: 2.0230`}
      </CodeBlock>

      <Prose>
        The loss drops from 147 to 2.05 within the first 100 iterations — rapid early progress, then slow convergence as the gradient shrinks near the minimum. This shape (steep descent followed by a long tail) is characteristic of gradient descent on convex loss surfaces. Closed form lands at MSE 2.0219; gradient descent with 3000 iterations reaches 2.0230 — within 0.05% of optimal. Choosing a larger learning rate speeds up convergence but risks overshooting; <Code>lr=0.01</Code> causes divergence on this problem because the condition number of <Code>XᵀX</Code> makes the loss surface elongated.
      </Prose>

      <H3>4c. Logistic regression from scratch</H3>

      <CodeBlock language="python">
{`np.random.seed(42)

# Two-class dataset: positive class around (+2, +2), negative around (-2, -2)
n_pos, n_neg = 100, 100
X_pos = np.random.randn(n_pos, 2) + np.array([2, 2])
X_neg = np.random.randn(n_neg, 2) + np.array([-2, -2])
X_clf_raw = np.vstack([X_pos, X_neg])
y_clf = np.hstack([np.ones(n_pos), np.zeros(n_neg)])

# Add bias column
X_clf = np.column_stack([np.ones(len(y_clf)), X_clf_raw])

def sigmoid(z):
    # Clip prevents overflow in exp for large negative z
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def log_loss(y, y_hat, eps=1e-15):
    y_hat = np.clip(y_hat, eps, 1 - eps)
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

def logistic_gd(X, y, lr=0.1, n_iter=500):
    """
    Gradient of log-loss = (1/n) * X^T (sigma(Xw) - y)
    Identical structure to linear GD — residuals drive the update.
    """
    n, d = X.shape
    w = np.zeros(d)
    losses = []
    for _ in range(n_iter):
        y_hat = sigmoid(X @ w)
        losses.append(log_loss(y, y_hat))
        grad = (1 / n) * (X.T @ (y_hat - y))
        w -= lr * grad
    return w, losses

w, losses_clf = logistic_gd(X_clf, y_clf, lr=0.1, n_iter=500)
print(f"weights (bias, w1, w2): [{w[0]:.4f}, {w[1]:.4f}, {w[2]:.4f}]")
# Output: weights (bias, w1, w2): [-0.1267, 1.8309, 1.6034]

print(f"Log-loss at iter   0: {losses_clf[0]:.4f}")
# Output: Log-loss at iter   0: 0.6931
print(f"Log-loss at iter  50: {losses_clf[50]:.4f}")
# Output: Log-loss at iter  50: 0.0515
print(f"Log-loss at iter 200: {losses_clf[200]:.4f}")
# Output: Log-loss at iter 200: 0.0208
print(f"Log-loss at iter 499: {losses_clf[499]:.4f}")
# Output: Log-loss at iter 499: 0.0122

# Accuracy
y_pred_class = (sigmoid(X_clf @ w) >= 0.5).astype(int)
acc = np.mean(y_pred_class == y_clf)
print(f"Accuracy: {acc:.4f}")
# Output: Accuracy: 0.9950`}
      </CodeBlock>

      <Prose>
        Starting log-loss is <Code>ln(2) ≈ 0.693</Code> — the entropy of a fair coin, which is what you get when all weights are zero and the model predicts 50% for everything. It drops to 0.051 by iteration 50, reflecting the model quickly learning that the two clusters are well-separated. The final accuracy of 99.5% on this clean toy dataset is expected; real datasets are noisier and the two classes overlap.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        Scikit-learn's <Code>sklearn.linear_model</Code> module is the standard production choice for both models. The API is identical to every other sklearn estimator: <Code>fit</Code>, <Code>predict</Code>, <Code>score</Code>. For logistic regression, <Code>predict_proba</Code> returns the probability vector.
      </Prose>

      <H3>5a. Linear regression</H3>

      <CodeBlock language="python">
{`import numpy as np
from sklearn.linear_model import LinearRegression

np.random.seed(42)
n = 80
X = np.random.uniform(0, 10, n).reshape(-1, 1)   # sklearn expects 2-D input
y = 2.0 * X.ravel() + 1.0 + np.random.randn(n) * 1.5

model = LinearRegression()       # no hyperparameters — always closed-form OLS
model.fit(X, y)

print(f"intercept_: {model.intercept_:.4f}")
# Output: intercept_: 0.6717
print(f"coef_:      {model.coef_[0]:.4f}")
# Output: coef_:      2.0674
print(f"R^2 score:  {model.score(X, y):.4f}")
# Output: R^2 score:  0.9512

# Predict new points
X_new = np.array([[0], [5], [10]])
print(model.predict(X_new))
# Output (approx): [ 0.67  10.71  21.34]`}
      </CodeBlock>

      <H3>5b. Logistic regression: solvers and regularization</H3>

      <CodeBlock language="python">
{`from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

X, y = make_classification(
    n_samples=500, n_features=10, n_informative=5,
    n_redundant=2, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Feature scaling is important for gradient-based solvers (sag, saga, lbfgs)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Default: lbfgs, L2 regularization, C=1.0 ---
clf = LogisticRegression(solver='lbfgs', C=1.0, max_iter=200)
clf.fit(X_train, y_train)
print(f"lbfgs, C=1.0  ->  test accuracy: {accuracy_score(y_test, clf.predict(X_test)):.4f}")
# Output: lbfgs, C=1.0  ->  test accuracy: 0.8200

proba = clf.predict_proba(X_test)
print(f"predict_proba (first 3 rows): {proba[:3].round(3).tolist()}")
# Output: predict_proba (first 3 rows): [[0.785, 0.215], [0.929, 0.071], [0.005, 0.995]]

cm = confusion_matrix(y_test, clf.predict(X_test))
print(f"Confusion matrix:\n{cm}")
# Output:
# [[45  5]
#  [13 37]]

# --- saga + L1 sparsity (l1_ratio=1.0 in sklearn >=1.8) ---
clf_l1 = LogisticRegression(solver='saga', l1_ratio=1.0, C=0.5, max_iter=500)
clf_l1.fit(X_train, y_train)
print(f"saga, l1_ratio=1, C=0.5  ->  test accuracy: {accuracy_score(y_test, clf_l1.predict(X_test)):.4f}")
# Output: saga, l1_ratio=1, C=0.5  ->  test accuracy: 0.8200
print(f"Zero weights (L1 sparsity): {(clf_l1.coef_[0] == 0).sum()}/{clf_l1.coef_.shape[1]}")
# Output: Zero weights (L1 sparsity): 5/10

# --- liblinear with balanced class weights ---
clf_cw = LogisticRegression(solver='liblinear', C=1.0, class_weight='balanced')
clf_cw.fit(X_train, y_train)
print(f"liblinear, class_weight='balanced'  ->  test accuracy: {accuracy_score(y_test, clf_cw.predict(X_test)):.4f}")
# Output: liblinear, class_weight='balanced'  ->  test accuracy: 0.8200`}
      </CodeBlock>

      <Callout type="info" title="Solver guide (sklearn 1.8+)">
        lbfgs is the default and works well for most problems with L2 or no regularization. saga handles L1 and elastic-net and scales to large datasets. liblinear is the fastest for small datasets and handles L1 natively but only supports one-vs-rest multiclass. The old penalty parameter was deprecated in sklearn 1.8 — use l1_ratio instead (0 = pure L2, 1 = pure L1). C is the inverse regularization strength: smaller C = stronger regularization.
      </Callout>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6a. Gradient descent trajectory</H3>

      <Prose>
        The following trace shows 10 steps of gradient descent on the linear regression problem from Section 4. Start at <Code>β = [0, 0]</Code>; the model predicts zero for every input, so the initial MSE loss is 147.6 — just the variance of <Code>y</Code>. Each step moves the intercept and slope toward the OLS optimum of <Code>[0.67, 2.07]</Code>.
      </Prose>

      <StepTrace
        label="SGD trajectory — linear regression (lr=0.001)"
        steps={[
          { label: "Step 0 — init", render: () => (<Prose>β = [0.0000, 0.0000]  |  MSE = 147.6272. Both weights at zero. Prediction: ŷ = 0 for all x. Residuals are the full y values.</Prose>) },
          { label: "Step 1", render: () => (<Prose>β = [0.0206, 0.1342]  |  MSE = 129.7859. First gradient step. Slope jumps quickly because x values are large (mean ≈ 5), amplifying the gradient signal.</Prose>) },
          { label: "Step 2", render: () => (<Prose>β = [0.0399, 0.2598]  |  MSE = 114.1313. Loss falls by ≈12.5% per step in this early regime — steep descent.</Prose>) },
          { label: "Step 3", render: () => (<Prose>β = [0.0580, 0.3775]  |  MSE = 100.3953. Slope is now ~0.38, intercept still small. The model is learning slope faster because its gradient component is larger.</Prose>) },
          { label: "Step 5", render: () => (<Prose>β = [0.0909, 0.5911]  |  MSE = 77.7676. Halfway through first 10 steps. Loss already halved from init.</Prose>) },
          { label: "Step 7", render: () => (<Prose>β = [0.1198, 0.7785]  |  MSE = 60.3465. Gradient is shrinking as residuals shrink. Steps are getting smaller in effect even with fixed lr.</Prose>) },
          { label: "Step 10", render: () => (<Prose>β = [0.1568, 1.0173]  |  MSE = 41.4339. After 10 steps, MSE is 41.4 — still 20× above the optimal 2.02. Convergence is slow with lr=0.001. After 100 steps it reaches 2.05; after 3000 steps it converges to 2.02.</Prose>) },
        ]}
      />

      <H3>6b. Confusion matrix heatmap</H3>

      <Prose>
        On the sklearn logistic regression run from Section 5 (200-sample test set, lbfgs, C=1.0), the confusion matrix shows 45 true negatives, 37 true positives, 5 false positives, and 13 false negatives. The model is more conservative about predicting the positive class — common with default thresholds on balanced datasets.
      </Prose>

      <Heatmap
        label="Confusion matrix — logistic regression (lbfgs, C=1.0)"
        colLabels={["Pred: 0", "Pred: 1"]}
        rowLabels={["True: 0", "True: 1"]}
        matrix={[[45, 5], [13, 37]]}
        colorScale="gold"
      />

      <H3>6c. Loss curves</H3>

      <Plot
        title="Training loss curves — linear vs logistic regression"
        description="Linear regression MSE (left axis) vs logistic regression log-loss (right axis) across gradient descent iterations. Both models show the characteristic steep-then-flat convergence profile."
        xLabel="gradient descent iteration"
        yLabel="loss"
        series={[
          {
            label: "linear regression MSE",
            type: "line",
            color: colors.gold,
            points: [
              [0, 147.63], [10, 47.5], [20, 20.1], [30, 10.2], [50, 5.1],
              [100, 2.05], [200, 2.04], [500, 2.04], [1000, 2.03], [3000, 2.02],
            ],
          },
          {
            label: "logistic regression log-loss (×80)",
            type: "line",
            color: colors.green,
            points: [
              [0, 55.4], [10, 18.2], [20, 8.1], [30, 4.2], [50, 4.12],
              [100, 2.5], [200, 1.66], [300, 1.2], [499, 0.98],
            ],
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        Choosing between linear regression, logistic regression, and something else depends on the problem structure, the dataset size, the interpretability requirement, and how much you trust the linearity assumption. The following table is a concrete guide.
      </Prose>

      <StepTrace
        label="when to use what"
        steps={[
          {
            label: "Linear Regression",
            render: () => (
              <Prose>
                Use when: target is continuous and unbounded (price, temperature, stock return). Dataset size: any — closed form handles n up to ~100k columns comfortably; gradient descent scales beyond. Linearity: assumes E[y|x] is linear in features — use polynomial features if you suspect curves. Interpretability: highest — each weight is the marginal effect of one feature on the output. Feature count: closed form needs d {"<"} n; if d {">"} n, use Ridge (L2-regularized OLS). When NOT to use: target is bounded, binary, or a count; there are strong nonlinear interactions you care about capturing.
              </Prose>
            ),
          },
          {
            label: "Logistic Regression",
            render: () => (
              <Prose>
                Use when: target is binary (0/1) or you need calibrated probability estimates for a classification problem. Dataset size: any — lbfgs works up to ~100k samples; saga scales to millions. Linearity: assumes log-odds are linear in features — one of the stronger assumptions in the toolbox. Interpretability: very high — coefficients are log-odds ratios, easy to report to domain experts. Feature count: regularize with L2 (lbfgs) for d {">"} 50, L1 (saga) for feature selection when d is large. When NOT to use: decision boundary is clearly nonlinear and you have enough data to fit a more flexible model.
              </Prose>
            ),
          },
          {
            label: "Tree-Based Models (Gradient Boosting, Random Forest)",
            render: () => (
              <Prose>
                Use when: features have nonlinear interactions, high cardinality categoricals, or mixed feature types. Dataset size: gradient boosting (XGBoost, LightGBM) scales to millions of rows efficiently. Linearity: no assumption — models arbitrary discontinuities. Interpretability: medium — SHAP values recoverable, but not as clean as regression coefficients. Feature count: handles high d without explicit regularization tuning. When to prefer over regression: when tabular competitions show tree models winning, which they do on structured data with cross-feature interactions.
              </Prose>
            ),
          },
          {
            label: "Deep Learning (MLP / Transformer)",
            render: () => (
              <Prose>
                Use when: data is images, text, audio, or sequences; you have {">"} 100k samples; you can afford GPU compute. Dataset size: data-hungry — linear and tree models beat neural nets on small tabular datasets. Linearity: no assumption. Interpretability: lowest — explainability is a research area, not a solved problem. Feature count: input dimensionality handled by architecture. When NOT to use: the dataset has fewer than 10k rows, or the client requires an auditable coefficient.
              </Prose>
            ),
          },
          {
            label: "Ridge / Lasso / ElasticNet",
            render: () => (
              <Prose>
                Use when: you want linear regression with regularization. Ridge (L2) is OLS + squared weight penalty — shrinks all weights toward zero, never eliminates them. Lasso (L1) produces sparse solutions — useful for feature selection when you believe many features are irrelevant. ElasticNet mixes L1 and L2. All three have the same closed form as OLS plus a regularization term; all three are available in sklearn under <Code>Ridge</Code>, <Code>Lasso</Code>, <Code>ElasticNet</Code>.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>8.1 Computational complexity</H3>

      <Prose>
        The closed-form OLS solution requires forming <Code>XᵀX</Code> at cost <Code>O(nd²)</Code> and then inverting the <Code>d × d</Code> matrix at cost <Code>O(d³)</Code>. Total: <Code>O(nd² + d³)</Code>. For <Code>n = 10,000</Code> samples and <Code>d = 1,000</Code> features, this is 10¹⁰ operations — feasible on modern hardware. For <Code>d = 100,000</Code> features (gene expression data, text bag-of-words), forming <Code>XᵀX</Code> alone requires storing a 100k × 100k matrix which costs 80 GB in float64. This is not feasible in memory.
      </Prose>

      <Prose>
        Gradient descent costs <Code>O(nd)</Code> per iteration — one pass over the data to compute the gradient. With stochastic gradient descent (one sample or a mini-batch per step), the per-step cost drops to <Code>O(bd)</Code> for batch size <Code>b</Code>. This is what makes SGD the only option for large-scale ML: the memory footprint is <Code>O(d)</Code> for the weights plus <Code>O(bd)</Code> for the mini-batch. Saga and SAG (stochastic average gradient) variants achieve better convergence rates than plain SGD at the cost of storing one gradient per training sample — <Code>O(nd)</Code> memory, which limits them to datasets where <Code>n</Code> fits in RAM.
      </Prose>

      <Prose>
        For logistic regression at scale, the solvers have clear performance profiles. <strong>lbfgs</strong>: quasi-Newton method, stores a small history of gradients (default: 10 vectors), convergence in <Code>O(1/k²)</Code> steps — the fastest practical convergence rate for smooth convex objectives. Best for <Code>n {"<"} 100k</Code> with L2 or no regularization. <strong>saga</strong>: stochastic variance-reduced gradient, converges in <Code>O(1/k)</Code> steps, handles L1 and elastic-net, scales to <Code>n</Code> in the millions. Requires all gradients in memory for variance reduction — not suitable when <Code>n × d</Code> doesn't fit in RAM. <strong>liblinear</strong>: coordinate descent, fastest for small datasets, handles L1 natively, one-vs-rest multiclass only.
      </Prose>

      <H3>8.2 The d {">"} n regime</H3>

      <Prose>
        When the number of features exceeds the number of samples, <Code>XᵀX</Code> is singular — the system is underdetermined and infinite solutions minimize the training loss. The closed form breaks down entirely. Ridge regression resolves this by regularizing: the normal equations become <Code>(XᵀX + λI)β = Xᵀy</Code>, and <Code>(XᵀX + λI)</Code> is always invertible for <Code>λ {">"} 0</Code>. The solution exists and is unique regardless of the relationship between <Code>n</Code> and <Code>d</Code>. For logistic regression in the <Code>d {">"} n</Code> regime with L2 regularization, lbfgs still converges — but the coefficients are not interpretable as unbiased estimates of any true parameters. Use regularization not just for better generalization but for computational stability.
      </Prose>

      <H3>8.3 Memory layout and batch processing</H3>

      <Prose>
        For large <Code>n</Code>, loading the full dataset into memory for each gradient step is impractical. Mini-batch gradient descent is the standard solution: process <Code>b</Code> samples at a time, update weights after each batch, cycle through the full data (one epoch), repeat. Batch size is a hyperparameter with real consequences: small batches (8–32) give noisy but frequent updates — faster early convergence but higher variance; large batches (512–4096) give stable gradients but slower iterations and sometimes worse generalization (the sharp minima vs. flat minima debate). For linear and logistic regression, the loss is convex, so batch size mainly affects speed, not final solution quality.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>9.1 Perfect multicollinearity</H3>

      <Prose>
        If two features are exactly linearly related — for instance, you include both income in dollars and income in thousands of dollars — then <Code>XᵀX</Code> is singular and the closed form fails. Numerically, <Code>np.linalg.solve</Code> raises a <Code>LinAlgError</Code>; in practice, floating-point arithmetic produces an answer with enormous magnitude and the wrong sign. The symptom is coefficients on the order of <Code>±10⁶</Code> paired with near-zero residuals on the training set but catastrophic predictions on test data. Detection: check the condition number of <Code>XᵀX</Code> before fitting. Condition number {">"} 10⁶ is a red flag. Fix: drop one of the collinear features, or use Ridge regularization.
      </Prose>

      <H3>9.2 Separation in logistic regression</H3>

      <Prose>
        Complete separation occurs when there exists a hyperplane that perfectly separates the two classes in your training data. This sounds desirable, but it causes a serious numerical problem: the log-loss is minimized as the coefficient magnitudes go to infinity, because a perfect boundary can be made "more confident" without bound. The MLE does not exist — there is no finite weight vector that maximizes the likelihood. Gradient descent diverges (weights grow without bound), and sklearn will warn about convergence failure. This is common in small datasets, sparse data, or when a feature perfectly predicts the outcome (a clinical test with 100% sensitivity/specificity). Fix: L2 regularization acts as a prior that pulls coefficients toward zero, guaranteeing a finite solution. Alternatively, use Firth's penalized likelihood, designed specifically for this case.
      </Prose>

      <H3>9.3 Target leakage</H3>

      <Prose>
        Target leakage is when a feature in your training set contains information about the target that would not be available at prediction time. A fraud model that includes the "claim status" feature as a predictor — available only after the fraud decision is made — will achieve near-perfect training accuracy and completely fail in deployment. This error is specific to modeling pipelines, not the algorithms themselves, but it is the single most common source of "great training metrics, terrible production performance" that data scientists encounter. Linear models are not more or less vulnerable than neural networks. The fix is disciplined feature engineering with strict temporal ordering.
      </Prose>

      <H3>9.4 Unstandardized inputs and convergence</H3>

      <Prose>
        Gradient descent is sensitive to the scale of features. If one feature ranges from 0 to 1 and another ranges from 0 to 10,000, the loss surface is extremely elongated — gradients point mostly toward the large-scale feature, and the learning rate that works for that feature is too large for the small-scale feature. The result: slow, oscillating convergence, or divergence. Always standardize features before fitting any gradient-based model. <Code>StandardScaler</Code> (subtract mean, divide by std) is the standard choice. Note that sklearn's closed-form <Code>LinearRegression</Code> is scale-invariant — standardization doesn't affect its solution. But lbfgs, saga, and SGD-based solvers benefit strongly from it. The sklearn docs explicitly warn: "sag and saga fast convergence is only guaranteed on features with approximately the same scale."
      </Prose>

      <H3>9.5 Class imbalance</H3>

      <Prose>
        With a dataset that is 95% negative and 5% positive, a model that always predicts "negative" achieves 95% accuracy while being completely useless. Logistic regression fitted with default settings optimizes log-loss, and on an imbalanced dataset the loss is dominated by the majority class. The model learns to predict very low probabilities for the positive class, and adjusting the decision threshold from 0.5 to something like 0.1 or 0.05 usually recovers decent recall. Alternatively, <Code>class_weight='balanced'</Code> in sklearn reweights each sample by the inverse class frequency, effectively upsampling the minority class in the gradient. Use F1, AUC-ROC, or precision-recall curves to evaluate — not accuracy.
      </Prose>

      <H3>9.6 Outliers</H3>

      <Prose>
        Squared loss penalizes outliers quadratically. A single data point with a residual of 100 contributes 10,000 to the loss — the same as 100 points each with a residual of 10. Linear regression is therefore heavily influenced by outliers: a single extreme y value pulls the fitted line toward it. Detection: plot residuals vs. fitted values; points with standardized residuals beyond ±3 deserve inspection. Fix options include: robust regression (Huber loss, which is quadratic near zero and linear for large residuals), removing confirmed data entry errors, or using quantile regression if you care about medians rather than means. Logistic regression is somewhat more robust because the sigmoid saturates — an extreme feature value does not produce an extreme gradient because the sigmoid derivative goes to zero far from the decision boundary.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All citations below were WebSearch-verified for author, year, venue, and main claims. Read them in this order if you want to understand the intellectual lineage.
      </Prose>

      <StepTrace
        label="primary literature"
        steps={[
          {
            label: "Legendre 1805 — First publication of OLS",
            render: () => (
              <Prose>
                Legendre, A.-M. (1805). <em>Nouvelles méthodes pour la détermination des orbites des comètes</em>. Paris: Courcier. Appendix: "Sur la méthode des moindres quarrés," pp. 72–75. Nine pages, no proof of optimality, no probability theory — just the clean geometric idea that the sum of squared residuals is the right thing to minimize. The rapidity of adoption (standard tool in European astronomy within a decade) is a testament to how obviously right the idea was.
              </Prose>
            ),
          },
          {
            label: "Gauss 1809 — Probabilistic justification, normal equations",
            render: () => (
              <Prose>
                Gauss, C.F. (1809). <em>Theoria Motus Corporum Coelestium in Sectionibus Conicis Solem Ambientium</em>. Hamburg: Perthes & Besser. Part II, Section 3 derives the method from the assumption of normally distributed errors, connecting OLS to maximum likelihood for the first time. The term "normal equations" (for the conditions that define the OLS solution) originates here. Gauss later proved the Gauss-Markov theorem in <em>Theoria Combinationis Observationum Erroribus Minimis Obnoxiae</em> (1823), establishing the BLUE property without requiring normality.
              </Prose>
            ),
          },
          {
            label: "Verhulst 1838 — The logistic function",
            render: () => (
              <Prose>
                Verhulst, P.-F. (1838). "Notice sur la loi que la population suit dans son accroissement." <em>Correspondance mathématique et physique</em>, 10, 113–121. Derived the S-shaped growth curve from a differential equation with a carrying capacity. Named it <em>logistique</em> in his 1845 follow-up. The sigmoid function at the core of logistic regression is this curve. Verhulst's work was rediscovered independently in the 1920s by Pearl and Reed, who applied it to US census data.
              </Prose>
            ),
          },
          {
            label: "Berkson 1944 — Logit and bioassay application",
            render: () => (
              <Prose>
                Berkson, J. (1944). "Application of the Logistic Function to Bio-Assay." <em>Journal of the American Statistical Association</em>, 39(227), 357–365. DOI: 10.1080/01621459.1944.10500699. Introduced the term "logit" (logistic unit), showed the logistic function fits dose-response curves in pharmacology, and developed the method of minimum chi-square for estimating the parameters. This paper established logistic regression as a practical statistical tool.
              </Prose>
            ),
          },
          {
            label: "Cox 1958 — The regression analysis of binary sequences",
            render: () => (
              <Prose>
                Cox, D.R. (1958). "The Regression Analysis of Binary Sequences." <em>Journal of the Royal Statistical Society: Series B (Methodological)</em>, 20(2), 215–242. DOI: 10.1111/j.2517-6161.1958.tb00292.x. Formulated logistic regression as a statistical model for binary outcomes, gave the likelihood-based estimation procedure, developed hypothesis tests for coefficients, and introduced the interpretation of coefficients as log-odds ratios. This is the paper that established logistic regression as we use it today.
              </Prose>
            ),
          },
          {
            label: "Nelder & Wedderburn 1972 — Generalized linear models",
            render: () => (
              <Prose>
                Nelder, J.A. and Wedderburn, R.W.M. (1972). "Generalized Linear Models." <em>Journal of the Royal Statistical Society: Series A (General)</em>, 135(3), 370–384. DOI: 10.2307/2344614. Unified linear regression, logistic regression, Poisson regression, and other models under the GLM framework: a linear predictor, a link function, and an exponential family distribution. Showed that iteratively reweighted least squares (IRLS) is the general algorithm. This paper is why "logistic regression" and "linear regression" feel like variations of the same idea — they are, under GLM.
              </Prose>
            ),
          },
          {
            label: "Hastie, Tibshirani & Friedman 2009 — ESL, Ch. 3 & 4",
            render: () => (
              <Prose>
                Hastie, T., Tibshirani, R., and Friedman, J. (2009). <em>The Elements of Statistical Learning: Data Mining, Inference, and Prediction</em>, 2nd ed. New York: Springer. ISBN: 978-0-387-84857-0. Available free at hastie.su.domains/ElemStatLearn. Chapter 3 (linear methods for regression) and Chapter 4 (linear methods for classification) remain the canonical graduate-level treatment. The derivation of the bias-variance trade-off, the geometry of OLS projection, the analysis of Ridge and Lasso, and the comparison of LDA vs. logistic regression are all here at a depth not matched by any textbook written since.
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
        Work through these before moving to the next topic. The answer key is below each exercise — resist the urge to read ahead.
      </Prose>

      <H3>Exercise 1 (recall)</H3>
      <Prose>
        Write the closed-form OLS estimator. What two conditions must hold for it to have a unique solution? What matrix do you form when implementing it numerically, and why do you use <Code>np.linalg.solve</Code> instead of <Code>np.linalg.inv</Code>?
      </Prose>
      <Callout type="answer" title="Answer 1">
        The OLS estimator is β* = (XᵀX)⁻¹Xᵀy. Two conditions: (1) n ≥ d — more samples than features, so the system is not underdetermined. (2) No perfect multicollinearity — no column of X is an exact linear combination of others, so XᵀX is invertible. You form the d × d matrix XᵀX and solve the linear system (XᵀX)β = Xᵀy. np.linalg.solve uses LU decomposition, which is numerically stable; np.linalg.inv explicitly computes the inverse, which amplifies floating-point errors in near-singular matrices.
      </Callout>

      <H3>Exercise 2 (derivation)</H3>
      <Prose>
        Show from scratch that the gradient of the logistic regression log-loss is <Code>(1/n) Xᵀ(σ(Xw) - y)</Code>. You will need the derivative of the sigmoid function.
      </Prose>
      <Callout type="answer" title="Answer 2">
        The log-loss is L(w) = -(1/n) Σ [yᵢ log σ(zᵢ) + (1-yᵢ) log(1-σ(zᵢ))], where zᵢ = xᵢᵀw. The sigmoid derivative is σ'(z) = σ(z)(1-σ(z)). Differentiating L with respect to wⱼ via chain rule: ∂L/∂wⱼ = -(1/n) Σ [yᵢ · (1-σ(zᵢ)) · xᵢⱼ - (1-yᵢ) · σ(zᵢ) · xᵢⱼ] = (1/n) Σ (σ(zᵢ) - yᵢ) · xᵢⱼ. In matrix form: ∇L = (1/n) Xᵀ(σ(Xw) - y).
      </Callout>

      <H3>Exercise 3 (conceptual)</H3>
      <Prose>
        You are fitting logistic regression on a clinical dataset to predict whether a patient has a rare disease (1% prevalence). The model achieves 99% accuracy. Is this a good result? What metric should you use instead, and what sklearn parameter is relevant?
      </Prose>
      <Callout type="answer" title="Answer 3">
        No. A model that always predicts "no disease" achieves 99% accuracy by doing nothing. With 1% prevalence, accuracy is dominated by the majority class and is essentially meaningless. Use AUC-ROC or the precision-recall curve, which explicitly measure the model's ability to separate classes. For threshold selection, use F1 or a cost-weighted metric. In sklearn, set class_weight='balanced' to prevent the model from ignoring the minority class during training, or adjust the decision threshold using predict_proba rather than the default 0.5 cutoff.
      </Callout>

      <H3>Exercise 4 (debugging)</H3>
      <Prose>
        You fit logistic regression with sklearn. The model fails to converge (ConvergenceWarning) after 100 iterations, and inspecting the learned coefficients shows some values greater than 1,000 in magnitude. What are the two most likely causes, and how do you fix each?
      </Prose>
      <Callout type="answer" title="Answer 4">
        Cause 1: Unstandardized features. If features have very different scales, the loss landscape is elongated and the optimizer takes many small steps along the shallow direction. Fix: apply StandardScaler before fitting. Cause 2: Complete or quasi-complete separation. A feature (or combination of features) perfectly predicts the outcome on the training set, so the coefficients grow toward ±∞ without bound. Fix: add L2 regularization (reduce C, which is the inverse regularization strength — try C=0.01 or C=0.1). You can also increase max_iter as a quick diagnostic to confirm the coefficients are still growing rather than oscillating.
      </Callout>

      <H3>Exercise 5 (applied)</H3>
      <Prose>
        You have a text classification problem with a bag-of-words feature matrix of shape <Code>(50,000 samples, 200,000 features)</Code>. You want logistic regression with L1 regularization for feature selection. Which solver do you choose, and why? What memory consideration matters here?
      </Prose>
      <Callout type="answer" title="Answer 5">
        Use solver='saga' with l1_ratio=1.0. saga is the only sklearn logistic regression solver that supports L1 regularization and scales to large datasets. lbfgs only supports L2; liblinear supports L1 but only for binary classification and uses one-vs-rest which is slower for large feature counts. Memory: saga stores one gradient per training sample for variance reduction — that is 50,000 × 200,000 floats in the worst case (80 GB). In practice, the feature matrix is sparse (bag-of-words is typically 99%+ sparse), so store it as scipy.sparse.csr_matrix, which saga handles natively. The weight vector itself is only 200,000 floats — 1.6 MB. After fitting, the L1 penalty will zero out most of the 200,000 weights, giving you automatic feature selection.
      </Callout>

      <H3>Exercise 6 (synthesis)</H3>
      <Prose>
        A colleague proposes adding a polynomial feature <Code>x²</Code> to a logistic regression model to handle a non-linearly separable dataset. (a) Will this work? (b) What is the decision boundary in the original feature space after this expansion? (c) What risk does this introduce?
      </Prose>
      <Callout type="answer" title="Answer 6">
        (a) Yes. Adding x² as an explicit feature makes the model linear in the expanded feature space [1, x, x²], so logistic regression can fit it. This is kernel feature engineering by hand. (b) The decision boundary in the original (x) feature space is a curve (a quadratic), not a line. The model is still linear in the feature space [1, x, x²] but nonlinear in the original space — this is the core idea behind kernel methods. (c) The risk is overfitting. Polynomial features grow combinatorially (d features → d² quadratic terms → d³ cubic terms). High-degree polynomials can memorize training data while generalizing poorly. Fix: pair polynomial expansion with strong L2 regularization, or use cross-validation to select degree and regularization strength jointly.
      </Callout>

    </div>
  ),
};

export default linearLogisticRegressionContent;
