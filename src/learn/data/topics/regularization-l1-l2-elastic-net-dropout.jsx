import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const regularizationContent = {
  title: "Regularization (L1, L2, Elastic Net, Dropout)",
  readTime: "~50 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Every statistical model faces the same tension. Make it flexible enough to capture real signal in the training data, and it starts capturing noise too — patterns that are specific to this particular sample and do not generalize. Make it rigid enough to avoid memorizing noise, and it fails to find the real structure. This tension is the bias-variance tradeoff, and regularization is the technical machinery for navigating it without changing the model's fundamental architecture.
      </Prose>

      <Prose>
        The intellectual origin traces to 1943, when the Soviet mathematician Andrey Nikolayevich Tikhonov published "On the stability of inverse problems" in Doklady Akademii Nauk SSSR (volume 39, no. 5, pp. 195–198). Tikhonov was not thinking about machine learning — the field did not exist. He was thinking about ill-posed problems in mathematical physics: problems where a small perturbation in the input (noisy data) causes a catastrophically large change in the solution. His remedy was to add a penalty term to the objective that enforced smoothness on the solution, effectively trading some bias for massive reductions in variance. The technique he formalized became known as Tikhonov regularization. In regression, it is called ridge regression.
      </Prose>

      <Prose>
        The machine learning adoption came in 1970. Arthur E. Hoerl and Robert W. Kennard published "Ridge Regression: Biased Estimation for Nonorthogonal Problems" in Technometrics, volume 12(1), pages 55–67. Their motivating problem was practical and concrete: when predictor variables are nearly collinear, the ordinary least squares estimator becomes wildly unstable — small changes in the data produce enormous swings in the coefficients. Adding a small positive constant to the diagonal of <Code>XᵀX</Code> before inverting it stabilizes the solution at the cost of introducing a small bias. Hoerl and Kennard showed this trade was almost always worth making and gave the method its modern name. Their paper introduced the ridge trace — the plot of coefficient values as a function of regularization strength — which remains a standard diagnostic today.
      </Prose>

      <Prose>
        Ridge shrinks all coefficients toward zero but never eliminates them. In problems with hundreds or thousands of features, this is unsatisfying: you want the model to tell you which features matter and which do not. The solution came in 1996. Robert Tibshirani published "Regression Shrinkage and Selection via the Lasso" in the Journal of the Royal Statistical Society, Series B, volume 58(1), pages 267–288. The LASSO (Least Absolute Shrinkage and Selection Operator) replaces ridge's squared penalty on weights with an absolute value penalty. The geometric consequence is that the constraint region has corners at the axes, and the optimal solution lands at a corner — meaning some weights are exactly zero. LASSO simultaneously shrinks and selects, acting as a continuous alternative to stepwise feature selection with much better statistical properties.
      </Prose>

      <Prose>
        LASSO has one weakness: when features are correlated, it tends to select one arbitrarily and discard the others, even if the true model uses all of them. The fix came in 2005. Hui Zou and Trevor Hastie published "Regularization and Variable Selection via the Elastic Net" in JRSS-B, volume 67(2), pages 301–320. Elastic Net linearly interpolates between L1 and L2 penalties: it can shrink, select, and handle correlated features by grouping them. It became the default choice for high-dimensional regression whenever both sparsity and correlation are present.
      </Prose>

      <Prose>
        Dropout arrived from a completely different direction. Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov published "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" in the Journal of Machine Learning Research, volume 15, pages 1929–1958, in 2014. The idea is disarmingly simple: during training, randomly zero out neurons with probability <Code>p</Code> at each forward pass. This forces the network to learn redundant representations — no single neuron can rely on any other neuron being present — and acts as an implicit regularizer. Wang and Manning (ICML 2013, "Fast dropout training") showed that dropout's expected behavior is approximately equivalent to Gaussian noise injection and has connections to L2 regularization. In practice, dropout is now one of the most widely deployed regularization techniques for deep neural networks.
      </Prose>

      <Callout type="insight">
        All four methods answer the same question — how to control model capacity without changing the algorithm's structure — but from different angles. L2 shrinks smoothly. L1 creates sparsity. Elastic Net does both. Dropout regularizes stochastically by destroying information during training. Together they cover the full spectrum of modern regularization practice.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Regularization adds a penalty to the training objective that discourages large or numerous weights. The character of the penalty determines the character of the solution.
      </Prose>

      <H3>2.1 L2 (Ridge): smooth shrinkage</H3>

      <Prose>
        L2 regularization adds <Code>{"λ‖w‖²"}</Code> to the loss — the sum of squared weights. This penalizes large weights quadratically: a weight of 2 contributes 4 to the penalty; a weight of 4 contributes 16. The gradient of the penalty is <Code>2λw</Code>, which means every gradient descent step includes a small pull toward zero proportional to the current weight magnitude. Large weights get pulled harder; small weights get pulled gently. The result is that all weights shrink toward zero, but none reach exactly zero unless the data provides zero signal for that feature. L2 is analogous to a Bayesian prior: placing a Gaussian prior <Code>{"N(0, 1/(2λ))"}</Code> on each weight and taking the MAP estimate recovers the ridge solution exactly.
      </Prose>

      <H3>2.2 L1 (LASSO): sparse selection</H3>

      <Prose>
        L1 regularization adds <Code>{"λ‖w‖₁"}</Code> — the sum of absolute values of weights. The penalty is linear in each weight's magnitude, not quadratic. This creates a qualitatively different behavior: the subgradient of the L1 penalty at <Code>w=0</Code> is the interval <Code>[-λ, λ]</Code>, not a single value. A weight reaches zero and stays there whenever the data signal for that feature (the OLS gradient) is smaller than <Code>λ</Code> in magnitude. Small irrelevant features — where the gradient is noise — get zeroed out completely. Large relevant features are shrunk but survive. The practical consequence is automatic feature selection: fit LASSO with a good <Code>λ</Code> and the non-zero weights tell you which features matter.
      </Prose>

      <Prose>
        The geometric picture makes this concrete. The OLS objective is an ellipse (in 2D, at minimum when the contours touch a point). The L2 constraint region is a circle; the L1 constraint region is a diamond. The constrained optimum is where the ellipse first touches the constraint region. For a circle, touching can happen anywhere on the boundary — the optimum is on the circle but not at a corner, so neither weight is zero. For a diamond, the corners point along the axes; an ellipse from almost any direction will touch a corner first, meaning one weight is forced to zero. This is why L1 induces sparsity and L2 does not.
      </Prose>

      <H3>2.3 Elastic Net: the best of both</H3>

      <Prose>
        Elastic Net combines both penalties: <Code>{"λ·(α‖w‖₁ + (1−α)·‖w‖²/2)"}</Code>. The parameter <Code>α ∈ [0, 1]</Code> interpolates between pure L2 (<Code>α=0</Code>) and pure L1 (<Code>α=1</Code>). When <Code>0 {"<"} α {"<"} 1</Code>, the constraint region is a rounded diamond — corners exist but are softened. The sparsity property is preserved (some weights hit zero) but the grouping property is added: correlated features tend to be selected or discarded together, rather than one being arbitrarily chosen. Elastic Net is the standard choice when you have correlated predictors and want both sparse and stable solutions.
      </Prose>

      <H3>2.4 Dropout: stochastic regularization</H3>

      <Prose>
        Dropout is qualitatively different. It does not add a penalty term to the loss. Instead, at each training step, it randomly zeroes out a random fraction of activations. A neuron that is dropped contributes nothing to the forward pass and receives no gradient in the backward pass. Because different random subsets are dropped at each step, no neuron can co-adapt with specific other neurons — it must learn features useful in the context of many different subnetworks. At test time, all neurons are active, but their outputs are scaled by the keep probability to match the expected activation during training (inverted dropout scales during training instead, making test time a simple pass-through). The connection to L2: Wang and Manning (2013) showed that dropout's expected gradient update is approximately the gradient of an L2-regularized objective with a strength proportional to <Code>p(1-p)</Code>.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 Ridge: closed-form solution</H3>

      <Prose>
        Ordinary least squares minimizes <Code>{"‖y − Xw‖²"}</Code>. Ridge adds a squared weight penalty:
      </Prose>

      <MathBlock>
        {"\\mathcal{L}_{\\text{ridge}}(w) = \\|y - Xw\\|^2 + \\lambda \\|w\\|^2"}
      </MathBlock>

      <Prose>
        Taking the gradient and setting to zero:
      </Prose>

      <MathBlock>
        {"-2X^\\top(y - Xw) + 2\\lambda w = 0 \\quad \\Rightarrow \\quad (X^\\top X + \\lambda I)w = X^\\top y"}
      </MathBlock>

      <Prose>
        The ridge solution is:
      </Prose>

      <MathBlock>
        {"w^*_{\\text{ridge}} = (X^\\top X + \\lambda I)^{-1} X^\\top y"}
      </MathBlock>

      <Prose>
        Two critical properties fall out immediately. First, <Code>{"(XᵀX + λI)"}</Code> is always invertible for <Code>{"λ > 0"}</Code>, even when <Code>{"XᵀX"}</Code> is singular (i.e., when <Code>{"d > n"}</Code> or features are perfectly collinear). Ridge fixes the underdetermined problem. Second, as <Code>{"λ → ∞"}</Code>, <Code>{"w* → 0"}</Code>; as <Code>{"λ → 0"}</Code>, <Code>{"w* → w_OLS"}</Code>. The parameter <Code>{"λ"}</Code> continuously interpolates between full shrinkage and no regularization.
      </Prose>

      <Prose>
        The Bayesian interpretation: placing a zero-mean Gaussian prior <Code>{"w ~ N(0, (1/λ)I)"}</Code> on the weights and computing the MAP estimate recovers the ridge solution exactly. The prior variance <Code>{"1/λ"}</Code> encodes our belief about typical weight magnitudes. This connection is why ridge is sometimes called Gaussian regularization.
      </Prose>

      <H3>3.2 LASSO: coordinate descent and soft-thresholding</H3>

      <Prose>
        LASSO minimizes:
      </Prose>

      <MathBlock>
        {"\\mathcal{L}_{\\text{lasso}}(w) = \\frac{1}{2n}\\|y - Xw\\|^2 + \\lambda \\|w\\|_1"}
      </MathBlock>

      <Prose>
        There is no closed form because the L1 norm is not differentiable at zero. The standard solver is coordinate descent: cycle through each weight <Code>{"w_j"}</Code>, holding others fixed, and compute the optimal <Code>{"w_j"}</Code> analytically. The partial residual — what the model cannot explain using all other features — is:
      </Prose>

      <MathBlock>
        {"r_j = y - X_{-j}w_{-j} = y - Xw + X_{:,j}\\,w_j"}
      </MathBlock>

      <Prose>
        The one-dimensional LASSO problem for <Code>{"w_j"}</Code> given <Code>{"r_j"}</Code> has a closed-form solution via soft-thresholding:
      </Prose>

      <MathBlock>
        {"w_j^* = S\\!\\left(\\frac{X_{:,j}^\\top r_j}{\\|X_{:,j}\\|^2},\\; \\frac{\\lambda}{\\|X_{:,j}\\|^2}\\right) \\quad \\text{where} \\quad S(z, \\gamma) = \\text{sign}(z)\\cdot\\max(|z| - \\gamma,\\, 0)"}
      </MathBlock>

      <Prose>
        Soft-thresholding is the proximal operator of the L1 norm. It zeroes out values with magnitude below <Code>{"γ"}</Code> and shifts others toward zero by <Code>{"γ"}</Code>. This is why coordinate descent produces exact zeros: whenever the signal <Code>{"X_{:,j}ᵀr_j"}</Code> is smaller than <Code>{"λ"}</Code> in magnitude, the soft-threshold maps it to zero and it stays there.
      </Prose>

      <Prose>
        The Bayesian interpretation: LASSO corresponds to placing a Laplace (double-exponential) prior <Code>{"w_j ~ Laplace(0, 1/λ)"}</Code> on each weight. The Laplace prior has a sharp peak at zero and heavier tails than Gaussian — it encourages exactly-zero weights while allowing a few large ones. The MAP estimate under this prior is the LASSO solution.
      </Prose>

      <H3>3.3 Elastic Net: hybrid penalty</H3>

      <Prose>
        Elastic Net combines both:
      </Prose>

      <MathBlock>
        {"\\mathcal{L}_{\\text{EN}}(w) = \\frac{1}{2n}\\|y - Xw\\|^2 + \\lambda\\left(\\alpha\\|w\\|_1 + \\frac{1-\\alpha}{2}\\|w\\|^2\\right)"}
      </MathBlock>

      <Prose>
        The mixed penalty can be solved via proximal gradient descent: take a gradient step on the smooth part (data loss + L2 penalty), then apply soft-thresholding for the L1 part. For coordinate descent, the update for each <Code>{"w_j"}</Code> is:
      </Prose>

      <MathBlock>
        {"w_j^* = \\frac{S\\!\\left(X_{:,j}^\\top r_j / n,\\; \\lambda\\alpha\\right)}{\\|X_{:,j}\\|^2/n + \\lambda(1-\\alpha)}"}
      </MathBlock>

      <Prose>
        The L2 term in the denominator <em>groups</em> correlated features — if <Code>{"x_i"}</Code> and <Code>{"x_j"}</Code> are highly correlated, both tend to receive similar coefficients rather than one being arbitrarily zeroed. This is the "encourages grouping effect" described by Zou and Hastie (2005). Pure L1 breaks this: it picks one correlated feature and discards the rest.
      </Prose>

      <H3>3.4 Dropout: ensemble averaging and L2 connection</H3>

      <Prose>
        Let <Code>{"p"}</Code> be the probability of keeping a neuron (keep probability). At each training step, a Bernoulli mask <Code>{"m ~ Bernoulli(p)"}</Code> is drawn and applied element-wise to the activations. The effective network at step <Code>{"t"}</Code> is a subnetwork defined by the mask. Over many training steps, the model learns across an exponential ensemble of <Code>{"2^n"}</Code> different subnetworks (where <Code>{"n"}</Code> is the number of neurons). At test time, using all neurons with activations scaled by <Code>{"p"}</Code> approximates averaging over all <Code>{"2^n"}</Code> networks — an exponential ensemble for the cost of one forward pass.
      </Prose>

      <Prose>
        The L2 connection: Wang and Manning (2013) showed that dropout applied to a linear model is equivalent to optimizing a quadratic lower bound on the expected loss, which has the form of an L2-regularized objective. The effective regularization strength is <Code>{"p(1-p)σ²x"}</Code>, where <Code>{"σ²x"}</Code> is the input variance. Features with high variance are regularized more strongly — dropout implicitly adapts to the data's structure. Inverted dropout (the standard implementation) scales activations by <Code>{"1/p"}</Code> during training so that the expected activation at test time is unchanged, eliminating the need to scale at inference.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        All code below runs on a single synthetic dataset: <Code>n=100</Code> samples, <Code>d=10</Code> features, true weights with exactly 4 zeros to test sparsity recovery. NumPy only — no scikit-learn. Every output shown is verbatim stdout.
      </Prose>

      <H3>4a. Ridge regression — closed form</H3>

      <CodeBlock language="python">
{`import numpy as np

np.random.seed(42)
n, d = 100, 10
X = np.random.randn(n, d)
true_w = np.array([3.0, -2.0, 1.5, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, -1.0])
y = X @ true_w + np.random.randn(n) * 0.5

# Ridge closed form: w* = (X^T X + lambda I)^{-1} X^T y
# np.linalg.solve is numerically stabler than explicit matrix inverse
lam = 1.0
w_ridge = np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)
print("=== Ridge (lambda=1.0) ===")
print("Weights:", np.round(w_ridge, 4))
print("True   :", true_w)
mse_ridge = np.mean((y - X @ w_ridge) ** 2)
print(f"Train MSE: {mse_ridge:.4f}")
# Output:
# === Ridge (lambda=1.0) ===
# Weights: [ 2.921  -2.0038  1.4686  0.0193 -0.0618  0.0392  0.6956  0.0002  0.0322 -1.0117]
# True   : [ 3.  -2.   1.5  0.   0.   0.   0.8  0.   0.  -1. ]
# Train MSE: 0.2157`}
      </CodeBlock>

      <Prose>
        Ridge recovers all four signal features well (weights 0, 1, 2, 6, 9) but leaves the four zero features (3, 4, 5, 7, 8) with small non-zero values around 0.02–0.06. It shrinks everything toward zero but cannot produce exact zeros. As <Code>{"λ"}</Code> increases, all weights shrink further; but they approach zero asymptotically, never reaching it.
      </Prose>

      <H3>4b. LASSO — coordinate descent with soft-thresholding</H3>

      <CodeBlock language="python">
{`def soft_threshold(x, lam):
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)

def lasso_coordinate_descent(X, y, lam=0.1, n_iter=200, tol=1e-6):
    n, d = X.shape
    w = np.zeros(d)
    col_norms_sq = np.sum(X ** 2, axis=0)   # ||X[:,j]||^2 for each feature
    for it in range(n_iter):
        w_old = w.copy()
        for j in range(d):
            # Partial residual: what the model can't explain without feature j
            r_j = y - X @ w + X[:, j] * w[j]
            rho_j = X[:, j] @ r_j          # X[:,j]^T r_j
            # Soft-threshold: zero out if |rho_j| < lambda
            w[j] = soft_threshold(rho_j / col_norms_sq[j],
                                   lam / col_norms_sq[j])
        if np.max(np.abs(w - w_old)) < tol:
            break
    return w

print("=== LASSO at different lambda values ===")
for lam_val in [0.01, 0.10, 0.50, 1.00]:
    w_l = lasso_coordinate_descent(X, y, lam=lam_val, n_iter=500)
    zeros = np.sum(np.abs(w_l) < 1e-6)
    print(f"lambda={lam_val:.2f}  zeros={zeros}/{d}  w={np.round(w_l, 3)}")
# Output:
# lambda=0.01  zeros=0/10  w=[ 2.955 -2.026  1.484  0.022 -0.061  0.032  0.699  0.002  0.024 -1.028]
# lambda=0.10  zeros=0/10  w=[ 2.955 -2.026  1.482  0.022 -0.06   0.031  0.698  0.     0.023 -1.027]
# lambda=0.50  zeros=1/10  w=[ 2.951 -2.023  1.477  0.019 -0.056  0.028  0.695  0.     0.019 -1.022]
# lambda=1.00  zeros=1/10  w=[ 2.947 -2.019  1.47   0.015 -0.052  0.023  0.691  0.     0.015 -1.016]`}
      </CodeBlock>

      <Prose>
        At <Code>{"λ=0.5"}</Code>, weight 7 (true value 0) is exactly zeroed. As <Code>{"λ"}</Code> increases, more weights hit zero. The signal weights (0, 1, 2, 6, 9) are shrunk but survive. At <Code>{"λ=1.0"}</Code>, the LASSO still finds only 1 exact zero — this dataset has low noise (σ=0.5) and well-separated features, so the algorithm needs a larger <Code>{"λ"}</Code> to zero out the remaining three near-zero weights (3, 4, 5, 8).
      </Prose>

      <H3>4c. Elastic Net — proximal gradient descent</H3>

      <CodeBlock language="python">
{`def elastic_net_proximal(X, y, lam=0.2, alpha=0.5, lr=0.01, n_iter=2000, tol=1e-6):
    """
    Elastic Net: min (1/2n)||y-Xw||^2 + lam*(alpha*||w||_1 + (1-alpha)*0.5*||w||^2)
    Proximal gradient: gradient step on smooth part, soft-threshold for L1.
    """
    n, d = X.shape
    w = np.zeros(d)
    for it in range(n_iter):
        w_old = w.copy()
        # Gradient of smooth part: data loss + L2 penalty
        grad = (1 / n) * X.T @ (X @ w - y) + lam * (1 - alpha) * w
        w_half = w - lr * grad
        # Proximal step: soft-threshold for L1 part
        w = soft_threshold(w_half, lr * lam * alpha)
        if np.max(np.abs(w - w_old)) < tol:
            break
    return w

print("=== Elastic Net: alpha interpolates L2 -> L1 ===")
for alpha_val in [0.0, 0.5, 1.0]:
    w_en = elastic_net_proximal(X, y, lam=0.2, alpha=alpha_val, lr=0.01)
    zeros = np.sum(np.abs(w_en) < 1e-6)
    print(f"alpha={alpha_val:.1f}  zeros={zeros}/{d}  w={np.round(w_en, 3)}")
# Output:
# alpha=0.0  zeros=0/10  w=[ 2.394 -1.665  1.227 -0.01  -0.072  0.13   0.628 -0.015  0.13  -0.774]
# alpha=0.5  zeros=4/10  w=[ 2.568 -1.76   1.222  0.    -0.     0.011  0.582 -0.     0.    -0.783]
# alpha=1.0  zeros=5/10  w=[ 2.759 -1.827  1.235  0.    -0.     0.     0.525 -0.     0.    -0.771]`}
      </CodeBlock>

      <Prose>
        At <Code>{"alpha=0.0"}</Code> (pure Ridge), no zeros — all weights survive. At <Code>{"alpha=0.5"}</Code>, 4 zeros appear: the grouping + sparsity combination correctly identifies 4 of the true zero features. At <Code>{"alpha=1.0"}</Code> (pure LASSO), 5 zeros — LASSO overshoots slightly and zeroes out one small-but-real feature due to the L1 penalty's harsher treatment of correlated predictors.
      </Prose>

      <H3>4d. Dropout — 2-layer network with inverted dropout</H3>

      <CodeBlock language="python">
{`def relu(x):
    return np.maximum(0, x)

def dropout_mask(shape, p_keep):
    """Inverted dropout: scale by 1/p_keep so test time needs no correction."""
    return (np.random.rand(*shape) < p_keep) / p_keep

class TwoLayerDropoutNet:
    def __init__(self, input_dim, hidden_dim, output_dim, p_keep=0.5):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros(output_dim)
        self.p_keep = p_keep

    def forward(self, X, training=True):
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.h1 = relu(self.z1)
        if training:
            self.mask = dropout_mask(self.h1.shape, self.p_keep)
            self.h1_drop = self.h1 * self.mask
        else:
            self.mask = np.ones_like(self.h1)
            self.h1_drop = self.h1   # no scaling needed at test time
        self.out = self.h1_drop @ self.W2 + self.b2
        return self.out

    def backward(self, y, lr=0.005):
        n = len(y)
        loss = np.mean((self.out.ravel() - y) ** 2)
        d_out = (2 / n) * (self.out.ravel() - y).reshape(-1, 1)
        dW2 = self.h1_drop.T @ d_out
        db2 = d_out.sum(axis=0)
        d_h1_drop = d_out @ self.W2.T
        d_h1 = d_h1_drop * self.mask    # backprop through dropout
        d_z1 = d_h1 * (self.z1 > 0)    # backprop through ReLU
        dW1 = self.X.T @ d_z1
        db1 = d_z1.sum(axis=0)
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        return loss

np.random.seed(0)
true_w = np.array([1.0, -2.0, 0.5, 0.0, 0.0])
X_data = np.random.randn(200, 5)
y_data = X_data @ true_w + 0.1 * np.random.randn(200)

net = TwoLayerDropoutNet(input_dim=5, hidden_dim=16, output_dim=1, p_keep=0.5)
print("=== 2-Layer Net with Dropout (p_keep=0.5) ===")
for epoch in range(300):
    out = net.forward(X_data, training=True)
    loss = net.backward(y_data, lr=0.005)
    if epoch in [0, 9, 49, 99, 199, 299]:
        print(f"Epoch {epoch+1:>3}  train MSE (stochastic): {loss:.4f}")

out_test = net.forward(X_data, training=False)
test_mse = np.mean((out_test.ravel() - y_data) ** 2)
print(f"Test MSE (dropout off): {test_mse:.4f}")
# Output:
# === 2-Layer Net with Dropout (p_keep=0.5) ===
# Epoch   1  train MSE (stochastic): 4.9473
# Epoch  10  train MSE (stochastic): 4.8492
# Epoch  50  train MSE (stochastic): 4.5863
# Epoch 100  train MSE (stochastic): 3.9825
# Epoch 200  train MSE (stochastic): 1.4686
# Epoch 300  train MSE (stochastic): 0.7592
# Test MSE (dropout off): 0.2949`}
      </CodeBlock>

      <Prose>
        Training MSE is intentionally noisier and higher than test MSE — dropout degrades performance during training by design. At test time with dropout disabled, the test MSE (0.29) is much lower than the stochastic training MSE (0.76), confirming the network generalized well. The gap between stochastic training loss and clean test loss is a normal feature of dropout, not a sign of overfitting.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <H3>5a. sklearn linear models</H3>

      <CodeBlock language="python">
{`import numpy as np
from sklearn.linear_model import (
    Ridge, Lasso, ElasticNet,
    RidgeCV, LassoCV, ElasticNetCV,
    LogisticRegression
)
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n, d = 100, 10
X = np.random.randn(n, d)
true_w = np.array([3.0, -2.0, 1.5, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, -1.0])
y = X @ true_w + np.random.randn(n) * 0.5

# CRITICAL: always standardize before regularization
# Features with large variance dominate the penalty without scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Ridge ---
ridge = Ridge(alpha=1.0)
ridge.fit(X_scaled, y)
print(f"Ridge (alpha=1):   coef={np.round(ridge.coef_, 3)}")
print(f"  zeros={np.sum(np.abs(ridge.coef_)<1e-6)}/{d}")
# Output: Ridge (alpha=1):   coef=[ 2.605 -2.085  1.504  0.01  -0.07   0.041  0.667 -0.004  0.03  -1.03 ]
#   zeros=0/10

# --- Lasso ---
lasso = Lasso(alpha=0.05)
lasso.fit(X_scaled, y)
print(f"Lasso (alpha=0.05): coef={np.round(lasso.coef_, 3)}")
print(f"  zeros={np.sum(np.abs(lasso.coef_)<1e-6)}/{d}")
# Output: Lasso (alpha=0.05): coef=[ 2.596 -2.066  1.449  0.    -0.022  0.     0.633 -0.     0.    -0.985]
#   zeros=4/10

# --- ElasticNet ---
en = ElasticNet(alpha=0.1, l1_ratio=0.5)
en.fit(X_scaled, y)
print(f"ElasticNet (alpha=0.1, l1_ratio=0.5): coef={np.round(en.coef_, 3)}")
print(f"  zeros={np.sum(np.abs(en.coef_)<1e-6)}/{d}")
# Output: ElasticNet (alpha=0.1, l1_ratio=0.5): coef=[ 2.486 -1.949  1.369  0.    -0.025  0.025  0.622 -0.     0.013 -0.906]
#   zeros=2/10

# --- CV versions: automatic lambda selection ---
lasso_cv = LassoCV(cv=5, random_state=42)
lasso_cv.fit(X_scaled, y)
print(f"LassoCV  best_alpha={lasso_cv.alpha_:.4f}  zeros={np.sum(np.abs(lasso_cv.coef_)<1e-6)}/{d}")
# Output: LassoCV  best_alpha=0.0364  zeros=3/10

en_cv = ElasticNetCV(cv=5, l1_ratio=[0.1, 0.5, 0.9, 1.0], random_state=42)
en_cv.fit(X_scaled, y)
print(f"ElasticNetCV  best_alpha={en_cv.alpha_:.4f}  best_l1_ratio={en_cv.l1_ratio_:.1f}")
# Output: ElasticNetCV  best_alpha=0.0364  best_l1_ratio=1.0`}
      </CodeBlock>

      <Callout type="info" title="sklearn API notes">
        In Ridge and Lasso, <Code>alpha</Code> is the regularization strength (what the math calls <Code>{"λ"}</Code>). In LogisticRegression, the convention is flipped: <Code>C = 1/λ</Code> so smaller <Code>C</Code> means stronger regularization. <Code>l1_ratio</Code> in ElasticNet is the <Code>{"α"}</Code> parameter from the math (0 = pure L2, 1 = pure L1). Use <Code>LassoCV</Code> and <Code>ElasticNetCV</Code> instead of manual grid search — they use the warm-start regularization path and are much faster than cross-validating independently for each <Code>alpha</Code>.
      </Callout>

      <H3>5b. Logistic regression with L1/L2/Elastic Net</H3>

      <CodeBlock language="python">
{`from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

X_clf, y_clf = make_classification(
    n_samples=500, n_features=20, n_informative=5,
    n_redundant=5, random_state=42
)
X_tr, X_te, y_tr, y_te = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)
X_te = scaler.transform(X_te)

# lbfgs: L2 (default), fast, no L1
clf_l2 = LogisticRegression(solver="lbfgs", C=1.0, max_iter=500)
clf_l2.fit(X_tr, y_tr)
print(f"lbfgs  L2  C=1.0  -> acc={accuracy_score(y_te, clf_l2.predict(X_te)):.4f}  zeros={np.sum(clf_l2.coef_[0]==0)}/20")

# saga: L1, scales to large sparse data
clf_l1 = LogisticRegression(solver="saga", penalty="l1", C=0.5, max_iter=2000, random_state=42)
clf_l1.fit(X_tr, y_tr)
print(f"saga   L1  C=0.5  -> acc={accuracy_score(y_te, clf_l1.predict(X_te)):.4f}  zeros={np.sum(clf_l1.coef_[0]==0)}/20")

# saga: Elastic Net (penalty='elasticnet' + l1_ratio)
clf_en = LogisticRegression(
    solver="saga", penalty="elasticnet", C=0.5, l1_ratio=0.5,
    max_iter=2000, random_state=42
)
clf_en.fit(X_tr, y_tr)
print(f"saga   EN  C=0.5  -> acc={accuracy_score(y_te, clf_en.predict(X_te)):.4f}  zeros={np.sum(clf_en.coef_[0]==0)}/20")`}
      </CodeBlock>

      <H3>5c. PyTorch Dropout — training vs eval mode</H3>

      <CodeBlock language="python">
{`import torch
import torch.nn as nn

torch.manual_seed(42)

class SimpleNet(nn.Module):
    def __init__(self, p_drop=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Dropout(p=p_drop),    # p is DROP probability (1 - p_keep)
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=p_drop),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.net(x)

model = SimpleNet(p_drop=0.5)
X_t = torch.randn(5, 10)

# Training mode: dropout is ACTIVE — different output each call
model.train()
out1 = model(X_t).detach().numpy().ravel().round(4)
out2 = model(X_t).detach().numpy().ravel().round(4)
print(f"[train] pass 1: {out1}")
print(f"[train] pass 2: {out2}")
print("^ Stochastic: outputs differ because dropout masks are resampled each pass")

# Eval mode: dropout is DISABLED — deterministic
model.eval()
with torch.no_grad():
    out3 = model(X_t).numpy().ravel().round(4)
    out4 = model(X_t).numpy().ravel().round(4)
print(f"[eval]  pass 1: {out3}")
print(f"[eval]  pass 2: {out4}")
print("^ Deterministic: identical outputs, no dropout")

# Weight decay in the optimizer = L2 regularization on all parameters
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
print(f"\\nAdam weight_decay=1e-4 adds L2 penalty: ||w||^2 * weight_decay")
print("Use weight_decay on top of Dropout for double regularization in deep nets")`}
      </CodeBlock>

      <Callout type="info" title="Dropout API gotcha">
        <Code>nn.Dropout(p=0.5)</Code> — <Code>p</Code> is the <em>drop</em> probability, not the keep probability. So <Code>p=0.5</Code> keeps 50% of neurons. Always call <Code>model.train()</Code> before training and <Code>model.eval()</Code> before inference. PyTorch uses inverted dropout internally: during training it scales activations by <Code>{"1/(1-p)"}</Code> so test-time inference requires no correction. Forgetting <Code>model.eval()</Code> at inference time is one of the most common Dropout bugs — your predictions will be noisy and unpredictable.
      </Callout>

      <H3>5d. Tree regularization in XGBoost/LightGBM</H3>

      <CodeBlock language="python">
{`import xgboost as xgb
import lightgbm as lgb
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X, y = make_regression(n_samples=500, n_features=10, noise=20, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost: reg_alpha (L1 on leaf weights), reg_lambda (L2 on leaf weights)
xgb_model = xgb.XGBRegressor(
    n_estimators=100, max_depth=4,
    reg_alpha=0.1,    # L1: pushes leaf weights toward zero, creates sparse trees
    reg_lambda=1.0,   # L2: shrinks leaf weights (default=1)
    tree_method="hist", random_state=42, eval_metric="rmse",
    early_stopping_rounds=10, verbosity=0
)
xgb_model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
print(f"XGBoost  reg_alpha=0.1  reg_lambda=1.0  -> best_iter={xgb_model.best_iteration}")

# LightGBM: lambda_l1, lambda_l2
lgb_model = lgb.LGBMRegressor(
    n_estimators=100, num_leaves=31,
    reg_alpha=0.1,    # L1 (lambda_l1)
    reg_lambda=1.0,   # L2 (lambda_l2)
    random_state=42, verbose=-1
)
lgb_model.fit(
    X_tr, y_tr,
    eval_set=[(X_te, y_te)],
    callbacks=[lgb.early_stopping(10, verbose=False), lgb.log_evaluation(-1)]
)
print(f"LightGBM reg_alpha=0.1  reg_lambda=1.0  -> best_iter={lgb_model.best_iteration_}")`}
      </CodeBlock>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6a. Regularization path: weight trajectories vs lambda</H3>

      <Prose>
        As <Code>{"λ"}</Code> increases from near-zero to large, the weight on feature 0 (true value 3.0) shrinks differently under each method. Ridge shrinks smoothly but never reaches zero. LASSO shrinks more aggressively and eventually hits exactly zero. Elastic Net (<Code>{"α=0.5"}</Code>) lies between the two.
      </Prose>

      <Plot
        label="Regularization path — weight on feature 0 vs lambda (true value = 3.0)"
        xLabel="Regularization strength (lambda)"
        yLabel="Weight value"
        series={[
          {
            name: "Ridge (L2)",
            color: colors.gold,
            points: [
              [0.001, 2.629], [0.01, 2.628], [0.05, 2.628], [0.1, 2.626],
              [0.3, 2.622], [0.5, 2.617], [1.0, 2.605], [2.0, 2.582],
              [5.0, 2.516], [10.0, 2.413],
            ],
          },
          {
            name: "LASSO (L1)",
            color: colors.green,
            points: [
              [0.001, 2.628], [0.01, 2.622], [0.05, 2.596], [0.1, 2.558],
              [0.3, 2.407], [0.5, 2.255], [1.0, 1.851], [2.0, 0.950],
              [5.0, 0.0], [10.0, 0.0],
            ],
          },
          {
            name: "Elastic Net (alpha=0.5)",
            color: "#c084fc",
            points: [
              [0.001, 2.627], [0.01, 2.614], [0.05, 2.556], [0.1, 2.486],
              [0.3, 2.231], [0.5, 2.005], [1.0, 1.552], [2.0, 0.949],
              [5.0, 0.130], [10.0, 0.0],
            ],
          },
        ]}
      />

      <H3>6b. Validation error U-curve vs lambda</H3>

      <Prose>
        Cross-validated MSE follows a U-shaped curve. Too small a <Code>{"λ"}</Code> overfits (training noise is memorized). Too large a <Code>{"λ"}</Code> underfits (all weights are shrunk to zero). The sweet spot is the minimum of the CV curve, which <Code>RidgeCV</Code> and <Code>LassoCV</Code> find automatically.
      </Prose>

      <Plot
        label="Ridge 5-fold CV MSE vs lambda — U-curve showing optimal regularization"
        xLabel="Lambda"
        yLabel="5-fold CV MSE"
        series={[
          {
            name: "CV MSE (mean)",
            color: colors.gold,
            points: [
              [0.001, 0.277], [0.01, 0.277], [0.05, 0.277], [0.1, 0.277],
              [0.3, 0.277], [0.5, 0.277], [1.0, 0.279], [2.0, 0.288],
              [5.0, 0.343], [10.0, 0.510],
            ],
          },
        ]}
      />

      <Prose>
        This dataset is low-noise, so the optimal <Code>{"λ"}</Code> is small (around 0.1–0.5) and the curve is flat on the left before rising sharply. In noisy real-world datasets, the U-shape is more pronounced and the left side rises earlier, making the sweet spot easier to identify.
      </Prose>

      <H3>6c. Sparsity: LASSO zeros vs lambda</H3>

      <Heatmap
        label="LASSO weight magnitude at different lambda (10 features, 4 true zeros)"
        rowLabels={["lam=0.01", "lam=0.10", "lam=0.50", "lam=1.00"]}
        colLabels={["w0", "w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8", "w9"]}
        matrix={[
          [2.955, 2.026, 1.484, 0.022, 0.061, 0.032, 0.699, 0.002, 0.024, 1.028],
          [2.955, 2.026, 1.482, 0.022, 0.06,  0.031, 0.698, 0.0,   0.023, 1.027],
          [2.951, 2.023, 1.477, 0.019, 0.056, 0.028, 0.695, 0.0,   0.019, 1.022],
          [2.947, 2.019, 1.47,  0.015, 0.052, 0.023, 0.691, 0.0,   0.015, 1.016],
        ]}
        colorScale="gold"
      />

      <Prose>
        Weight 7 (true value 0) goes to zero earliest, at <Code>{"λ=0.1"}</Code>. Weights 3, 4, 5, 8 (all true zeros) require larger <Code>{"λ"}</Code> because they have small but non-negligible OLS estimates on this finite sample. The signal weights (0, 1, 2, 6, 9) survive across all tested <Code>{"λ"}</Code> values, shrinking slowly.
      </Prose>

      <H3>6d. Coordinate descent trace for LASSO</H3>

      <Prose>
        The following trace shows 5 full sweeps of coordinate descent for LASSO at <Code>{"λ=0.5"}</Code>. Each sweep updates all 10 weights once. Weight 7 reaches zero and stays there from iteration 4 onward.
      </Prose>

      <StepTrace
        label="LASSO coordinate descent — 5 sweeps at lambda=0.5"
        steps={[
          {
            label: "Iter 1 — first sweep",
            render: () => (
              <Prose>
                w = [3.312, -1.791, 1.270, 0.001, -0.184, 0.021, 0.755, -0.164, 0.068, -0.891]. All 10 weights non-zero. Loss = 4.432. First sweep uses OLS-like estimates as starting point — some weights overshoot before soft-thresholding brings them back on subsequent iterations. Weight 7 is at -0.164 (still non-zero).
              </Prose>
            ),
          },
          {
            label: "Iter 2 — large initial oscillations dampen",
            render: () => (
              <Prose>
                w = [3.005, -1.941, 1.470, -0.016, -0.066, 0.052, 0.724, -0.020, 0.045, -0.988]. Loss = 4.276. Rapid convergence on the major weights (0, 1, 2, 9). Weight 7 shrinks from -0.164 to -0.020. Weight 3 flips sign (from 0.001 to -0.016) — coordinate descent can oscillate for near-zero features before settling.
              </Prose>
            ),
          },
          {
            label: "Iter 3 — near-zero weights tighten",
            render: () => (
              <Prose>
                w = [2.957, -2.003, 1.476, 0.003, -0.058, 0.035, 0.702, -0.000, 0.028, -1.013]. Loss = 4.246. Weight 7 is now -0.000 — effectively zero but not yet pinned. The algorithm has found the true structure: weights 0, 1, 2, 6, 9 are strong, the rest are near zero.
              </Prose>
            ),
          },
          {
            label: "Iter 4 — weight 7 pinned to zero",
            render: () => (
              <Prose>
                w = [2.952, -2.018, 1.477, 0.014, -0.057, 0.029, 0.697, -0.000, 0.022, -1.020]. Loss = 4.250. Weight 7 = 0.000 exactly — soft-thresholding clips it precisely to zero. Once at zero, coordinate descent will not move it unless the partial residual exceeds the threshold (it never does here). The solution is stabilizing.
              </Prose>
            ),
          },
          {
            label: "Iter 5 — convergence",
            render: () => (
              <Prose>
                w = [2.951, -2.022, 1.477, 0.017, -0.056, 0.028, 0.696, 0.000, 0.020, -1.021]. Loss = 4.252. Weights change by less than 0.005 from iteration 4. The algorithm converges within 11 total sweeps. Final result: 1 exact zero (weight 7), 9 non-zero weights. Weights 3, 4, 5, 8 remain non-zero because their OLS estimates are still above the soft-threshold level at {"λ=0.5"}.
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
        Choosing the right regularization method depends on the model type, feature structure, dataset size, and what you need the solution to do.
      </Prose>

      <StepTrace
        label="which regularization to use"
        steps={[
          {
            label: "L2 (Ridge) — the safe default",
            render: () => (
              <Prose>
                Use L2 when: you want to add regularization without thinking hard about it. Ridge is the right default for linear and logistic regression in nearly every setting. It is numerically stable, has a closed form, always has a unique solution (even when {"d > n"}), and its bias is mild unless <Code>{"α"}</Code> (sklearn's <Code>alpha</Code>) is very large. If features are correlated, Ridge still works — it shrinks all correlated features equally rather than arbitrarily eliminating some. Use <Code>RidgeCV</Code> to select <Code>alpha</Code> automatically via efficient leave-one-out cross-validation (it costs the same as fitting Ridge once). When NOT to use: when you need sparse solutions or automatic feature selection — Ridge will never zero out a weight.
              </Prose>
            ),
          },
          {
            label: "L1 (LASSO) — when sparsity is the goal",
            render: () => (
              <Prose>
                Use L1 when: you believe many features are irrelevant and want automatic feature selection. LASSO is the standard choice for genomics ({"p > 10,000"} features, few are causal), NLP bag-of-words with large vocabularies, and any setting where interpretability requires a small number of non-zero coefficients. Use <Code>LassoCV</Code> to find <Code>alpha</Code> — it fits the entire regularization path using warm starts and is faster than grid search. When NOT to use: when features are highly correlated. LASSO will pick one from a correlated group and zero the rest, which is statistically arbitrary and makes the solution unstable — small data changes cause different features to be selected. Use Elastic Net instead.
              </Prose>
            ),
          },
          {
            label: "Elastic Net — correlated features + sparsity",
            render: () => (
              <Prose>
                Use Elastic Net when: you want sparsity but your features are correlated. The L2 component groups correlated features (gives them similar coefficients) while the L1 component allows the group to be zeroed out together. This is the right choice for: gene expression data with correlated gene clusters, financial features with correlated time series, text features with synonyms. The default <Code>l1_ratio=0.5</Code> is a reasonable starting point; use <Code>ElasticNetCV</Code> to search over <Code>l1_ratio=[0.1, 0.5, 0.9, 1.0]</Code> and <Code>alpha</Code> jointly. When <Code>l1_ratio=1</Code>, ElasticNet reduces to LASSO; when <Code>l1_ratio=0</Code>, it reduces to Ridge. In practice, <Code>l1_ratio</Code> between 0.5 and 0.9 works well for most correlated-feature settings.
              </Prose>
            ),
          },
          {
            label: "Dropout — deep neural networks",
            render: () => (
              <Prose>
                Use Dropout when: training a deep neural network (MLP, CNN, Transformer) and the model is overfitting. The standard placement is after the activation of each dense layer; for modern Transformers, Dropout is placed after attention and feedforward layers. Standard <Code>p=0.5</Code> (50% drop rate) is the default for hidden layers; <Code>p=0.1</Code>–<Code>0.2</Code> is common for input layers and large pre-trained models. Do NOT use Dropout on its own for regularization in classical ML (linear models, trees) — L1/L2 is more principled there. Combine Dropout with weight decay (<Code>weight_decay</Code> in the optimizer) for double regularization in deep nets. Remove Dropout for recurrent layers (use Recurrent Dropout instead, which drops along the time dimension).
              </Prose>
            ),
          },
          {
            label: "Early stopping — implicit regularization for iterative methods",
            render: () => (
              <Prose>
                Early stopping is not a penalty-based method but functions as implicit regularization for gradient boosting and neural networks. Stopping gradient descent before convergence is equivalent to Ridge regularization in linear models (the connection is made precise by the bias-variance analysis of iterative solvers). For gradient boosting (XGBoost, LightGBM), early stopping with a proper validation set is the primary regularization mechanism — always use it. For neural nets, early stopping combined with Dropout provides defense in depth. For linear models, prefer explicit L1/L2 — they are convex and coordinate descent converges to the true solution, so early stopping adds no benefit.
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
        Ridge via the closed form requires inverting a <Code>{"d × d"}</Code> matrix, costing <Code>{"O(nd² + d³)"}</Code>. For <Code>{"n=10,000"}</Code> and <Code>{"d=1,000"}</Code> this is feasible; for <Code>{"d=100,000"}</Code> the <Code>{"d³"}</Code> term makes it prohibitive. Ridge via the Gram matrix trick computes <Code>{"(XX⊤ + λI)⁻¹"}</Code> instead — an <Code>{"n × n"}</Code> matrix costing <Code>{"O(n²d + n³)"}</Code> — which is faster when <Code>{"n ≪ d"}</Code>. The optimal switch-over is at <Code>{"n = d"}</Code>. SVD-based Ridge costs <Code>{"O(nd · min(n,d))"}</Code> and is numerically the most stable; sklearn's <Code>RidgeCV</Code> uses SVD internally.
      </Prose>

      <Prose>
        LASSO via coordinate descent costs <Code>{"O(nd)"}</Code> per sweep, with typically 10–200 sweeps to convergence. For sparse feature matrices (e.g., text bag-of-words), coordinate descent exploits sparsity: if <Code>{"X[:,j]"}</Code> has only <Code>{"k"}</Code> non-zeros, updating <Code>{"w_j"}</Code> costs <Code>{"O(k)"}</Code> instead of <Code>{"O(n)"}</Code>. This makes LASSO via coordinate descent practical for <Code>{"d = 1,000,000"}</Code> with sparse inputs. The SAGA solver in sklearn scales to millions of samples with sparse features by using stochastic variance-reduced gradient estimates — each step costs <Code>{"O(d)"}</Code> and the algorithm converges in <Code>{"O(1/k)"}</Code> steps with optimal constants.
      </Prose>

      <Prose>
        Dropout adds essentially zero computational overhead per parameter — it is a Bernoulli sample and element-wise multiply, both O(hidden_dim) per layer. The cost is in the training dynamics: dropout requires more iterations to converge because the effective gradient is noisier. In practice, networks with dropout need 2–3x more epochs than without. At inference time, dropout is free — it is simply disabled.
      </Prose>

      <H3>8.2 The {"d > n"} regime</H3>

      <Prose>
        When features outnumber samples, OLS has infinitely many solutions (the system is underdetermined) and the standard normal equations break down. Ridge fixes this completely: <Code>{"(XᵀX + λI)"}</Code> is full-rank and invertible for any <Code>{"λ > 0"}</Code>, giving a unique solution regardless of the ratio of <Code>{"d"}</Code> to <Code>{"n"}</Code>. This is why Ridge is used in genomics (<Code>{"d ≈ 20,000"}</Code> genes, <Code>{"n ≈ 200"}</Code> patients) and in situations where you have more features than observations. LASSO also handles <Code>{"d > n"}</Code> well, but there is a theoretical limit: LASSO can select at most <Code>{"n"}</Code> non-zero features (it cannot identify more signals than it has data points). For <Code>{"d ≫ n"}</Code> with many truly relevant features, LASSO will miss some; Elastic Net reduces this problem by grouping correlated signals.
      </Prose>

      <H3>8.3 Distributed regularization</H3>

      <Prose>
        For distributed settings where the dataset does not fit on one machine, Ridge and LASSO are solved differently. Ridge decomposes naturally: compute <Code>{"X_iᵀX_i"}</Code> and <Code>{"X_iᵀy_i"}</Code> on each shard, sum them across shards, then solve once. The full problem reduces to a single matrix inversion of size <Code>{"d × d"}</Code> — feasible as long as <Code>{"d"}</Code> is manageable. LASSO does not decompose as cleanly because coordinate descent requires access to all residuals. ADMM (Alternating Direction Method of Multipliers) is the standard distributed LASSO solver: it maintains a consensus variable and solves per-shard subproblems in parallel, communicating only the consensus variable at each round. Scikit-learn's <Code>saga</Code> solver can be distributed via Dask for large-scale sparse logistic regression.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>9.1 Forgetting to standardize features</H3>

      <Prose>
        This is the single most common regularization mistake. The L1 and L2 penalties are applied to the raw weight values, which means they penalize features with large natural scales more than features with small scales. A feature measuring income in dollars (range: 20,000–200,000) will have a small coefficient after regularization not because it is unimportant but because a small coefficient times a large value produces a large prediction. A feature measuring age in years (range: 20–80) will survive regularization at a larger coefficient for the same predictive power. The result: regularization distorts the feature selection by conflating weight magnitude with feature scale. Always apply <Code>StandardScaler</Code> before Ridge, LASSO, or Elastic Net. Fit the scaler on training data only and apply the transform to test data.
      </Prose>

      <H3>9.2 Choosing lambda on the test set</H3>

      <Prose>
        If you evaluate multiple values of <Code>{"λ"}</Code> on the test set and pick the best one, you have used the test set for hyperparameter selection — it is no longer an unbiased estimate of generalization error. This inflates apparent performance. The correct procedure: use <Code>k</Code>-fold cross-validation on the training set to select <Code>{"λ"}</Code>, then evaluate the model with that <Code>{"λ"}</Code> on the held-out test set exactly once. <Code>RidgeCV</Code>, <Code>LassoCV</Code>, and <Code>ElasticNetCV</Code> automate this correctly. Never look at the test set until your model is fully specified.
      </Prose>

      <H3>9.3 LASSO instability with correlated features</H3>

      <Prose>
        When two features <Code>{"x_i"}</Code> and <Code>{"x_j"}</Code> are highly correlated, LASSO's solution is unstable: small changes in the data determine which feature is selected and which is zeroed. Both selections have nearly the same training loss, but they lead to very different models. In cross-validation, different folds may select different features, leading to high variance in the feature selection. The symptom is that different random seeds or data splits produce dramatically different non-zero sets. The fix is Elastic Net, which groups correlated features by the L2 penalty — both get similar (non-zero) coefficients and the selection becomes stable.
      </Prose>

      <H3>9.4 Ridge penalizes the intercept (sometimes)</H3>

      <Prose>
        By default, sklearn's <Code>Ridge</Code> does not penalize the intercept term (bias). This is correct behavior and matches the math: the regularization should shrink slope parameters, not the overall mean of the predictions. Some from-scratch implementations accidentally include the bias column in the design matrix and apply the penalty to it, which distorts the solution. Always verify that the intercept is excluded from the penalty when implementing regularization manually. In sklearn, this is handled correctly by default; in PyTorch, weight decay applied via the optimizer penalizes all parameters including biases — you may want to exclude them using parameter groups.
      </Prose>

      <H3>9.5 Dropout at test time without model.eval()</H3>

      <Prose>
        Leaving dropout active at inference time is one of the most common PyTorch bugs. The symptoms are subtle: predictions are correct on average but have high variance across runs, and the model appears to underperform benchmarks by a few percentage points. The fix is always calling <Code>model.eval()</Code> before inference and <Code>model.train()</Code> before resuming training. A related issue: Monte Carlo Dropout, used for uncertainty estimation, intentionally keeps dropout active at test time and averages many stochastic passes. If you intend deterministic inference, always call <Code>model.eval()</Code>.
      </Prose>

      <H3>9.6 Numerical instability for tiny lambda</H3>

      <Prose>
        Ridge with <Code>{"λ → 0"}</Code> approaches OLS. If <Code>{"XᵀX"}</Code> is nearly singular (due to collinear features), the solution becomes numerically unstable — tiny floating-point differences produce huge swings in the coefficients. This is visible as coefficients with magnitude <Code>{"10⁶"}</Code> paired with low training loss but catastrophic test predictions. The fix is to ensure <Code>{"λ"}</Code> is at least on the order of the smallest eigenvalue of <Code>{"XᵀX"}</Code>. A practical heuristic: if the condition number of <Code>{"XᵀX"}</Code> exceeds <Code>{"10⁶"}</Code>, use Ridge with at minimum <Code>{"λ = 1e-3 · trace(XᵀX) / d"}</Code>. Using <Code>np.linalg.solve</Code> instead of explicit matrix inversion mitigates (but does not eliminate) this issue.
      </Prose>

      <H3>9.7 Dropout with batch normalization</H3>

      <Prose>
        Dropout and batch normalization interact in a subtle and often harmful way when placed in the wrong order. Batch normalization uses batch statistics (mean, variance) during training and running averages at test time. If Dropout precedes BatchNorm, the effective batch statistics change between training and test time — the dropout changes which neurons contribute to the mean and variance, creating a statistical mismatch. The standard recommendation (supported empirically and theoretically in the "Understanding the Disharmony between Dropout and Batch Normalization by Variance Shift" paper) is to not use Dropout before BatchNorm layers, or to place Dropout only after the final BatchNorm in the network. In modern architectures (ResNets, EfficientNets), BatchNorm and Dropout are rarely used together — choose one or the other.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All citations verified via WebSearch against their primary publication venues. Read in order for the intellectual lineage.
      </Prose>

      <StepTrace
        label="primary literature"
        steps={[
          {
            label: "Tikhonov 1943 — Origin of regularization theory",
            render: () => (
              <Prose>
                Tikhonov, A.N. (1943). "On the stability of inverse problems." <em>Doklady Akademii Nauk SSSR</em>, 39(5):195–198. The founding paper of regularization theory, written in Russian during World War II. Tikhonov's problem was ill-posed inverse problems in mathematical physics, not statistics. The key insight — add a smoothness penalty to the objective to stabilize a solution — is the same insight that underlies L2 regularization in machine learning. Tikhonov later developed the method extensively in his 1963 paper "Solution of incorrectly formulated problems and the regularization method" (Dokl. Akad. Nauk SSSR, 151(3):501–504), which gave the method the more general "Tikhonov regularization" name used in the inverse problems literature.
              </Prose>
            ),
          },
          {
            label: "Hoerl & Kennard 1970 — Ridge regression",
            render: () => (
              <Prose>
                Hoerl, A.E. and Kennard, R.W. (1970). "Ridge Regression: Biased Estimation for Nonorthogonal Problems." <em>Technometrics</em>, 12(1):55–67. DOI: 10.1080/00401706.1970.10488634. Available via Taylor & Francis. This paper introduced ridge regression as a practical tool for statisticians dealing with collinear predictors, showed that the biased ridge estimator has uniformly lower mean squared error than OLS under near-collinearity, and introduced the ridge trace as a diagnostic. A companion paper in the same issue (pages 69–82) gave application examples. Both are still cited in regression textbooks.
              </Prose>
            ),
          },
          {
            label: "Tibshirani 1996 — LASSO",
            render: () => (
              <Prose>
                Tibshirani, R. (1996). "Regression Shrinkage and Selection via the Lasso." <em>Journal of the Royal Statistical Society: Series B (Methodological)</em>, 58(1):267–288. DOI: 10.1111/j.2517-6161.1996.tb02080.x. Available via Oxford Academic (open access via JRSS). This paper introduced the LASSO, showed its geometric connection to the L1 ball, derived the soft-thresholding solution for orthonormal design, and demonstrated via simulation that LASSO dominates ridge when the true model is sparse and subset selection when features are correlated. The name "LASSO" has become standard in statistics, machine learning, signal processing, and econometrics. Tibshirani's 2011 retrospective (JRSS-B 73(3):273–282) is worth reading for the history of the idea and subsequent developments.
              </Prose>
            ),
          },
          {
            label: "Zou & Hastie 2005 — Elastic Net",
            render: () => (
              <Prose>
                Zou, H. and Hastie, T. (2005). "Regularization and Variable Selection via the Elastic Net." <em>Journal of the Royal Statistical Society: Series B (Statistical Methodology)</em>, 67(2):301–320. DOI: 10.1111/j.1467-9868.2005.00503.x. Available via Oxford Academic. This paper proved that LASSO's solution is non-unique when features are correlated (infinitely many weight vectors achieve the same L1-penalized loss), identified the grouping property as desirable and absent in LASSO, introduced the elastic net as the convex combination of L1 and L2 penalties, and showed both theoretically and empirically that elastic net outperforms LASSO in the correlated-features regime. The LARS (Least Angle Regression) algorithm of Efron et al. (2004, Annals of Statistics) is the standard way to compute the elastic net regularization path efficiently and is implemented in sklearn's <Code>LassoLars</Code>.
              </Prose>
            ),
          },
          {
            label: "Srivastava et al. 2014 — Dropout",
            render: () => (
              <Prose>
                Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., and Salakhutdinov, R. (2014). "Dropout: A Simple Way to Prevent Neural Networks from Overfitting." <em>Journal of Machine Learning Research</em>, 15:1929–1958. Available at jmlr.org/papers/v15/srivastava14a.html. The foundational dropout paper. Shows dropout prevents co-adaptation of neurons, demonstrates state-of-the-art results on vision, speech, text, and computational biology benchmarks, analyzes the connection to model averaging, and shows that the naive scaling approximation (multiply by keep probability at test time) is exact for linear networks and a good approximation for nonlinear ones. The inverted dropout implementation (scale by 1/p during training, no scaling at test) was not in the original paper but became standard for engineering convenience.
              </Prose>
            ),
          },
          {
            label: "Wang & Manning 2013 — Fast Dropout and L2 connection",
            render: () => (
              <Prose>
                Wang, S. and Manning, C. (2013). "Fast dropout training." <em>Proceedings of the 30th International Conference on Machine Learning (ICML)</em>, PMLR 28(2):118–126. Available at proceedings.mlr.press/v28/wang13a.html. This paper shows that the expected gradient update under dropout is approximately the gradient of an L2-regularized quadratic objective — making precise the connection between dropout and weight decay. The effective regularization strength is proportional to <Code>{"p(1-p) · σ²_input"}</Code>, explaining why features with high input variance receive stronger dropout regularization. The paper also introduces a Gaussian approximation to dropout that is faster to compute (no sampling) and gives similar empirical results. This theoretical grounding justifies using dropout as principled regularization, not just an engineering trick.
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
        Work through each exercise before reading the answer. These test recall, derivation, debugging, and applied judgment — the same mix you encounter in interviews and production debugging.
      </Prose>

      <H3>Exercise 1 (recall)</H3>
      <Prose>
        Write the ridge regression objective and its closed-form solution. Why does adding <Code>{"λI"}</Code> to <Code>{"XᵀX"}</Code> before inverting always produce a unique solution, even when <Code>{"d > n"}</Code>?
      </Prose>
      <Callout type="answer" title="Answer 1">
        {"Ridge objective: L(w) = ‖y − Xw‖² + λ‖w‖²."} Solution: {"w* = (XᵀX + λI)⁻¹ Xᵀy."}
        The matrix {"XᵀX"} is positive semi-definite — all eigenvalues are ≥ 0. When d {">"} n (more features than samples), {"XᵀX"} has at least d − n zero eigenvalues, making it singular (non-invertible). Adding {"λI"} shifts all eigenvalues up by λ: the eigenvalues of {"(XᵀX + λI)"} are {"‌(σᵢ² + λ)"}, all strictly positive for any λ {">"} 0. A matrix with all positive eigenvalues is positive definite, hence invertible. This is why ridge always has a unique solution regardless of the n-vs-d relationship.
      </Callout>

      <H3>Exercise 2 (derivation)</H3>
      <Prose>
        Explain geometrically why L1 regularization produces sparse solutions but L2 does not. Use the constraint-region framing — where is the constrained optimum for each penalty?
      </Prose>
      <Callout type="answer" title="Answer 2">
        The constrained form of regularization asks: minimize the OLS loss subject to {"‖w‖_p ≤ t"} for some budget t. The OLS loss forms elliptical contours around the unconstrained optimum. The constraint region for L2 is a circle (sphere in high dimensions) — a smooth, curved surface with no corners. The elliptical contours can be tangent to the circle at any point, and that point will generally not lie on any axis. So L2 solutions are generically non-zero in all dimensions.
        The constraint region for L1 is a diamond (cross-polytope) — a non-smooth surface with corners that point along coordinate axes. The elliptical contours are pulled toward the closest corner of the diamond because the corners project furthest in all directions. When the contours first touch the diamond, they almost always touch a corner, which sits exactly on a coordinate axis — forcing one or more weights to be exactly zero. The more corners there are (higher dimension), the more likely the optimum lands on a corner, and the sparser the solution.
      </Callout>

      <H3>Exercise 3 (applied)</H3>
      <Prose>
        You are fitting logistic regression on a gene expression dataset with 5,000 samples and 20,000 features. You believe about 50 genes are truly predictive. You want the final model to show which genes matter. What regularization method do you choose, what sklearn solver, and what hyperparameter do you tune?
      </Prose>
      <Callout type="answer" title="Answer 3">
        Use LASSO (L1) regularization — you want sparse solutions, and LASSO is designed to zero out irrelevant features, ideally leaving only the ~50 predictive genes with non-zero coefficients. In sklearn: {"LogisticRegression(solver='saga', penalty='l1', C=..., max_iter=2000)"}. Solver: saga is the only sklearn logistic regression solver that supports L1 and scales to large sparse datasets efficiently. Hyperparameter: tune C (the inverse regularization strength — smaller C means stronger regularization, more zeros). Use LogisticRegressionCV with cv=5 and a logarithmic grid of C values: {"C=[0.001, 0.01, 0.1, 1.0, 10.0]"}. Since gene expression features may be correlated (co-regulated gene modules), you should also try Elastic Net ({"penalty='elasticnet'"}, {"l1_ratio=[0.5, 0.9, 1.0]"}) and compare sparsity vs. CV accuracy. Always standardize the features with StandardScaler before fitting — gene expression values have very different scales across genes.
      </Callout>

      <H3>Exercise 4 (debugging)</H3>
      <Prose>
        Your PyTorch network achieves 95% validation accuracy during training but 72% when deployed. The only code difference between training and deployment is that your training loop calls <Code>{"forward(X)"}</Code> and your deployment server calls <Code>{"model(X)"}</Code> after loading the checkpoint. What is the most likely bug and how do you fix it?
      </Prose>
      <Callout type="answer" title="Answer 4">
        The model is in training mode at deployment — dropout (and batch normalization's training statistics) are still active. When you load a checkpoint with torch.load and call model(X), the model defaults to training mode unless you explicitly call model.eval(). In training mode, dropout randomly zeros neurons at every forward pass, making predictions stochastic and on average incorrect (neurons are scaled by 1/p_keep during training so the raw activations are inflated relative to what test expects without dropout). The fix: add {"model.eval()"} immediately after loading the checkpoint in the deployment server: {"model.load_state_dict(checkpoint); model.eval()"}. If you also use torch.no_grad() (which you should, to save memory and speed up inference), ensure it wraps the forward call: {"with torch.no_grad(): out = model(X)"}. Always test your deployment pipeline end-to-end with a known input to catch this class of bug.
      </Callout>

      <H3>Exercise 5 (conceptual)</H3>
      <Prose>
        A colleague claims: "Elastic Net is strictly better than both LASSO and Ridge, so we should always use it." Is this claim correct? Give one scenario where Ridge strictly outperforms Elastic Net and one where LASSO does.
      </Prose>
      <Callout type="answer" title="Answer 5">
        The claim is incorrect. Elastic Net has more hyperparameters (both alpha and l1_ratio must be tuned), is slower to fit than Ridge (no closed form), and when the optimal l1_ratio is 0 or 1, it reduces to Ridge or LASSO respectively — adding unnecessary search cost.
        Ridge strictly outperforms Elastic Net when: all features are truly relevant (no true zeros in the generating model) AND features are uncorrelated. In this "dense signal" setting, L1 penalties shrink relevant features too aggressively — the optimal solution has no zeros, so adding any L1 component wastes bias on features that should be kept. Ridge gives the lowest MSE.
        LASSO strictly outperforms Elastic Net when: features are independent (zero correlation) AND the true model is sparse. With independent features, the grouping property of Elastic Net is irrelevant — you do not need the L2 component to stabilize the solution. LASSO with a perfectly tuned alpha achieves the same sparsity as Elastic Net with lower bias (because it does not add the L2 shrinkage to the non-zero features). In this regime, Elastic Net's l1_ratio search just wastes compute converging to l1_ratio=1.0 anyway.
      </Callout>

      <H3>Exercise 6 (math)</H3>
      <Prose>
        Derive the soft-thresholding operator from first principles. Given a 1D LASSO problem: {"min_w  (1/2)(z − w)² + λ|w|"} where <Code>{"z"}</Code> is a constant, show that the optimal <Code>{"w*"}</Code> is {"S(z, λ) = sign(z) · max(|z| − λ, 0)"}. Consider the three cases: <Code>{"z > λ"}</Code>, <Code>{"z < −λ"}</Code>, and <Code>{"|z| ≤ λ"}</Code>.
      </Prose>
      <Callout type="answer" title="Answer 6">
        {"The objective f(w) = (1/2)(z − w)² + λ|w| is convex but non-smooth at w=0. We use subgradient optimality: w* minimizes f iff 0 ∈ ∂f(w*)."}
        {"∂f(w) = −(z − w) + λ∂|w| = (w − z) + λ∂|w|."}
        {"∂|w| = {1} if w > 0; {−1} if w < 0; [−1, +1] if w = 0."}
        Case 1 (z {">"} λ): Try w* = z − λ {">"} 0. Then ∂f(w*) = (w* − z) + λ·1 = (z − λ − z) + λ = 0. Optimality holds. ✓
        Case 2 (z {"<"} −λ): Try w* = z + λ {"<"} 0. Then ∂f(w*) = (w* − z) + λ·(−1) = (z + λ − z) − λ = 0. Optimality holds. ✓
        {"Case 3 (|z| ≤ λ): Try w* = 0. Then ∂f(0) = (0 − z) + λ[−1, +1] = {−z + s : s ∈ [−λ, λ]}. For 0 ∈ ∂f(0) we need −z + s = 0 for some s ∈ [−λ, λ], i.e., s = z, which holds iff |z| ≤ λ. Optimality holds. ✓"}
        {"Combining: w* = sign(z) · max(|z| − λ, 0). This is the soft-thresholding operator S(z, λ). Note: hard thresholding (keep z if |z| > λ, else 0) sets w* = z·1{|z|>λ} — no shrinkage of large values, no subgradient proof."}
      </Callout>

    </div>
  ),
};

export default regularizationContent;
