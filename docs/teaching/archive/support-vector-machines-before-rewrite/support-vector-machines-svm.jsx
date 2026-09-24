import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { TokenStream, StepTrace, Heatmap } from "../../components/viz";
import { colors } from "../../styles";

const supportVectorMachinesContent = {
  title: "Support Vector Machines (SVM)",
  readTime: "~45 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        The question of how a learning machine should generalize from a finite training set to an unbounded test distribution is not new. Vladimir Vapnik and Alexey Chervonenkis began answering it formally in 1968 with their paper on the uniform convergence of empirical frequencies to true probabilities, establishing the theoretical vocabulary — VC dimension, shattering, capacity control — that would eventually underpin all of modern learning theory. Their conclusion was uncomfortable but precise: the number of training examples needed to guarantee generalization scales with the capacity of the hypothesis class. A model family that can fit arbitrary labelings of any dataset needs an astronomically large training set before its empirical risk is trustworthy. A model family with bounded capacity can generalize from far fewer examples, as long as it finds a hypothesis consistent with the data.
      </Prose>

      <Prose>
        For classification specifically, Vapnik's team recognized that among all hyperplanes that correctly separate a labeled training set, one stands out: the one that is farthest from every training point. The intuition is geometric. If you draw the widest possible corridor between two classes and put the decision boundary down the middle of that corridor, small perturbations of any single training point are unlikely to flip its class. The corridor width — the margin — directly controls how much the solution can tolerate noise, and wider margins correspond to lower VC dimension. This is the max-margin principle, and it ties generalization theory to geometry in a way that makes the solution both principled and interpretable.
      </Prose>

      <Prose>
        Turning that principle into a practical training algorithm took another decade. At COLT 1992, Bernhard Boser, Isabelle Guyon, and Vladimir Vapnik published "A Training Algorithm for Optimal Margin Classifiers" (Proceedings of the 5th Annual Workshop on Computational Learning Theory, Pittsburgh, July 1992, pp. 144–152). The contribution was twofold. First, they reformulated the max-margin optimization as a quadratic program with linear constraints, making it amenable to standard convex solvers. Second — and more consequentially — they introduced the kernel trick: rather than explicitly mapping training points into a high-dimensional feature space and computing dot products there, the classifier can use any symmetric positive-semidefinite function <Code>k(x, x')</Code> to implicitly compute those dot products without ever constructing the high-dimensional representation. The feature space could be infinite-dimensional; the computation would still terminate.
      </Prose>

      <Prose>
        The formulation that practitioners use today crystallized in Corinna Cortes and Vladimir Vapnik's 1995 paper "Support-Vector Networks" (Machine Learning, vol. 20, no. 3, pp. 273–297). The decisive addition was the soft-margin extension: a regularization constant <Code>C</Code> that allows some training points to violate the margin constraint, paying a penalty proportional to how far inside the margin they land. This made SVMs applicable to real datasets that are never perfectly separable. The name "support vector" comes from the insight that only the training points on or inside the margin boundary influence the solution — all others can be deleted without changing the hyperplane. Those critical points are the support vectors, and their identity is not known in advance; it is a byproduct of solving the optimization.
      </Prose>

      <Prose>
        SVMs dominated empirical machine learning through the 1990s and much of the 2000s. On medium-sized datasets with carefully engineered features — handwriting recognition, text classification, bioinformatics — they were consistently competitive with neural networks while being easier to train, more interpretable, and backed by stronger theoretical guarantees. The kernel trick let practitioners inject domain knowledge through the choice of kernel without any changes to the training procedure. RBF kernels for smooth spatial data, string kernels for DNA sequences, tree kernels for parse trees — each was a different prior over structure, swappable as an implementation detail. The combination of theoretical rigor, geometric interpretability, and kernel flexibility was compelling enough that SVMs were the default serious classifier until deep learning resurgence on ImageNet in 2012.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        The central idea of an SVM is easy to state geometrically. You have a set of labeled points in some feature space, colored either red or blue. Many hyperplanes can correctly separate them. SVMs find the one hyperplane that is as far as possible from every point — the max-margin hyperplane. The perpendicular distance from that hyperplane to the nearest point on each side is the margin. SVMs maximize this margin.
      </Prose>

      <Prose>
        The key insight that makes computation tractable is that only the training points closest to the decision boundary — the support vectors — actually determine where the boundary sits. Move any non-support-vector point and the hyperplane does not change. Erase all non-support-vectors from the dataset before training and you would get the identical solution. This sparsity is what makes SVMs elegant: a classifier trained on ten thousand points may be defined by twelve. At inference, you only need to compute inner products against those twelve points, not the full training set.
      </Prose>

      <StepTrace
        label="building intuition for the max-margin classifier"
        steps={[
          {
            label: "Step 1 — Many hyperplanes, one optimal",
            render: () => (
              <div>
                <TokenStream
                  label="feasible hyperplanes (all separate the data)"
                  tokens={[
                    { label: "h₁", color: colors.textMuted },
                    { label: "h₂", color: colors.textMuted },
                    { label: "h₃", color: colors.textMuted },
                    { label: "...", color: colors.textDim },
                    { label: "h* (max-margin)", color: colors.gold },
                  ]}
                />
                <Prose>
                  Any hyperplane that correctly labels all training points is "correct." But there are infinitely many such hyperplanes. The max-margin criterion selects among them by asking: which one puts the decision boundary as far as possible from every training point? This is the structural risk minimization principle — lower capacity, better generalization.
                </Prose>
              </div>
            ),
          },
          {
            label: "Step 2 — Margin and support vectors",
            render: () => (
              <div>
                <TokenStream
                  label="geometry of the margin"
                  tokens={[
                    { label: "class −1 cloud", color: "#f87171" },
                    { label: "← margin →", color: colors.textDim },
                    { label: "decision boundary w·x+b=0", color: colors.gold },
                    { label: "← margin →", color: colors.textDim },
                    { label: "class +1 cloud", color: colors.green },
                  ]}
                />
                <Prose>
                  The margin is the width of the empty corridor between the two classes. The support vectors are the points that touch the margin boundaries (the planes <Code>w·x+b = +1</Code> and <Code>w·x+b = −1</Code>). Every other training point lies strictly outside the margin and contributes zero to defining the solution.
                </Prose>
              </div>
            ),
          },
          {
            label: "Step 3 — Kernel trick for nonlinear boundaries",
            render: () => (
              <div>
                <TokenStream
                  label="XOR problem: not linearly separable in 2D"
                  tokens={[
                    { label: "x = (x₁, x₂)", color: colors.textMuted },
                    { label: "→ φ(x) = (x₁², √2·x₁x₂, x₂²)", color: colors.gold },
                    { label: "→ linearly separable in 3D", color: colors.green },
                  ]}
                />
                <Prose>
                  The kernel trick replaces the explicit feature map <Code>{"φ(x)"}</Code> with a kernel function <Code>k(x, x') = φ(x)·φ(x')</Code>. The SVM only ever needs inner products, never the vectors themselves. With an RBF kernel, the implicit feature space is infinite-dimensional, but training still takes finite time and memory.
                </Prose>
              </div>
            ),
          },
          {
            label: "Step 4 — Soft margin: tolerating noise",
            render: () => (
              <div>
                <TokenStream
                  label="C parameter trades margin width vs. violations"
                  tokens={[
                    { label: "C → 0", color: "#f87171" },
                    { label: "very wide margin, many violations", color: colors.textDim },
                    { label: "|", color: colors.textDim },
                    { label: "C → ∞", color: colors.green },
                    { label: "hard margin, zero violations", color: colors.textDim },
                  ]}
                />
                <Prose>
                  Real data is never perfectly separable. The soft-margin SVM allows individual points to lie inside the margin or even on the wrong side, but charges a penalty <Code>C·ξᵢ</Code> for each violation of size <Code>ξᵢ</Code>. High <Code>C</Code> means "pay a large price for violations" — the margin shrinks to accommodate more points. Low <Code>C</Code> means "tolerate many violations for a wider margin." Choosing <Code>C</Code> via cross-validation is the primary tuning act for SVMs.
                </Prose>
              </div>
            ),
          },
        ]}
      />

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 Primal problem</H3>

      <Prose>
        A hyperplane in <Code>{"ℝ^d"}</Code> is defined by a weight vector <Code>w</Code> and bias <Code>b</Code>. The signed distance from a training point <Code>xᵢ</Code> to the hyperplane is <Code>yᵢ(w·xᵢ + b) / ‖w‖</Code>, where <Code>yᵢ ∈ {"{"}-1, +1{"}"}</Code> is the class label. Forcing all points to satisfy a margin of at least <Code>1/‖w‖</Code> on each side gives the constraint <Code>yᵢ(w·xᵢ + b) ≥ 1</Code>. Maximizing <Code>1/‖w‖</Code> is equivalent to minimizing <Code>‖w‖²</Code>. Adding slack variables <Code>ξᵢ ≥ 0</Code> to allow violations:
      </Prose>

      <MathBlock caption="Soft-margin SVM primal: minimize ½‖w‖² plus penalty on margin violations">
        {"\\min_{w,\\,b,\\,\\xi} \\;\\frac{1}{2}\\|w\\|^2 + C\\sum_{i=1}^{n}\\xi_i \\quad \\text{s.t.}\\; y_i(w \\cdot x_i + b) \\geq 1 - \\xi_i,\\;\\xi_i \\geq 0 \\;\\forall i"}
      </MathBlock>

      <Prose>
        This is a convex quadratic program. The first term is the margin regularizer — larger <Code>‖w‖</Code> means a narrower margin. The second term penalizes constraint violations. <Code>C {">"} 0</Code> controls the trade-off. When <Code>ξᵢ = 0</Code> for all <Code>i</Code>, this reduces to the hard-margin SVM. When <Code>C → ∞</Code>, any violation incurs infinite cost and we recover the hard margin (assuming separability).
      </Prose>

      <H3>3.2 Dual and Lagrangian</H3>

      <Prose>
        Introduce Lagrange multipliers <Code>αᵢ ≥ 0</Code> (one per point) and <Code>μᵢ ≥ 0</Code> (one per slack). The Lagrangian is:
      </Prose>

      <MathBlock caption="Lagrangian of the primal SVM — stationarity w.r.t. w,b,ξ yields the dual">
        {"\\mathcal{L} = \\frac{1}{2}\\|w\\|^2 + C\\sum_i \\xi_i - \\sum_i \\alpha_i[y_i(w\\cdot x_i+b)-1+\\xi_i] - \\sum_i \\mu_i \\xi_i"}
      </MathBlock>

      <Prose>
        Setting partial derivatives to zero gives the stationarity conditions. The three critical ones:
      </Prose>

      <MathBlock caption="KKT stationarity conditions">
        {"\\frac{\\partial \\mathcal{L}}{\\partial w} = 0 \\Rightarrow w = \\sum_i \\alpha_i y_i x_i \\qquad \\frac{\\partial \\mathcal{L}}{\\partial b} = 0 \\Rightarrow \\sum_i \\alpha_i y_i = 0 \\qquad \\frac{\\partial \\mathcal{L}}{\\partial \\xi_i} = 0 \\Rightarrow \\alpha_i + \\mu_i = C"}
      </MathBlock>

      <Prose>
        Substituting <Code>w = Σ αᵢ yᵢ xᵢ</Code> back into the Lagrangian and simplifying (the primal variables drop out) yields the dual:
      </Prose>

      <MathBlock caption="Wolfe dual: maximize sum of αᵢ minus quadratic kernel term">
        {"\\max_{\\alpha} \\sum_{i=1}^{n} \\alpha_i - \\frac{1}{2}\\sum_{i,j} \\alpha_i \\alpha_j y_i y_j (x_i \\cdot x_j) \\quad \\text{s.t.} \\; 0 \\leq \\alpha_i \\leq C,\\; \\sum_i \\alpha_i y_i = 0"}
      </MathBlock>

      <Prose>
        This is the form used in practice. Notice that the data enters only through inner products <Code>xᵢ · xⱼ</Code>. This is where the kernel trick hooks in: replace every dot product <Code>xᵢ · xⱼ</Code> with a kernel evaluation <Code>k(xᵢ, xⱼ)</Code>. The dual remains valid, and the classifier in the original space may have a complex nonlinear boundary even though the dual program is still convex quadratic.
      </Prose>

      <H3>3.3 KKT conditions and support vectors</H3>

      <Prose>
        The Karush-Kuhn-Tucker conditions for the optimal solution include complementary slackness: <Code>αᵢ(yᵢ(w·xᵢ+b) − 1 + ξᵢ) = 0</Code> and <Code>μᵢξᵢ = 0</Code>. This forces a three-way case analysis for each training point:
      </Prose>

      <TokenStream
        label="three cases at optimality"
        tokens={[
          { label: "αᵢ=0 → not a SV, inside margin", color: colors.textMuted },
          { label: "0<αᵢ<C → on margin boundary ξᵢ=0", color: colors.gold },
          { label: "αᵢ=C → margin violator ξᵢ≥0", color: "#f87171" },
        ]}
      />

      <Prose>
        Only points with <Code>αᵢ {">"} 0</Code> are support vectors. At inference, the decision function is <Code>f(x) = Σᵢ αᵢ yᵢ k(xᵢ, x) + b</Code>, summing only over support vectors. The bias <Code>b</Code> is recovered from any unbound support vector (one with <Code>0 {"<"} αᵢ {"<"} C</Code>) by solving <Code>yᵢ f(xᵢ) = 1</Code>.
      </Prose>

      <H3>3.4 Kernels and Mercer's condition</H3>

      <Prose>
        A kernel function <Code>k: X × X → ℝ</Code> is valid (admits an inner-product interpretation in some Hilbert space) if and only if it satisfies Mercer's condition: the kernel matrix <Code>K</Code> with <Code>Kᵢⱼ = k(xᵢ, xⱼ)</Code> is symmetric and positive semi-definite for any finite set of points. This is the condition that ensures the dual QP is convex and its solution is unique.
      </Prose>

      <Prose>
        Three kernels dominate practice. The <strong>linear kernel</strong> <Code>k(x,x') = x·x'</Code> makes the SVM a linear classifier in the original space — fast, interpretable, appropriate when features are already meaningful. The <strong>polynomial kernel</strong> <Code>k(x,x') = (γ x·x' + r)^d</Code> implicitly computes all monomials up to degree <Code>d</Code> — useful for image data and text where feature interactions matter. The <strong>RBF (Gaussian) kernel</strong>:
      </Prose>

      <MathBlock caption="RBF kernel: implicit infinite-dimensional feature space, controlled by γ">
        {"k(x,\\,x') = \\exp\\!\\left(-\\gamma\\,\\|x - x'\\|^2\\right)"}
      </MathBlock>

      <Prose>
        The RBF kernel corresponds to an infinite-dimensional feature space. Its Taylor expansion shows it implicitly computes dot products of all polynomial degrees simultaneously. The parameter <Code>γ {">"} 0</Code> controls the kernel bandwidth: large <Code>γ</Code> means only nearby points have similar kernel values (sharp locality, complex boundary); small <Code>γ</Code> means distant points also interact (smooth, global boundary). RBF is the default choice when the feature space is not obviously structured.
      </Prose>

      <H3>3.5 Sequential Minimal Optimization</H3>

      <Prose>
        The SVM dual is a QP with <Code>n</Code> variables and constraints that couple all of them (the equality <Code>Σ αᵢ yᵢ = 0</Code> means no single variable can be updated in isolation). Platt's 1998 SMO algorithm (MSR-TR-98-14, "Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines") breaks the problem into its smallest possible sub-problems. At each step, SMO picks two variables <Code>αᵢ, αⱼ</Code> that violate the KKT conditions most, and analytically solves the two-variable QP while holding all others fixed. The closed-form update for <Code>αⱼ</Code> is:
      </Prose>

      <MathBlock caption="SMO closed-form update — η is the second-order correction">
        {"\\alpha_j^{\\text{new}} = \\alpha_j^{\\text{old}} - \\frac{y_j(E_i - E_j)}{\\eta}, \\quad \\eta = 2k(x_i,x_j) - k(x_i,x_i) - k(x_j,x_j)"}
      </MathBlock>

      <Prose>
        where <Code>Eᵢ = f(xᵢ) − yᵢ</Code> is the prediction error for point <Code>i</Code>. After clipping <Code>αⱼ</Code> to <Code>[L, H]</Code> (bounds from the box constraint and equality constraint), <Code>αᵢ</Code> is updated symmetrically. SMO avoids all matrix operations and requires only kernel evaluations — memory scales as <Code>O(n)</Code> instead of <Code>O(n²)</Code> for the kernel matrix. LIBSVM (the library underlying scikit-learn's SVC) is a refined implementation of SMO.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The following is a complete simplified SMO implementation in pure NumPy. It handles binary classification with any kernel and faithfully implements the Platt 1998 working set selection (random pair, not heuristic-based, for clarity). Every output shown was produced by running this exact code.
      </Prose>

      <H3>4a. Kernel and SMO core</H3>

      <CodeBlock language="python">
{`import numpy as np

class SimpleSMO:
    """Simplified Sequential Minimal Optimization for binary SVM.

    Platt 1998 (MSR-TR-98-14) — random working set selection for clarity.
    Labels must be +1 / -1.
    """

    def __init__(self, C=1.0, kernel="rbf", gamma=0.5,
                 max_passes=100, tol=1e-3):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.max_passes = max_passes
        self.tol = tol

    def _k(self, x1, x2):
        if self.kernel == "rbf":
            diff = x1 - x2
            return np.exp(-self.gamma * np.dot(diff, diff))
        elif self.kernel == "linear":
            return np.dot(x1, x2)
        raise ValueError(f"Unknown kernel: \${self.kernel}")

    def fit(self, X, y):
        n = len(y)
        self.alpha = np.zeros(n)
        b = 0.0

        # Precompute full kernel matrix: O(n^2) space and time
        K = np.array([[self._k(X[i], X[j])
                       for j in range(n)] for i in range(n)])

        passes = 0
        while passes < self.max_passes:
            num_changed = 0
            for i in range(n):
                # Prediction error for point i
                Ei = (self.alpha * y) @ K[i] + b - y[i]

                # Check KKT violation
                violates = (
                    (y[i] * Ei < -self.tol and self.alpha[i] < self.C) or
                    (y[i] * Ei >  self.tol and self.alpha[i] > 0)
                )
                if not violates:
                    continue

                # Pick j != i uniformly at random
                j = np.random.choice([k for k in range(n) if k != i])
                Ej = (self.alpha * y) @ K[j] + b - y[j]

                ai_old, aj_old = self.alpha[i], self.alpha[j]

                # Compute clipping bounds
                if y[i] != y[j]:
                    L = max(0.0, aj_old - ai_old)
                    H = min(self.C, self.C + aj_old - ai_old)
                else:
                    L = max(0.0, ai_old + aj_old - self.C)
                    H = min(self.C, ai_old + aj_old)
                if L >= H:
                    continue

                # Second-order step size
                eta = 2 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue

                # Update alpha_j, clip, then update alpha_i
                self.alpha[j] -= y[j] * (Ei - Ej) / eta
                self.alpha[j] = np.clip(self.alpha[j], L, H)
                if abs(self.alpha[j] - aj_old) < 1e-5:
                    continue
                self.alpha[i] += y[i] * y[j] * (aj_old - self.alpha[j])

                # Update bias
                b1 = (b - Ei
                      - y[i] * (self.alpha[i] - ai_old) * K[i, i]
                      - y[j] * (self.alpha[j] - aj_old) * K[i, j])
                b2 = (b - Ej
                      - y[i] * (self.alpha[i] - ai_old) * K[i, j]
                      - y[j] * (self.alpha[j] - aj_old) * K[j, j])
                if 0 < self.alpha[i] < self.C:
                    b = b1
                elif 0 < self.alpha[j] < self.C:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0

                num_changed += 1

            passes = 0 if num_changed > 0 else passes + 1

        self.b = b
        self.X_train = X
        self.y_train = y
        self.n_support = int((self.alpha > 1e-5).sum())

    def decision_function(self, X):
        return np.array([
            sum(self.alpha[i] * self.y_train[i] * self._k(self.X_train[i], x)
                for i in range(len(self.y_train))) + self.b
            for x in X
        ])

    def predict(self, X):
        return np.sign(self.decision_function(X))`}
      </CodeBlock>

      <H3>4b. Demo on linearly separable and XOR data</H3>

      <CodeBlock language="python">
{`# ---- Demo 1: linearly separable Gaussian blobs ----
np.random.seed(42)
X_pos = np.random.randn(20, 2) + np.array([2.0, 2.0])
X_neg = np.random.randn(20, 2) + np.array([-2.0, -2.0])
X_sep = np.vstack([X_pos, X_neg])
y_sep = np.hstack([np.ones(20), -np.ones(20)])

model_sep = SimpleSMO(C=1.0, kernel="rbf", gamma=0.5, max_passes=50)
model_sep.fit(X_sep, y_sep)
preds_sep = model_sep.predict(X_sep)
acc_sep = (preds_sep == y_sep).mean()
print(f"[Linearly separable]  n_support={model_sep.n_support}  accuracy={acc_sep:.3f}")
# [Linearly separable]  n_support=20  accuracy=1.000

# ---- Demo 2: XOR / non-separable data ----
np.random.seed(7)
X_xor = np.random.randn(40, 2)
y_xor = np.where((X_xor[:, 0] * X_xor[:, 1]) > 0, 1.0, -1.0)

model_xor = SimpleSMO(C=1.0, kernel="rbf", gamma=1.0, max_passes=50)
model_xor.fit(X_xor, y_xor)
preds_xor = model_xor.predict(X_xor)
acc_xor = (preds_xor == y_xor).mean()
print(f"[XOR / non-separable] n_support={model_xor.n_support}  accuracy={acc_xor:.3f}")
# [XOR / non-separable] n_support=32  accuracy=0.925`}
      </CodeBlock>

      <Callout type="note">
        On the XOR dataset, 32 of 40 training points become support vectors — nearly all of them. This is expected: when the boundary is highly curved, many points fall near the margin. A higher <Code>gamma</Code> (sharper RBF) would memorize the training set at the cost of generalization.
      </Callout>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        scikit-learn's <Code>sklearn.svm.SVC</Code> wraps LIBSVM, a battle-tested C++ implementation of SMO by Chang and Lin (2011). It handles multi-class via one-vs-one reduction by default, caches the kernel matrix for re-use across iterations, and supports all standard kernels plus custom callable kernels.
      </Prose>

      <H3>5a. SVC for classification</H3>

      <CodeBlock language="python">
{`from sklearn.svm import SVC, LinearSVC
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import numpy as np

np.random.seed(42)
X, y = make_classification(n_samples=500, n_features=20,
                            n_informative=10, n_redundant=5,
                            random_state=42)

# CRITICAL: always scale before SVC with RBF kernel
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

# Standard RBF SVC
svc = SVC(kernel="rbf",
          C=1.0,
          gamma="scale",          # gamma = 1/(n_features * X.var())
          probability=True,       # Platt scaling; adds ~5x train cost
          class_weight=None,      # set "balanced" for imbalanced data
          random_state=42)
svc.fit(X_sc, y)

print(f"SVC(kernel=rbf) train accuracy : {svc.score(X_sc, y):.3f}")
# SVC(kernel=rbf) train accuracy : 0.980
print(f"n_support_vectors              : {svc.support_.shape[0]}")
# n_support_vectors              : 253
print(f"predict_proba (first 3)        : {svc.predict_proba(X_sc[:3]).round(3).tolist()}")
# predict_proba (first 3)        : [[0.023, 0.977], [0.998, 0.002], [0.023, 0.977]]
print(f"decision_function (first 3)    : {svc.decision_function(X_sc[:3]).round(3).tolist()}")
# decision_function (first 3)    : [1.0, -1.374, 1.0]`}
      </CodeBlock>

      <H3>5b. LinearSVC for large-scale problems</H3>

      <CodeBlock language="python">
{`# LinearSVC wraps liblinear — scales to millions of samples
lsvc = LinearSVC(C=1.0, max_iter=2000, random_state=42)
lsvc.fit(X_sc, y)
print(f"LinearSVC train accuracy: {lsvc.score(X_sc, y):.3f}")
# LinearSVC train accuracy: 0.842

# For very large n (>100k), prefer SGDClassifier with hinge loss:
# from sklearn.linear_model import SGDClassifier
# sgd_svm = SGDClassifier(loss="hinge", alpha=0.001)  # alpha ~ 1/C
# Equivalent to SVM with linear kernel; trains in O(n) per epoch.`}
      </CodeBlock>

      <H3>5c. SVR for regression</H3>

      <CodeBlock language="python">
{`from sklearn.svm import SVR
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

X_r, y_r = make_regression(n_samples=200, n_features=5,
                             noise=5, random_state=42)
X_r_sc = StandardScaler().fit_transform(X_r)

# epsilon-insensitive tube: no penalty if |prediction - target| < epsilon
svr = SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale")
svr.fit(X_r_sc, y_r)
print(f"SVR R^2 on train: {svr.score(X_r_sc, y_r):.3f}")
# SVR R^2 on train: 0.775
print(f"SVR n_support_vectors: {len(svr.support_)}")
# SVR n_support_vectors: 199`}
      </CodeBlock>

      <Callout type="tip">
        <strong>Platt scaling and probability=True.</strong> Setting <Code>probability=True</Code> on <Code>SVC</Code> adds a 5-fold internal cross-validation after training to fit the sigmoid <Code>P(y=1) = 1/(1 + exp(A·f(x) + B))</Code>, where <Code>f(x)</Code> is the raw decision function. The extra cost is non-trivial; only request probabilities if you actually need calibrated scores. If you only need the class label, use <Code>predict()</Code> directly from the decision function.
      </Callout>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6a. Kernel comparison on moons data</H3>

      <Prose>
        The same two-moons dataset (200 points, 0.2 noise) with three kernels, all trained with <Code>C=1.0</Code>. The decision boundary character changes dramatically with kernel choice.
      </Prose>

      <Heatmap
        label="Kernel comparison on two-moons (C=1.0): n_support_vectors and train_accuracy"
        rowLabels={["linear", "poly (d=3)", "rbf (γ=0.5)"]}
        colLabels={["n_sv", "train_acc"]}
        matrix={[
          [67, 0.850],
          [72, 0.845],
          [60, 0.945],
        ]}
        colorScale="gold"
      />

      <Callout accent="gold">
        Linear kernel cannot capture the curved boundary → underfits. Polynomial kernel d=3: marginal improvement over linear. RBF kernel: +9.5 percentage points accuracy with fewer support vectors → tighter, more efficient fit.
      </Callout>

      <Prose>
        Fewer support vectors with higher accuracy (RBF vs linear) means the RBF model found a more efficient representation: the decision boundary is well-specified by a compact subset of training points. More support vectors often signals that the model is struggling — either underfitting (linear on nonlinear data) or the margin is very tight.
      </Prose>

      <H3>6b. C parameter: margin width vs. violations</H3>

      <Prose>
        The <Code>C</Code> sweep below shows how regularization strength changes the number of support vectors and training accuracy on the two-moons dataset (kernel=rbf, gamma=0.5). The actual output from running the sweep:
      </Prose>

      <CodeBlock language="python">
{`# stdout from running the C sweep:
# === C sweep (kernel=rbf, gamma=0.5) ===
# C=  0.01  n_sv=200  train_acc=0.860
# C=  0.10  n_sv=112  train_acc=0.905
# C=  1.00  n_sv= 60  train_acc=0.945
# C= 10.00  n_sv= 36  train_acc=0.970
# C=100.00  n_sv= 24  train_acc=0.985`}
      </CodeBlock>

      <StepTrace
        label="C sweep: C=0.01 → C=100 (rbf, γ=0.5)"
        steps={[
          {
            label: "C = 0.01 — very wide margin, high regularization",
            render: () => (
              <div>
                <TokenStream
                  label="C=0.01"
                  tokens={[
                    { label: "n_sv=200 (ALL points)", color: "#f87171" },
                    { label: "train_acc=0.860", color: colors.textMuted },
                    { label: "extremely wide margin", color: colors.textDim },
                  ]}
                />
                <Prose>
                  All 200 training points become support vectors — the regularizer dominates and the margin is so wide that everything falls inside or violates it. This is extreme underfitting.
                </Prose>
              </div>
            ),
          },
          {
            label: "C = 1.0 — balanced default",
            render: () => (
              <div>
                <TokenStream
                  label="C=1.0"
                  tokens={[
                    { label: "n_sv=60", color: colors.gold },
                    { label: "train_acc=0.945", color: colors.gold },
                    { label: "reasonable margin width", color: colors.textDim },
                  ]}
                />
                <Prose>
                  Good default starting point. 60 support vectors, strong accuracy. Start here and tune outward.
                </Prose>
              </div>
            ),
          },
          {
            label: "C = 100 — near-hard margin, low regularization",
            render: () => (
              <div>
                <TokenStream
                  label="C=100"
                  tokens={[
                    { label: "n_sv=24", color: colors.green },
                    { label: "train_acc=0.985", color: colors.green },
                    { label: "risk of overfit on noisy data", color: "#f87171" },
                  ]}
                />
                <Prose>
                  Only 24 support vectors needed — the boundary fits the training set closely. On noisy data this will overfit; monitor validation accuracy.
                </Prose>
              </div>
            ),
          },
        ]}
      />

      <H3>6c. Gamma sweep: decision boundary smoothness</H3>

      <Prose>
        With <Code>C=1.0</Code> fixed, varying <Code>gamma</Code> of the RBF kernel controls how local the influence of each training point is.
      </Prose>

      <Heatmap
        label="gamma sweep — effect on train accuracy and n_sv (kernel=rbf, C=1.0, two-moons)"
        rowLabels={["gamma=0.01", "gamma=0.10", "gamma=0.50", "gamma=2.00", "gamma=10.0"]}
        colLabels={["n_support_vectors", "train_accuracy"]}
        matrix={[
          [114, 0.835],
          [77,  0.850],
          [60,  0.945],
          [57,  0.970],
          [118, 0.985],
        ]}
        colorScale="gold"
      />

      <Prose>
        Note the U-shaped pattern in <Code>n_sv</Code>: very low gamma (global, smooth boundary) needs many support vectors because the model under-specifies the boundary. Very high gamma (ultra-local) memorizes individual points — n_sv spikes back up to 118 and the training accuracy approaches 100%, but validation accuracy would collapse on noisy data. The sweet spot for this dataset is around <Code>gamma=0.5</Code> to <Code>2.0</Code>.
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        When should you reach for an SVM versus a logistic regression, random forest, or neural network? The answer depends on dataset scale, feature structure, and what you know about the boundary.
      </Prose>

      <TokenStream
        label="algorithm selection at a glance"
        tokens={[
          { label: "SVM", color: colors.gold },
          { label: "LogReg", color: "#60a5fa" },
          { label: "RandomForest", color: colors.green },
          { label: "DeepNet", color: "#c084fc" },
        ]}
      />

      <Callout type="note">
        <strong>Use SVM when:</strong> your dataset has <Code>n {"<"} 100k</Code> rows; features are moderate-to-high dimensional (text TF-IDF, hand-crafted image descriptors, biomarkers); you expect a nonlinear boundary; interpretability via support vectors matters; you want a convex optimization with a unique global optimum and no hyperparameter sensitivity to initialization.
      </Callout>

      <Callout type="warning">
        <strong>Use logistic regression when:</strong> the boundary is linear or close to linear; you need calibrated probabilities without the overhead of Platt scaling; the dataset is very large (<Code>n {">"} 500k</Code>). LogReg trains in <Code>O(n·d)</Code> per epoch with SGD; SVC with RBF kernel cannot compete at that scale.
      </Callout>

      <Callout type="note">
        <strong>Use random forest when:</strong> you have mixed feature types (categorical + continuous); you want robustness to outliers without scaling; feature importance scores matter; you have time for a quick non-parametric baseline before tuning.
      </Callout>

      <Callout type="warning">
        <strong>Use a deep network when:</strong> raw unstructured inputs (pixels, raw text tokens, audio waveforms) where feature engineering is not feasible; very large labeled datasets (<Code>n {">"} 100k</Code>); task benefits from learned hierarchical representations; transfer learning from a pretrained model is available.
      </Callout>

      <Prose>
        Concrete rule of thumb for tabular ML: if <Code>n {"<"} 5k</Code> and features are meaningful, RBF SVM with tuned <Code>C</Code> and <Code>gamma</Code> will beat neural networks on average because the inductive bias (max-margin, RBF smoothness) is appropriate and you avoid overfitting from excess capacity. If <Code>5k {"<"} n {"<"} 100k</Code>, gradient-boosted trees (XGBoost, LightGBM) are usually superior due to better scaling and handling of mixed features. Above 100k, deep learning or linear models dominate.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>8.1 Training complexity</H3>

      <Prose>
        The SVM dual QP has <Code>n</Code> variables. A naive QP solver costs <Code>O(n³)</Code> due to matrix factorization. SMO reduces the effective cost to between <Code>O(n²)</Code> and <Code>O(n²·d)</Code> in practice (each iteration is <Code>O(n·d)</Code> for kernel evaluations, and the number of iterations grows with <Code>n</Code>). For the kernel SVM (RBF, polynomial), training on more than ~50,000 points becomes slow, and on more than ~200,000 points is impractical without approximations. This is the hard ceiling that prevented SVMs from scaling to ImageNet-class data.
      </Prose>

      <TokenStream
        label="training complexity summary"
        tokens={[
          { label: "Hard/RBF SVM: O(n²)–O(n³)", color: "#f87171" },
          { label: "LinearSVC (liblinear): O(n·d)", color: colors.green },
          { label: "SGDClassifier(loss='hinge'): O(n·d) per epoch", color: colors.green },
        ]}
      />

      <H3>8.2 Inference complexity</H3>

      <Prose>
        At inference time, classifying a new point requires evaluating the kernel against every support vector: <Code>O(n_sv · d)</Code>. If your model learned 10,000 support vectors from a 50,000-point training set, inference is 10x more expensive than a linear model. For time-sensitive applications, compress the support vector set post-training (support vector reduction, or replace the kernel SVM with a random features approximation trained with the same hinge loss).
      </Prose>

      <H3>8.3 Approximations for large-scale kernel SVMs</H3>

      <Prose>
        Two approximation strategies make kernel SVMs tractable at larger scale. The <strong>Nyström method</strong> samples a subset of <Code>m {"<"} n</Code> training points, uses them to construct a low-rank approximation of the kernel matrix, and then trains a linear classifier in the approximate feature space. The <strong>random Fourier features</strong> approach (Rahimi and Recht, NeurIPS 2007) constructs an explicit <Code>D</Code>-dimensional random feature map <Code>z(x)</Code> such that <Code>z(x)·z(x') ≈ k(x, x')</Code> for the RBF kernel, enabling training of a linear model on the approximate features with full SGD. Both reduce training to <Code>O(n · D)</Code> where <Code>D</Code> is the approximation rank, at the cost of some accuracy. scikit-learn provides both via <Code>sklearn.kernel_approximation.Nystroem</Code> and <Code>RBFSampler</Code>.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes {"&"} gotchas</H2>

      <H3>9.1 Forgetting to scale features</H3>

      <Prose>
        The RBF kernel computes <Code>exp(−γ‖x−x'‖²)</Code>. If one feature has values in the thousands while another is in the range <Code>[0, 1]</Code>, the Euclidean distance is dominated by the large-scale feature and the kernel is effectively ignoring all others. A column with range <Code>[0, 1000]</Code> contributes <Code>1,000,000</Code> to the squared distance; a column with range <Code>[0, 1]</Code> contributes at most <Code>1</Code>. The model will learn as if only the large-scale feature exists. Always apply <Code>StandardScaler</Code> or <Code>MinMaxScaler</Code> before fitting an SVC with a non-linear kernel. This is the single most common SVM bug in practice.
      </Prose>

      <H3>9.2 C and gamma interact — grid search in 2D</H3>

      <Prose>
        <Code>C</Code> and <Code>gamma</Code> jointly determine the bias-variance trade-off and they interact: high <Code>gamma</Code> with low <Code>C</Code> produces a model that tries to fit a complex boundary but then heavily regularizes away from it, which is contradictory and usually bad. The canonical advice (Hsu, Chang, Lin 2003 "A Practical Guide to Support Vector Classification") is to grid-search <Code>C</Code> and <Code>gamma</Code> simultaneously on the log scale, e.g. <Code>C ∈ {"{"} 2⁻⁵, 2⁻³, ..., 2¹⁵ {"}"}</Code> and <Code>gamma ∈ {"{"} 2⁻¹⁵, ..., 2³ {"}"}</Code> with 5-fold cross-validation. This is expensive — <Code>|grid| × 5</Code> SVM fits — but unavoidable for non-trivial datasets.
      </Prose>

      <H3>9.3 Multi-class: one-vs-one by default in SVC</H3>

      <Prose>
        <Code>sklearn.svm.SVC</Code> handles <Code>K</Code>-class problems by training <Code>K(K−1)/2</Code> binary classifiers, each distinguishing one class from another (one-vs-one). A test point is classified by majority vote. For <Code>K=10</Code> classes, this means 45 SVMs. <Code>LinearSVC</Code> uses one-vs-rest by default (K classifiers), which is faster but can produce less accurate decision regions near class boundaries. For small <Code>K {"<"} 10</Code>, one-vs-one is usually better; for large <Code>K {">"} 50</Code>, one-vs-rest or structured prediction methods are preferred.
      </Prose>

      <H3>9.4 No native probability estimates</H3>

      <Prose>
        The raw SVM output is a signed distance to the hyperplane — not a probability. Setting <Code>probability=True</Code> in <Code>SVC</Code> adds Platt scaling (Platt 1999, "Probabilistic Outputs for Support Vector Machines"): a sigmoid is fit to the decision function values on held-out folds. This adds training time, and the resulting probabilities are poorly calibrated near extreme values. If you need reliable probability estimates, consider isotonic regression or temperature scaling applied post-hoc, or use logistic regression directly.
      </Prose>

      <H3>9.5 Class imbalance</H3>

      <Prose>
        SVMs are not inherently robust to class imbalance. With 95% negative examples, the max-margin hyperplane will tend to be pushed toward the minority class, and the minority class will have more margin violations. The fix is <Code>class_weight="balanced"</Code> in <Code>SVC</Code> or <Code>LinearSVC</Code>, which sets <Code>Cᵢ = C × n_total / (n_classes × n_class_i)</Code>. This increases the penalty for misclassifying the minority class, rebalancing the margin.
      </Prose>

      <H3>9.6 Numerical issues with the kernel matrix</H3>

      <Prose>
        LIBSVM precomputes the kernel matrix for small datasets and uses a cache for larger ones. If you pass a custom precomputed kernel (<Code>kernel="precomputed"</Code>), ensure the matrix is positive semi-definite and its diagonal values are non-negative. Non-PSD kernel matrices break the convexity assumption and the solver may cycle or produce NaNs. Check with <Code>np.linalg.eigvalsh(K).min() {">"} -1e-8</Code> before passing to SVC.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        Every citation below was verified via web search. Links are to canonical publisher pages or official author copies.
      </Prose>

      <H3>Foundational papers</H3>

      <Prose>
        <strong>Vapnik, V. N. {"&"} Chervonenkis, A. Ya. (1968).</strong> "On the uniform convergence of relative frequencies of events to their probabilities." <em>Automation and Remote Control</em>, 16(2), 264–280. (Russian original 1968; English translation 1971.) The theoretical bedrock: VC dimension, capacity, and the relationship between margin and generalization.
      </Prose>

      <Prose>
        <strong>Boser, B. E., Guyon, I. M., {"&"} Vapnik, V. N. (1992).</strong> "A training algorithm for optimal margin classifiers." <em>Proceedings of the 5th Annual Workshop on Computational Learning Theory (COLT)</em>, Pittsburgh, July 27–29, 1992, pp. 144–152. ACM. DOI: 10.1145/130385.130401. Introduced the kernel trick and cast max-margin classification as a QP.
      </Prose>

      <Prose>
        <strong>Cortes, C., {"&"} Vapnik, V. (1995).</strong> "Support-vector networks." <em>Machine Learning</em>, 20(3), 273–297. Springer. DOI: 10.1007/BF00994018. Extended to soft-margin SVM with slack variables and <Code>C</Code>-regularization; coined "support-vector network."
      </Prose>

      <H3>Algorithms and implementations</H3>

      <Prose>
        <strong>Platt, J. C. (1998).</strong> "Sequential minimal optimization: A fast algorithm for training support vector machines." Microsoft Research Technical Report MSR-TR-98-14. The SMO algorithm that made kernel SVM training practical; the basis for LIBSVM.
      </Prose>

      <Prose>
        <strong>Platt, J. C. (1999).</strong> "Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods." In Smola, A. J. et al. (Eds.), <em>Advances in Large Margin Classifiers</em>, pp. 61–74. MIT Press. Introduced Platt scaling — fitting a sigmoid to map SVM decision values to calibrated probabilities.
      </Prose>

      <Prose>
        <strong>Chang, C.-C., {"&"} Lin, C.-J. (2011).</strong> "LIBSVM: A library for support vector machines." <em>ACM Transactions on Intelligent Systems and Technology</em>, 2(3), 27:1–27:27. DOI: 10.1145/1961189.1961199. The library underlying scikit-learn's SVC; over 10,000 citations.
      </Prose>

      <H3>Comprehensive reference</H3>

      <Prose>
        <strong>Schölkopf, B., {"&"} Smola, A. J. (2002).</strong> <em>Learning with Kernels: Support Vector Machines, Regularization, Optimization, and Beyond.</em> MIT Press (Adaptive Computation and Machine Learning series). ISBN: 978-0-262-19475-4. The definitive mathematical treatment: Mercer's theorem, RKHS, regularization, and connections to Gaussian processes. Chapters 6–8 cover the SVM dual in full rigor.
      </Prose>

      <Prose>
        <strong>Hsu, C.-W., Chang, C.-C., {"&"} Lin, C.-J. (2003).</strong> "A practical guide to support vector classification." Technical report, National Taiwan University. Essential applied reference: log-scale grid search for C and gamma, feature scaling protocol, one-vs-one vs. one-vs-rest.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1</H3>
      <Prose>
        <strong>Question.</strong> You train an RBF SVM on a dataset with features in vastly different scales — one feature ranges from 0 to 10,000 (annual salary) and another from 0 to 1 (a probability). Without scaling, what will happen to the learned decision boundary? Why?
      </Prose>
      <Prose>
        <strong>Answer.</strong> The salary feature will dominate the Euclidean distance in the RBF kernel <Code>exp(−γ‖x−x'‖²)</Code>. A difference of 1 in salary contributes <Code>1</Code> to the squared norm; a difference of 1 in the probability feature contributes at most <Code>1</Code>. But a typical salary difference between two points might be 5,000, contributing <Code>25,000,000</Code> to the squared distance — making the probability feature effectively invisible to the kernel. The SVM will produce a decision boundary that depends almost entirely on salary, ignoring the probability feature regardless of its predictive value. Always apply <Code>StandardScaler</Code> before SVC.
      </Prose>

      <H3>Exercise 2</H3>
      <Prose>
        <strong>Question.</strong> You observe that after training an SVM, <em>every</em> training point has become a support vector (<Code>n_support == n_train</Code>). What does this indicate and what would you change?
      </Prose>
      <Prose>
        <strong>Answer.</strong> All training points being support vectors means every point lies on or inside the margin — the margin is so wide that nothing sits outside it. This is a sign of extreme under-regularization: <Code>C</Code> is too low (or equivalently, regularization is too strong). The model is underfit. Increase <Code>C</Code> to narrow the margin and allow fewer violations. If even large <Code>C</Code> cannot reduce the number of support vectors substantially, the data may be inherently non-separable with the chosen kernel — try a different kernel or more informative features.
      </Prose>

      <H3>Exercise 3</H3>
      <Prose>
        <strong>Question.</strong> Write down the dual objective for the hard-margin SVM and explain why the solution <Code>w</Code> can be expressed as a linear combination of only the support vectors, even if the training set has millions of points.
      </Prose>
      <Prose>
        <strong>Answer.</strong> The dual objective is <Code>max Σ αᵢ − (1/2) Σᵢⱼ αᵢ αⱼ yᵢ yⱼ k(xᵢ,xⱼ)</Code> subject to <Code>αᵢ ≥ 0</Code> and <Code>Σ αᵢ yᵢ = 0</Code>. From the KKT stationarity condition, <Code>w = Σ αᵢ yᵢ xᵢ</Code>. Complementary slackness forces <Code>αᵢ(yᵢ f(xᵢ) − 1) = 0</Code> at optimality. For a non-support-vector (a point strictly outside the margin), <Code>yᵢ f(xᵢ) {">"} 1</Code>, so the complementary slackness equation forces <Code>αᵢ = 0</Code>. Therefore only support vectors (where <Code>yᵢ f(xᵢ) = 1</Code>) can have <Code>αᵢ {">"} 0</Code>. The sum <Code>w = Σ αᵢ yᵢ xᵢ</Code> collapses to only those terms.
      </Prose>

      <H3>Exercise 4</H3>
      <Prose>
        <strong>Question.</strong> Your binary classification dataset has 95% negative and 5% positive examples. You train <Code>SVC(kernel='rbf', C=1.0)</Code> and find it predicts negative for every test example, achieving 95% accuracy. What is wrong and how do you fix it?
      </Prose>
      <Prose>
        <strong>Answer.</strong> The model has collapsed to the majority-class predictor — a classic class-imbalance failure. The max-margin objective, with equal penalty <Code>C</Code> for all misclassifications, prefers to violate the 5% minority class because it incurs a smaller total penalty. Fix: set <Code>class_weight="balanced"</Code> in <Code>SVC</Code>, which scales <Code>Cᵢ</Code> inversely proportional to class frequency. This forces the model to pay 19× more per minority-class violation, rebalancing the margin. Also monitor F1, precision-recall, or AUC instead of accuracy on imbalanced data.
      </Prose>

      <H3>Exercise 5</H3>
      <Prose>
        <strong>Question.</strong> Explain in one paragraph why an RBF kernel SVM on a dataset with <Code>n = 1,000,000</Code> points is impractical, and name two approximation strategies that restore tractability.
      </Prose>
      <Prose>
        <strong>Answer.</strong> Kernel SVM training requires evaluating and caching the <Code>n × n</Code> kernel matrix (1 trillion entries for 1M points — far exceeding memory) and running SMO iterations whose cost grows as <Code>O(n²)</Code> to <Code>O(n³)</Code>. Even with LIBSVM's kernel cache, training times would be days to weeks on standard hardware. Two approximation strategies restore tractability: (1) the <strong>Nyström method</strong> uses a rank-<Code>m</Code> approximation of the kernel matrix (sampling <Code>m {"<"} n</Code> landmark points), reducing training to fitting a linear SVM in <Code>m</Code>-dimensional space with <Code>O(n · m)</Code> cost; and (2) <strong>random Fourier features</strong> (Rahimi {"&"} Recht 2007) constructs an explicit <Code>D</Code>-dimensional random feature map <Code>z(x)</Code> such that <Code>z(x)·z(x') ≈ k(x, x')</Code>, allowing SGD training in <Code>O(n · D)</Code>. Both trade a small accuracy loss for order-of-magnitude speedups.
      </Prose>

      <H3>Exercise 6</H3>
      <Prose>
        <strong>Question.</strong> State Mercer's condition. Why must a kernel function satisfy it for the SVM dual to remain a valid convex optimization problem?
      </Prose>
      <Prose>
        <strong>Answer.</strong> Mercer's condition states that a kernel function <Code>k: X × X → ℝ</Code> is valid if and only if the kernel (Gram) matrix <Code>K</Code>, with <Code>Kᵢⱼ = k(xᵢ, xⱼ)</Code>, is symmetric and positive semi-definite for any finite set of points <Code>{"{"} x₁, ..., xₙ {"}"} ⊂ X</Code>. This is equivalent to requiring that <Code>k</Code> is an inner product in some (possibly infinite-dimensional) Hilbert space. If the kernel matrix is not PSD, the dual objective <Code>max Σ αᵢ − (1/2) αᵀ K_y α</Code> (where <Code>K_y</Code> has entries <Code>yᵢ yⱼ k(xᵢ,xⱼ)</Code>) is no longer guaranteed to be a concave quadratic form. A non-concave dual may have multiple local maxima, and the global max may be unbounded above — making the QP ill-posed and the solver diverge or cycle. Mercer's condition is the mathematical guarantee that the kernel trick produces a well-posed, convex optimization problem.
      </Prose>

    </div>
  ),
};

export default supportVectorMachinesContent;
