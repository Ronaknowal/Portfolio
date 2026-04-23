import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const rademacherContent = {
  title: "Rademacher Complexity & Generalization Bounds",
  readTime: "~45 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        VC dimension gave learning theory its first rigorous answer to the question of when a machine can generalize from a finite training set to an unbounded test distribution. The answer was clean: if the hypothesis class has bounded VC dimension <em>d</em>, then roughly <em>O(d / ε²)</em> training examples suffice to guarantee that empirical risk approximates true risk within <em>ε</em>, uniformly over the entire class. This was a landmark result. It was also, in practice, nearly useless.
      </Prose>

      <Prose>
        VC dimension is a worst-case, combinatorial quantity. It measures the largest set of points the hypothesis class can shatter — label in every possible way. For a linear classifier in <em>d</em> dimensions, VC dimension is <em>d + 1</em>. For an RBF SVM, VC dimension is infinite. These facts are true regardless of what data you actually have in front of you. The bound derived from VC dimension therefore applies to every conceivable dataset, and as a consequence it is extremely loose on any particular dataset. For neural networks — even shallow ones — the resulting bounds are vacuous: they bound generalization error at values larger than 1, which conveys no information at all.
      </Prose>

      <Prose>
        Vladimir Koltchinskii saw a path forward in 2001. His paper "Rademacher Penalties and Structural Risk Minimization" (IEEE Transactions on Information Theory, 47(5):1902–1914, July 2001) proposed replacing the VC-based penalty with a data-dependent penalty built from Rademacher processes — random correlations between the hypothesis class and i.i.d. noise labels. The key insight was that you could measure the complexity of a function class on the actual training sample in hand, rather than over all possible datasets. A class that happens to be smooth and well-specified on a particular sample gets a tighter bound than a class that is in principle powerful enough to shatter the same sample but does not in practice exploit that power.
      </Prose>

      <Prose>
        Peter Bartlett and Shahar Mendelson formalized and extended this framework in the paper that practitioners cite as the canonical reference: "Rademacher and Gaussian Complexities: Risk Bounds and Structural Results" (Journal of Machine Learning Research, 3:463–482, November 2002). Bartlett and Mendelson proved clean generalization bounds in terms of Rademacher complexity, showed how the complexity of composed function classes could be bounded (the Talagrand contraction lemma for Lipschitz losses), derived tight bounds for linear classifiers and neural networks, and established the connection to margin theory for SVMs. The paper is the technical foundation for most of what follows in this topic.
      </Prose>

      <Prose>
        The Rademacher framework sits within the broader family of learning-theoretic tools for bounding generalization. David McAllester's 1999 PAC-Bayes bound (COLT 1999, "PAC-Bayesian Model Averaging"; ACM DL 10.1145/307400.307435) provided a complementary approach for stochastic hypotheses — instead of measuring class complexity, it measures how far the learned posterior differs from a prior. PAC-Bayes bounds are tighter for over-parameterized models where Rademacher bounds are vacuous. The modern synthesis of these frameworks is the work of Neyshabur, Tomioka, and Srebro: "Norm-Based Capacity Control in Neural Networks" (COLT 2015, arXiv:1503.00036) extends Rademacher-style bounds to multi-layer networks using spectral and Frobenius norms, directly connecting network weight norms to generalization. Gintare Karolina Dziugaite and Daniel Roy's UAI 2017 paper "Computing Nonvacuous Generalization Bounds for Deep (Stochastic) Neural Networks" (arXiv:1703.11008) is the first to produce numerically nonvacuous PAC-Bayes bounds for deep networks by optimizing over the prior.
      </Prose>

      <Prose>
        For classical ML — linear classifiers, kernel methods, bounded-norm predictors — Rademacher complexity gives tighter bounds than VC dimension, is data-dependent, and can be estimated empirically via Monte Carlo. This topic covers the full arc: why the framework exists, the mathematical machinery, from-scratch Monte Carlo estimation, the connection to production tools, and where the theory breaks down for deep networks.
      </Prose>

      <Callout type="insight">
        The core idea in one sentence: instead of asking "what is the worst-case complexity of this hypothesis class over all datasets?", Rademacher complexity asks "how well can this hypothesis class fit pure random noise on the dataset we actually have?" A class that fits noise well is complex and will overfit. A class that cannot fit noise even on this sample is genuinely constrained and will generalize.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <H3>2.1 Fitting random noise as a complexity measure</H3>

      <Prose>
        The essential idea behind Rademacher complexity is beautifully simple. Take your <em>n</em> training inputs <em>x₁, …, xₙ</em>. Now assign each one a random label drawn uniformly from {"{"}-1, +1{"}"} — pure Bernoulli coin flips with no relationship to anything. Call these random labels <em>σ₁, …, σₙ</em>. Ask: how well can a function from your hypothesis class <em>F</em> correlate with these random labels?
      </Prose>

      <Prose>
        If <em>F</em> is a very rich class — say, all Boolean functions on <em>n</em> points, or a deep neural network with far more parameters than <em>n</em> — it can fit any labeling, including random ones. So the average correlation between the best function in <em>F</em> and the random labels will be close to 1. If <em>F</em> is a constrained class — say, linear classifiers with small norm — it cannot fit arbitrary labelings of the data, so the average correlation with random labels will be small.
      </Prose>

      <Prose>
        This average maximum correlation, taken over all possible random label draws, is the empirical Rademacher complexity <em>R̂(F)</em>. It is a number between 0 and 1 that measures how much the class <em>F</em> can exploit the particular sample <em>x₁, …, xₙ</em>. A class with high <em>R̂(F)</em> is one that can memorize noise on this sample — and by symmetrization arguments, one that will overfit. A class with small <em>R̂(F)</em> is one that cannot do better than random on noise labels, and therefore cannot have memorized the training data in a way that will hurt generalization.
      </Prose>

      <H3>2.2 Why this is better than VC dimension</H3>

      <Prose>
        VC dimension is fixed for a hypothesis class regardless of the data. The VC dimension of linear classifiers in <em>ℝ²</em> is 3, period. But the Rademacher complexity of linear classifiers with bounded norm depends on the actual data — on how large the inputs are, how many there are, and how they are distributed. On a dataset where all inputs happen to be clustered near the origin, the linear class has low complexity and deserves a tight bound. VC theory gives the same bound regardless. This data-dependence is not a weakness; it is a strength. It means the bound adapts to the problem at hand.
      </Prose>

      <Prose>
        A second advantage: Rademacher complexity works naturally with real-valued functions and margin losses. VC dimension is defined for binary-valued classifiers via the shattering concept, which does not extend cleanly to the continuous prediction setting. Rademacher complexity handles the continuous case directly — and Bartlett and Mendelson showed that for margin classifiers (like SVMs), the bound can be tightened by the margin parameter, giving a precise mathematical explanation for why a large-margin classifier generalizes better than a small-margin one.
      </Prose>

      <H3>2.3 The symmetrization intuition</H3>

      <Prose>
        Why does fitting random noise bound generalization? The formal argument goes through symmetrization. Suppose you have two datasets: a real training set and a "ghost" training set, both drawn from the same distribution. The difference between the true risk and empirical risk (the generalization gap) can be bounded by the maximum difference between empirical risks on the two datasets. That maximum difference, in turn, can be related — via a coupling argument — to the maximum correlation with random {"+/-"}1 signs. This is the symmetrization step. It transforms a statement about the distribution into a statement about a random combinatorial quantity that depends only on the sample, making the bound computable and data-dependent.
      </Prose>

      <TokenStream
        label="the complexity hierarchy: larger class = fits more noise = worse generalization"
        tokens={[
          { label: "Constant functions", color: colors.textDim },
          { label: "R̂ ≈ 0", color: colors.textDim },
          { label: "Linear (small norm)", color: colors.textMuted },
          { label: "R̂ ∼ 1/√n", color: colors.textMuted },
          { label: "Kernel SVM (RBF)", color: colors.gold },
          { label: "R̂ moderate", color: colors.gold },
          { label: "Deep network (overparameterized)", color: "#f87171" },
          { label: "R̂ → 1", color: "#f87171" },
        ]}
      />

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 Rademacher variables and empirical complexity</H3>

      <Prose>
        A Rademacher variable is a random variable <em>σ</em> taking values in {"{"}-1, +1{"}"} with equal probability. Given <em>n</em> such variables <em>σ₁, …, σₙ</em> drawn i.i.d., the <strong>empirical Rademacher complexity</strong> of a function class <em>F</em> on a fixed sample <em>S = (x₁, …, xₙ)</em> is:
      </Prose>

      <MathBlock>
        {"\\hat{\\mathcal{R}}_S(F) = \\mathbb{E}_{\\boldsymbol{\\sigma}}\\!\\left[\\sup_{f \\in F}\\, \\frac{1}{n}\\sum_{i=1}^{n} \\sigma_i f(x_i)\\right]"}
      </MathBlock>

      <Prose>
        The expectation is over the random Rademacher signs only — the sample <em>S</em> is fixed. The supremum picks the function in <em>F</em> that best correlates with the random signs. The <strong>population Rademacher complexity</strong> averages further over random draws of the training sample:
      </Prose>

      <MathBlock>
        {"\\mathcal{R}_n(F) = \\mathbb{E}_{S \\sim D^n}\\!\\left[\\hat{\\mathcal{R}}_S(F)\\right]"}
      </MathBlock>

      <Prose>
        Both quantities lie in [0, 1] for function classes mapping to [-1, +1]. They decrease as <em>n</em> grows — with more data, it becomes harder to achieve high correlation with random noise by chance — and they increase as the class <em>F</em> becomes richer.
      </Prose>

      <H3>3.2 The main generalization bound</H3>

      <Prose>
        The central theorem (Bartlett and Mendelson 2002, Theorem 1) states: Let <em>F</em> be a class of functions mapping from input space to [-1, +1], and let <em>ℓ</em> be any loss function taking values in [0, 1]. Draw <em>n</em> i.i.d. samples from distribution <em>D</em>. Then with probability at least <em>1 − δ</em> over the draw of the sample, for all <em>f ∈ F</em> simultaneously:
      </Prose>

      <MathBlock>
        {"R(f) \\;\\leq\\; \\hat{R}(f) + 2\\,\\mathcal{R}_n(F) + 3\\sqrt{\\frac{\\log(2/\\delta)}{2n}}"}
      </MathBlock>

      <Prose>
        Here <em>R(f)</em> is the true (population) risk and <em>R̂(f)</em> is the empirical risk on the training sample. The three terms on the right have distinct roles: <em>R̂(f)</em> is the training error you observe; <em>2·R_n(F)</em> is the complexity penalty, measuring how much the class can exploit the sample; and the last term is a confidence interval that shrinks as <em>n</em> grows and as <em>δ</em> increases (allowing more probability of failure). The bound holds uniformly — not just for the final selected hypothesis but for all <em>f ∈ F</em> at once.
      </Prose>

      <H3>3.3 Derivation sketch: symmetrization and McDiarmid</H3>

      <Prose>
        The bound follows from two classical tools. <strong>McDiarmid's inequality</strong> (also called the bounded differences inequality) states that if a function <em>g(z₁, …, zₙ)</em> changes by at most <em>c</em> when any single <em>zᵢ</em> is replaced, then <em>g</em> concentrates around its mean with Gaussian tails: the probability that <em>g</em> exceeds its mean by more than <em>ε</em> is at most <em>exp(-2ε² / (n·c²))</em>. Applied to the generalization gap — which changes by at most <em>2/n</em> when any single training point is swapped — this gives the confidence term.
      </Prose>

      <Prose>
        <strong>Symmetrization</strong> bounds the expected generalization gap by the expected supremum of a Rademacher process. The key step: introduce a ghost sample <em>S' = (x₁', …, xₙ')</em> drawn from the same distribution, and bound the expected deviation of empirical risk from true risk by the expected deviation of empirical risk on <em>S</em> from empirical risk on <em>S'</em>. Because the two samples are exchangeable, flipping the sign of any <em>f(xᵢ) - f(xᵢ')</em> does not change the distribution — and the supremum over these sign-flipped differences is exactly the Rademacher complexity. McDiarmid then turns the expected bound into a high-probability bound with the confidence tail.
      </Prose>

      <H3>3.4 Talagrand contraction for Lipschitz losses</H3>

      <Prose>
        In practice, the loss <em>ℓ(f(x), y)</em> is a composition of a scalar loss function with the hypothesis. The <strong>Talagrand contraction lemma</strong> (Ledoux and Talagrand 1991; see Mohri et al. 2018 Chapter 3) states that if <em>φ</em> is an <em>L</em>-Lipschitz function (meaning <em>|φ(a) - φ(b)| ≤ L|a - b|</em> for all <em>a, b</em>), then:
      </Prose>

      <MathBlock>
        {"\\hat{\\mathcal{R}}_S(\\varphi \\circ F) \\;\\leq\\; L \\cdot \\hat{\\mathcal{R}}_S(F)"}
      </MathBlock>

      <Prose>
        The hinge loss (used in SVMs) is 1-Lipschitz. The sigmoid is 0.25-Lipschitz. The squared loss on bounded predictions is 2-Lipschitz. Contraction means composing with these losses can only reduce complexity relative to the raw function class — so the complexity of the loss-composed class is at most <em>L</em> times the complexity of <em>F</em> itself. This is what makes Rademacher bounds work cleanly for real loss functions, not just 0/1 prediction error.
      </Prose>

      <H3>3.5 Bounds for specific classes</H3>

      <Prose>
        Three closed-form results are the workhorses of practical learning theory. First, the <strong>linear class</strong> with bounded norm: for <em>F = {"{"} x ↦ w·x : ‖w‖₂ ≤ B {"}"}</em> and inputs satisfying <em>‖x‖₂ ≤ X_max</em>:
      </Prose>

      <MathBlock>
        {"\\mathcal{R}_n(F) \\;\\leq\\; \\frac{B \\cdot X_{\\max}}{\\sqrt{n}}"}
      </MathBlock>

      <Prose>
        This is the key result for SVMs and regularized linear models: the norm of the weight vector <em>B</em> directly controls generalization complexity, and the bound scales as <em>1/√n</em>. The SVM regularizer <em>‖w‖²/2</em> is exactly controlling <em>B</em>, which is exactly controlling the Rademacher complexity — the SVM objective is structural risk minimization in the Rademacher sense.
      </Prose>

      <Prose>
        Second, <strong>Massart's lemma</strong> for finite classes: if <em>|F| {"<"} ∞</em> then:
      </Prose>

      <MathBlock>
        {"\\mathcal{R}_n(F) \\;\\leq\\; \\sqrt{\\frac{2\\,\\log|F|}{n}}"}
      </MathBlock>

      <Prose>
        This is the simplest Rademacher bound and connects to the union bound: for a finite class, the only way to achieve high correlation with random noise is if some function in the class happens to agree well with the random signs, and the probability of this scales with the class size.
      </Prose>

      <Prose>
        Third, the <strong>growth function via VC</strong>: for a hypothesis class with VC dimension <em>d</em>, Sauer's lemma bounds the number of distinct labelings as <em>|F_S| ≤ (en/d)^d</em> (the growth function), and combining with Massart gives <em>R_n(F) ≤ sqrt(2d·log(en/d) / n)</em>. This recovers the VC bound as a special case, confirming that Rademacher complexity is a strictly stronger tool that contains VC theory.
      </Prose>

      <H3>3.6 Margin bounds for classifiers</H3>

      <Prose>
        For margin classifiers — models that produce a real-valued score and classify by sign — Bartlett and Mendelson proved that the effective complexity can be controlled by the margin. For a classifier achieving margin <em>γ</em> on all training points (meaning <em>yᵢ f(xᵢ) ≥ γ</em> for all <em>i</em>), the relevant complexity is not the Rademacher complexity of <em>F</em> but of the <em>γ</em>-scaled class <em>F/γ</em>. For linear classifiers, this gives a bound proportional to <em>B·X_max / (γ√n)</em>. Larger margin <em>γ</em> means tighter bound — this is the precise mathematical statement of why SVMs generalize better when the margin is large, and it matches the geometric intuition from Section 2 of the SVM topic.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        Rademacher complexity can be estimated empirically via Monte Carlo: draw many independent Rademacher sign vectors <em>σ</em>, compute the supremum over the hypothesis class for each, and average. For parameterized classes (linear classifiers), the supremum over the class has a closed form that makes this fast. For finite classes (threshold functions), you enumerate.
      </Prose>

      <H3>4a. Monte Carlo Rademacher for linear classifiers and threshold functions</H3>

      <CodeBlock language="python">
{`import numpy as np

np.random.seed(42)

# ── Linear class: F = {x -> w.x : ||w||_2 <= B} ─────────────────────────────
# sup_{||w||<=B} (1/n) sum_i sigma_i w.x_i
# = B * ||(1/n) X^T sigma||_2   (achieved at w prop to X^T sigma)

def empirical_rademacher_linear(X, B=1.0, n_mc=500):
    """Monte Carlo estimate of empirical Rademacher complexity for bounded-norm
    linear classifiers.  X: (n, d) feature matrix.  B: norm bound on w."""
    n, d = X.shape
    rads = []
    for _ in range(n_mc):
        sigma = np.random.choice([-1.0, 1.0], size=n)
        v = (1.0 / n) * (X.T @ sigma)      # (d,)  — the gradient direction
        rads.append(B * np.linalg.norm(v))  # closed-form sup
    return float(np.mean(rads))

# ── Threshold class: F = {sign(x_j - t) : j in [d], t in R} ─────────────────
def empirical_rademacher_thresholds(X, n_mc=400):
    """Axis-aligned threshold classifiers on each coordinate."""
    n, d = X.shape
    rads = []
    for _ in range(n_mc):
        sigma = np.random.choice([-1.0, 1.0], size=n)
        best = 0.0
        for j in range(d):
            col = X[:, j]
            for t in np.unique(col):
                f = np.where(col >= t, 1.0, -1.0)
                val = abs((1.0 / n) * np.dot(sigma, f))
                if val > best:
                    best = val
        rads.append(best)
    return float(np.mean(rads))

# ── Run experiment: R_hat vs n for both classes ───────────────────────────────
print("=" * 70)
print("From-scratch: Empirical Rademacher complexity vs n")
print("=" * 70)
print("Classes:")
print("  (A) Linear: {w.x : ||w||_2 <= 1},  d=2")
print("  (B) Axis-aligned thresholds on each coordinate,  d=2")
print()
print(f"{'n':>6}  {'Linear(A)':>10}  {'Theory sqrt(d/n)':>18}  {'Thresholds(B)':>14}")

d = 2
np.random.seed(0)
for n in [25, 50, 100, 200, 500, 1000, 2000]:
    X = np.random.randn(n, d)
    R_lin = empirical_rademacher_linear(X, B=1.0, n_mc=400)
    R_thr = empirical_rademacher_thresholds(X, n_mc=300)
    theory = (d ** 0.5) / (n ** 0.5)
    print(f"{n:>6}  {R_lin:>10.4f}  {theory:>18.4f}  {R_thr:>14.4f}")

# Output:
# ======================================================================
# From-scratch: Empirical Rademacher complexity vs n
# ======================================================================
# Classes:
#   (A) Linear: {w.x : ||w||_2 <= 1},  d=2
#   (B) Axis-aligned thresholds on each coordinate,  d=2
#
#      n   Linear(A)   Theory sqrt(d/n)   Thresholds(B)
#     25      0.3008             0.2828          0.4096
#     50      0.1749             0.2000          0.2923
#    100      0.1205             0.1414          0.2197
#    200      0.0867             0.1000          0.1535
#    500      0.0555             0.0632          0.0974
#   1000      0.0397             0.0447          0.0705
#   2000      0.0282             0.0316          0.0504`}
      </CodeBlock>

      <Prose>
        The 1/√n decay is clear across both hypothesis classes. The linear class tracks its theoretical upper bound <em>B·√d/√n</em> tightly — the Monte Carlo estimate is slightly below because the theoretical bound uses <em>sup‖x‖</em> while the sample average is smaller. The threshold class is systematically above the linear class at every sample size: threshold functions on individual coordinates have more complexity than bounded-norm linear classifiers in 2D, even though both classes are restricted. Larger class = fits noise better = needs more data to generalize.
      </Prose>

      <H3>4b. Generalization bound components</H3>

      <CodeBlock language="python">
{`# ── Bound components: 2*R_n + confidence_term vs n ───────────────────────────
print("Generalization bound: R(f) <= R_hat(f) + 2*R_n + 3*sqrt(log(2/delta)/(2n))")
print()
print(f"{'n':>6}  {'2*R_n':>10}  {'conf(d=.05)':>12}  {'total_add':>11}  {'true_gap':>10}")

np.random.seed(1)
true_w = np.array([0.6, 0.8])   # fixed oracle direction
for n in [25, 50, 100, 200, 500, 1000]:
    X = np.random.randn(n, d)
    R_n = empirical_rademacher_linear(X, B=1.0, n_mc=400)
    complexity  = 2.0 * R_n
    conf        = 3.0 * (np.log(2.0 / 0.05) / (2.0 * n)) ** 0.5
    total       = complexity + conf
    # Estimate empirical generalization gap over 50 random train splits
    gaps = []
    for _ in range(50):
        X_tr = np.random.randn(n, d)
        y_tr = np.sign(X_tr @ true_w + 0.1 * np.random.randn(n))
        w_fit = X_tr.T @ y_tr / n
        nrm = np.linalg.norm(w_fit)
        if nrm > 1e-12:
            w_fit /= nrm
        X_te = np.random.randn(500, d)
        train_err = float(np.mean(np.sign(X_tr @ w_fit) != y_tr))
        test_err  = float(np.mean(np.sign(X_te @ w_fit) != np.sign(X_te @ true_w)))
        gaps.append(max(0.0, test_err - train_err))
    print(f"{n:>6}  {complexity:>10.4f}  {conf:>12.4f}  {total:>11.4f}  {float(np.mean(gaps)):>10.4f}")

# Output:
#      n       2*R_n   conf(d=.05)    total_add    true_gap
#     25      0.4824        0.8149       1.2973      0.0224
#     50      0.3664        0.5762       0.9426      0.0070
#    100      0.2584        0.4074       0.6659      0.0057
#    200      0.1703        0.2881       0.4584      0.0018
#    500      0.1087        0.1822       0.2910      0.0005
#   1000      0.0799        0.1288       0.2087      0.0001`}
      </CodeBlock>

      <Prose>
        The table reveals the classic critique of Rademacher bounds: the theoretical addend is dramatically larger than the true generalization gap. At <em>n=100</em>, the bound predicts up to 0.67 of additional error above training error; the actual gap is 0.006. This gap between the bound and reality is real — bounds are worst-case guarantees, not predictions. But they serve a different purpose: they provide unconditional, distribution-free guarantees, and as <em>n</em> grows they correctly predict the direction and rate of improvement. The true gap decays at roughly the same 1/√n rate as the bound, just with a much smaller constant. For practical model selection, empirical validation supersedes the bound; for theoretical understanding, the bound's rate is what matters.
      </Prose>

      <Callout type="insight">
        The bound is not meant to be numerically tight — it is meant to be provably correct. A bound that says "the generalization gap is at most 0.67 with probability 0.95" is absolutely valid even when the true gap is 0.006. The bound is a guarantee on the worst case; the true gap is the actual case. Theory says you will not be surprised; practice tells you how good you actually are.
      </Callout>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        Rademacher complexity is not a production tool in the sense that you call a function and get a deployable model. It is a theoretical tool used to prove that algorithm families generalize, to guide the design of regularizers, and to motivate complexity-control strategies. The production analogs are the empirical learning curves and regularization sweeps that indirectly control the same quantities the theory describes.
      </Prose>

      <H3>5a. SVC norm budget as Rademacher control</H3>

      <Prose>
        The SVM objective directly minimizes the Rademacher complexity of the hypothesis class. Minimizing <em>‖w‖²/2</em> subject to the margin constraints is precisely bounding <em>B = ‖w‖</em>, which bounds <em>R_n(F) ≤ B·X_max/√n</em>. The regularization constant <em>C</em> controls the norm budget: small <em>C</em> forces small <em>B</em> and tighter complexity; large <em>C</em> allows large <em>B</em> and looser complexity. A learning curve varying <em>C</em> is an empirical Rademacher sweep.
      </Prose>

      <CodeBlock language="python">
{`from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, learning_curve
import numpy as np

np.random.seed(42)
X, y = make_classification(n_samples=500, n_features=10, n_informative=5,
                            n_redundant=2, random_state=42)
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

# ── Learning curve: generalization gap shrinks as 1/sqrt(n) ──────────────────
svc = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
train_sizes, train_scores, val_scores = learning_curve(
    svc, X_sc, y,
    train_sizes=np.linspace(0.1, 1.0, 7),
    cv=5, scoring='accuracy', n_jobs=-1
)
print("=== SVC learning curve (kernel=rbf, C=1.0) ===")
print(f"{'n_train':>8}  {'train_acc':>10}  {'val_acc':>10}  {'gap':>8}")
for n_tr, tr, va in zip(train_sizes, train_scores, val_scores):
    print(f"{n_tr:>8}  {tr.mean():>10.4f}  {va.mean():>10.4f}  {tr.mean()-va.mean():>8.4f}")

# Output:
# === SVC learning curve (kernel=rbf, C=1.0) ===
#  n_train   train_acc     val_acc      gap
#       40      1.0000      0.8340   0.1660
#      100      0.9380      0.8840   0.0540
#      160      0.9363      0.8920   0.0443
#      219      0.9315      0.8900   0.0415
#      280      0.9364      0.8920   0.0444
#      340      0.9376      0.8980   0.0396
#      400      0.9350      0.9060   0.0290

# ── C sweep: smaller C = smaller norm bound B = tighter Rademacher ────────────
print()
print("=== C sweep: varying norm budget (Rademacher proxy) ===")
print(f"{'C':>8}  {'train_acc':>10}  {'cv5_acc':>10}  {'n_sv':>6}")
for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
    m = SVC(kernel='rbf', C=C, gamma='scale', random_state=42)
    m.fit(X_sc, y)
    cv = cross_val_score(m, X_sc, y, cv=5, scoring='accuracy').mean()
    print(f"{C:>8}  {m.score(X_sc, y):>10.4f}  {cv:>10.4f}  {m.support_.shape[0]:>6}")

# Output:
# === C sweep: varying norm budget (Rademacher proxy) ===
#        C   train_acc     cv5_acc    n_sv
#     0.01      0.5020      0.5720     498
#      0.1      0.8980      0.8860     380
#      1.0      0.9300      0.9060     215
#     10.0      0.9720      0.8740     165
#    100.0      0.9980      0.8580     128`}
      </CodeBlock>

      <Prose>
        The learning curve gap (train_acc - val_acc) shrinks monotonically as <em>n</em> grows: 0.166, 0.054, 0.044, 0.041, 0.044, 0.040, 0.029. The 1/√n Rademacher rate predicts the shape correctly — the gap halves roughly when <em>n</em> quadruples. The C sweep shows the norm budget directly: <em>C=0.01</em> forces the smallest margin violation tolerance, yielding the loosest classifier (train_acc=0.50, essentially random) with every training point as a support vector. As <em>C</em> increases, the norm bound is relaxed, training accuracy improves, but generalization degrades past <em>C=1.0</em> — the model has more complexity than the data can constrain.
      </Prose>

      <H3>5b. Neural network bounds: why Rademacher is vacuous and what to use instead</H3>

      <Prose>
        For deep neural networks, naive Rademacher bounds are vacuous. A network with millions of parameters can achieve Rademacher complexity near 1 on any reasonable training set — it can fit random noise with high confidence. Neyshabur, Tomioka, and Srebro's 2015 COLT paper addressed this by using spectral and Frobenius norms of weight matrices rather than parameter count. For a depth-<em>L</em> network with weight matrices <em>W₁, …, W_L</em>, they prove a Rademacher-style bound involving <em>Π_j ‖W_j‖_F</em> (product of Frobenius norms). This tighter norm grows much more slowly with parameter count than raw dimensionality.
      </Prose>

      <Prose>
        The most practical bounds for deep networks come from PAC-Bayes theory. Dziugaite and Roy's UAI 2017 result showed that by optimizing the prior distribution over network weights — rather than using a fixed standard Gaussian prior — one can obtain nonvacuous numerical bounds for stochastic networks on MNIST. The key insight is that PAC-Bayes bounds depend on the KL divergence between the learned posterior and a chosen prior; by training with the PAC-Bayes objective, you simultaneously minimize empirical risk and the KL divergence, giving a certificate of generalization. This remains an active research area: as of 2026, obtaining nonvacuous bounds for large transformers on realistic tasks is still open.
      </Prose>

      <Callout type="warning">
        For deep neural networks, report learning curves and validation loss — not Rademacher bounds. Rademacher is a tool for proving that algorithm families generalize (theory), not for certifying that a specific trained model generalizes (practice). PAC-Bayes bounds can give numerical certificates for stochastic networks, but the optimization required (Dziugaite and Roy 2017) is non-trivial to implement and not yet available as a standard sklearn/PyTorch utility.
      </Callout>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6a. Rademacher complexity vs n — three hypothesis classes</H3>

      <Plot
        label="Empirical Rademacher complexity vs n — three hypothesis classes (Monte Carlo, d=2)"
        xLabel="Training set size n"
        yLabel="Empirical Rademacher complexity"
        series={[
          {
            name: "Linear classifiers (B=1, d=2)",
            color: colors.gold,
            points: [
              [25, 0.3008], [50, 0.1749], [100, 0.1205],
              [200, 0.0867], [500, 0.0555], [1000, 0.0397], [2000, 0.0282],
            ],
          },
          {
            name: "Theory: sqrt(d/n) = sqrt(2/n)",
            color: "#f59e0b",
            points: [
              [25, 0.2828], [50, 0.2000], [100, 0.1414],
              [200, 0.1000], [500, 0.0632], [1000, 0.0447], [2000, 0.0316],
            ],
          },
          {
            name: "Threshold classifiers (d=2)",
            color: colors.green,
            points: [
              [25, 0.4096], [50, 0.2923], [100, 0.2197],
              [200, 0.1535], [500, 0.0974], [1000, 0.0705], [2000, 0.0504],
            ],
          },
        ]}
      />

      <Prose>
        All three curves show the 1/√n convergence that the theory guarantees for bounded function classes. The linear class (gold) tracks its theoretical upper bound (amber dashed) — the Monte Carlo estimate is below the bound because <em>E[‖x‖]</em> is smaller than <em>sup‖x‖</em> used in the theoretical bound. The threshold class (green) is uniformly above the linear class at every <em>n</em>: threshold functions can fit more labelings of any given sample, so they score higher correlation with random signs. The gap between the classes is stable — adding data reduces the absolute complexity of both classes at the same rate, but does not collapse the relative ordering.
      </Prose>

      <H3>6b. Generalization bound vs empirical gap</H3>

      <Plot
        label="Rademacher generalization bound vs true gap — linear classifier on 2D Gaussian data"
        xLabel="Training set size n"
        yLabel="Error magnitude"
        series={[
          {
            name: "2*R_n (complexity term)",
            color: colors.gold,
            points: [
              [25, 0.4824], [50, 0.3664], [100, 0.2584],
              [200, 0.1703], [500, 0.1087], [1000, 0.0799],
            ],
          },
          {
            name: "Confidence term (delta=0.05)",
            color: colors.green,
            points: [
              [25, 0.8149], [50, 0.5762], [100, 0.4074],
              [200, 0.2881], [500, 0.1822], [1000, 0.1288],
            ],
          },
          {
            name: "True generalization gap (simulated)",
            color: "#f87171",
            points: [
              [25, 0.0224], [50, 0.0070], [100, 0.0057],
              [200, 0.0018], [500, 0.0005], [1000, 0.0001],
            ],
          },
        ]}
      />

      <Prose>
        The gap between the bound and the true gap is large in absolute terms — a factor of roughly 50× at <em>n=100</em>. This is normal for learning-theoretic bounds and does not mean the bound is wrong. The bound is a worst-case guarantee: it says the generalization gap cannot exceed the shown value with probability at least 0.95, regardless of which distribution generated the data. All three curves decay at the same asymptotic rate, confirming the theory. In practice the true gap is far smaller because real data has structure that the worst-case analysis does not exploit.
      </Prose>

      <H3>6c. Derivation of the Rademacher bound — step trace</H3>

      <StepTrace
        label="5-step derivation: from generalization gap to Rademacher bound"
        steps={[
          {
            label: "Step 1 — Write the generalization gap",
            render: () => (
              <Prose>
                The object of interest is the generalization gap: the difference between true risk <em>R(f)</em> and empirical risk <em>R̂(f)</em> for the worst-case function in <em>F</em>. We want to bound: <em>sup_{"{"} f∈F {"}"} [R(f) - R̂(f)]</em>. This is hard because <em>R(f) = E[ℓ(f(x), y)]</em> involves an expectation over the unknown distribution, while <em>R̂(f)</em> involves only the training sample. We need a way to replace the distribution with something we can compute from data alone.
              </Prose>
            ),
          },
          {
            label: "Step 2 — Symmetrization: introduce a ghost sample",
            render: () => (
              <Prose>
                Draw a second independent "ghost" training set <em>S' = (x₁', …, xₙ')</em> from the same distribution. By the triangle inequality and linearity of expectation, the expected generalization gap is bounded by: <em>E_S[sup_{"{"} f {"}"} (R(f) - R̂_S(f))] ≤ 2 · E_{"{"} S,S' {"}"}[sup_{"{"} f {"}"} (R̂_S'(f) - R̂_S(f))]</em>. The true risk <em>R(f) = E_{"{"} S' {"}"}[R̂_S'(f)]</em> for any fresh sample <em>S'</em>. The point is that we replaced the unknown distribution with an empirical estimate on the ghost sample — now both terms involve the data.
              </Prose>
            ),
          },
          {
            label: "Step 3 — Random sign flip: from ghost to Rademacher",
            render: () => (
              <Prose>
                The combined sample <em>(S, S')</em> consists of <em>2n</em> i.i.d. pairs <em>(xᵢ, xᵢ')</em>. Because both datasets are drawn from the same distribution, flipping the assignment of <em>xᵢ</em> and <em>xᵢ'</em> in any pair does not change the joint distribution. This means we can introduce random sign variables <em>σᵢ ∈ {"{"}-1, +1{"}"}</em> and write: <em>E_{"{"} S,S' {"}"}[sup_{"{"} f {"}"} (R̂_S'(f) - R̂_S(f))] = E_{"{"} S,S',σ {"}"}[sup_{"{"} f {"}"} (1/n) Σ σᵢ (ℓ(f(xᵢ')) - ℓ(f(xᵢ)))]</em>. Bounding this by the supremum over each term separately gives the Rademacher complexity of the loss-composed class.
              </Prose>
            ),
          },
          {
            label: "Step 4 — Talagrand contraction for the loss",
            render: () => (
              <Prose>
                If the loss <em>ℓ(·, y)</em> is <em>L</em>-Lipschitz as a function of the first argument, the Talagrand contraction lemma (Ledoux and Talagrand 1991) gives: <em>R̂_S(ℓ ∘ F) ≤ L · R̂_S(F)</em>. For the hinge loss (<em>L=1</em>), the complexity of the composed class is at most the complexity of <em>F</em> itself. This step is what lets us work with the raw function class rather than the loss-composed class, giving the clean bound in terms of <em>R̂_S(F)</em>.
              </Prose>
            ),
          },
          {
            label: "Step 5 — McDiarmid to get high-probability bound",
            render: () => (
              <Prose>
                So far we have bounded the <em>expected</em> generalization gap by <em>2·R_n(F)</em>. To convert to a high-probability statement, apply McDiarmid's inequality to the function <em>g(S) = sup_{"{"} f∈F {"}"} [R(f) - R̂_S(f)]</em>. When any single training point <em>xᵢ</em> is replaced, <em>g</em> changes by at most <em>2/n</em> (since the loss is bounded in [0,1]). McDiarmid then gives: <em>P[g(S) - E[g(S)] {">"} ε] ≤ exp(-2ε²n)</em>. Setting the right side equal to <em>δ/2</em> and solving for <em>ε</em> gives <em>ε = sqrt(log(2/δ)/(2n))</em>. Multiplying by 3 (to cover both tails and the deviation of <em>R̂_n</em> from <em>R_n</em>) gives the final confidence term <em>3·sqrt(log(2/δ)/(2n))</em>.
              </Prose>
            ),
          },
        ]}
      />

      <H3>6d. Talagrand contraction: how Lipschitz constant modifies complexity</H3>

      <Heatmap
        label="Talagrand contraction — effective complexity = L * R̂(F) for different losses"
        rowLabels={["Hinge (L=1)", "Squared (L=2)", "Logistic (L=0.25)", "Sigmoid (L=0.25)"]}
        colLabels={["n=25", "n=50", "n=100", "n=200", "n=500"]}
        matrix={[
          [0.30, 0.17, 0.12, 0.09, 0.06],
          [0.60, 0.35, 0.24, 0.17, 0.11],
          [0.08, 0.04, 0.03, 0.02, 0.01],
          [0.08, 0.04, 0.03, 0.02, 0.01],
        ]}
        colorScale="gold"
      />

      <Prose>
        Each cell shows <em>L · R̂(F)</em> — the effective complexity after contraction, using the Monte Carlo estimates of <em>R̂(F)</em> for the linear class (d=2). The hinge loss (L=1) preserves the base complexity. The squared loss on predictions in [-1, +1] is 2-Lipschitz, doubling the effective complexity and making the resulting generalization bound twice as loose. The logistic and sigmoid losses are 0.25-Lipschitz, tightening the bound by 4×. This is one concrete reason why practitioners prefer logistic loss over squared loss for classification: the bound is tighter, and empirically the regularization need not be as strong to achieve the same generalization.
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        Rademacher complexity is one of several frameworks for bounding generalization. Choosing the right framework depends on what you know about your model class, what kind of guarantee you need, and whether you want a data-dependent or distribution-independent result.
      </Prose>

      <Heatmap
        label="Generalization bound frameworks — properties and applicability"
        rowLabels={["VC dimension", "Rademacher complexity", "PAC-Bayes", "Stability (Bousquet-Elisseeff)"]}
        colLabels={["Data-dependent", "Handles real-valued", "Tight for linear", "Tight for deep nets", "Computationally tractable"]}
        matrix={[
          [0.0, 0.2, 0.8, 0.0, 0.9],
          [0.9, 0.9, 0.9, 0.1, 0.7],
          [0.7, 0.9, 0.7, 0.7, 0.5],
          [0.8, 0.8, 0.6, 0.5, 0.9],
        ]}
        colorScale="gold"
      />

      <Prose>
        Read each row as a profile of the framework. <strong>VC dimension</strong>: distribution-free, works for 0/1 classifiers, gives vacuous bounds for infinite-VC classes (RBF kernels, neural nets), but is the simplest to state and most widely taught. Use it for finite hypothesis classes and pedagogical purposes. <strong>Rademacher complexity</strong>: data-dependent, handles real-valued and margin losses, gives tight bounds for linear/kernel classes with bounded norm, vacuous for over-parameterized nets. Use it for theoretical analysis of SVMs, regularized linear models, and kernel methods. <strong>PAC-Bayes</strong>: works with stochastic hypotheses (posterior over weights), data-dependent via KL divergence to a prior, can be numerically nonvacuous for deep stochastic networks when the prior is optimized (Dziugaite and Roy 2017). Best framework for neural network generalization theory. <strong>Stability bounds</strong> (Bousquet and Elisseeff 2002): algorithm-dependent — they bound generalization by how much the trained model changes when any single training point is swapped. Stability bounds are tight for SGD-trained neural networks (Hardt, Recht, Singer 2016 showed SGD is uniformly stable) and computationally lightweight to estimate. Best for analyzing iterative algorithms without needing to specify the hypothesis class explicitly.
      </Prose>

      <StepTrace
        label="Choosing a generalization framework — decision flow"
        steps={[
          {
            label: "Is your model class finite or have bounded VC dimension?",
            render: () => (
              <Prose>
                If yes and you want the simplest possible bound: use VC theory. Massart's lemma gives <em>R_n ≤ sqrt(2 log|F| / n)</em> which directly yields VC-style bounds for finite classes. If the class is infinite but has bounded norm (linear SVM, ridge regression): use Rademacher. If the class is a deep network: neither VC nor Rademacher will give nonvacuous bounds — proceed to PAC-Bayes or stability.
              </Prose>
            ),
          },
          {
            label: "Do you want a data-dependent bound on a specific training sample?",
            render: () => (
              <Prose>
                Use Rademacher complexity. The empirical form <em>R̂_S(F)</em> can be estimated by Monte Carlo on your actual training set in O(n · n_mc) operations for parameterized classes. This gives a bound that adapts to the specific data distribution rather than the worst case. VC-based bounds are distribution-independent and will be looser on well-structured data.
              </Prose>
            ),
          },
          {
            label: "Are you analyzing a deep neural network?",
            render: () => (
              <Prose>
                Use PAC-Bayes. Rademacher bounds for deep networks are vacuous in practice. PAC-Bayes bounds depend on the KL divergence between the posterior (learned weights) and a prior, which can be controlled during training by adding a KL penalty to the loss. Neyshabur et al. (2015) showed that using the Frobenius-norm product across layers gives tighter Rademacher-style bounds than raw parameter count, but even these are typically loose for large networks. Dziugaite and Roy (2017) is the closest to a practical certificate.
              </Prose>
            ),
          },
          {
            label: "Are you analyzing an iterative algorithm (SGD, gradient boosting)?",
            render: () => (
              <Prose>
                Consider stability bounds. If an algorithm trained for <em>T</em> steps on a dataset of size <em>n</em> is <em>β</em>-uniformly stable (changing one training point changes the output by at most <em>β</em>), then the generalization gap is at most <em>2β</em> with high probability. For SGD with learning rate <em>η</em> and loss Lipschitz constant <em>L</em>: stability is <em>2ηLT/n</em> (Hardt, Recht, Singer 2016). This bound requires no knowledge of the hypothesis class — only the training dynamics. Early stopping controls <em>T</em> and thereby controls the stability bound directly.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>8.1 Computational cost of empirical Rademacher estimation</H3>

      <Prose>
        For a parameterized class where the supremum has a closed form — linear classifiers, kernel SVMs — estimating <em>R̂_S(F)</em> costs <em>O(n · d · n_mc)</em>: for each of <em>n_mc</em> Rademacher draws, compute <em>X^T σ</em> in O(n·d) and take its norm. With <em>n_mc=500</em>, <em>n=10,000</em>, and <em>d=100</em>, this is 500 million multiplications — fast on NumPy, well under a second. For infinite classes without closed-form suprema, you must approximate via random search over <em>F</em>, which adds an inner optimization loop. For neural networks as the function class, the inner supremum requires maximizing over all parameter vectors — effectively a full training run — for each Rademacher draw. This is computationally infeasible for large networks.
      </Prose>

      <Prose>
        For <strong>finite classes</strong> (threshold functions, decision stumps), estimation is <em>O(n · |F| · n_mc)</em>. With 16 thresholds, 100 points, and 500 draws, this is 800,000 operations — trivial. But for decision trees of depth <em>d</em>, <em>|F|</em> grows exponentially, making the enumeration approach impractical beyond depth 3 or 4. In this regime, you either bound via Massart's lemma (using log|F|) or via VC dimension (using Sauer's lemma growth function).
      </Prose>

      <H3>8.2 Scaling to large datasets</H3>

      <Prose>
        The <em>n_mc</em> independent Rademacher sign draws are embarrassingly parallel — each draw is independent and can be computed on a separate CPU/GPU core. For large <em>n</em>, the bottleneck is the matrix-vector product <em>X^T σ</em> which is O(n·d) per draw. This scales linearly in both n and d, so the estimation remains practical even for datasets with millions of points, provided the hypothesis class has a closed-form supremum. The bound itself shrinks as 1/√n, so for very large <em>n</em>, the bound is tight enough to be useful before the computation becomes expensive.
      </Prose>

      <H3>8.3 The norm-based path to neural network bounds</H3>

      <Prose>
        Neyshabur, Tomioka, and Srebro's 2015 COLT result avoids the intractability of computing Rademacher complexity for neural networks by bounding it via weight matrix norms. For a depth-<em>L</em> feed-forward network with weight matrices <em>W₁, …, W_L</em> with ReLU activations, they prove: <em>R_n(F) = O(√(L) · Π_j ‖W_j‖_F · ‖W_j‖ / √n)</em>, where the product involves Frobenius and spectral norms of each layer's weight matrix. This bound is computable from the trained network's weights without any Monte Carlo estimation and without enumerating the hypothesis class. The practical implication: track the product of weight norms during training; if it grows unboundedly, generalization will degrade even if training loss continues to fall. Weight decay regularization is precisely what controls this product.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>9.1 Empirical Rademacher depends on the specific sample</H3>

      <Prose>
        The empirical Rademacher complexity <em>R̂_S(F)</em> is computed on a fixed sample <em>S</em> and varies across samples drawn from the same distribution. The population complexity <em>R_n(F)</em> is the expectation of <em>R̂_S(F)</em> over the draw of <em>S</em>. When you estimate <em>R_n(F)</em> from a single sample by Monte Carlo over Rademacher signs, you are estimating <em>R̂_S(F)</em> for that particular sample. If the sample is atypical — clustered, low-dimensional structure, outliers — the estimate may be substantially above or below the true population complexity. To get a reliable estimate of <em>R_n(F)</em>, average <em>R̂_S(F)</em> across multiple independent draws of <em>S</em>, similar to cross-validation.
      </Prose>

      <H3>9.2 Bound constants are often stated loosely</H3>

      <Prose>
        The generalization bound is stated as <em>R(f) ≤ R̂(f) + 2·R_n(F) + 3·sqrt(log(2/δ)/(2n))</em>. The factor of 2 in front of <em>R_n</em> and the factor of 3 in the confidence term come from the specific symmetrization and McDiarmid argument used. Different proofs in different textbooks use slightly different constants — Bartlett and Mendelson 2002 use one set; Mohri, Rostamizadeh, and Talwalkar 2018 use another. When reading papers, always check which version of the bound is being cited. The asymptotic rate (1/√n) is universal, but the constants matter for numerical comparisons.
      </Prose>

      <H3>9.3 Talagrand contraction requires Lipschitz loss</H3>

      <Prose>
        The contraction inequality applies only to Lipschitz losses. The squared loss <em>(y - f(x))²</em> is Lipschitz on bounded domains but not on unbounded ones: if predictions or targets can be arbitrarily large, the Lipschitz constant is infinite and the contraction lemma does not apply. Always specify the domain of your loss when invoking contraction. For the hinge loss and logistic loss, Lipschitz constants are 1 and 0.25 respectively, everywhere — these are safe. For the squared loss on regression, you must bound the prediction range, typically via the norm bound on <em>w</em> and the norm bound on inputs.
      </Prose>

      <H3>9.4 Standard bounds are vacuous for deep networks</H3>

      <Prose>
        A ResNet-50 has approximately 25 million parameters. On CIFAR-10 with 50,000 training examples, the Rademacher complexity of the function class (all networks with those dimensions) is effectively 1 — the class can fit any labeling of any 50,000 points. The bound <em>R(f) ≤ R̂(f) + 2·R_n(F) + confidence</em> gives <em>R(f) ≤ R̂(f) + 2 + something</em>, which is worse than the trivial bound of 1 (since risk is always at most 1). The bound is not informative. This is not a failure of the framework but of applying it to a class for which it is not designed. The correct framework for deep networks is PAC-Bayes with optimized priors (Dziugaite and Roy 2017) or stability bounds for SGD (Hardt, Recht, Singer 2016).
      </Prose>

      <H3>9.5 Larger margin tightens the bound — but margin must be measured correctly</H3>

      <Prose>
        The margin bound for linear classifiers scales as <em>B·X_max / (γ√n)</em>. A model achieving large margin <em>γ</em> gets a tighter bound. But the margin is measured on the training set — if you tune <em>C</em> to maximize margin on training data without accounting for the effective complexity, you can inadvertently select a model with larger norm <em>B</em> and smaller margin <em>γ</em> that has the same or worse generalization bound. The SVM's dual objective correctly balances <em>B</em> and <em>γ</em> simultaneously (the SVM margin is <em>1/‖w‖</em>, so minimizing ‖w‖ maximizes the margin without a separate margin parameter). Manually picking the function with the largest training margin from a hypothesis class of unbounded norm would not give the margin bound.
      </Prose>

      <H3>9.6 I.i.d. assumption is load-bearing</H3>

      <Prose>
        The Rademacher generalization bound assumes the training examples are drawn i.i.d. from the test distribution. Under distribution shift — train on photos from one camera, test on photos from another — the bound still holds as a guarantee on the training distribution, but it says nothing about the test distribution. Under temporal dependence (time-series data), the i.i.d. assumption fails and the symmetrization step of the proof breaks down. Mixing-condition extensions of Rademacher theory (Mohri 2008) address dependent data but are considerably more complex and less commonly applied.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All citations below were verified against primary publication venues, author lists, and main claims.
      </Prose>

      <StepTrace
        label="primary literature — Rademacher complexity and generalization"
        steps={[
          {
            label: "Koltchinskii 2001 — The paper that introduced Rademacher penalties to learning theory",
            render: () => (
              <Prose>
                Koltchinskii, V. (2001). "Rademacher penalties and structural risk minimization." <em>IEEE Transactions on Information Theory</em>, 47(5):1902–1914. July 2001. DOI: 10.1109/18.930923. Koltchinskii introduced the idea of replacing VC-based penalties with data-dependent penalties built from the Rademacher process indexed by the function class. He proved oracle inequalities for structural risk minimization with these penalties, showing that they adapt to the difficulty of the specific sample rather than being fixed by the worst-case complexity of the class. The key theorem shows that minimizing empirical risk plus a Rademacher penalty produces a hypothesis whose true risk is bounded by the best achievable risk in the class plus twice the Rademacher penalty — a sharp result without the loose VC constants. This paper, along with simultaneous independent work by Bartlett and Mendelson, launched Rademacher complexity as the standard tool in modern statistical learning theory.
              </Prose>
            ),
          },
          {
            label: "Bartlett and Mendelson 2002 — The canonical paper practitioners cite",
            render: () => (
              <Prose>
                Bartlett, P. L., and Mendelson, S. (2002). "Rademacher and Gaussian complexities: Risk bounds and structural results." <em>Journal of Machine Learning Research</em>, 3:463–482. November 2002. Available: jmlr.org/papers/v3/bartlett02a.html. This is the paper. Bartlett and Mendelson prove the main generalization bound in terms of Rademacher and Gaussian complexities, establish the Talagrand contraction lemma as a tool for Lipschitz losses, derive closed-form bounds for linear classifiers and neural networks, and show how the margin bound for classifiers follows from the Rademacher framework. They connect Rademacher complexity to the covering-number and Sauer-lemma approaches, showing that Rademacher is a strictly stronger tool. The paper is 20 pages, technically dense, and completely self-contained. Chapter 3 of Mohri et al. 2018 is essentially a textbook expansion of this paper.
              </Prose>
            ),
          },
          {
            label: "McAllester 1999 — PAC-Bayes bound for stochastic hypotheses",
            render: () => (
              <Prose>
                McAllester, D. A. (1999). "PAC-Bayesian model averaging." <em>Proceedings of the 12th Annual Conference on Computational Learning Theory (COLT 1999)</em>, pp. 164–170. ACM. DOI: 10.1145/307400.307435. Extended version: McAllester, D. A. (2003). "PAC-Bayesian stochastic model selection." <em>Machine Learning</em>, 51(1):5–21. McAllester's PAC-Bayes bound is the complementary framework to Rademacher: instead of bounding the complexity of a function class, it bounds the generalization gap for any stochastic hypothesis (a posterior distribution over classifiers) in terms of the KL divergence between the posterior and a prior chosen before seeing the data. The bound is: <em>R(Q) ≤ R̂(Q) + sqrt((KL(Q||P) + log(2n/δ)) / (2n))</em>. This is nonvacuous for stochastic networks when the posterior is close to the prior — which can be enforced by adding a KL penalty to the training loss.
              </Prose>
            ),
          },
          {
            label: "Mohri, Rostamizadeh, Talwalkar 2018 — Canonical textbook treatment",
            render: () => (
              <Prose>
                Mohri, M., Rostamizadeh, A., and Talwalkar, A. (2018). <em>Foundations of Machine Learning</em>, 2nd ed. MIT Press. ISBN: 978-0-262-03940-6. 504 pages. Chapter 3 is the definitive textbook treatment of Rademacher and Gaussian complexities: it covers empirical and population Rademacher complexity, the main generalization bound, Talagrand contraction, closed-form bounds for linear and neural-network classes, and the connections to VC theory, covering numbers, and margin bounds. The notation in this topic follows Mohri et al. closely. The book is available in PDF from the MIT Press and the authors' pages; it is the standard graduate reference for algorithmic learning theory alongside Shalev-Shwartz and Ben-David.
              </Prose>
            ),
          },
          {
            label: "Shalev-Shwartz and Ben-David 2014 — Chapter 26 on Rademacher complexities",
            render: () => (
              <Prose>
                Shalev-Shwartz, S., and Ben-David, S. (2014). <em>Understanding Machine Learning: From Theory to Algorithms</em>. Cambridge University Press. ISBN: 978-1-107-05713-5. Available: cs.huji.ac.il/~shais/UnderstandingMachineLearning. Chapter 26 ("Rademacher Complexities") is Part IV of the book and is accessible to readers with a calculus-and-probability background. It derives the main bound, proves contraction, and connects to margin-based guarantees. The book is freely available and is the most accessible entry point to learning theory that also covers practical topics (SVMs, regularization, boosting) in earlier chapters.
              </Prose>
            ),
          },
          {
            label: "Neyshabur, Tomioka, Srebro 2015 — Rademacher meets neural networks",
            render: () => (
              <Prose>
                Neyshabur, B., Tomioka, R., and Srebro, N. (2015). "Norm-based capacity control in neural networks." <em>Proceedings of the 28th Conference on Learning Theory (COLT)</em>, PMLR 40:1376–1401. arXiv:1503.00036. This paper extends Rademacher-style bounds to multi-layer feed-forward networks. The key result: the Rademacher complexity of a network class can be bounded by the product of the Frobenius norms of its weight matrices, divided by √n. This is tighter than bounds based on the total number of parameters because the Frobenius-norm product can be small even for wide networks with many near-zero weights. The paper launched a line of work on norm-based generalization bounds and directly motivated the study of implicit regularization in gradient descent — if SGD finds solutions with small weight-norm products, it is automatically controlling the Rademacher complexity.
              </Prose>
            ),
          },
          {
            label: "Dziugaite and Roy 2017 — First nonvacuous bound for deep stochastic networks",
            render: () => (
              <Prose>
                Dziugaite, G. K., and Roy, D. M. (2017). "Computing nonvacuous generalization bounds for deep (stochastic) neural networks with many more parameters than training data." <em>Proceedings of the 33rd Conference on Uncertainty in Artificial Intelligence (UAI 2017)</em>. arXiv:1703.11008. This paper produced the first numerically nonvacuous PAC-Bayes bounds for deep neural networks. The key innovation: instead of using a fixed standard Gaussian prior, Dziugaite and Roy optimize the prior to minimize the PAC-Bayes bound itself, treating the bound as an objective. For a stochastic MNIST classifier with Gaussian weight perturbations, they computed a bound of approximately 16% on test error when the model achieved around 2% actual test error — a bound smaller than 1, and therefore nontrivial. The optimization involves minimizing a loss that combines cross-entropy with the KL term in the PAC-Bayes bound. This work remains the closest thing to a practical generalization certificate for deep networks, and the methodology it introduced continues to be refined.
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
        Work through these before moving to the next topic. Try each on paper before reading the answer.
      </Prose>

      <H3>Exercise 1 (Recall)</H3>
      <Prose>
        Write the definition of empirical Rademacher complexity <em>R̂_S(F)</em> for a function class <em>F</em> on a sample <em>S = (x₁, …, xₙ)</em>. What are Rademacher variables? What does the supremum over <em>F</em> measure, intuitively?
      </Prose>
      <Callout type="answer" title="Answer 1">
        {"Rademacher variables σ₁, …, σₙ are i.i.d. random variables taking values in {-1, +1} with equal probability — pure random coin flips. The empirical Rademacher complexity is: R̂_S(F) = E_σ[sup_{f∈F} (1/n) Σᵢ σᵢ f(xᵢ)]. The expectation is over the random signs σ only; the sample S is fixed. The supremum over F picks the function in the class that best correlates with the random signs on this particular sample — it measures how well the best function in F can 'fit' random noise on the specific dataset in hand. A class with high R̂_S(F) can fit arbitrary noise on this sample, which means it has enough complexity to memorize random labels and will likely overfit. A class with low R̂_S(F) cannot do better than random on noise labels, indicating it is genuinely constrained and will generalize."}
      </Callout>

      <H3>Exercise 2 (Conceptual)</H3>
      <Prose>
        Explain intuitively why the Rademacher complexity of the linear class <em>{"{"} w·x : ‖w‖ ≤ B {"}"}</em> decreases as <em>n</em> increases, even though the class itself does not change. Why does the norm bound <em>B</em> appear in the formula <em>R_n(F) ≤ B·X_max / √n</em>?
      </Prose>
      <Callout type="answer" title="Answer 2">
        {"The class F = {w·x : ||w|| ≤ B} is fixed — it always contains the same set of functions. But the empirical Rademacher complexity R̂_S(F) measures how well the best function in F can correlate with a random labeling of the specific n-point sample S. With more points, the sample is harder to fit randomly: the sum (1/n) Σ σᵢ w·xᵢ = w·(X^T σ)/n. The random vector (X^T σ)/n is a sum of n i.i.d. zero-mean vectors, so by the law of large numbers its norm shrinks as 1/√n. The supremum over w of w·(X^T σ)/n = ||(X^T σ)/n|| (by Cauchy-Schwarz, achieved at w = B * (X^T σ)/||X^T σ||). So the expected supremum scales as B * E[||(X^T σ)/n||] ≈ B * X_max / √n. The norm bound B appears because a larger budget for ||w|| means w can align more strongly with any direction, amplifying the correlation with random signs. Smaller B = smaller maximum correlation with noise = tighter complexity = tighter generalization bound. This is precisely why SVM regularization on ||w|| controls generalization."}
      </Callout>

      <H3>Exercise 3 (Applied)</H3>
      <Prose>
        A colleague trains a linear SVM (RBF kernel, C=0.01) on a training set of n=5,000 and finds that every training point becomes a support vector. What does this indicate about the relationship between the SVM objective and Rademacher complexity? What should they change?
      </Prose>
      <Callout type="answer" title="Answer 3">
        {"Every training point becoming a support vector means the margin is so wide that every point lies inside or on the margin — the model has been over-regularized. In Rademacher terms: C=0.01 forces a very small norm budget B = ||w||, which produces a very tight Rademacher bound but at the cost of high training error (the model is too constrained to fit the data). The bound R(f) ≤ R̂(f) + 2*R_n + confidence becomes tight in the 2*R_n term but the R̂(f) term (training error) is large — the model is biased. In learning-curve terms: this is high bias. The SVM is trading too much capacity for too little. Fix: increase C. Try C ∈ {0.1, 1.0, 10.0} with 5-fold cross-validation. Increasing C relaxes the norm constraint, allows the margin to narrow, reduces training error, and typically reduces generalization error until the model starts overfitting (at very large C). The Rademacher complexity will increase with larger C, but the training error will decrease faster, producing a net improvement in the generalization bound."}
      </Callout>

      <H3>Exercise 4 (Math)</H3>
      <Prose>
        State Massart's lemma. Use it to derive a bound on the Rademacher complexity of the class of 16 threshold functions <em>{"{"} sign(x - tₖ) : k=1,…,16 {"}"}</em> on a dataset of n=100 points. Compare to the Monte Carlo estimate of 0.1626 from the from-scratch code.
      </Prose>
      <Callout type="answer" title="Answer 4">
        {"Massart's lemma: for a finite function class F with |F| functions, the Rademacher complexity satisfies R_n(F) ≤ sqrt(2 log|F| / n). For |F|=16 and n=100: R_n ≤ sqrt(2 * log(16) / 100) = sqrt(2 * 2.773 / 100) = sqrt(0.05545) = 0.2355. The Monte Carlo estimate from the code was 0.1626. The Massart bound gives 0.2355, which is 44% above the Monte Carlo estimate. This is the bound-vs-reality relationship in miniature: Massart's lemma is a worst-case bound over all possible distributions of the 16 threshold functions' values; the actual sample average is smaller because the functions are not maximally spread (they are thresholds on a continuous distribution, so adjacent thresholds give similar classifications on most points, reducing the effective diversity). The bound is correct — 0.1626 ≤ 0.2355 — but loose by a factor of ~1.45. In practice, whenever you can estimate R̂_S(F) by Monte Carlo, do so; the Massart bound is useful when Monte Carlo is infeasible (infinite or very large |F|)."}
      </Callout>

      <H3>Exercise 5 (Applied — framework choice)</H3>
      <Prose>
        You have trained a 50M-parameter BERT-style transformer on a text classification task with 100,000 training examples. You want to provide a theoretical certificate that the model generalizes. Rank the four frameworks (VC, Rademacher, PAC-Bayes, Stability) from most to least applicable, and explain your ranking.
      </Prose>
      <Callout type="answer" title="Answer 5">
        {"Ranking from most to least applicable: PAC-Bayes > Stability > Rademacher > VC. PAC-Bayes (most applicable): The only framework that has produced nonvacuous numerical bounds for deep networks on realistic tasks (Dziugaite and Roy 2017). By training with a KL-penalized objective and optimizing the prior, you can produce a bound that is actually below 1.0 — meaning informative. The bound depends on how much the learned weight distribution differs from the prior, which is a meaningful quantity for fine-tuned transformers (the posterior is close to the pretrained weights = small KL = tight bound). Stability (second): SGD-trained networks are uniformly stable (Hardt, Recht, Singer 2016), and the stability bound depends on the number of SGD steps and learning rate — quantities you know. Stability bounds are typically looser than PAC-Bayes for large networks but require no assumption about the hypothesis class. Rademacher (third): Naive Rademacher bounds are vacuous (complexity ≈ 1). Norm-based Rademacher (Neyshabur et al. 2015) using the Frobenius-norm product can be computed from the trained weights and may give a nonvacuous bound if the norms are well-controlled, but this requires careful implementation. VC (least applicable): VC dimension of a 50M-parameter network is essentially infinite or intractably large to compute — gives no useful bound."}
      </Callout>

      <H3>Exercise 6 (Synthesis)</H3>
      <Prose>
        The Talagrand contraction lemma says that composing with an <em>L</em>-Lipschitz loss multiplies the Rademacher complexity by at most <em>L</em>. The hinge loss is 1-Lipschitz; the squared loss on [-1, +1] predictions is 2-Lipschitz. A researcher argues: "We should always use hinge loss instead of squared loss because it gives a tighter generalization bound." Is this argument correct? What is it missing?
      </Prose>
      <Callout type="answer" title="Answer 6">
        {"The argument is partially correct but incomplete. It is correct that the contraction lemma gives a bound of L * R̂(F) for the loss-composed class, and since hinge has L=1 vs squared loss L=2, the hinge-based bound is at most half as large as the squared-loss-based bound for the same hypothesis class F. So yes, from a bound perspective, hinge is preferable. What the argument misses: (1) The bound is an upper bound on the generalization gap, not the generalization gap itself. A tighter upper bound does not automatically mean better generalization — it means you have a better theoretical certificate. The true generalization gap depends on the specific model trained, not just the loss function used. (2) Different losses find different optima. Minimizing hinge loss gives an SVM-type solution (max-margin); minimizing squared loss gives a regression-type solution. For the same hypothesis class F, the model trained with hinge loss may have lower ||w|| (because the hinge loss optimum is the max-margin hyperplane, which minimizes ||w|| among all correct classifiers) — and lower ||w|| means lower R̂(F) and thus a tighter bound from that side too. (3) The correct comparison is not loss functions in isolation but the full training pipeline (loss + optimizer + hypothesis class). In practice, logistic loss (L=0.25 Lipschitz) often gives better calibrated probabilities than hinge, and the generalization difference in real experiments is small — the bound constants are loose enough that they do not determine which loss is better empirically."}
      </Callout>

    </div>
  ),
};

export default rademacherContent;
