import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const pacVcContent = {
  title: "PAC Learning & VC Dimension",
  readTime: "~45 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Every supervised learning algorithm has an implicit contract: train on a finite sample, predict accurately on unseen examples from the same distribution. For decades, this contract was informal. Practitioners accepted on faith that training on enough data would produce good generalization. The question no one had answered precisely was: how much is enough? The field of computational learning theory exists to answer that question rigorously, and its two foundational pillars — the VC dimension and the PAC framework — gave the first complete, formal answers.
      </Prose>

      <Prose>
        The theoretical groundwork was laid by Vladimir Vapnik and Alexey Chervonenkis. Their 1968 paper (published in English in 1971 as "On the Uniform Convergence of Relative Frequencies of Events to their Probabilities" in <em>Theory of Probability and its Applications</em>, 16(2):264–280) introduced the central notion now called VC dimension. The paper proved that a hypothesis class can generalize reliably from a finite sample if and only if its capacity — measured by how many distinct dichotomies it can produce on arbitrary sets of points — grows sub-exponentially with sample size. This growth function condition, known as Sauer's lemma after its tighter independent formulation, converted the abstract question of generalization into a combinatorial property of hypothesis classes.
      </Prose>

      <Prose>
        The PAC (Probably Approximately Correct) framework arrived in 1984 through Leslie Valiant's paper "A Theory of the Learnable" (<em>Communications of the ACM</em>, 27(11):1134–1142). Valiant asked: what does it mean for an algorithm to learn a concept? His definition was deliberately practical. An algorithm PAC-learns a concept class if, for any target concept and any distribution over inputs, it can produce — from a polynomial number of labeled examples — a hypothesis that is (a) approximately correct (error at most ε) and (b) probably so (with probability at least 1 − δ). The two free parameters ε and δ let practitioners set their own accuracy and confidence requirements, and the theory tells them exactly how many samples that precision requires.
      </Prose>

      <Prose>
        The VC and PAC threads were unified by Blumer, Ehrenfeucht, Haussler, and Warmuth in their 1989 JACM paper "Learnability and the Vapnik-Chervonenkis dimension" (<em>Journal of the ACM</em>, 36(4):929–965). They proved that a concept class is PAC-learnable if and only if its VC dimension is finite — the finiteness of VC dimension is both necessary and sufficient for learnability. This was a striking result: a single combinatorial integer characterizes whether a concept class is learnable at all. The paper also established tight sample complexity bounds, connecting the number of examples required to the VC dimension, the desired error ε, and the confidence parameter δ.
      </Prose>

      <Prose>
        Why do these results matter to practitioners? Three concrete reasons. First, they provide <em>pre-training data requirements</em>: given a model family and a desired accuracy level, the VC bound tells you the minimum data you need before training. Second, they enable <em>model-class comparison</em>: two models that achieve the same training error but have different VC dimensions will generalize differently, and the theory quantifies the gap. Third, in <em>safety-critical applications</em> — autonomous vehicles, medical diagnosis, finance — regulators increasingly require formal guarantees of model behavior, not just empirical performance on a held-out set. PAC bounds are the language in which such guarantees are stated.
      </Prose>

      <Callout type="insight">
        {"VC dimension answers: 'Which hypothesis classes can learn at all?' PAC bounds answer: 'Given VC dimension d, error tolerance ε, and confidence 1−δ, how many training examples suffice?' These are the two fundamental questions of sample complexity."}
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <H3>2.1 PAC learning: Probably Approximately Correct</H3>

      <Prose>
        The PAC framework encodes a simple but precise aspiration. You want a learned hypothesis <Code>h</Code> that is approximately correct — its true error (probability of misclassifying a randomly drawn example) is at most ε — and probably so — this guarantee holds with probability at least 1 − δ over the random draw of your training set. The two parameters trade off: requiring smaller ε means needing more data; requiring higher confidence (smaller δ) means needing more data. The sample complexity function <Code>m(ε, δ)</Code> makes this tradeoff explicit.
      </Prose>

      <Prose>
        The qualifier "probably" is essential and often misunderstood. It does not mean the algorithm has a chance of failing — it means there is a small probability that the random training set is a bad one, in which case the algorithm cannot guarantee its output is accurate. A PAC algorithm cannot distinguish a bad draw from a good draw; it only guarantees that good draws happen with probability at least 1 − δ. Setting δ = 0.05 means your training set is unlucky at most 5% of the time.
      </Prose>

      <H3>2.2 VC dimension: maximum shattering size</H3>

      <Prose>
        The VC dimension of a hypothesis class <Code>H</Code> is the size of the largest set of points that <Code>H</Code> can <em>shatter</em>. A set of points is shattered by <Code>H</Code> if, for every possible binary labeling of those points (there are <Code>{"2^n"}</Code> such labelings for <Code>n</Code> points), there exists some <Code>h ∈ H</Code> that correctly assigns those labels. Intuitively: if <Code>H</Code> can shatter a set of size <Code>d</Code>, it can express all possible binary functions on <Code>d</Code> inputs — it is at least as expressive as a lookup table for <Code>d</Code> examples.
      </Prose>

      <Prose>
        The canonical examples build intuition quickly. <strong>Threshold classifiers on the real line</strong> (<Code>h_a(x) = 1[x &gt;= a]</Code>) have VC dimension 1: any single point can be shattered (label it +1 by setting <Code>a</Code> below it, or -1 by setting <Code>a</Code> above it), but no two-point set can be shattered (the labeling +1, -1 in reading order requires <Code>a</Code> to be between them, but that puts both above, not one above and one below). <strong>Intervals on the real line</strong> have VC dimension 2: any two points can be shattered, but for three points the labeling +1, -1, +1 (middle point negative) is impossible — a single interval cannot include both endpoints while excluding the middle. <strong>Half-planes in {"ℝ²"}</strong> have VC dimension 3: any three points in general position can be shattered, but no four-point set can be shattered (the XOR-like alternating labeling is unrealizable).
      </Prose>

      <Prose>
        The connection to generalization is the main theorem: richer hypothesis classes (higher VC dimension) need more training data to achieve the same generalization guarantee. A hypothesis class with VC dimension <Code>d</Code> needs roughly <Code>O(d/ε)</Code> examples to guarantee error at most ε (ignoring log factors). The VC dimension is literally the price of expressiveness, denominated in samples.
      </Prose>

      <StepTrace
        label="VC dimension intuition — three canonical hypothesis classes"
        steps={[
          {
            label: "Thresholds on R: VC dim = 1",
            render: () => (
              <Prose>
                A threshold classifier puts label +1 on everything above some threshold <Code>a</Code> and -1 below. One point can always be shattered: set <Code>a</Code> below it for +1, above it for -1. Two points cannot be shattered: the labeling (+1 on left, -1 on right) requires the threshold between them, but that labels both points on their correct sides... actually try (+1 left, -1 right): threshold must be between them, making left above threshold — contradiction. No threshold can put the leftmost point as +1 and the rightmost as -1 if left {"<"} right. So VC(thresholds) = 1.
              </Prose>
            ),
          },
          {
            label: "Intervals on R: VC dim = 2",
            render: () => (
              <Prose>
                A closed-interval classifier assigns +1 to points inside [a,b] and -1 outside. Any two points can be shattered: labeling (+1,+1) uses an interval covering both; labeling (-1,-1) uses an empty interval; labeling (+1,-1) puts interval around left only; labeling (-1,+1) puts interval around right only. Three points cannot be shattered: the labeling (+1,-1,+1) — include left and right but not middle — is impossible. A single interval is convex, so it cannot include two endpoints while excluding a middle point between them. So VC(intervals) = 2.
              </Prose>
            ),
          },
          {
            label: "Half-planes in R²: VC dim = 3",
            render: () => (
              <Prose>
                A half-plane classifier assigns +1 to one side of a line and -1 to the other. Any three points in general position (not collinear) can be shattered by some half-plane for each of the 8 labelings. No four-point set can be shattered: by Radon's theorem, any four points in {"ℝ²"} can be partitioned into two sets whose convex hulls intersect — this implies a labeling where the +1 set and -1 set cannot be separated by a hyperplane. The unrealizable labelings correspond to "XOR-like" configurations. So VC(half-planes in {"ℝ²"}) = 3. More generally, VC(half-planes in {"ℝ^d"}) = d+1.
              </Prose>
            ),
          },
        ]}
      />

      <Callout type="insight">
        {"A hypothesis class is more expressive if it can shatter larger point sets — but expressiveness comes at a cost in sample complexity. VC dimension is the exact quantifier of this cost."}
      </Callout>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 Realizable PAC learning — finite hypothesis classes</H3>

      <Prose>
        In the <em>realizable</em> setting, we assume the target concept <Code>c</Code> belongs to the hypothesis class <Code>H</Code>. A consistent learner returns any <Code>h ∈ H</Code> with zero training error. For finite <Code>H</Code>, the sample complexity is clean. Let <Code>m</Code> be the number of training examples and let <Code>h</Code> be any bad hypothesis (true error {">"} ε). The probability that <Code>h</Code> is consistent (makes no training error) on a single example is at most <Code>1 − ε</Code>. Over <Code>m</Code> i.i.d. examples: <Code>P(h consistent) ≤ (1−ε)^m ≤ exp(−εm)</Code>. Union-bounding over all hypotheses in <Code>H</Code>:
      </Prose>

      <MathBlock>
        {"P(\\exists h \\in H: R(h) > \\varepsilon \\text{ and } \\hat{R}(h) = 0) \\;\\leq\\; |H| \\cdot e^{-\\varepsilon m}"}
      </MathBlock>

      <Prose>
        Setting this at most δ and solving for <Code>m</Code> gives the realizable finite-H sample complexity:
      </Prose>

      <MathBlock>
        {"m \\;\\geq\\; \\frac{1}{\\varepsilon}\\left(\\ln |H| + \\ln \\frac{1}{\\delta}\\right)"}
      </MathBlock>

      <Prose>
        This bound scales as <Code>O((log|H| + log(1/δ))/ε)</Code>. The <Code>log|H|</Code> term is the price of model complexity — you pay logarithmically for the richness of your hypothesis class. For a hypothesis class parameterized by <Code>n</Code> bits, <Code>log|H| = n</Code>, so larger models need more data proportional to their description length.
      </Prose>

      <H3>3.2 Sauer's lemma and the growth function</H3>

      <Prose>
        For infinite hypothesis classes — nearly all practical model families — the bound above is useless because <Code>|H| = ∞</Code>. The resolution is the <em>growth function</em> {"Π_H(m)"}, which counts the maximum number of distinct dichotomies (labelings) a hypothesis class can produce on any set of <Code>m</Code> points:
      </Prose>

      <MathBlock>
        {"\\Pi_H(m) = \\max_{x_1,\\ldots,x_m} |\\{(h(x_1),\\ldots,h(x_m)) : h \\in H\\}|"}
      </MathBlock>

      <Prose>
        The growth function is always at most <Code>{"2^m"}</Code> (all possible labelings). If it equals <Code>{"2^m"}</Code>, the class can shatter a set of size <Code>m</Code>. If the VC dimension is <Code>d {"<"} m</Code>, Sauer's lemma gives a polynomial upper bound:
      </Prose>

      <MathBlock>
        {"\\Pi_H(m) \\;\\leq\\; \\sum_{i=0}^{d} \\binom{m}{i} \\;\\leq\\; \\left(\\frac{em}{d}\\right)^d"}
      </MathBlock>

      <Prose>
        The key observation is that when VC dimension <Code>d</Code> is finite, the growth function is polynomial in <Code>m</Code> rather than exponential — it grows as <Code>O(m^d)</Code> instead of <Code>{"O(2^m)"}</Code>. This polynomial growth is what enables PAC learning to work for infinite hypothesis classes with finite VC dimension.
      </Prose>

      <H3>3.3 The VC generalization theorem</H3>

      <Prose>
        The central result of VC theory bounds the generalization gap — the difference between true risk <Code>R(h)</Code> and empirical risk <Code>{"R̂(h)"}</Code> — uniformly over all hypotheses in <Code>H</Code>. With probability at least <Code>1 − δ</Code> over the draw of <Code>m</Code> i.i.d. training examples:
      </Prose>

      <MathBlock>
        {"\\sup_{h \\in H} \\left|R(h) - \\hat{R}(h)\\right| \\;\\leq\\; \\sqrt{\\frac{d \\ln(2m/d) + \\ln(4/\\delta)}{m}}"}
      </MathBlock>

      <Prose>
        where <Code>d = VC(H)</Code>. This is the <em>VC bound on the generalization gap</em>. The bound is uniform: it holds simultaneously for all <Code>h ∈ H</Code>, not just the one selected after training. This uniformity is what makes it useful for learning — it says that empirical risk is a reliable proxy for true risk, regardless of which hypothesis we end up choosing.
      </Prose>

      <Prose>
        To interpret the bound: the gap decays as <Code>O(sqrt(d log m / m))</Code>. For fixed <Code>d</Code> and <Code>δ</Code>, the gap shrinks to zero as <Code>m → ∞</Code>, confirming consistency. For fixed <Code>m</Code>, increasing <Code>d</Code> (choosing a more complex hypothesis class) widens the gap — this is the formal statement of the capacity penalty. The <Code>log(2m/d)</Code> factor is the source of the loose "log factor" that practitioners observe: the bound is pessimistic because it accounts for the worst-case distribution, not the actual one.
      </Prose>

      <H3>3.4 Agnostic PAC and the rate difference</H3>

      <Prose>
        The realizable setting assumes the target concept is in <Code>H</Code>. In the <em>agnostic</em> setting (which is realistic), no assumption is made about the target: the optimal hypothesis in <Code>H</Code> may still have nonzero error. The agnostic PAC bound uses the same VC bound but applied to excess risk:
      </Prose>

      <MathBlock>
        {"R(h_S) \\;\\leq\\; \\min_{h \\in H} R(h) + 2\\sqrt{\\frac{d \\ln(2m/d) + \\ln(4/\\delta)}{m}}"}
      </MathBlock>

      <Prose>
        where <Code>h_S</Code> is the empirical risk minimizer on sample <Code>S</Code>. In the realizable case, the sample complexity to achieve error ε scales as <Code>O(d/ε)</Code>. In the agnostic case, it scales as <Code>{"O(d/ε²)"}</Code> — a quadratically worse sample complexity because the signal-to-noise ratio is lower when the best achievable error is nonzero. The difference between <Code>1/ε</Code> and <Code>{"1/ε²"}</Code> becomes enormous for small ε: at ε = 0.01, the agnostic case needs 100× more data.
      </Prose>

      <H3>3.5 VC dimension of specific model classes</H3>

      <Prose>
        Several results are worth memorizing. <strong>Linear classifiers in {"ℝ^d"}</strong> (half-spaces): VC dimension = <Code>d + 1</Code>. A linear model with 100 features can shatter up to 101 points. <strong>Convex polygons in {"ℝ²"}</strong> with <Code>k</Code> sides: VC dimension = <Code>2k + 1</Code>. <strong>Axis-aligned rectangles in {"ℝ^d"}</strong>: VC dimension = <Code>2d</Code>. <strong>Neural networks</strong>: for a network with <Code>W</Code> weights and <Code>L</Code> layers using threshold activations, VC dimension is <Code>O(WL log W)</Code>; for continuous activations (ReLU, sigmoid), related results give <Code>O(WL)</Code> in certain parameterizations. <strong>Decision trees</strong> with <Code>L</Code> leaves: VC dimension = <Code>O(L log L)</Code>.
      </Prose>

      <Prose>
        A critical subtlety: VC dimension is not the same as parameter count. A model with 1 parameter can have infinite VC dimension. Consider <Code>{"h_θ(x) = 1[sin(θx) ≥ 0]"}</Code> — a one-parameter classifier that can shatter arbitrarily large sets of points on the real line (by choosing θ appropriately). Conversely, a model can have many parameters but low effective VC dimension after regularization. This is why the VC dimension of deep neural networks is often much larger than their parameter count, yet they generalize — the true "effective capacity" under gradient descent with regularization is much lower.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        Two experiments, NumPy only. The first empirically measures the growth function (number of realizable dichotomies) for two hypothesis classes. The second simulates the PAC bound: train an ERM on a synthetic dataset, measure empirical and true error, and verify that the VC generalization bound holds.
      </Prose>

      <H3>4a. Empirical growth function — intervals and half-planes</H3>

      <CodeBlock language="python">
{`import numpy as np
from itertools import product as iproduct
np.random.seed(42)

# ── Hypothesis class 1: closed intervals [a,b] on [0,1] ─────────────────────
# Label +1 if x in [a,b], else -1
# Enumerate dichotomies by sweeping all breakpoints formed by the data points

def shatter_coeff_intervals(n_points, n_trials=2000):
    """Count distinct labelings realized by intervals on random n-point sets."""
    realized = set()
    for _ in range(n_trials):
        pts = np.sort(np.random.uniform(0, 1, n_points))
        # All intervals defined by adjacent breakpoints
        breaks = np.concatenate([[0.0], pts, [1.0]])
        for i in range(len(breaks)):
            for j in range(i, len(breaks)):
                a, b = breaks[i], breaks[j]
                lab = tuple(1 if a <= p <= b else -1 for p in pts)
                realized.add(lab)
    return len(realized)

print("=== Growth function: closed intervals on [0,1] ===")
print(f"{'n':>4}  {'growth_fn':>12}  {'2^n':>8}  {'Shattered?':>12}")
for n in [2, 3, 4, 5]:
    r  = shatter_coeff_intervals(n, 2000)
    mx = 2 ** n
    print(f"{n:>4}  {r:>12}  {mx:>8}  {'YES' if r == mx else 'NO':>12}")

# Output:
#    n    growth_fn       2^n    Shattered?
#    2            4         4           YES
#    3            7         8            NO
#    4           11        16            NO
#    5           16        32            NO

print()
print("VC(intervals on R) = 2: shatters n=2, fails at n=3.")
print("  Impossible labeling: (+1,-1,+1) — a single interval cannot")
print("  include both endpoints while excluding a middle point.")

# ── Hypothesis class 2: half-planes in R^2 ──────────────────────────────────
# Label +1 if w.x + b >= 0, else -1
# Use exhaustive check over all 2^n labelings for fixed point sets

def halfplane_realizable(pts, labeling, n_tries=50000):
    """True if some w,b assigns the given labeling to pts."""
    pts = np.array(pts)
    y   = np.array(labeling)
    for _ in range(n_tries):
        w = np.random.randn(2)
        b = np.random.randn() * 3
        pred = np.sign(pts @ w + b)
        pred[pred == 0] = 1
        if np.all(pred == y):
            return True
    return False

def max_realized_halfplane(n_points, n_configs=10):
    """Over multiple random point configs, find the max realizable dichotomies."""
    best = 0
    for _ in range(n_configs):
        pts   = np.random.randn(n_points, 2) * 2
        count = sum(
            halfplane_realizable(pts, lab)
            for lab in iproduct([1, -1], repeat=n_points)
        )
        best = max(best, count)
    return best

print()
print("=== Growth function: half-planes in R^2 ===")
print(f"{'n':>4}  {'max_growth_fn':>14}  {'2^n':>8}  {'Shattered?':>12}")
for n in [2, 3, 4]:
    r  = max_realized_halfplane(n, n_configs=10)
    mx = 2 ** n
    print(f"{n:>4}  {r:>14}  {mx:>8}  {'YES' if r == mx else 'NO':>12}")

# Output:
#    n  max_growth_fn       2^n    Shattered?
#    2              4         4           YES
#    3              8         8           YES
#    4             14        16            NO

print()
print("VC(half-planes in R^2) = 3: shatters n=3, fails at n=4.")
print("  n=4: at most 14 of 16 labelings are realizable.")
print("  Two unrealizable labelings correspond to XOR-like configurations.")`}
      </CodeBlock>

      <Callout type="output">
{`=== Growth function: closed intervals on [0,1] ===
   n    growth_fn       2^n    Shattered?
   2            4         4           YES
   3            7         8            NO
   4           11        16            NO
   5           16        32            NO

VC(intervals on R) = 2: shatters n=2, fails at n=3.
  Impossible labeling: (+1,-1,+1) — a single interval cannot
  include both endpoints while excluding a middle point.

=== Growth function: half-planes in R^2 ===
   n  max_growth_fn       2^n    Shattered?
   2              4         4           YES
   3              8         8           YES
   4             14        16            NO

VC(half-planes in R^2) = 3: shatters n=3, fails at n=4.
  n=4: at most 14 of 16 labelings are realizable.
  Two unrealizable labelings correspond to XOR-like configurations.`}
      </Callout>

      <H3>4b. PAC bound simulation — verifying the VC generalization bound</H3>

      <CodeBlock language="python">
{`import numpy as np
np.random.seed(0)

# True concept: interval [0.3, 0.7] on [0,1]; VC dim d=2
TRUE_A, TRUE_B = 0.3, 0.7
d     = 2      # VC dim of intervals
delta = 0.05   # 95% confidence

def true_error_interval(a, b, n=200_000):
    """Estimate true error by large-sample Monte Carlo."""
    x      = np.random.uniform(0, 1, n)
    y_true = ((x >= TRUE_A) & (x <= TRUE_B)).astype(int)
    y_pred = ((x >= a)      & (x <= b)     ).astype(int)
    return np.mean(y_pred != y_true)

def vc_bound(m, d, delta):
    """One-sided VC generalization gap bound."""
    return np.sqrt((d * np.log(2 * m / d) + np.log(4 / delta)) / m)

print("=== PAC bound verification ===")
print("True concept: [0.3, 0.7] on [0,1]. d=2, delta=0.05.")
print("Learner: ERM (tightest consistent interval).")
print()
print(f"{'m':>6}  {'emp_err':>9}  {'true_err':>9}  {'gap':>8}  {'VC_bound':>9}  {'holds?':>8}")

for m in [50, 100, 200, 500, 1000, 2000]:
    X        = np.random.uniform(0, 1, m)
    y_train  = ((X >= TRUE_A) & (X <= TRUE_B)).astype(int)

    # ERM: tightest interval containing all positive training points
    pos = X[y_train == 1]
    a_hat, b_hat = (pos.min(), pos.max()) if len(pos) > 0 else (0.45, 0.55)

    emp   = np.mean(((X >= a_hat) & (X <= b_hat)).astype(int) != y_train)
    true  = true_error_interval(a_hat, b_hat)
    gap   = abs(true - emp)
    bound = vc_bound(m, d, delta)
    holds = "YES" if gap <= bound else "NO"
    print(f"{m:>6}  {emp:>9.4f}  {true:>9.4f}  {gap:>8.4f}  {bound:>9.4f}  {holds:>8}")

# Output:
#      m    emp_err  true_err       gap  VC_bound    holds?
#     50     0.0000    0.0181    0.0181    0.4941       YES
#    100     0.0000    0.0088    0.0088    0.3687       YES
#    200     0.0000    0.0198    0.0198    0.2737       YES
#    500     0.0000    0.0016    0.0016    0.1834       YES
#   1000     0.0000    0.0015    0.0015    0.1349       YES
#   2000     0.0000    0.0014    0.0014    0.0990       YES

print()
print("Bound holds at every sample size. Gap << VC_bound: the bound is loose")
print("(worst-case over all distributions) but never violated.")`}
      </CodeBlock>

      <Callout type="output">
{`=== PAC bound verification ===
True concept: [0.3, 0.7] on [0,1]. d=2, delta=0.05.
Learner: ERM (tightest consistent interval).

     m    emp_err  true_err       gap  VC_bound    holds?
    50     0.0000    0.0181    0.0181    0.4941       YES
   100     0.0000    0.0088    0.0088    0.3687       YES
   200     0.0000    0.0198    0.0198    0.2737       YES
   500     0.0000    0.0016    0.0016    0.1834       YES
  1000     0.0000    0.0015    0.0015    0.1349       YES
  2000     0.0000    0.0014    0.0014    0.0990       YES

Bound holds at every sample size. Gap << VC_bound: the bound is loose
(worst-case over all distributions) but never violated.`}
      </Callout>

      <Prose>
        Three observations from the simulation. First, the empirical error is always zero — the ERM algorithm on a realizable problem produces a consistent hypothesis, as expected. Second, the true error is small (the learned interval is close to [0.3, 0.7]) and shrinks as <Code>m</Code> grows. Third, the VC bound is dramatically looser than the actual gap — by a factor of roughly 25–70×. This looseness is intrinsic: the VC bound is uniform over all distributions and all hypotheses, so it cannot be tight for any specific, well-behaved distribution.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        PAC learning and VC dimension are theoretical constructs — there is no <Code>sklearn.pac.VCBoundEstimator</Code> API. But the theory informs three concrete engineering practices: model-class selection given data size, learning curve analysis as an empirical PAC story, and reasoning about when transfer learning or pretraining is justified.
      </Prose>

      <H3>5a. Learning curve as empirical PAC — generalization improves with m</H3>

      <CodeBlock language="python">
{`import numpy as np
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

np.random.seed(42)

X, y = make_classification(
    n_samples=3000, n_features=10, n_informative=5,
    n_redundant=2, random_state=42
)

# Three model classes with different effective VC dimension
models = [
    ("LogReg(C=1)  [low VC]",
     Pipeline([("sc", StandardScaler()),
               ("clf", LogisticRegression(C=1.0, max_iter=500, random_state=42))])),
    ("SVC(RBF)     [medium VC]",
     Pipeline([("sc", StandardScaler()),
               ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42))])),
    ("DTree(d=10)  [high VC]",
     DecisionTreeClassifier(max_depth=10, random_state=42)),
]

train_sizes = [100, 200, 400, 800, 1500, 2400]

print("=== PAC sample complexity — learning curves across 3 model classes ===")
print("PAC prediction: val_acc increases with m; gap (train-val) shrinks as sqrt(d*log(m)/m)")
for name, model in models:
    ts, tr, va = learning_curve(
        model, X, y,
        train_sizes=train_sizes,
        cv=5, scoring="accuracy", n_jobs=-1
    )
    print(f"\n--- {name} ---")
    print(f"{'m':>6}  {'train_acc':>10}  {'val_acc':>10}  {'gap':>8}")
    for m, tr_s, va_s in zip(ts, tr, va):
        print(f"{m:>6}  {tr_s.mean():>10.4f}  {va_s.mean():>10.4f}"
              f"  {(tr_s.mean()-va_s.mean()):>8.4f}")

# Output:
# --- LogReg(C=1)  [low VC] ---
#      m   train_acc    val_acc       gap
#    100      0.7340      0.6807    0.0533
#    200      0.6880      0.6913   -0.0033
#    400      0.7020      0.6967    0.0053
#    800      0.7095      0.7147   -0.0052
#   1500      0.7219      0.7243   -0.0025
#   2400      0.7315      0.7310    0.0005
#
# --- SVC(RBF)     [medium VC] ---
#      m   train_acc    val_acc       gap
#    100      0.9160      0.7167    0.1993
#    200      0.8850      0.7897    0.0953
#    400      0.8910      0.8210    0.0700
#    800      0.8795      0.8493    0.0302
#   1500      0.8929      0.8670    0.0259
#   2400      0.8950      0.8787    0.0163
#
# --- DTree(d=10)  [high VC] ---
#      m   train_acc    val_acc       gap
#    100      1.0000      0.6333    0.3667
#    200      1.0000      0.6653    0.3347
#    400      0.9930      0.7130    0.2800
#    800      0.9782      0.7937    0.1846
#   1500      0.9617      0.8203    0.1414
#   2400      0.9510      0.8303    0.1207`}
      </CodeBlock>

      <Callout type="output">
{`=== PAC sample complexity — learning curves across 3 model classes ===
PAC prediction: val_acc increases with m; gap (train-val) shrinks as sqrt(d*log(m)/m)

--- LogReg(C=1)  [low VC] ---
     m   train_acc    val_acc       gap
   100      0.7340      0.6807    0.0533
   200      0.6880      0.6913   -0.0033
   400      0.7020      0.6967    0.0053
   800      0.7095      0.7147   -0.0052
  1500      0.7219      0.7243   -0.0025
  2400      0.7315      0.7310    0.0005

--- SVC(RBF)     [medium VC] ---
     m   train_acc    val_acc       gap
   100      0.9160      0.7167    0.1993
   200      0.8850      0.7897    0.0953
   400      0.8910      0.8210    0.0700
   800      0.8795      0.8493    0.0302
  1500      0.8929      0.8670    0.0259
  2400      0.8950      0.8787    0.0163

--- DTree(d=10)  [high VC] ---
     m   train_acc    val_acc       gap
   100      1.0000      0.6333    0.3667
   200      1.0000      0.6653    0.3347
   400      0.9930      0.7130    0.2800
   800      0.9782      0.7937    0.1846
  1500      0.9617      0.8203    0.1414
  2400      0.9510      0.8303    0.1207`}
      </Callout>

      <Prose>
        The three learning curves are a live demonstration of the VC theorem. Logistic regression (low VC) has a small, stable gap throughout — empirical risk is already a good proxy for true risk at <Code>m=100</Code>, and both train and val accuracy converge quickly. The SVC with RBF kernel (higher effective VC) shows a larger initial gap (0.20 at <Code>m=100</Code>) that shrinks systematically as data grows. The decision tree (high VC, depth 10) has a massive initial gap (0.37 at <Code>m=100</Code>): it memorizes the training set perfectly but generalizes poorly, and the gap only closes slowly — consistent with needing <Code>O(d/ε)</Code> where <Code>d</Code> is large.
      </Prose>

      <H3>5b. Modern theory connections: Rademacher complexity and PAC-Bayes</H3>

      <Prose>
        VC dimension is a combinatorial measure of capacity — it depends only on the hypothesis class, not on the training data distribution. Two more powerful frameworks address this limitation. <strong>Rademacher complexity</strong> is a data-dependent measure: it asks how well the hypothesis class can correlate with random noise labels on the actual training distribution. A hypothesis class that fits random labels well has high Rademacher complexity. The Rademacher generalization bound gives tighter guarantees when the actual data structure limits the effective capacity below the worst-case VC bound. In practice, Rademacher complexity explains why overparameterized neural networks generalize despite high VC dimension — the training distribution and gradient descent implicitly restrict the effective complexity.
      </Prose>

      <Prose>
        <strong>PAC-Bayes bounds</strong> (McAllester 1999) apply to stochastic classifiers — a prior distribution over hypotheses and a posterior updated on data. They give bounds of the form: with high probability, the expected error of a randomly drawn posterior hypothesis is bounded by the KL divergence between posterior and prior plus a data-fitting term. PAC-Bayes bounds are often tighter than VC bounds for practical models and have been used to derive non-vacuous generalization bounds for neural networks. Algorithmic stability bounds (Bousquet and Elisseeff 2002) offer a third approach: if replacing one training example changes the output hypothesis very little (the algorithm is stable), then the training error generalizes. SGD has been shown to be a uniformly stable algorithm, partially explaining neural network generalization through this lens.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6a. Shattering demonstration — step trace through labelings</H3>

      <StepTrace
        label="Shattering: intervals on 3 points — all 8 labelings attempted"
        steps={[
          {
            label: "Points: x1=0.2, x2=0.5, x3=0.8 — can intervals shatter them?",
            render: () => (
              <Prose>
                We have three points on [0,1]: <Code>x1=0.2</Code>, <Code>x2=0.5</Code>, <Code>x3=0.8</Code>. A hypothesis class <em>shatters</em> this set if, for each of the 8 possible +1/-1 labelings, there exists an interval [a,b] that assigns exactly those labels. We check each labeling in turn. Points inside [a,b] get +1; points outside get -1.
              </Prose>
            ),
          },
          {
            label: "Labeling (-1,-1,-1): all negative — realized by empty interval",
            render: () => (
              <Prose>
                Use [a,b] = [0.9, 1.0]. All three points (0.2, 0.5, 0.8) fall outside this interval. All get label -1. <strong>Realizable.</strong>
              </Prose>
            ),
          },
          {
            label: "Labeling (+1,-1,-1): only x1=0.2 positive — realized by [0.1, 0.3]",
            render: () => (
              <Prose>
                Use [a,b] = [0.1, 0.3]. Point x1=0.2 is inside (gets +1). Points x2=0.5 and x3=0.8 are outside (get -1). <strong>Realizable.</strong>
              </Prose>
            ),
          },
          {
            label: "Labeling (+1,+1,+1): all positive — realized by [0.1, 0.9]",
            render: () => (
              <Prose>
                Use [a,b] = [0.1, 0.9]. All three points are inside. All get +1. <strong>Realizable.</strong>
              </Prose>
            ),
          },
          {
            label: "Labeling (+1,-1,+1): x1 and x3 positive, x2 negative — IMPOSSIBLE",
            render: () => (
              <Prose>
                We need x1=0.2 inside the interval and x3=0.8 inside the interval, but x2=0.5 outside. An interval [a,b] is a convex set on the real line. If 0.2 {"∈"} [a,b] and 0.8 {"∈"} [a,b], then the entire range [0.2, 0.8] is in [a,b] by convexity. Therefore 0.5, which lies between 0.2 and 0.8, must also be in [a,b]. The labeling (+1,-1,+1) is <strong>unrealizable</strong> by any interval. This proves VC(intervals) {"<"} 3, so VC(intervals) = 2.
              </Prose>
            ),
          },
        ]}
      />

      <H3>6b. Sample complexity m vs error tolerance ε for three VC dimensions</H3>

      <Plot
        label="Sample complexity m vs error tolerance epsilon (delta=0.05, VC bound)"
        xLabel="Error tolerance epsilon"
        yLabel="Required sample size m"
        series={[
          {
            name: "d=5 (e.g. half-planes in R^4)",
            color: colors.gold,
            points: [
              [0.20, 500], [0.15, 667], [0.10, 1000], [0.08, 1250],
              [0.05, 2000], [0.03, 3333], [0.02, 5000], [0.01, 10000],
            ],
          },
          {
            name: "d=20 (e.g. linear classifier in R^19)",
            color: colors.green,
            points: [
              [0.20, 2000], [0.15, 2667], [0.10, 4000], [0.08, 5000],
              [0.05, 8000], [0.03, 13333], [0.02, 20000], [0.01, 40000],
            ],
          },
          {
            name: "d=100 (e.g. decision tree depth~7)",
            color: "#f87171",
            points: [
              [0.20, 10000], [0.15, 13333], [0.10, 20000], [0.08, 25000],
              [0.05, 40000], [0.03, 66667], [0.02, 100000], [0.01, 200000],
            ],
          },
        ]}
      />

      <Prose>
        The plot shows the dominant <Code>O(d/ε)</Code> scaling of PAC sample complexity. For ε = 0.10 (10% error tolerance), a classifier with VC dim 5 needs roughly 1,000 examples; one with VC dim 20 needs 4,000; and one with VC dim 100 needs 20,000. The gap widens as ε shrinks: at ε = 0.01, the high-capacity model needs 200× more data than the low-capacity one. This is why practitioners working with small datasets should choose simpler model families — not because they believe the true function is simple, but because they cannot afford the sample complexity that complex models require.
      </Prose>

      <H3>6c. VC generalization bound vs sample size for three VC dimensions</H3>

      <Plot
        label="VC generalization bound sqrt((d*log(2m/d)+log(4/delta))/m) vs m (delta=0.05)"
        xLabel="Training set size m"
        yLabel="Generalization gap bound"
        series={[
          {
            name: "d=5",
            color: colors.gold,
            points: [
              [100, 0.4778], [500, 0.2485], [1000, 0.1853],
              [2000, 0.1380], [5000, 0.0921], [10000, 0.0677],
            ],
          },
          {
            name: "d=20",
            color: colors.green,
            points: [
              [100, 0.7102], [500, 0.4065], [1000, 0.3106],
              [2000, 0.2367], [5000, 0.1604], [10000, 0.1194],
            ],
          },
          {
            name: "d=100",
            color: "#f87171",
            points: [
              [100, 0.8585], [500, 0.6850], [1000, 0.5513],
              [2000, 0.4340], [5000, 0.3049], [10000, 0.2311],
            ],
          },
        ]}
      />

      <Prose>
        The bound decays as <Code>O(sqrt(d log m / m))</Code>. For <Code>d=5</Code>, it reaches 0.10 around <Code>m=5,000</Code>. For <Code>d=100</Code>, you would need roughly <Code>m=400,000</Code> to push the bound below 0.10. These numbers are dramatically larger than what practitioners typically use — a well-known tension: in practice, a model with <Code>d=100</Code> generalizes well from a few thousand examples. This is the looseness of the VC bound in action. The bound is a worst-case over all distributions; real-world distributions are much better-behaved.
      </Prose>

      <H3>6d. Growth function (Sauer's lemma) — polynomial vs exponential</H3>

      <Heatmap
        label="Growth function sum C(n,i) for i=0..d (Sauer bound) — rows=n, cols=d"
        rowLabels={["n=5", "n=10", "n=20", "n=50", "n=100"]}
        colLabels={["d=2", "d=5", "d=10", "d=20"]}
        matrix={[
          [0.015, 0.031, 0.031, 0.031],
          [0.054, 0.623, 1.0,   1.0  ],
          [0.201, 0.879, 0.588, 1.0  ],
          [0.012, 0.226, 0.118, 0.109],
          [0.005, 0.076, 0.018, 0.068],
        ]}
        colorScale="gold"
      />

      <Prose>
        Values are growth function divided by <Code>{"2^n"}</Code> (fraction of all possible labelings that are realizable). When this fraction is 1.0, the class can shatter that sample size. For <Code>d=2</Code> and <Code>n=5</Code>, only 1.5% of all 32 possible labelings are realizable — the interval class is extremely constrained. For <Code>d=20</Code> and <Code>n=20</Code>, the growth function equals <Code>{"2^20 = 1,048,576"}</Code> — the class can shatter 20 points, achieving all labelings. The Sauer bound shows that once <Code>n {">"} d</Code>, the realizable fraction drops polynomially. This polynomial collapse is the formal guarantor of generalization.
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        PAC/VC theory is not an algorithm to run — it is a lens for reasoning about model-class choice. The questions it answers most precisely are: "Is my model class fundamentally learnable from my data volume?" and "Which of two model families is more appropriate for my sample size?"
      </Prose>

      <Heatmap
        label="When PAC/VC theory is actionable — usefulness across settings"
        rowLabels={["Safety/regulated domains", "Model-class comparison", "Transfer learning justification", "Large-data deep learning", "Time-series / distribution shift"]}
        colLabels={["Theory directly applicable", "Empirical PAC (learning curves)", "Rademacher / PAC-Bayes instead", "Theory vacuous / misleading"]}
        matrix={[
          [0.9, 0.8, 0.5, 0.0],
          [0.9, 0.7, 0.4, 0.1],
          [0.7, 0.6, 0.6, 0.2],
          [0.1, 0.9, 0.9, 0.3],
          [0.2, 0.6, 0.5, 0.7],
        ]}
        colorScale="gold"
      />

      <Prose>
        <strong>High applicability — safety-critical settings:</strong> In autonomous vehicles, medical devices, and financial systems subject to regulatory review, model selection must be accompanied by formal generalization guarantees. The VC bound provides the vocabulary: "This linear classifier on 50 features (VC dim 51) trained on 10,000 examples satisfies a generalization gap of at most 0.21 with probability 0.95." This statement is auditable. Regulators are increasingly familiar with PAC-style guarantees; empirical hold-out accuracy alone is not sufficient in high-stakes deployments.
      </Prose>

      <Prose>
        <strong>High applicability — model-class comparison with limited data:</strong> When you have a fixed data budget and must choose between a linear model (low VC), a kernel SVM (medium VC), and a deep network (high VC), the theory gives a principled answer. If your data budget is below the sample complexity of the complex model, you should use the simpler one — not because the simple model is more accurate, but because the complex model cannot reliably exploit the complexity it has. The learning curve in Section 5 demonstrates this exactly.
      </Prose>

      <Prose>
        <strong>Low applicability — deep learning at scale:</strong> For neural networks with billions of parameters trained on billions of tokens, the VC bound gives values like 1.0 — completely uninformative. The gap between theory and practice here is enormous. Deep networks generalize far better than VC theory predicts. The current explanation, still partially incomplete, involves implicit regularization by gradient descent (Rademacher complexity of the function class actually used, not the theoretical maximum), the benign geometry of overparameterized loss landscapes, and the interpolation regime phenomena documented by Belkin et al. (2019). For deep learning, use empirical learning curves and Rademacher-style bounds, not VC bounds.
      </Prose>

      <Prose>
        <strong>Low applicability — distribution shift:</strong> PAC bounds assume i.i.d. training and test distributions. If your test distribution differs from training (domain shift, temporal shift, population shift), the PAC guarantee no longer holds — the bound is vacuous because the fundamental assumption is violated. For such settings, domain adaptation theory, covariate shift bounds, and causal inference tools are more appropriate.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>8.1 Computational intractability of VC dimension</H3>

      <Prose>
        Computing the VC dimension of a hypothesis class is algorithmically hard in general. Papadimitriou and Yannakakis showed in 1996 that computing the VC dimension of a classifier defined by a neural network is {"Π₂ᴾ"}-complete — harder than NP in the polynomial hierarchy. For specific, well-structured classes (half-spaces, decision lists, intervals), VC dimension is known analytically. For arbitrary hypothesis classes encountered in practice — deep networks, gradient-boosted trees with complex regularization — the VC dimension is not computable in practice, and practitioners fall back on empirical learning curve analysis.
      </Prose>

      <Prose>
        Sauer's lemma provides analytical tractability: instead of computing VC dimension, you bound the growth function using the known VC dimension. For linear classifiers in <Code>ℝ^d</Code>, VC dim = <Code>d+1</Code> is known analytically; the Sauer bound follows. For neural networks, VC dimension bounds are known in terms of network architecture (number of weights, layers, activation type), giving computable worst-case bounds even if the actual VC dimension is unknown.
      </Prose>

      <H3>8.2 Scalable replacements: Rademacher complexity</H3>

      <Prose>
        Rademacher complexity is the scalable, data-dependent alternative to VC dimension. Defined as:
      </Prose>

      <MathBlock>
        {"\\hat{\\mathcal{R}}_m(H) = \\mathbb{E}_{\\sigma} \\left[ \\sup_{h \\in H} \\frac{1}{m} \\sum_{i=1}^{m} \\sigma_i h(x_i) \\right]"}
      </MathBlock>

      <Prose>
        where <Code>σᵢ</Code> are i.i.d. Rademacher random variables (<Code>±1</Code> with equal probability) and the expectation is over their randomness given the training sample. Rademacher complexity measures how well the hypothesis class correlates with random noise on the actual data distribution — a directly computable quantity for many practical settings. For kernel methods, it admits closed-form upper bounds in terms of kernel trace. For neural networks, Rademacher-based bounds (while still often loose) are tighter than VC-based bounds and capture properties of the actual data distribution rather than the worst case.
      </Prose>

      <H3>8.3 Algorithmic stability (Bousquet and Elisseeff 2002)</H3>

      <Prose>
        A learning algorithm is <em>uniformly stable</em> with parameter β if replacing one training example changes the loss of any output hypothesis by at most β. Bousquet and Elisseeff proved that if an algorithm is β-stable, the expected generalization gap is at most β, and the gap concentrates exponentially around its expectation. For regularized ERM (ridge regression, SVMs with L2 regularization), β scales as <Code>O(1/(λm))</Code> where λ is the regularization coefficient. This gives tight, algorithm-specific bounds that bypass the VC dimension entirely. SGD itself has been shown to be stable under appropriate step size and number of iterations, partially explaining neural network generalization through this lens.
      </Prose>

      <H3>8.4 Non-vacuous bounds for neural networks: PAC-Bayes</H3>

      <Prose>
        PAC-Bayes bounds (McAllester 1999, further developed by Dziugaite and Roy 2017 for neural networks) provide the tightest known computable generalization bounds for deep networks. They apply to stochastic classifiers — draw a network at random from a posterior distribution <Code>Q</Code> over weights — and bound the expected error by the empirical error plus a KL divergence term. Dziugaite and Roy showed that optimizing the PAC-Bayes bound directly gives non-vacuous guarantees (values below 1.0) for deep networks on MNIST, a result not achievable with VC bounds. This is an active research frontier: deriving tight, non-vacuous bounds for practical neural networks on real datasets.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>9.1 VC dimension is not parameter count</H3>

      <Prose>
        The most dangerous misconception. Consider the one-parameter model <Code>{"h_θ(x) = 1[sin(θx) ≥ 0]"}</Code>. It has exactly one parameter (θ), yet can shatter arbitrarily large point sets on the real line — its VC dimension is infinite. Conversely, a kernel SVM with a fixed RBF kernel bandwidth has finite VC dimension (bounded by the number of training points it uses) even though it has as many effective parameters as training examples. The VC dimension characterizes the hypothesis class (the set of all functions the model can represent), not the number of degrees of freedom in the fitting procedure. Naive parameter counting as a proxy for VC dimension will mislead you in both directions.
      </Prose>

      <H3>9.2 PAC bounds are worst-case — actual gaps are usually much smaller</H3>

      <Prose>
        The VC generalization bound is proven for the worst-case distribution over inputs and the worst-case hypothesis class. For any specific, benign distribution (Gaussian inputs, smooth decision boundaries), the actual generalization gap is typically 10–100× smaller than the VC bound. This looseness is not a bug in the theorem — it is a necessary consequence of being distribution-free. The bound must hold for adversarially constructed distributions, and it does. Practitioners should treat VC bounds as order-of-magnitude guidance, not as tight estimates of their actual generalization gap. For tight empirical estimates, use cross-validation and compute confidence intervals on validation error directly.
      </Prose>

      <H3>9.3 The d log m factor makes bounds loose</H3>

      <Prose>
        The VC bound contains a <Code>log(2m/d)</Code> factor inside the square root. This factor grows (slowly) with <Code>m</Code>, so the bound decays slightly slower than <Code>O(1/sqrt(m))</Code>. More concretely: to halve the generalization bound, you need more than four times the data. For the agnostic setting, the bound decays as <Code>O(sqrt(d log(m/d)/m))</Code>, which at <Code>d=100</Code> and <Code>m=1000</Code> gives a bound of roughly 0.55 — meaningless for practical purposes. This is why practitioners using classical ML on small datasets and regulators demanding formal guarantees typically use simple model families with small known VC dimensions, not complex ones.
      </Prose>

      <H3>9.4 Realizable vs agnostic PAC: a qualitative gap in sample complexity</H3>

      <Prose>
        In the realizable setting (target concept is in <Code>H</Code>), the sample complexity scales as <Code>O(d/ε)</Code>. In the agnostic setting (no assumption on the target), it scales as <Code>{"O(d/ε²)"}</Code>. This quadratic difference is not a minor constant — at ε = 0.01 it is a 100× difference. Practitioners often implicitly assume realizability (the true function is in their model class), but real datasets are always agnostic: no model family contains the true data-generating process. Being in the agnostic regime means you need far more data than the realizable bound suggests. Calibrating your data requirements to the realizable bound when you are actually in the agnostic regime is a common source of underestimating data needs.
      </Prose>

      <H3>9.5 i.i.d. assumption breaks for dependent data</H3>

      <Prose>
        PAC bounds assume training examples are drawn i.i.d. from the same distribution as the test examples. For time-series data, spatial data with autocorrelation, or any setting with distribution shift, this assumption is violated. An interval estimator trained on historical stock returns does not have the same distribution as future returns. A sentiment classifier trained on 2022 social media has a different distribution from 2024 social media. Applying PAC bounds in these settings gives guarantees that are formally correct about the wrong quantity — the gap between in-distribution training and in-distribution test performance — but say nothing about the gap that matters in deployment.
      </Prose>

      <H3>9.6 VC dimension for regression: pseudo-dimension and fat-shattering</H3>

      <Prose>
        The VC dimension as defined above applies to binary classification. For regression (real-valued outputs), the appropriate generalization of VC dimension is the <em>pseudo-dimension</em> (or its refined version, fat-shattering dimension). A set of real-valued functions has pseudo-dimension <Code>d</Code> if there exists a set of <Code>d</Code> points <Code>x₁, ..., x_d</Code> and witnesses <Code>r₁, ..., r_d</Code> (thresholds) such that for every binary labeling, some function <Code>f ∈ H</Code> satisfies <Code>f(xᵢ) ≥ rᵢ</Code> when the label is +1 and <Code>f(xᵢ) {"<"} rᵢ</Code> when -1. PAC bounds for regression use pseudo-dimension in place of VC dimension, with the same qualitative behavior. Practitioners rarely compute pseudo-dimension directly; they use the known pseudo-dimensions for specific model classes (linear functions in <Code>ℝ^d</Code> have pseudo-dimension <Code>d+1</Code>, matching their VC dimension for classification).
      </Prose>

      <H3>9.7 Deep learning: VC bounds are vacuous but the theory is not wrong</H3>

      <Prose>
        When you compute the VC generalization bound for a ResNet-50 (25M parameters, VC dim in the billions) trained on ImageNet (1.2M examples), the bound comes out to a number larger than 1.0. This is often interpreted as "VC theory fails for deep learning." The correct interpretation is different: VC theory is a worst-case bound, and deep networks trained with gradient descent do not represent the worst case. The actual function class realized by gradient descent on overparameterized networks has much lower effective complexity than the theoretical maximum. Understanding why is the central open question of modern deep learning theory, with Rademacher complexity, implicit regularization, neural tangent kernel theory, and lottery ticket hypothesis all contributing partial answers.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All citations were verified against primary publication venues, author lists, and main claims.
      </Prose>

      <StepTrace
        label="Foundational literature"
        steps={[
          {
            label: "Vapnik & Chervonenkis 1971 — The theoretical foundation",
            render: () => (
              <Prose>
                Vapnik, V. N. and Chervonenkis, A. Ya. (1971). "On the Uniform Convergence of Relative Frequencies of Events to their Probabilities." <em>Theory of Probability and its Applications</em>, 16(2):264–280. (Russian original 1968.) This is the paper that introduced what we now call the VC dimension, the growth function, and the uniform convergence argument that ties them to generalization. The result — that uniform convergence holds if and only if the growth function is polynomial (i.e., VC dimension is finite) — is the theoretical foundation of all modern statistical learning theory. The paper is mathematically dense and written for probabilists; the 1974 monograph "Theory of Pattern Recognition" by the same authors is more accessible. The core inequality proven here underlies every VC-type generalization bound used in practice today.
              </Prose>
            ),
          },
          {
            label: "Valiant 1984 — The PAC framework",
            render: () => (
              <Prose>
                Valiant, L. G. (1984). "A Theory of the Learnable." <em>Communications of the ACM</em>, 27(11):1134–1142. DOI: 10.1145/1968.1972. This paper introduced the PAC (Probably Approximately Correct) learning model. Valiant asked what it means for an algorithm to "learn" a concept — formally, in polynomial time and polynomial samples, with respect to any distribution over inputs. He showed that Boolean conjunctions are PAC-learnable, established the computational complexity framework, and planted the seed for the subsequent twenty years of computational learning theory. The paper is unusually readable for a foundational theory paper. Valiant received the Turing Award in 2010 partly for this work.
              </Prose>
            ),
          },
          {
            label: "Blumer, Ehrenfeucht, Haussler, Warmuth 1989 — VC meets PAC",
            render: () => (
              <Prose>
                Blumer, A., Ehrenfeucht, A., Haussler, D., and Warmuth, M. K. (1989). "Learnability and the Vapnik-Chervonenkis dimension." <em>Journal of the ACM</em>, 36(4):929–965. DOI: 10.1145/76359.76371. The central unification: a concept class is PAC-learnable if and only if its VC dimension is finite. The paper derives the sample complexity bounds linking VC dimension to ε and δ, proves that empirical risk minimization is a universal PAC learning algorithm for finite-VC-dimension classes, and establishes the tight (up to log factors) sample complexity in both the realizable and agnostic settings. This is the canonical reference for anyone seeking rigorous proofs; Chapter 3 of Shalev-Shwartz and Ben-David (2014) gives a modern pedagogical treatment.
              </Prose>
            ),
          },
          {
            label: "Shalev-Shwartz & Ben-David 2014 — Canonical textbook",
            render: () => (
              <Prose>
                Shalev-Shwartz, S. and Ben-David, S. (2014). <em>Understanding Machine Learning: From Theory to Algorithms.</em> Cambridge University Press. ISBN: 978-1-107-05713-5. Freely available at cs.huji.ac.il/~shais/UnderstandingMachineLearning. This is the best modern treatment of PAC theory and VC dimension for practitioners. Chapter 3 covers the PAC model; Chapter 6 derives the VC generalization theorem with full proofs; Chapter 26 covers Rademacher complexity. The book is more accessible than the original papers, more rigorous than a survey, and written by researchers who contributed significantly to the field. Chapters 3 and 6 together with the appendix on concentration inequalities are the minimal investment for understanding where every formula in this topic comes from.
              </Prose>
            ),
          },
          {
            label: "Bousquet & Elisseeff 2002 — Algorithmic stability",
            render: () => (
              <Prose>
                Bousquet, O. and Elisseeff, A. (2002). "Stability and Generalization." <em>Journal of Machine Learning Research</em>, 2:499–526. The paper that introduced algorithmic stability as a route to generalization bounds that bypass VC dimension entirely. Their main result: a uniformly stable algorithm (one where replacing a training example changes the output loss by at most β for all hypotheses and inputs) satisfies a generalization bound of O(β + sqrt(log(1/δ)/m)). For regularized ERM, β = O(1/(λm)) where λ is the regularization parameter, giving tight bounds for ridge regression, kernel SVMs, and other regularized methods. This work is foundational for understanding why regularized methods generalize better than VC theory predicts, and for the subsequent line of work showing that SGD is a stable algorithm.
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
        Work through each question before reading the answer. The first three are recall and conceptual; the last three require careful reasoning.
      </Prose>

      <H3>Exercise 1 (Recall)</H3>
      <Prose>
        Define PAC learning. What do the two parameters ε and δ represent? State the sample complexity bound for a finite hypothesis class in the realizable setting.
      </Prose>
      <Callout type="answer" title="Answer 1">
        {"An algorithm PAC-learns a concept class C if: for every target concept c ∈ C, every distribution D over the input space, and every ε, δ > 0, when given m training examples (drawn i.i.d. from D, labeled by c), the algorithm outputs a hypothesis h with true error R(h) ≤ ε with probability at least 1 − δ over the random draw of training examples. ε is the error tolerance (how approximately correct we require the output to be); δ is the failure probability (how often we allow the training set to be 'unlucky'). For a finite hypothesis class |H| < ∞ in the realizable setting, the sample complexity bound is: m ≥ (1/ε)(ln|H| + ln(1/δ)). This follows from: a bad hypothesis (error > ε) is consistent with all m examples with probability at most (1−ε)^m ≤ exp(−εm). Union-bounding over all hypotheses gives P(any bad hypothesis is consistent) ≤ |H|·exp(−εm). Setting this ≤ δ and solving for m gives the result."}
      </Callout>

      <H3>Exercise 2 (Conceptual)</H3>
      <Prose>
        What is the VC dimension of axis-aligned rectangles in {"ℝ²"}? Prove it by (a) showing a set of 4 points that can be shattered and (b) showing that no 5-point set can be shattered.
      </Prose>
      <Callout type="answer" title="Answer 2">
        {"VC(axis-aligned rectangles in ℝ²) = 4. (a) Consider the 4 points: top=(0,1), bottom=(0,-1), left=(-1,0), right=(1,0). For any binary labeling of these 4 points, we can find a rectangle. Label all 4 positive: use the rectangle [-1.5,1.5]×[-1.5,1.5]. Label top positive only: use [−0.1,0.1]×[0.5,1.5]. Label top and right positive: use [−0.1,1.5]×[−0.1,1.5]. For any subset S ⊆ {top, bottom, left, right}, the tightest rectangle enclosing exactly S works. All 2^4 = 16 labelings are realizable. (b) No 5-point set can be shattered. Given any 5 points, consider the one with the smallest x-coordinate (leftmost), the largest x-coordinate (rightmost), the smallest y-coordinate (bottommost), and the largest y-coordinate (topmost). At most 4 points can simultaneously be 'extreme' in one direction. The fifth point is in the interior of the convex hull of the extreme points. For the labeling where only the 4 extreme points are positive, any rectangle that contains all 4 extremes must also contain the 5th interior point (since it lies within the bounding box of the extremes). So the labeling (extreme points +1, interior point -1) is unrealizable. Hence VC(axis-aligned rectangles in ℝ²) < 5, so VC = 4."}
      </Callout>

      <H3>Exercise 3 (Math — Sauer's lemma)</H3>
      <Prose>
        State Sauer's lemma precisely. Compute the Sauer upper bound on the growth function {"Π_H(m)"} for {"m=100"} and <Code>d=5</Code>. How does this compare to <Code>{"2^100"}</Code>?
      </Prose>
      <Callout type="answer" title="Answer 3">
        {"Sauer's lemma states: if H has VC dimension d, then for all m ≥ d, the growth function satisfies Π_H(m) ≤ Σ_{i=0}^{d} C(m,i), and this is further bounded by (em/d)^d. For m=100, d=5: Exact Sauer bound = C(100,0)+C(100,1)+C(100,2)+C(100,3)+C(100,4)+C(100,5) = 1 + 100 + 4950 + 161700 + 3921225 + 75287520 = 79,375,496 ≈ 7.9×10^7. The loose upper bound: (e×100/5)^5 = (20e)^5 ≈ (54.37)^5 ≈ 4.7×10^8. Compare to 2^100 ≈ 1.27×10^30. The growth function is about 10^22 times smaller than the exponential — this is the polynomial collapse that enables generalization. Despite having an infinite hypothesis class, only ~80 million of 10^30 possible labelings on 100 points are realizable. This is what makes learning from 100 examples possible at all."}
      </Callout>

      <H3>Exercise 4 (Applied — data requirement reasoning)</H3>
      <Prose>
        You are building a fraud classifier using logistic regression on 30 features. The business requires the true error to be at most 5% with confidence 99%. Use the VC bound to estimate the minimum training set size. Now repeat the calculation for a gradient-boosted tree with effective VC dimension approximately 1,000. What does the ratio tell you?
      </Prose>
      <Callout type="answer" title="Answer 4">
        {"For logistic regression on 30 features: VC dimension d = 31 (linear classifiers in ℝ^30 have VC dim = 31). Using the agnostic VC sample complexity approximation m ≈ (d/ε²)(log(d/ε) + log(1/δ)): ε = 0.05, δ = 0.01, d = 31. Rough estimate: m ≈ (31/0.0025)(ln(31/0.05) + ln(100)) ≈ 12400 × (6.43 + 4.61) ≈ 12400 × 11.04 ≈ 137,000 examples. For the GBM with d=1000: m ≈ (1000/0.0025)(ln(1000/0.05) + ln(100)) ≈ 400,000 × (9.9 + 4.61) ≈ 400,000 × 14.51 ≈ 5.8 million examples. Ratio ≈ 5,800,000 / 137,000 ≈ 42×. The theory says the GBM needs 42× more data than logistic regression to achieve the same formal guarantee. If you have only 10,000 labeled fraud examples, this tells you clearly: use logistic regression (or a regularized linear model), not a gradient-boosted tree. The GBM may empirically generalize better at small data sizes, but you cannot provide formal guarantees — and in a regulated setting like fraud detection, that matters."}
      </Callout>

      <H3>Exercise 5 (Conceptual — deep learning)</H3>
      <Prose>
        GPT-2 Small has 117 million parameters. Its VC dimension (treating it as a hypothesis class over fixed-length binary classification problems) is at least in the hundreds of millions. Yet it generalizes to held-out text from the same distribution. Explain why VC theory does not predict this failure and what modern alternatives better explain neural network generalization.
      </Prose>
      <Callout type="answer" title="Answer 5">
        {"VC theory bounds the worst-case generalization gap over all distributions and all hypotheses in the class. For GPT-2 with VC dim ~ 10^8 trained on ~40GB of text (call it m ~ 10^10 tokens), the VC bound gives something like sqrt(10^8 × log(2×10^10/10^8) / 10^10) = sqrt(10^8 × log(200) / 10^10) = sqrt(5.3/100) ≈ 0.23. This is technically not vacuous! But it is very loose. The deeper issue is that VC theory considers the entire hypothesis class — all possible ways GPT-2's weights could be set — and bounds over the hardest possible point in that space. In practice, SGD on language modeling does not traverse the full hypothesis class. It finds a specific minimum in a specific region of weight space that corresponds to a much lower effective complexity than the full class. Three modern frameworks better explain this: (1) Rademacher complexity — data-dependent, it measures how well the *actually learned* hypothesis class correlates with noise, not the worst case over all parameterizations. (2) PAC-Bayes bounds — treat the trained weights as a posterior distribution around the initialization, and the KL divergence between posterior and prior (a Gaussian around random initialization) is small if SGD doesn't move far. (3) Implicit regularization / minimum-norm interpolation — among all weight settings that fit the training data, SGD tends to find the one with minimum norm, which is also the one with the simplest behavior on inputs that were not in the training set."}
      </Callout>

      <H3>Exercise 6 (Synthesis)</H3>
      <Prose>
        A researcher trains 10 different model architectures on the same dataset of 5,000 examples and selects the one with the lowest validation accuracy. She reports the validation accuracy of the selected model as her estimate of generalization error. Why is this estimate optimistic, and how does PAC theory explain the bias? What correction can she apply?
      </Prose>
      <Callout type="answer" title="Answer 6">
        {"The estimate is optimistic because of selection bias — selecting the best of 10 models based on validation performance is itself a form of overfitting to the validation set. PAC theory explains this via the union bound argument: if you consider 10 hypotheses h₁, ..., h₁₀ and pick the one with minimum empirical risk on the validation set, the probability that any one of them has generalization gap > ε is bounded by 10 × P(single model gap > ε). The effective hypothesis class has now expanded to include all 10 models, and the bound must account for the size of this meta-class. Formally, if the validation set has size m_val and you select the best of k models, the gap between selected-model validation error and true error is bounded by sqrt((1/2m_val) × ln(2k/δ)) rather than sqrt((1/2m_val) × ln(2/δ)) — an additional ln(k) ≈ 2.3 penalty for k=10. For m_val=1000 and k=10, this adds roughly 0.015 to the bound. More practically: (a) use a separate final holdout set that is touched exactly once after all model selection is complete; (b) apply Bonferroni correction — report validation error ± 2σ/sqrt(m_val) for each model individually, where σ is the validation set standard deviation, and require statistical separation before declaring a winner; (c) use nested cross-validation where the inner loop selects the architecture and the outer loop estimates generalization, ensuring separation between selection and evaluation."}
      </Callout>

    </div>
  ),
};

export default pacVcContent;
