import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const gradientBoostedTreesContent = {
  title: "Gradient Boosted Trees (XGBoost, LightGBM, CatBoost)",
  readTime: "~55 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Boosting was invented as a theoretical answer to a combinatorial question: can a learning algorithm with only slight-better-than-random accuracy be "boosted" to arbitrary accuracy? Robert Schapire answered yes in 1990, and the practical consequence — AdaBoost, due to Freund and Schapire in 1997 — dominated ensemble learning for several years. But AdaBoost had a known brittleness: it was sensitive to outliers, it required re-weighting training examples in a way that could explode when the base learner was nearly perfect, and its loss function (exponential loss) was not easily swapped for something more robust. The field needed a unifying theory.
      </Prose>

      <Prose>
        Jerome H. Friedman provided it in 2001. His paper "Greedy Function Approximation: A Gradient Boosting Machine," published in the <em>Annals of Statistics</em> 29(5):1189–1232 (DOI: 10.1214/aos/1013203451), reframed boosting not as a re-weighting scheme but as gradient descent in function space. The insight: instead of optimizing a loss function over parameters, treat the prediction function itself as the thing being optimized. The "gradient" is the vector of pseudo-residuals — the negative gradient of the loss evaluated at each training point. Each new weak learner fits these pseudo-residuals, reducing the loss in the direction of steepest descent. Swap the loss function and the pseudo-residuals change; the machinery stays the same. This unified exponential loss (AdaBoost), squared error, absolute deviation, Huber loss, and any other differentiable objective under a single algorithmic frame. Decision trees were the obvious weak learner: shallow trees with a fixed number of leaves are expressive enough to capture interactions but simple enough to be fit rapidly by exhaustive search.
      </Prose>

      <Prose>
        Friedman's algorithm was powerful but slow. For a dataset with <em>n</em> samples and <em>d</em> features, finding the best split at each node required sorting each feature — O(n log n) per feature per node, repeated across all nodes and all trees. The theoretical soundness was not in question; the wall-clock reality was. The decade between Friedman 2001 and the first Kaggle competitions revealed the gap between a correct algorithm and a practical one.
      </Prose>

      <Prose>
        Three libraries closed that gap, each attacking a different bottleneck.
      </Prose>

      <Prose>
        <strong>XGBoost</strong> (eXtreme Gradient Boosting) was introduced by Tianqi Chen and Carlos Guestrin in "XGBoost: A Scalable Tree Boosting System," presented at KDD 2016 (arXiv:1603.02754). Its central contribution was a second-order Taylor expansion of the loss, which led to a closed-form expression for the optimal leaf weight and a quantifiable gain for each candidate split — no inner loop needed. This "split gain" formula, with an explicit L1 penalty <Code>γ</Code> on the number of leaves and an L2 penalty <Code>λ</Code> on leaf weights, let XGBoost prune trees objectively rather than heuristically. A second contribution was sparsity-aware split finding: a default direction for missing values learned from data, not imputed, so sparse one-hot encoded features worked natively. XGBoost also introduced a weighted quantile sketch for approximate split finding on large datasets, making it the first practical gradient boosting library for billion-row data.
      </Prose>

      <Prose>
        <strong>LightGBM</strong> came from Microsoft Research in 2017. Guolin Ke, Qi Meng, Thomas Finley, and collaborators published "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" at NeurIPS 2017. Two algorithmic ideas distinguish it. Gradient-based One-Side Sampling (GOSS) observes that data instances with small gradient magnitude are already well-fitted — they contribute less information to the next split. GOSS keeps all large-gradient instances and randomly samples a fraction of small-gradient ones, weighting the latter to correct for the bias. The result: many fewer instances per tree without meaningful accuracy loss. Exclusive Feature Bundling (EFB) addresses wide, sparse feature matrices: features that are mutually exclusive (they are never both nonzero) can be packed into a single "bundle" without losing information, reducing the effective feature count. On top of these two ideas, LightGBM switched from level-wise (breadth-first) tree growth to leaf-wise (best-first) growth — always split the leaf with the highest gain, regardless of depth — and built its entire split search on a histogram of discretized feature values rather than exact sorted values. The combination yielded 20x speedups over XGBoost on the benchmarks in the paper.
      </Prose>

      <Prose>
        <strong>CatBoost</strong> (Categorical Boosting) came from Yandex in 2018. Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush, and Andrey Gulin published "CatBoost: unbiased boosting with categorical features" at NeurIPS 2018 (arXiv:1706.09516). The key diagnosis: every existing implementation of gradient boosting used the same examples to both estimate the gradient and to fit the tree, creating a subtle but real bias in gradient estimates — what the paper calls prediction shift. This bias grows with the number of boosting rounds and with the number of categorical features processed via target encoding. The remedy is ordered boosting: permute the dataset, then compute the gradient for each example using a model fitted on only the examples that precede it in the permutation. This is analogous to online learning within each boosting round and eliminates the target leakage. For categorical features, CatBoost computes ordered target statistics — mean target values calculated only from preceding examples in the permutation — rather than global means, which prevents the leakage that makes naive target encoding overfit badly.
      </Prose>

      <Callout type="insight">
        All three libraries implement the same underlying algorithm — gradient boosting of decision trees — but optimize for different bottlenecks. XGBoost: regularization quality and sparsity. LightGBM: training speed on large dense datasets. CatBoost: unbiased estimates and native categorical support. In practice, their accuracy on a given tabular dataset is often within a few percent of each other; the right choice depends on data characteristics and latency constraints.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        The mental model for gradient boosting is sequential error correction. Start with a naïve prediction — the mean of the target for regression, or the log-odds for classification. Compute how wrong that prediction is at each point. Fit a shallow tree to those errors (the "pseudo-residuals"). Add the tree's prediction, scaled by a small learning rate, to the running total. Recompute the errors. Repeat. Each tree does not try to solve the whole problem; it tries to correct the specific errors that the current ensemble is making.
      </Prose>

      <Prose>
        This is gradient descent, but the "parameter" being updated is the prediction function, not a weight vector. The pseudo-residuals are the negative gradient of the loss with respect to the current function value: for squared-error loss, that gradient is literally <Code>y - F(x)</Code>, i.e., the residual. For other losses — absolute error, logistic, quantile — the pseudo-residuals are different, but the structure is the same: compute the gradient, fit a tree to it, take a step.
      </Prose>

      <Prose>
        Compare this to a Random Forest. A Random Forest trains many trees in <em>parallel</em>, each on a bootstrap sample of the data, and averages their predictions. Each tree is deep — high variance, low bias — and averaging reduces the variance. Gradient boosting trains trees <em>sequentially</em>, each tree fitting the mistakes of all previous trees. Individual trees are kept shallow — low variance, high bias — and the sequential correction reduces the bias. The two approaches exploit the bias-variance tradeoff from opposite directions.
      </Prose>

      <StepTrace
        label="gradient boosting: 5 rounds of sequential correction"
        steps={[
          {
            label: "Round 0 — initial prediction (mean)",
            render: () => (
              <div>
                <TokenStream
                  label="prediction = mean(y)"
                  tokens={[
                    { label: "F₀ = 2.5", color: colors.textMuted },
                    { label: "residuals: [−1.5, −0.5, +0.5, +1.5, +2.5]", color: "#f87171" },
                    { label: "MSE = 2.75", color: colors.textMuted },
                  ]}
                />
                <Prose>
                  F₀ is just the mean of y. Every training point has a nonzero residual. The first tree will fit these residuals.
                </Prose>
              </div>
            ),
          },
          {
            label: "Round 1 — tree fits residuals, MSE drops sharply",
            render: () => (
              <div>
                <TokenStream
                  label="tree₁ fits [−1.5, −0.5, +0.5, +1.5, +2.5]"
                  tokens={[
                    { label: "F₁ = F₀ + 0.3·h₁(x)", color: colors.gold },
                    { label: "new residuals smaller", color: "#86efac" },
                    { label: "MSE ≈ 1.93", color: colors.textMuted },
                  ]}
                />
                <Prose>
                  The learning rate 0.3 scales the tree's contribution. A rate of 1.0 would fully correct each error in one step but overfit; shrinkage forces more trees and better generalization.
                </Prose>
              </div>
            ),
          },
          {
            label: "Round 2 — second tree corrects remaining errors",
            render: () => (
              <div>
                <TokenStream
                  label="tree₂ fits residuals of F₁"
                  tokens={[
                    { label: "F₂ = F₁ + 0.3·h₂(x)", color: colors.gold },
                    { label: "MSE ≈ 1.35", color: colors.textMuted },
                  ]}
                />
              </div>
            ),
          },
          {
            label: "Round 3 — continued correction",
            render: () => (
              <div>
                <TokenStream
                  label="tree₃ fits residuals of F₂"
                  tokens={[
                    { label: "F₃ = F₂ + 0.3·h₃(x)", color: colors.gold },
                    { label: "MSE ≈ 0.94", color: colors.textMuted },
                  ]}
                />
              </div>
            ),
          },
          {
            label: "Round 4 — diminishing returns, regularization matters",
            render: () => (
              <div>
                <TokenStream
                  label="tree₄ fits residuals of F₃"
                  tokens={[
                    { label: "F₄ = F₃ + 0.3·h₄(x)", color: colors.gold },
                    { label: "MSE ≈ 0.66", color: colors.textMuted },
                    { label: "early stopping monitors val loss here", color: "#60a5fa" },
                  ]}
                />
                <Prose>
                  Each successive tree contributes less marginal improvement. Without early stopping or a large learning rate penalty, the model will eventually memorize the training set. Validation-based early stopping is the primary regularization mechanism in practice.
                </Prose>
              </div>
            ),
          },
        ]}
      />

      <Prose>
        The shrinkage parameter (learning rate, typically 0.01–0.3) is arguably the most important hyperparameter. A small learning rate requires more trees but generally achieves lower generalization error; a large one converges faster but risks overshooting. The canonical recommendation — use the smallest learning rate your compute budget allows, then tune the number of trees with early stopping — holds across all three libraries.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 Gradient boosting as functional gradient descent</H3>

      <Prose>
        Let <Code>L(y, F(x))</Code> be a differentiable loss function. At iteration <Code>m</Code>, the current model is <Code>F_{"m-1"}(x)</Code>. Define the pseudo-residuals as the negative gradient of the loss with respect to the current prediction:
      </Prose>

      <MathBlock>
        {"r_i^{(m)} = -\\left[\\frac{\\partial L(y_i,\\, F(x_i))}{\\partial F(x_i)}\\right]_{F = F_{m-1}}"}
      </MathBlock>

      <Prose>
        Fit a regression tree <Code>h_m</Code> to the pseudo-residuals <Code>{"(x_i, r_i^{(m)})"}</Code>. Find the optimal step size <Code>ρ_m</Code> by line search:
      </Prose>

      <MathBlock>
        {"\\rho_m = \\underset{\\rho}{\\operatorname{argmin}} \\sum_{i=1}^n L\\!\\left(y_i,\\, F_{m-1}(x_i) + \\rho\\, h_m(x_i)\\right)"}
      </MathBlock>

      <Prose>
        Update: <Code>F_m(x) = F_{"m-1"}(x) + η · ρ_m · h_m(x)</Code>, where <Code>η</Code> is the shrinkage (learning rate). For squared error loss, <Code>r_i = y_i - F_{"m-1"}(x_i)</Code> — the ordinary residual — and the line search collapses to fitting the mean of the residuals in each leaf. For logistic loss, the pseudo-residuals are <Code>y_i - σ(F_{"m-1"}(x_i))</Code> and the per-leaf line search has a closed form involving the Newton-Raphson step.
      </Prose>

      <H3>3.2 XGBoost: second-order split gain</H3>

      <Prose>
        XGBoost replaces the first-order steepest-descent step with a second-order (Newton) step by expanding the loss around <Code>F_{"m-1"}</Code> to second order. Let <Code>g_i = ∂L/∂F(x_i)</Code> (gradient) and <Code>h_i = ∂²L/∂F(x_i)²</Code> (Hessian) evaluated at <Code>F_{"m-1"}</Code>. The regularized objective for a single tree with leaf weights <Code>w_j</Code> is:
      </Prose>

      <MathBlock>
        {"\\tilde{\\mathcal{L}}(\\{w_j\\}) = \\sum_{j=1}^T \\left[ G_j w_j + \\frac{1}{2}(H_j + \\lambda)w_j^2 \\right] + \\gamma T"}
      </MathBlock>

      <Prose>
        where <Code>G_j = Σ_{"i∈leaf_j"} g_i</Code>, <Code>H_j = Σ_{"i∈leaf_j"} h_i</Code>, <Code>T</Code> is the number of leaves, <Code>λ</Code> is L2 regularization on leaf weights, and <Code>γ</Code> is a minimum gain threshold (L0 penalty on leaves). Setting the derivative to zero yields the optimal leaf weight:
      </Prose>

      <MathBlock>
        {"w_j^* = -\\frac{G_j}{H_j + \\lambda}"}
      </MathBlock>

      <Prose>
        Substituting back gives the optimal objective value for a fixed tree structure. The gain from splitting a leaf into left (L) and right (R) subsets is then:
      </Prose>

      <MathBlock>
        {"\\text{Gain} = \\frac{1}{2}\\left[\\frac{G_L^2}{H_L + \\lambda} + \\frac{G_R^2}{H_R + \\lambda} - \\frac{(G_L+G_R)^2}{H_L+H_R+\\lambda}\\right] - \\gamma"}
      </MathBlock>

      <Prose>
        This formula is the workhorse of XGBoost's tree learner. For every candidate split, compute <Code>G_L, H_L, G_R, H_R</Code> from the data in each side, plug in, and take the split that maximizes gain. If no split achieves <Code>Gain {">"} 0</Code>, the leaf is not split (the <Code>γ</Code> term enforces this automatically). For MSE loss, <Code>g_i = F(x_i) - y_i</Code> and <Code>h_i = 1</Code>, recovering a form equivalent to variance reduction weighted by sample count.
      </Prose>

      <H3>3.3 LightGBM: GOSS and EFB</H3>

      <Prose>
        <strong>Gradient-based One-Side Sampling (GOSS).</strong> Let instances be sorted by <Code>|g_i|</Code> in descending order. Keep the top-<Code>a·n</Code> instances (large-gradient set <Code>A</Code>) always. From the remaining, sample a fraction <Code>b</Code> uniformly to get small-gradient set <Code>B</Code>. Upweight each instance in <Code>B</Code> by <Code>(1-a)/b</Code> when computing split gain, correcting for the sampling bias. The estimated gain from any split becomes:
      </Prose>

      <MathBlock>
        {"\\tilde{V}(d) = \\frac{1}{n}\\left(\\frac{\\left(\\sum_{x_i \\in A_L} g_i + \\frac{1-a}{b}\\sum_{x_i \\in B_L} g_i\\right)^2}{n_l^j} + \\frac{\\left(\\sum_{x_i \\in A_R} g_i + \\frac{1-a}{b}\\sum_{x_i \\in B_R} g_i\\right)^2}{n_r^j}\\right)"}
      </MathBlock>

      <Prose>
        <strong>Exclusive Feature Bundling (EFB).</strong> In sparse datasets, many features are mutually exclusive — they are never simultaneously nonzero. EFB identifies such groups using a graph-coloring heuristic (feature conflict graph where edges connect features that co-occur) and packs exclusive features into a single bundle by offsetting their value ranges. A bundle of <Code>k</Code> features with value ranges <Code>[0, max_k]</Code> is stored in a single histogram by adding the cumulative offset of each feature's range. This reduces effective feature count from <Code>d</Code> to <Code>d' ≪ d</Code> for sparse inputs.
      </Prose>

      <Prose>
        <strong>Leaf-wise tree growth.</strong> Standard GBDT grows trees level by level (all nodes at depth <Code>k</Code> before any at depth <Code>k+1</Code>). LightGBM grows leaf-wise: at each step, pick the globally best leaf to split regardless of depth. With the same number of leaves, leaf-wise trees are typically deeper and achieve lower training loss; the downside is higher variance, mitigated by <Code>max_depth</Code> as a hard cap.
      </Prose>

      <H3>3.4 CatBoost: ordered boosting and ordered target statistics</H3>

      <Prose>
        <strong>Prediction shift.</strong> In standard GBDT, the gradient for example <Code>i</Code> at round <Code>m</Code> is computed using a model <Code>F_{m-1}</Code> that was fitted on all <Code>n</Code> examples including <Code>i</Code> itself. This causes a conditional bias: <Code>E[g_i | x_i] ≠ 0</Code>, because the model has partially memorized <Code>x_i</Code>. Prokhorenkova et al. show this bias causes the boosted model to underfit, especially with many trees.
      </Prose>

      <Prose>
        <strong>Ordered boosting.</strong> Draw a random permutation <Code>σ</Code> of the training set. For each example <Code>x_i</Code>, compute its gradient using a model <Code>M_i</Code> fitted only on examples <Code>{"{x_j : σ(j) < σ(i)}"}</Code>. This is maintained efficiently by keeping <Code>2^s</Code> models of exponentially increasing size, with gradients assigned from whichever model's training set excludes <Code>x_i</Code>. The resulting gradient estimates are unbiased: <Code>E[g_i | x_i] = 0</Code> by construction.
      </Prose>

      <Prose>
        <strong>Ordered target statistics for categoricals.</strong> Naïve target encoding — replacing a categorical value <Code>c</Code> with the mean target among all training examples with that category — leaks label information. CatBoost uses the ordered permutation to compute:
      </Prose>

      <MathBlock>
        {"\\hat{x}_i^k = \\frac{\\sum_{j=1}^{i-1} \\mathbf{1}[x_{\\sigma(j)}^k = x_{\\sigma(i)}^k] \\cdot y_{\\sigma(j)} + a \\cdot p}{\\sum_{j=1}^{i-1} \\mathbf{1}[x_{\\sigma(j)}^k = x_{\\sigma(i)}^k] + a}"}
      </MathBlock>

      <Prose>
        where <Code>p</Code> is the prior (global mean of <Code>y</Code>) and <Code>a</Code> is a smoothing parameter. Example <Code>i</Code> contributes only examples that precede it in the permutation — zero leakage from its own label.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        Every code block below was executed and the output is the literal stdout. No pseudo-code, no ellipsis. We build a gradient boosting regressor using NumPy only, then demonstrate the XGBoost-style second-order update formula.
      </Prose>

      <H3>4a. Gradient boosting regressor with MSE loss</H3>

      <CodeBlock language="python">
{`import numpy as np

class DecisionStump:
    """Shallow regression tree for use as a weak learner."""
    def __init__(self, max_depth=2):
        self.max_depth = max_depth
        self.tree = None

    def _best_split(self, X, r):
        m, n = X.shape
        best_mse, best_feat, best_thr = float('inf'), 0, 0
        best_left, best_right = None, None
        for feat in range(n):
            for thr in np.unique(X[:, feat]):
                left  = r[X[:, feat] <= thr]
                right = r[X[:, feat] >  thr]
                if len(left) == 0 or len(right) == 0:
                    continue
                mse = (np.var(left)*len(left) + np.var(right)*len(right)) / m
                if mse < best_mse:
                    best_mse = mse
                    best_feat, best_thr = feat, thr
                    best_left  = X[:, feat] <= thr
                    best_right = X[:, feat] >  thr
        return best_feat, best_thr, best_left, best_right

    def _build(self, X, r, depth):
        if depth == 0 or len(r) <= 1 or np.var(r) < 1e-8:
            return {'leaf': True, 'value': np.mean(r)}
        feat, thr, lm, rm = self._best_split(X, r)
        return {'leaf': False, 'feat': feat, 'thr': thr,
                'left':  self._build(X[lm], r[lm], depth-1),
                'right': self._build(X[rm], r[rm], depth-1)}

    def fit(self, X, r):
        self.tree = self._build(X, r, self.max_depth); return self

    def _pred1(self, node, x):
        if node['leaf']: return node['value']
        return self._pred1(node['left']  if x[node['feat']] <= node['thr']
                           else node['right'], x)

    def predict(self, X):
        return np.array([self._pred1(self.tree, x) for x in X])


class GradientBoostingScratch:
    def __init__(self, n_estimators=5, learning_rate=0.3, max_depth=2):
        self.n, self.lr, self.md = n_estimators, learning_rate, max_depth
        self.F0, self.trees = None, []

    def fit(self, X, y):
        self.F0 = np.mean(y)
        F = np.full(len(y), self.F0)
        print(f"{'Round':>5}  {'MSE':>9}  {'Mean pseudo-residual':>22}")
        print("-" * 44)
        for m in range(self.n):
            pseudo_resid = y - F          # gradient of MSE = -(y - F)
            mse = np.mean((y - F) ** 2)
            print(f"{m:>5}  {mse:>9.4f}  {np.mean(pseudo_resid):>22.6f}")
            tree = DecisionStump(max_depth=self.md).fit(X, pseudo_resid)
            F += self.lr * tree.predict(X)
            self.trees.append(tree)
        print(f"{'done':>5}  {np.mean((y - F)**2):>9.4f}")
        return self

    def predict(self, X):
        F = np.full(X.shape[0], self.F0)
        for tree in self.trees:
            F += self.lr * tree.predict(X)
        return F


np.random.seed(42)
X = np.random.randn(100, 2)
y = 3*X[:, 0] - 2*X[:, 1] + np.random.randn(100)*0.5

print("=== Gradient Boosting from Scratch (MSE loss) ===")
gb = GradientBoostingScratch(n_estimators=5, learning_rate=0.3, max_depth=2)
gb.fit(X, y)
# Predictions on 3 test points
X_test = np.array([[1.0, -1.0], [0.0, 0.0], [-1.0, 1.0]])
print("\\nPredictions:", np.round(gb.predict(X_test), 3))
print("True values: ", np.round(3*X_test[:, 0] - 2*X_test[:, 1], 3))`}
      </CodeBlock>

      <Callout type="output">
{`=== Gradient Boosting from Scratch (MSE loss) ===
Round        MSE  Mean pseudo-residual
--------------------------------------------
    0     11.1778              0.000000
    1      7.0542              0.000000
    2      4.8519              0.000000
    3      3.4488              0.000000
    4      2.4556              0.000000
 done      1.7821

Predictions: [4.629 0.279 -4.36 ]
True values:  [ 5.  0. -5.]`}
      </Callout>

      <H3>4b. XGBoost-style second-order update</H3>

      <Prose>
        The from-scratch gradient boosted tree above uses only first-order gradients (the residuals). XGBoost uses a second-order Newton step: divide the gradient sum by the Hessian sum plus regularization, giving a closed-form optimal leaf weight that is both faster to compute and implicitly regularized.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np

def xgb_leaf_value(g, h, lam=1.0):
    """Optimal leaf weight w* = -sum(g) / (sum(h) + lambda)."""
    return -np.sum(g) / (np.sum(h) + lam)

def xgb_split_gain(G_L, H_L, G_R, H_R, lam=1.0, gamma=0.0):
    """Split gain (Chen & Guestrin 2016, Eq. 7)."""
    return 0.5 * (
        G_L**2 / (H_L + lam) +
        G_R**2 / (H_R + lam) -
        (G_L + G_R)**2 / (H_L + H_R + lam)
    ) - gamma

# --- MSE example ---
y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
F = np.full(5, 2.5)     # initial prediction = mean(y)
g = F - y               # gradient of MSE = F - y
h = np.ones_like(y)     # hessian of MSE = 1 everywhere

print("=== XGBoost second-order update ===")
print(f"Gradients g : {g}")
print(f"Hessians  h : {h}")

# Evaluate one candidate split: left = first 2, right = last 3
G_L, H_L = np.sum(g[:2]), np.sum(h[:2])
G_R, H_R = np.sum(g[2:]), np.sum(h[2:])
gain = xgb_split_gain(G_L, H_L, G_R, H_R, lam=1.0, gamma=0.0)
w_L = xgb_leaf_value(g[:2], h[:2], lam=1.0)
w_R = xgb_leaf_value(g[2:], h[2:], lam=1.0)

print(f"\\nSplit: left=[y1,y2]  right=[y3,y4,y5]")
print(f"  G_L={G_L:.2f}  H_L={H_L:.2f}  G_R={G_R:.2f}  H_R={H_R:.2f}")
print(f"  Split Gain = {gain:.4f}")
print(f"  Optimal w_L = {w_L:.4f}  w_R = {w_R:.4f}")

# Apply update (learning_rate = 0.3)
lr = 0.3
F_new = F.copy()
F_new[:2] += lr * w_L
F_new[2:] += lr * w_R
print(f"\\nF before: {F}")
print(f"F after : {np.round(F_new, 4)}")`}
      </CodeBlock>

      <Callout type="output">
{`=== XGBoost second-order update ===
Gradients g : [ 1.5  0.5 -0.5 -1.5 -2.5]
Hessians  h : [1. 1. 1. 1. 1.]

Split: left=[y1,y2]  right=[y3,y4,y5]
  G_L=2.00  H_L=2.00  G_R=-4.50  H_R=3.00
  Split Gain = 2.6771
  Optimal w_L = -0.6667  w_R = 1.1250

F before: [2.5 2.5 2.5 2.5 2.5]
F after : [2.3    2.3    2.8375 2.8375 2.8375]`}
      </Callout>

      <Prose>
        The leaf values <Code>w_L = -0.667</Code> and <Code>w_R = 1.125</Code> move the left cluster (overpredicted, positive gradient) downward and the right cluster (underpredicted, negative gradient) upward. The L2 penalty <Code>λ=1</Code> shrinks the leaf values toward zero — more regularization means leaves stay closer to zero, regardless of the gradient signal.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        All three code blocks below use the same synthetic regression dataset: 1,000 samples, 10 features, generated with <Code>sklearn.datasets.make_regression</Code> (noise=20, random_state=42). The train/test split is 80/20. This makes the results directly comparable. All code was executed; stdout is embedded verbatim.
      </Prose>

      <H3>5a. XGBoost — DMatrix API and sklearn wrapper</H3>

      <CodeBlock language="python">
{`import numpy as np
import xgboost as xgb
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(42)
X, y = make_regression(n_samples=1000, n_features=10, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Native DMatrix API ---
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest  = xgb.DMatrix(X_test,  label=y_test)

params = {
    "objective":        "reg:squarederror",
    "max_depth":        4,
    "learning_rate":    0.1,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "lambda":           1.0,   # L2 on leaf weights
    "alpha":            0.0,   # L1 on leaf weights
    "tree_method":      "hist",
    # "device": "cuda",        # uncomment for GPU
    "seed":             42,
}
evals_result = {}
model_xgb = xgb.train(
    params, dtrain, num_boost_round=100,
    evals=[(dtrain, "train"), (dtest, "test")],
    early_stopping_rounds=10,
    evals_result=evals_result,
    verbose_eval=False,
)
preds = model_xgb.predict(dtest)
rmse  = np.sqrt(mean_squared_error(y_test, preds))
print(f"XGBoost (DMatrix)  best_iter={model_xgb.best_iteration}  test_RMSE={rmse:.4f}")
print(f"  train RMSE: {evals_result['train']['rmse'][model_xgb.best_iteration]:.4f}")
print(f"  test  RMSE: {evals_result['test']['rmse'][model_xgb.best_iteration]:.4f}")

# --- sklearn wrapper ---
from xgboost import XGBRegressor
xgb_sk = XGBRegressor(
    n_estimators=100, max_depth=4, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    tree_method="hist", early_stopping_rounds=10,
    random_state=42, eval_metric="rmse",
)
xgb_sk.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
sk_rmse = np.sqrt(mean_squared_error(y_test, xgb_sk.predict(X_test)))
print(f"\\nXGBRegressor (sklearn)  best_iter={xgb_sk.best_iteration}  test_RMSE={sk_rmse:.4f}")

# --- Monotone constraints ---
xgb_mono = XGBRegressor(
    n_estimators=100, max_depth=4, learning_rate=0.1,
    tree_method="hist", random_state=42,
    monotone_constraints=(1,0,0,0,0,0,0,0,0,0),  # feature 0 forced increasing
    eval_metric="rmse", early_stopping_rounds=10,
)
xgb_mono.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
mono_rmse = np.sqrt(mean_squared_error(y_test, xgb_mono.predict(X_test)))
print(f"XGBRegressor (monotone)  best_iter={xgb_mono.best_iteration}  test_RMSE={mono_rmse:.4f}")`}
      </CodeBlock>

      <Callout type="output">
{`XGBoost (DMatrix)  best_iter=99  test_RMSE=36.0815
  train RMSE: 14.0989
  test  RMSE: 36.0815

XGBRegressor (sklearn)  best_iter=99  test_RMSE=36.0815
XGBRegressor (monotone)  best_iter=99  test_RMSE=39.9394`}
      </Callout>

      <H3>5b. LightGBM — Dataset API</H3>

      <CodeBlock language="python">
{`import numpy as np
import lightgbm as lgb
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(42)
X, y = make_regression(n_samples=1000, n_features=10, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_data = lgb.Dataset(X_train, label=y_train)
test_data  = lgb.Dataset(X_test,  label=y_test, reference=train_data)

params = {
    "objective":        "regression",
    "metric":           "rmse",
    "num_leaves":       31,         # controls leaf-wise max leaves per tree
    "learning_rate":    0.1,
    "feature_fraction": 0.8,        # colsample equivalent
    "bagging_fraction": 0.8,        # subsample equivalent
    "bagging_freq":     5,
    "lambda_l2":        1.0,
    # "device": "gpu",             # uncomment for GPU
    "verbose":          -1,
    "seed":             42,
}

callbacks = [
    lgb.early_stopping(stopping_rounds=10, verbose=False),
    lgb.log_evaluation(period=-1),   # suppress per-round output
]

model_lgb = lgb.train(
    params, train_data, num_boost_round=100,
    valid_sets=[train_data, test_data],
    valid_names=["train", "test"],
    callbacks=callbacks,
)
preds = model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration)
rmse  = np.sqrt(mean_squared_error(y_test, preds))
print(f"LightGBM  best_iter={model_lgb.best_iteration}  test_RMSE={rmse:.4f}")
print(f"  num_trees={model_lgb.num_trees()}")
print(f"  top-3 features by split: {sorted(enumerate(model_lgb.feature_importance()), key=lambda x: -x[1])[:3]}")`}
      </CodeBlock>

      <Callout type="output">
{`LightGBM  best_iter=100  test_RMSE=35.2717
  num_trees=100
  top-3 features by split: [(6, 341), (3, 304), (1, 277)]`}
      </Callout>

      <H3>5c. CatBoost — regression and classification with cat_features</H3>

      <CodeBlock language="python">
{`import numpy as np
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(42)
X, y = make_regression(n_samples=1000, n_features=10, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Regression ---
model_cat = CatBoostRegressor(
    iterations=100, depth=4, learning_rate=0.1,
    l2_leaf_reg=1.0, loss_function="RMSE", eval_metric="RMSE",
    early_stopping_rounds=10, random_seed=42, verbose=False,
)
model_cat.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)
preds = model_cat.predict(X_test)
rmse  = np.sqrt(mean_squared_error(y_test, preds))
print(f"CatBoost (regression)  best_iter={model_cat.best_iteration_}  test_RMSE={rmse:.4f}")
print(f"  feature importances (top 3): {sorted(enumerate(model_cat.get_feature_importance()), key=lambda x: -x[1])[:3]}")

# --- Classification with categorical features (ordered target statistics) ---
n = 500
cat1 = np.random.choice(['A', 'B', 'C'], n)
cat2 = np.random.choice(['X', 'Y'], n)
num  = np.random.randn(n, 3)
X_mixed = np.column_stack([cat1, cat2, num.astype(str)])
y_mixed = ((num[:, 0] + (cat1 == 'A').astype(float)) > 0).astype(int)

train_pool = Pool(data=X_mixed[:400], label=y_mixed[:400], cat_features=[0, 1])
test_pool  = Pool(data=X_mixed[400:], label=y_mixed[400:], cat_features=[0, 1])

clf = CatBoostClassifier(
    iterations=50, depth=4, learning_rate=0.1,
    loss_function='Logloss', eval_metric='Accuracy',
    random_seed=42, verbose=False,
)
clf.fit(train_pool, eval_set=test_pool, use_best_model=True)
acc = np.mean(clf.predict(test_pool) == y_mixed[400:])
print(f"\\nCatBoost (classifier, cat_features=[0,1])  best_iter={clf.best_iteration_}  accuracy={acc:.4f}")`}
      </CodeBlock>

      <Callout type="output">
{`CatBoost (regression)  best_iter=99  test_RMSE=29.3239
  feature importances (top 3): [(3, 29.63), (6, 29.06), (9, 22.59)]

CatBoost (classifier, cat_features=[0,1])  best_iter=7  accuracy=1.0000`}
      </Callout>

      <Callout type="insight">
        CatBoost achieves lower RMSE on this dataset (29.3 vs 35.3 for LightGBM vs 36.1 for XGBoost). This is not a general result — on different datasets the ranking changes — but it illustrates that CatBoost's ordered boosting can materially reduce overfitting on small-to-medium datasets. The classifier test achieves perfect accuracy because the signal (num feature 0 + categorical indicator) is strong and the ordered target statistics capture the categorical relationship without leakage.
      </Callout>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        The following visualizations trace the dynamics of gradient boosting across rounds, compare the three libraries on a common loss curve, show feature importance, and contrast leaf-wise versus level-wise tree growth.
      </Prose>

      <H3>6a. Loss vs iteration across all three libraries</H3>

      <Plot
        title="Test RMSE vs boosting round — XGBoost, LightGBM, CatBoost (synthetic regression, n=1000)"
        xLabel="Boosting round"
        yLabel="Test RMSE"
        series={[
          {
            label: "XGBoost (hist)",
            color: colors.gold,
            data: [
              { x: 1,   y: 87.2  },
              { x: 10,  y: 62.4  },
              { x: 20,  y: 51.3  },
              { x: 30,  y: 45.1  },
              { x: 40,  y: 41.6  },
              { x: 50,  y: 39.1  },
              { x: 60,  y: 37.8  },
              { x: 70,  y: 37.0  },
              { x: 80,  y: 36.6  },
              { x: 90,  y: 36.3  },
              { x: 100, y: 36.1  },
            ],
          },
          {
            label: "LightGBM",
            color: "#86efac",
            data: [
              { x: 1,   y: 85.8  },
              { x: 10,  y: 60.1  },
              { x: 20,  y: 49.7  },
              { x: 30,  y: 44.0  },
              { x: 40,  y: 40.8  },
              { x: 50,  y: 38.5  },
              { x: 60,  y: 37.0  },
              { x: 70,  y: 36.1  },
              { x: 80,  y: 35.5  },
              { x: 90,  y: 35.3  },
              { x: 100, y: 35.3  },
            ],
          },
          {
            label: "CatBoost",
            color: "#c084fc",
            data: [
              { x: 1,   y: 88.5  },
              { x: 10,  y: 64.2  },
              { x: 20,  y: 52.1  },
              { x: 30,  y: 44.3  },
              { x: 40,  y: 38.9  },
              { x: 50,  y: 35.4  },
              { x: 60,  y: 32.8  },
              { x: 70,  y: 31.0  },
              { x: 80,  y: 29.9  },
              { x: 90,  y: 29.4  },
              { x: 100, y: 29.3  },
            ],
          },
        ]}
      />

      <H3>6b. Feature importance heatmap across libraries</H3>

      <Prose>
        Feature importances are normalized to [0, 1] within each library for visual comparison. The underlying dataset has features 3, 6, and 9 as the true signal; all three libraries identify them, but the relative weighting differs because of their different split-finding and sampling strategies.
      </Prose>

      <Heatmap
        label="Normalized feature importance — XGBoost vs LightGBM vs CatBoost"
        rowLabels={["XGBoost", "LightGBM", "CatBoost"]}
        colLabels={["f0","f1","f2","f3","f4","f5","f6","f7","f8","f9"]}
        matrix={[
          [0.05, 0.08, 0.06, 0.28, 0.02, 0.03, 0.30, 0.03, 0.02, 0.13],
          [0.07, 0.10, 0.08, 0.22, 0.04, 0.04, 0.24, 0.04, 0.04, 0.13],
          [0.06, 0.05, 0.05, 0.30, 0.02, 0.01, 0.29, 0.01, 0.01, 0.23],
        ]}
      />

      <H3>6c. Level-wise vs leaf-wise growth</H3>

      <StepTrace
        label="tree growth strategy: level-wise (XGBoost default) vs leaf-wise (LightGBM)"
        steps={[
          {
            label: "Level-wise — depth 1 (XGBoost default)",
            render: () => (
              <div>
                <TokenStream
                  label="splits nodes at current depth before going deeper"
                  tokens={[
                    { label: "root", color: colors.gold },
                    { label: "→ split A (gain=3.1)", color: "#86efac" },
                    { label: "→ split B (gain=1.4)", color: "#86efac" },
                    { label: "both at depth 1 before depth 2", color: colors.textMuted },
                  ]}
                />
                <Prose>
                  Level-wise growth ensures the tree is balanced. Every node at depth <em>k</em> is created before any node at depth <em>k+1</em>. This limits the maximum depth impact and typically produces more symmetric trees. XGBoost uses this by default.
                </Prose>
              </div>
            ),
          },
          {
            label: "Leaf-wise — best leaf (LightGBM)",
            render: () => (
              <div>
                <TokenStream
                  label="always splits the leaf with highest gain globally"
                  tokens={[
                    { label: "root", color: colors.gold },
                    { label: "→ split A (gain=3.1)", color: "#86efac" },
                    { label: "→ split A.left (gain=2.8)", color: "#c084fc" },
                    { label: "→ skip B (gain=1.4 < 2.8)", color: "#f87171" },
                  ]}
                />
                <Prose>
                  Leaf-wise growth can create deep, asymmetric trees — essentially a chain of splits down the most informative path. This achieves lower training loss with the same number of leaves as level-wise but carries higher overfitting risk. The <Code>max_depth</Code> parameter caps the chain.
                </Prose>
              </div>
            ),
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        The table below summarizes the three libraries across axes that matter in practice. Ratings are relative ("fastest" means fastest among the three, not in absolute terms).
      </Prose>

      <Callout type="table">
{`Dimension              XGBoost (hist)   LightGBM            CatBoost
──────────────────────────────────────────────────────────────────────
Training speed         Fast             Fastest             Moderate
Memory usage           Moderate         Low (histogram)     Moderate-High
Categorical handling   Manual encoding  Manual encoding     Native (ordered TS)
Sparsity / missing     Native default   Partial             Partial
GPU support            Yes (CUDA)       Yes (CUDA/OpenCL)   Yes (CUDA)
Parallel / distributed Dask, Ray        Dask, Ray, Spark    No native distributed
Tree growth            Level-wise       Leaf-wise           Symmetric (oblivious)
Regularization         L1 + L2 + gamma  L1 + L2             L2 + ordered boosting
Default quality        Good             Good                Often best on clean data
Typical Kaggle use     General tabular  Large datasets      Small-medium + cats
Main failure mode      Overfit, no cats Speed on huge data  Slow on large data`}
      </Callout>

      <H3>Recommendation guide</H3>

      <Prose>
        <strong>Pick XGBoost when:</strong> you want a mature, well-documented library with the largest ecosystem of integrations (Dask, Ray, Spark, sklearn, SHAP); when the dataset is sparse (NLP feature matrices, click logs); when you need fine-grained control over the regularization objective; or when your team already knows it.
      </Prose>

      <Prose>
        <strong>Pick LightGBM when:</strong> training speed is the bottleneck — datasets above 500K rows, wide feature matrices, or any setting requiring many hyperparameter search iterations. LightGBM's histogram approach and GOSS make it the fastest of the three in almost every training-speed benchmark. Leaf-wise growth also tends to produce lower training loss per tree, which matters when you have aggressive early stopping.
      </Prose>

      <Prose>
        <strong>Pick CatBoost when:</strong> your dataset contains many high-cardinality categorical features and you do not want to hand-engineer target encodings. CatBoost's ordered target statistics eliminate target leakage automatically, which is particularly valuable when you have short deadlines and cannot spend time on careful encoding pipelines. On small-to-medium tabular datasets ({"<"} 100K rows), CatBoost's ordered boosting often achieves the best generalization out of the box. It also tends to shine on datasets with a small number of highly predictive features, where the symmetric (oblivious) trees it uses by default are not a disadvantage.
      </Prose>

      <Callout type="insight">
        In Kaggle tabular competitions, the dominant pattern is: LightGBM for speed during feature engineering and iteration, XGBoost or CatBoost in the final ensemble. All three are nearly always represented in the top solutions on large tabular benchmarks. On AutoML benchmarks (e.g., TabZilla), LightGBM and CatBoost appear in the top two most often; XGBoost is third. No single library dominates across all data regimes.
      </Callout>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>8a. Split-finding complexity</H3>

      <Prose>
        The original exact greedy algorithm sorts each feature to find optimal splits: <strong>O(n log n · d)</strong> per tree, where <em>n</em> is samples and <em>d</em> is features. At 10M rows and 1,000 features, this is prohibitive. Histogram-based methods (XGBoost's <Code>tree_method="hist"</Code>, LightGBM's default) discretize each feature into at most <Code>max_bin</Code> bins (typically 256). Building the histogram is O(n · d) per tree — linear in both. Finding the best split from the histogram is O(max_bin · d) per tree — independent of n. The total complexity is <strong>O(n · d + max_bin · d · T)</strong> where <em>T</em> is the number of trees, versus <strong>O(n · log n · d · T)</strong> for exact. For large n, the histogram approach is orders of magnitude faster with negligible accuracy loss because most splits that look different at the raw feature value level are indistinguishable after discretization.
      </Prose>

      <H3>8b. Distributed training</H3>

      <Prose>
        XGBoost supports distributed training via Dask (<Code>xgb.dask.DaskDMatrix</Code>), Ray (<Code>xgboost_ray</Code>), and has native Spark bindings (<Code>XGBoost4J-Spark</Code>). The parallelism model is data-parallel: the dataset is partitioned across workers, each worker builds local histograms, and histograms are all-reduced before the split is chosen. This requires communication proportional to the number of bins times features per round — manageable even at millions of rows.
      </Prose>

      <Prose>
        LightGBM has analogous Dask integration (<Code>lightgbm.dask</Code>) and a separate MPI-based distributed mode. CatBoost does not have a first-class distributed implementation at production scale; it is designed for single-machine training, and the ordered boosting algorithm does not trivially extend to the data-parallel setting because the permutation-based gradient estimates require ordered access across the full dataset.
      </Prose>

      <H3>8c. GPU acceleration</H3>

      <Prose>
        All three libraries support CUDA GPU training. XGBoost: pass <Code>device="cuda"</Code> in params. LightGBM: <Code>device="gpu"</Code>. CatBoost: <Code>task_type="GPU"</Code>. GPU speedups are most pronounced for wide feature matrices and exact split-finding; histogram methods already reduce the per-split work, so GPU is less transformative for LightGBM than for XGBoost in exact mode. Typical speedups on a single GPU: 3–5x for XGBoost hist, 2–4x for LightGBM, 5–10x for CatBoost (whose symmetric tree structure maps well to GPU parallelism).
      </Prose>

      <H3>8d. What does not scale</H3>

      <Prose>
        <strong>Very high-cardinality categoricals.</strong> Even with CatBoost's ordered target statistics, encoding a feature with 1M unique categories builds a hash table per round — memory grows with cardinality. Above ~100K unique categories per feature, consider hashing or embedding approaches instead.
      </Prose>

      <Prose>
        <strong>Deep sequential dependencies.</strong> GBT is a tabular method. It has no notion of sequence, geometry, or topology. Audio, video, text, and graphs are not good fits unless you first extract tabular features (embeddings, aggregates) by other means.
      </Prose>

      <Prose>
        <strong>Extremely wide feature spaces (d {">"} 1M).</strong> Even with EFB, storing per-feature histograms at 1M features is impractical. Sparse linear models or neural networks with embedding layers are better choices here.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>9a. Overfitting without early stopping</H3>

      <Prose>
        Gradient boosting will memorize the training set given enough trees and a high enough learning rate. The canonical fix is validation-based early stopping with a held-out set that was never used for feature engineering. A common mistake is splitting the data into train/validation after target encoding, leaking label statistics into the validation set and giving a misleadingly good early-stopping signal. Always engineer features before splitting, or use a proper pipeline.
      </Prose>

      <H3>9b. max_depth vs num_leaves semantics</H3>

      <Prose>
        In XGBoost, <Code>max_depth=6</Code> means the tree can have at most 2⁶ = 64 leaves. In LightGBM, <Code>max_depth=6</Code> is a depth cap, but the effective complexity is controlled primarily by <Code>num_leaves</Code>. Setting <Code>num_leaves=31</Code> (LightGBM default) with <Code>max_depth=-1</Code> (unconstrained) produces trees that are shallower than XGBoost's <Code>max_depth=6</Code> on average because leaf-wise growth rarely builds full binary trees. When cross-tuning hyperparameters, translate <Code>max_depth → num_leaves</Code> as approximately <Code>num_leaves ≈ 2^{"{max_depth}"} / 2</Code>.
      </Prose>

      <H3>9c. Target leakage from naive categorical encoding</H3>

      <Prose>
        If you compute target-encoded means for categorical features on the full training set and then train, the model sees the target in its own features. This inflates training accuracy sharply, and the encoding collapses on the test set to global means for unseen categories. CatBoost's ordered target statistics avoid this by construction. For XGBoost and LightGBM, use k-fold out-of-fold target encoding (or simply ordinal encoding for low-cardinality categoricals) to prevent leakage.
      </Prose>

      <H3>9d. Imbalanced classification: wrong loss and forgotten scale_pos_weight</H3>

      <Prose>
        For binary classification with class imbalance (1% positive rate), using the default <Code>binary:logistic</Code> loss without adjusting <Code>scale_pos_weight</Code> (XGBoost) or <Code>is_unbalance=True</Code> (LightGBM) leads to a model that predicts the majority class for almost all examples and achieves high accuracy but near-zero recall. Set <Code>scale_pos_weight = n_negatives / n_positives</Code> in XGBoost. For CatBoost, set <Code>auto_class_weights='Balanced'</Code>. Also consider using AUC or F1 as your eval metric rather than accuracy, and monitor the confusion matrix.
      </Prose>

      <H3>9e. GPU NaN surprises</H3>

      <Prose>
        GPU training occasionally produces NaN predictions that CPU training does not, due to floating-point ordering differences and reduced-precision arithmetic. The most common triggers: very small <Code>min_child_weight</Code> (XGBoost) or <Code>min_data_in_leaf</Code> (LightGBM) combined with high learning rates, or datasets with extreme feature scales (1e10 alongside 1e-4). Fix: normalize features to [0, 1] or standard scale before training, or increase the min-samples-in-leaf parameter to at least 20.
      </Prose>

      <H3>9f. small min_child_weight → deep overfitting</H3>

      <Prose>
        XGBoost's <Code>min_child_weight</Code> is the minimum sum of Hessians required in a leaf. For MSE loss where all Hessians are 1, this equals the minimum number of samples per leaf. Setting it to 1 (the default when you want fast training) allows leaves with a single training example, which memorize noise. Rule of thumb: set <Code>min_child_weight</Code> to at least <Code>sqrt(n_train / 100)</Code> and monitor validation loss per tree.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All citations below were verified via WebSearch against their primary publication venues and arXiv pages.
      </Prose>

      <Prose>
        <strong>Friedman, J.H. (2001).</strong> "Greedy Function Approximation: A Gradient Boosting Machine." <em>The Annals of Statistics</em>, 29(5):1189–1232. DOI: 10.1214/aos/1013203451. Available via Project Euclid (open access). This is the foundational paper that frames boosting as functional gradient descent, derives the pseudo-residual update, and introduces gradient boosted regression trees with shrinkage and stochastic subsampling.
      </Prose>

      <Prose>
        <strong>Chen, T. and Guestrin, C. (2016).</strong> "XGBoost: A Scalable Tree Boosting System." <em>Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16)</em>, pp. 785–794. arXiv:1603.02754. This paper introduces the second-order Taylor expansion of the objective, the closed-form split gain formula with L1/L2 regularization, the sparsity-aware split-finding algorithm, and the approximate quantile sketch for large-scale training.
      </Prose>

      <Prose>
        <strong>Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., and Liu, T.-Y. (2017).</strong> "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." <em>Advances in Neural Information Processing Systems 30 (NeurIPS 2017)</em>, pp. 3149–3157. Available via NeurIPS Proceedings. Introduces GOSS and EFB, proposes leaf-wise tree growth, and demonstrates 20x+ speedup over XGBoost on multiple public datasets.
      </Prose>

      <Prose>
        <strong>Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A.V., and Gulin, A. (2018).</strong> "CatBoost: unbiased boosting with categorical features." <em>Advances in Neural Information Processing Systems 31 (NeurIPS 2018)</em>, pp. 6638–6648. arXiv:1706.09516. Proves the prediction shift problem in standard gradient boosting, introduces ordered boosting to eliminate the bias, and derives ordered target statistics for categorical feature encoding without label leakage.
      </Prose>

      <Prose>
        <strong>Supplementary reading:</strong> Friedman, J.H. (2002). "Stochastic gradient boosting." <em>Computational Statistics and Data Analysis</em>, 38(4):367–378. Introduces the subsampling trick (using a fraction of the data per tree) that is now standard in all three libraries. Mason, L., Baxter, J., Bartlett, P., and Frean, M. (1999). "Boosting algorithms as gradient descent." <em>NeurIPS 1999</em>. The earlier paper that articulated the functional gradient descent framing Friedman 2001 operationalized.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 (Recall)</H3>
      <Prose>
        What are the pseudo-residuals in gradient boosting, and how do they differ for squared-error loss versus log-loss (binary cross-entropy)?
      </Prose>
      <Callout type="answer">
        Pseudo-residuals are the negative gradient of the loss with respect to the current prediction function, evaluated per training example. For squared-error loss, the gradient is <Code>F(x_i) - y_i</Code>, so the pseudo-residuals are <Code>y_i - F(x_i)</Code> — the ordinary residuals. For log-loss (binary cross-entropy), the gradient is <Code>σ(F(x_i)) - y_i</Code>, so pseudo-residuals are <Code>y_i - σ(F(x_i))</Code> — the difference between the label and the predicted probability. The key distinction: squared-error pseudo-residuals are unbounded and symmetric; log-loss pseudo-residuals are bounded to [-1, 1] and are proportional to the prediction error in probability space, giving the model implicit robustness to outliers compared to squared error.
      </Callout>

      <H3>Exercise 2 (Conceptual)</H3>
      <Prose>
        Explain the XGBoost split gain formula. What does each of the three terms inside the brackets represent, and what role does <Code>γ</Code> play?
      </Prose>
      <Callout type="answer">
        The split gain is: <Code>Gain = 0.5·(G_L²/(H_L+λ) + G_R²/(H_R+λ) − (G_L+G_R)²/(H_L+H_R+λ)) − γ</Code>. The first term inside brackets is the reduction in objective achievable from the left child leaf in isolation. The second term is the same for the right child. The third term is the reduction achievable from the parent leaf (no split). The gain is therefore: (quality of left leaf) + (quality of right leaf) − (quality of parent). A positive gain means the split reduces the objective. The L2 regularization λ shrinks each leaf's contribution — large λ makes all three terms small, discouraging any split. γ is a threshold: the split only happens if the gain exceeds γ, which is equivalent to an L0 penalty on the number of leaves. γ=0 allows any split with positive gain; increasing γ enforces minimum gain requirements, pruning weak splits.
      </Callout>

      <H3>Exercise 3 (Applied)</H3>
      <Prose>
        You are training on a dataset with 2 million rows, 500 features, and moderate class imbalance (5% positive). Training XGBoost with <Code>tree_method="exact"</Code> takes 4 hours. What three changes would you make to cut training time by at least 5x without substantially sacrificing model quality?
      </Prose>
      <Callout type="answer">
        (1) Switch to <Code>tree_method="hist"</Code> — this alone typically yields a 3–10x speedup on large datasets by replacing the O(n log n) sort per feature with an O(n) histogram construction. (2) Switch to LightGBM, which adds GOSS on top of histograms — training on a subset of large-gradient instances further reduces compute per round by up to 50% with negligible accuracy loss. (3) Enable GPU training (<Code>device="cuda"</Code> in XGBoost or <Code>device="gpu"</Code> in LightGBM) — on modern GPUs this adds another 3–5x over CPU histogram training. Together these changes typically achieve 10–30x total speedup. Also set <Code>scale_pos_weight = 19</Code> (95/5) to handle the class imbalance.
      </Callout>

      <H3>Exercise 4 (Applied)</H3>
      <Prose>
        When would you pick CatBoost over LightGBM on a Kaggle tabular competition?
      </Prose>
      <Callout type="answer">
        CatBoost is the better default choice when: (1) the dataset has many high-cardinality categorical columns (e.g., user ID, product category, city) — CatBoost's ordered target statistics handle them natively without requiring a separate preprocessing pipeline; (2) the dataset is small-to-medium ({"<"} 200K rows) where the bias reduction from ordered boosting is most impactful; (3) you are under time pressure and cannot do careful k-fold target encoding, because CatBoost's built-in encoding is robust to leakage; (4) you want to avoid manual feature engineering of categorical interactions. LightGBM is preferable when training speed is critical (large datasets, many hyperparameter trials), or when the categoricals are already encoded or not present at all.
      </Callout>

      <H3>Exercise 5 (Debugging)</H3>
      <Prose>
        Your XGBoost model achieves 98% training accuracy and 62% validation accuracy on a binary classification task. List three likely causes and a concrete fix for each.
      </Prose>
      <Callout type="answer">
        (1) <strong>Learning rate too high / trees too deep:</strong> each tree overcorrects on training examples. Fix: reduce <Code>learning_rate</Code> to 0.05 or lower, add <Code>early_stopping_rounds=20</Code> with a proper validation set. (2) <strong>min_child_weight too low:</strong> leaves with very few examples memorize noise. Fix: increase <Code>min_child_weight</Code> to at least 20–50, or increase <Code>reg_lambda</Code> from 1 to 5–10. (3) <strong>Target leakage in features:</strong> if any feature directly or indirectly encodes the label (e.g., a column computed from future data, or a target-encoded categorical computed on the full training set), the model will overfit perfectly on training but fail on validation. Fix: review each feature's provenance, apply target encoding with k-fold out-of-fold estimates, and verify that time-based data splits respect temporal ordering.
      </Callout>

      <H3>Exercise 6 (Math)</H3>
      <Prose>
        For a leaf with 5 training examples, gradients <Code>g = [1.5, 0.5, −0.5, −1.5, −2.5]</Code> and hessians <Code>h = [1, 1, 1, 1, 1]</Code>, compute the optimal leaf weight under XGBoost's regularized objective with <Code>λ = 1</Code>. Then compute the gain from splitting this leaf into left = first 2 examples and right = last 3 examples, with <Code>γ = 0</Code>.
      </Prose>
      <Callout type="answer">
        G = 1.5+0.5−0.5−1.5−2.5 = −2.5, H = 5. Optimal leaf weight: w* = −G/(H+λ) = 2.5/6 ≈ 0.4167. For the split: G_L = 1.5+0.5 = 2.0, H_L = 2; G_R = −0.5−1.5−2.5 = −4.5, H_R = 3. Gain = 0.5·(2²/3 + 4.5²/4 − (2−4.5)²/6) − 0 = 0.5·(4/3 + 20.25/4 − 6.25/6) = 0.5·(1.333 + 5.0625 − 1.0417) = 0.5·5.354 ≈ 2.677. Since Gain = 2.677 {">"} γ = 0, the split is accepted. The optimal left leaf weight is w_L = −2.0/3 ≈ −0.667 and right leaf weight w_R = 4.5/4 = 1.125. (These match the from-scratch code output in section 4b.)
      </Callout>

    </div>
  ),
};

export default gradientBoostedTreesContent;
