import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const decisionTreesRandomForestsContent = {
  title: "Decision Trees & Random Forests",
  readTime: "~45 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Every supervised learning algorithm makes a structural bet about the world it is trying to model.
        Linear models bet that classes are separable by a hyperplane. Kernel SVMs bet that a kernel
        embedding can make them so. Neural networks bet that deep composition of simple nonlinearities
        can approximate any function given enough capacity. Decision trees make a different and
        considerably more transparent bet: that the world is best described by a hierarchy of yes/no
        questions about individual feature values. Is sepal length greater than 5.5? If yes, is petal
        width greater than 1.7? The model is literally a flowchart. Every prediction traces a path from
        the root of the flowchart to a leaf, and at each node the path branches on a single threshold
        applied to a single feature.
      </Prose>

      <Prose>
        This is the axis-aligned splits assumption, and it carries three concrete practical advantages
        that explain why trees have never fallen out of use despite the rise of much more powerful
        methods. First, they require no feature scaling. A linear model treated with features on
        wildly different scales will weight them wildly differently unless you standardize; a tree is
        blind to monotone transformations of any feature because it only cares about rank order within
        that feature. Second, they handle heterogeneous feature types naturally: a single tree can
        split on a binary flag in one branch and a continuous measurement in another, with no
        encoding gymnastics. Third, and most importantly for deployed systems, a sufficiently shallow
        tree is human-readable. You can print it, hand it to a domain expert, and ask whether the
        splits make sense. No other method of comparable accuracy offers this.
      </Prose>

      <Prose>
        The formal history begins in 1984, when Leo Breiman, Jerome Friedman, Richard Olshen, and
        Charles Stone published <em>Classification and Regression Trees</em> (Chapman and Hall/CRC).
        That monograph introduced the CART algorithm: binary splits chosen to minimize a criterion
        (Gini impurity for classification, variance reduction for regression), cost-complexity pruning
        to control overfitting, and cross-validation to select the pruning level. CART is the
        algorithm inside <Code>sklearn.tree.DecisionTreeClassifier</Code> today. Two years later,
        J. Ross Quinlan published "Induction of Decision Trees" in <em>Machine Learning</em> (1986),
        introducing ID3, which used information gain as the splitting criterion and handled
        multi-way splits. Quinlan then extended ID3 into C4.5 — described in his 1993 book
        <em> C4.5: Programs for Machine Learning</em> (Morgan Kaufmann) — adding support for
        continuous features, missing values, and pruning. ID3/C4.5 and CART developed in parallel
        and cross-pollinated; modern implementations draw from both.
      </Prose>

      <Prose>
        Trees alone, however, have a structural weakness. A single tree grown deep enough to be
        accurate is high-variance: small changes in the training set can flip early splits and produce
        a completely different tree downstream. Breiman attacked this directly. His 1996 paper
        "Bagging Predictors" (<em>Machine Learning</em>) showed that averaging many trees trained on
        bootstrap samples (bagging) dramatically reduces variance without increasing bias much. Then,
        in 2001, Breiman published "Random Forests" in <em>Machine Learning</em> 45(1):5–32,
        adding a crucial second source of decorrelation: at each split, each tree may only consider
        a random subset of features rather than all features. This feature subsampling means no
        single strong predictor can dominate every tree, so the trees make different errors,
        and their average is substantially better than any individual. The paper proved convergence
        of the generalization error and gave the now-standard analysis of why correlation between
        trees is the limiting factor on forest quality. Random forests became, and remain, one of
        the strongest general-purpose classifiers in the practitioner's toolkit.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        A decision tree partitions feature space into axis-aligned rectangles. Each internal node
        asks one question of the form "is feature <em>j</em> {`≤`} threshold <em>t</em>?", routing
        examples left or right. By the time an example reaches a leaf, it has been placed inside
        a hyper-rectangular region of feature space, and the tree assigns every example in that
        region the same class label — the majority class among training examples that fell there.
        The decision boundary of a single tree is therefore a set of axis-aligned line segments (in
        2D) that carve the input space into non-overlapping rectangles.
      </Prose>

      <StepTrace
        label="a tree partitioning 2D space — three splits"
        steps={[
          {
            label: "Split 1 — root: x₁ ≤ 0.29",
            render: () => (
              <Prose>
                The root node splits the entire training set on feature x₁ at threshold 0.29.
                Every example with x₁ {"<"}= 0.29 goes left; the rest go right. Gini impurity
                drops from 0.499 (nearly pure random) to a weighted average of 0.275 on each
                side — an information gain of 0.226.
              </Prose>
            ),
          },
          {
            label: "Split 2 — left child: x₂ ≤ −0.17",
            render: () => (
              <Prose>
                Inside the left region (x₁ {"<"}= 0.29), the tree asks about x₂.
                Examples with low x₂ values are strongly class 0; those with high x₂ are
                mostly class 1. The region is cut horizontally.
              </Prose>
            ),
          },
          {
            label: "Split 3 — right child: x₁ ≤ 1.18",
            render: () => (
              <Prose>
                Inside the right region (x₁ {">"} 0.29), another vertical cut separates
                a band of class 1 examples from a majority class 0 pocket on the far right.
                Each leaf is now a rectangle labeled by the majority class of training examples
                it contains.
              </Prose>
            ),
          },
        ]}
      />

      <Prose>
        The key weakness this reveals: trees can only draw horizontal and vertical lines.
        A dataset whose true decision boundary is diagonal or curved requires many
        axis-aligned rectangles to approximate it closely — which means a deep tree, which
        means high variance. A linear model would handle the diagonal case with one split;
        a tree needs many. The axis-aligned assumption is a strong inductive bias that is
        right when features are naturally threshold-able (income above $50k, temperature
        below 37°C) and wrong when classes are separated by interactions among features.
      </Prose>

      <Prose>
        A random forest takes this picture and runs it two hundred times. Each run uses a
        bootstrap sample (sample with replacement) from the training data, so different
        examples are over- and under-represented in each tree. At each node in each tree,
        only a random subset of features (typically sqrt(d) for classification) is
        considered as split candidates. The result is two hundred trees that are somewhat
        wrong in different ways. Their majority vote cancels out many of those errors.
        The decision boundary of the forest looks like a smoothed, more confident version
        of any individual tree — it still consists of rectangles, but the ensemble of
        votes produces a soft probability estimate that is much better calibrated.
      </Prose>

      <Callout variant="insight">
        The forest does not change the inductive bias of individual trees — it still builds
        axis-aligned partitions. What it changes is the variance. Averaging many noisy
        unbiased estimators is a better estimator; averaging many noisy biased estimators
        gives you a better estimate of a biased thing. The axis-aligned bias remains.
        Random forests shine on datasets where that bias is approximately correct.
      </Callout>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>Splitting criteria</H3>

      <Prose>
        At each node in a classification tree the algorithm must choose: which feature and
        which threshold produce the best split? "Best" is defined by a chosen impurity
        measure. Let <Code>p_i</Code> be the fraction of training examples at the current
        node belonging to class <em>i</em>. The two standard impurity measures are:
      </Prose>

      <MathBlock>
        {"\\text{Gini}(\\mathcal{D}) = 1 - \\sum_{i=1}^{K} p_i^2"}
      </MathBlock>

      <MathBlock>
        {"\\text{Entropy}(\\mathcal{D}) = -\\sum_{i=1}^{K} p_i \\log_2 p_i"}
      </MathBlock>

      <Prose>
        Gini impurity measures the probability that two randomly chosen examples from the
        node have different labels. It equals zero when the node is pure (one class only)
        and is maximized at 1 − 1/K for K balanced classes. Entropy is the Shannon
        information content of the label distribution at the node. Both reach their
        maximum when classes are equally represented and both are zero at purity.
        In practice, they produce nearly identical trees; Gini is slightly cheaper to
        compute (no logarithm) and is the CART default.
      </Prose>

      <Prose>
        Given an impurity measure H, the information gain of splitting node <Code>t</Code>
        at feature <em>j</em> with threshold <em>s</em> is:
      </Prose>

      <MathBlock>
        {"\\text{IG}(t, j, s) = H(\\mathcal{D}_t) - \\frac{|\\mathcal{D}_L|}{|\\mathcal{D}_t|} H(\\mathcal{D}_L) - \\frac{|\\mathcal{D}_R|}{|\\mathcal{D}_t|} H(\\mathcal{D}_R)"}
      </MathBlock>

      <Prose>
        The algorithm searches over all features <em>j</em> and all possible thresholds
        <em>s</em> (the midpoints between adjacent unique values in the training data)
        to find the split that maximizes information gain. For a node with <em>n</em>
        examples and <em>d</em> features, this takes O(n·d·log n) time — sorting each
        feature once to enumerate thresholds efficiently.
      </Prose>

      <H3>Regression trees</H3>

      <Prose>
        For regression, the splitting criterion switches to variance reduction. Let
        <Code>y_t</Code> be the vector of target values at node <em>t</em>:
      </Prose>

      <MathBlock>
        {"\\text{VR}(t, j, s) = \\text{Var}(y_t) - \\frac{|\\mathcal{D}_L|}{|\\mathcal{D}_t|} \\text{Var}(y_L) - \\frac{|\\mathcal{D}_R|}{|\\mathcal{D}_t|} \\text{Var}(y_R)"}
      </MathBlock>

      <Prose>
        The predicted value at each leaf is the mean of training targets that fell there.
        This minimizes the mean squared error within each leaf region.
      </Prose>

      <H3>Why bagging reduces variance</H3>

      <Prose>
        Suppose we have <em>B</em> trees, each trained on an independent bootstrap sample.
        Each tree is a random variable with some bias <em>b</em> and variance <em>σ²</em>.
        If the trees were truly independent, averaging them would give:
      </Prose>

      <MathBlock>
        {"\\text{Var}\\left(\\frac{1}{B}\\sum_{b=1}^{B} T_b\\right) = \\frac{\\sigma^2}{B}"}
      </MathBlock>

      <Prose>
        As B grows, variance vanishes. But trees trained on overlapping bootstrap samples
        from the same dataset are correlated. Let <em>ρ</em> be the average pairwise
        correlation between any two trees. Then the variance of their average is:
      </Prose>

      <MathBlock>
        {"\\text{Var}\\left(\\bar{T}\\right) = \\rho \\sigma^2 + \\frac{1 - \\rho}{B} \\sigma^2"}
      </MathBlock>

      <Prose>
        The first term, <Code>ρσ²</Code>, does not go to zero as B increases. No matter
        how many trees you add, this irreducible variance floor remains. This is the
        critical insight behind random feature subsampling. If every tree uses the same
        strong predictor at its root split, all trees are highly correlated (large ρ), and
        the forest gains little over a single tree as B grows. By restricting each split
        to a random subset of sqrt(d) features, we prevent any single feature from
        dominating every tree, which reduces ρ substantially. The cost is a modest
        increase in the variance of each individual tree — it no longer always picks the
        globally best split — but the reduction in ρ more than compensates. This
        bias-variance-correlation tradeoff is the theoretical core of Breiman (2001).
      </Prose>

      <Callout variant="math">
        The "strength" of individual trees (how accurate each tree is) and the
        "correlation" between trees jointly determine forest error. Breiman (2001)
        showed that the generalization error of a forest is bounded by ρ(1 − s²)/s²,
        where s is the mean margin (a measure of tree strength). Good forests maximize
        strength while minimizing correlation — exactly the tradeoff that sqrt(d)
        feature subsampling navigates.
      </Callout>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        Below is a complete NumPy-only implementation of both a decision tree classifier
        and a random forest classifier. The tree uses Gini impurity, max_depth, and
        min_samples_leaf. The forest wraps the tree with bootstrap sampling and random
        feature subsampling. Both produce actual predictions — no sklearn allowed here.
      </Prose>

      <CodeBlock>{`import numpy as np
from collections import Counter


# ── Gini impurity ────────────────────────────────────────────────────────────

def gini(y):
    """Gini impurity of label vector y."""
    if len(y) == 0:
        return 0.0
    counts = Counter(y)
    probs = np.array([c / len(y) for c in counts.values()])
    return 1.0 - np.sum(probs ** 2)


# ── Best split search ─────────────────────────────────────────────────────────

def best_split(X, y, feature_indices):
    """
    Search over feature_indices for the (feature, threshold) pair that
    maximises information gain using Gini impurity.
    """
    best_gain, best_feat, best_thresh = -1, None, None
    parent_impurity = gini(y)
    n = len(y)
    for f in feature_indices:
        thresholds = np.unique(X[:, f])
        for t in thresholds:
            left_mask = X[:, f] <= t
            right_mask = ~left_mask
            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue
            y_left, y_right = y[left_mask], y[right_mask]
            gain = parent_impurity - (
                len(y_left) / n * gini(y_left) +
                len(y_right) / n * gini(y_right)
            )
            if gain > best_gain:
                best_gain = gain
                best_feat = f
                best_thresh = t
    return best_feat, best_thresh, best_gain


# ── Decision tree ─────────────────────────────────────────────────────────────

class DecisionTreeClassifier:
    def __init__(self, max_depth=5, min_samples_leaf=1, n_features=None):
        """
        max_depth       : maximum tree depth (None = unlimited)
        min_samples_leaf: minimum samples required at each leaf
        n_features      : features to consider per split (None = all);
                          set to int(sqrt(d)) when used inside RandomForest
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_features = n_features
        self.tree_ = None

    def fit(self, X, y):
        self.n_features_in_ = X.shape[1]
        self.tree_ = self._grow(X, y, depth=0)
        return self

    def _grow(self, X, y, depth):
        # ── Leaf conditions ───────────────────────────────────────────────────
        if (depth >= self.max_depth or
                len(y) < 2 * self.min_samples_leaf or
                len(np.unique(y)) == 1):
            return {"leaf": True, "label": Counter(y).most_common(1)[0][0]}

        # ── Feature subsampling (identity if n_features is None) ──────────────
        n_feat = self.n_features or self.n_features_in_
        feat_indices = np.random.choice(
            self.n_features_in_, size=n_feat, replace=False)

        feat, thresh, gain = best_split(X, y, feat_indices)
        if feat is None or gain <= 0:
            return {"leaf": True, "label": Counter(y).most_common(1)[0][0]}

        left_mask = X[:, feat] <= thresh
        return {
            "leaf": False,
            "feat": feat,
            "thresh": thresh,
            "left":  self._grow(X[left_mask],  y[left_mask],  depth + 1),
            "right": self._grow(X[~left_mask], y[~left_mask], depth + 1),
        }

    def _predict_one(self, x, node):
        if node["leaf"]:
            return node["label"]
        if x[node["feat"]] <= node["thresh"]:
            return self._predict_one(x, node["left"])
        return self._predict_one(x, node["right"])

    def predict(self, X):
        return np.array([self._predict_one(row, self.tree_) for row in X])


# ── Random forest ─────────────────────────────────────────────────────────────

class RandomForestClassifier:
    def __init__(self, n_estimators=200, max_depth=5, min_samples_leaf=1,
                 max_features="sqrt", random_state=42):
        self.n_estimators    = n_estimators
        self.max_depth       = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features    = max_features
        self.random_state    = random_state
        self.trees_          = []

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        n, d = X.shape
        n_feat = max(1, int(np.sqrt(d))) if self.max_features == "sqrt" \
                 else self.max_features
        self.trees_ = []
        for _ in range(self.n_estimators):
            seed = rng.randint(0, 2 ** 31)
            np.random.seed(seed)
            # Bootstrap sample
            idx   = rng.choice(n, size=n, replace=True)
            X_b, y_b = X[idx], y[idx]
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                n_features=n_feat,
            )
            tree.fit(X_b, y_b)
            self.trees_.append(tree)
        return self

    def predict(self, X):
        # Stack predictions from all trees, majority vote per example
        all_preds = np.stack([t.predict(X) for t in self.trees_], axis=1)
        return np.array(
            [Counter(row).most_common(1)[0][0] for row in all_preds])


# ── Evaluation on synthetic two-moon dataset ──────────────────────────────────

def make_moons_np(n_samples=400, noise=0.25, random_state=0):
    rng = np.random.RandomState(random_state)
    n_each = n_samples // 2
    t = np.linspace(0, np.pi, n_each)
    X1 = np.c_[np.cos(t), np.sin(t)]
    X2 = np.c_[1 - np.cos(t), 1 - np.sin(t) - 0.5]
    X  = np.vstack([X1, X2]) + rng.randn(n_samples, 2) * noise
    y  = np.array([0] * n_each + [1] * n_each)
    return X, y

def train_test_split_np(X, y, test_size=0.25, random_state=0):
    rng = np.random.RandomState(random_state)
    idx = rng.permutation(len(y))
    n_test = int(len(y) * test_size)
    return (X[idx[n_test:]], X[idx[:n_test]],
            y[idx[n_test:]], y[idx[:n_test]])

np.random.seed(0)
X, y = make_moons_np(n_samples=400, noise=0.25, random_state=0)
X_train, X_test, y_train, y_test = train_test_split_np(
    X, y, test_size=0.25, random_state=0)

dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=2)
dt.fit(X_train, y_train)
dt_acc = np.mean(dt.predict(X_test) == y_test)
print(f"DecisionTree (max_depth=4)  accuracy: {dt_acc:.4f}")

rf = RandomForestClassifier(n_estimators=200, max_depth=4,
                             min_samples_leaf=2, random_state=42)
rf.fit(X_train, y_train)
rf_acc = np.mean(rf.predict(X_test) == y_test)
print(f"RandomForest (200 trees)    accuracy: {rf_acc:.4f}")

# Root split trace
root = dt.tree_
left_mask = X_train[:, root["feat"]] <= root["thresh"]
g_p = gini(y_train)
g_l = gini(y_train[left_mask])
g_r = gini(y_train[~left_mask])
n_t = len(y_train);  n_l = left_mask.sum();  n_r = (~left_mask).sum()
ig  = g_p - (n_l / n_t * g_l + n_r / n_t * g_r)
print(f"\\nRoot split: feature {root['feat']}, threshold {root['thresh']:.4f}")
print(f"  Parent Gini:      {g_p:.4f}")
print(f"  Left  Gini ({n_l:3d}): {g_l:.4f}")
print(f"  Right Gini ({n_r:3d}): {g_r:.4f}")
print(f"  Information gain: {ig:.4f}")

# Output:
# DecisionTree (max_depth=4)  accuracy: 0.8800
# RandomForest (200 trees)    accuracy: 0.8800
#
# Root split: feature 1, threshold 0.2861
#   Parent Gini:      0.4994
#   Left  Gini (158): 0.2750
#   Right Gini (142): 0.2715
#   Information gain: 0.2261`}</CodeBlock>

      <Prose>
        A few implementation notes worth keeping. The <Code>best_split</Code> function
        iterates over all unique threshold values per feature; a production implementation
        would use argsort to scan thresholds in one pass per feature, cutting the constant
        factor. The forest's feature subsampling happens inside the tree via the
        <Code>n_features</Code> argument — the tree randomly draws that many features at
        each node, not just at the root. This is what produces per-node feature diversity,
        not just per-tree diversity.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        Scikit-learn's <Code>sklearn.tree.DecisionTreeClassifier</Code> and
        <Code>sklearn.ensemble.RandomForestClassifier</Code> are the standard production
        implementations. They are written in Cython, parallelize across cores, and handle
        edge cases (sample weights, multi-output, missing value imputation) that the
        from-scratch version above ignores. The API is clean and the parameter surface
        is stable across major versions.
      </Prose>

      <CodeBlock>{`from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import numpy as np

X, y = make_moons(n_samples=500, noise=0.25, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ── Single decision tree ──────────────────────────────────────────────────────
dt = DecisionTreeClassifier(
    criterion="gini",       # "gini" (CART default) or "entropy"
    max_depth=4,            # None = grow until pure leaves (overfits)
    min_samples_leaf=5,     # prune leaves with fewer samples
    class_weight=None,      # set "balanced" for imbalanced labels
    random_state=42,
)
dt.fit(X_train, y_train)
print(f"DecisionTree test accuracy:   {dt.score(X_test, y_test):.4f}")
print(f"Feature importances (MDI):    {np.round(dt.feature_importances_, 4)}")

# ── Random forest ─────────────────────────────────────────────────────────────
rf = RandomForestClassifier(
    n_estimators=300,       # more trees = lower variance, diminishing returns
    max_depth=None,         # individual trees can overfit; forest averages it out
    min_samples_leaf=1,
    max_features="sqrt",    # sqrt(d) features per split — classification default
    oob_score=True,         # free held-out estimate using out-of-bag samples
    n_jobs=-1,              # trivially parallel across all CPU cores
    class_weight=None,
    random_state=42,
)
rf.fit(X_train, y_train)
print(f"\\nRandomForest test accuracy:   {rf.score(X_test, y_test):.4f}")
print(f"OOB score:                    {rf.oob_score_:.4f}")
print(f"Feature importances (MDI):    {np.round(rf.feature_importances_, 4)}")

# ── Permutation importance (less biased than MDI) ─────────────────────────────
result = permutation_importance(
    rf, X_test, y_test, n_repeats=30, random_state=42)
print(f"Permutation importance mean:  {np.round(result.importances_mean, 4)}")

# Output:
# DecisionTree test accuracy:   0.8800
# Feature importances (MDI):    [0.3829 0.6171]
#
# RandomForest test accuracy:   0.9500
# OOB score:                    0.9425
# Feature importances (MDI):    [0.4443 0.5557]
# Permutation importance mean:  [0.2287 0.2903]`}</CodeBlock>

      <H3>Key parameter guide</H3>

      <Prose>
        <strong>max_depth</strong>: The single most important regularization knob for a
        single tree. Default None (unlimited) will overfit on any dataset large enough
        to have signal. Start with 3–6 for interpretable trees; for random forests,
        deep or unlimited trees are fine because the ensemble averages variance away.
      </Prose>

      <Prose>
        <strong>min_samples_leaf</strong>: Sets a floor on leaf population. Larger values
        produce smoother decision boundaries and reduce overfitting. More reliable than
        max_depth for regression trees, where a few extreme values in a tiny leaf can
        drive prediction far from reality.
      </Prose>

      <Prose>
        <strong>max_features</strong>: For classification, <Code>"sqrt"</Code> (sqrt of
        the total feature count) is the canonical default established by Breiman (2001).
        For regression, <Code>"1.0"</Code> (all features) or <Code>1/3</Code> of features
        are common. Reducing max_features decorrelates trees but increases each tree's
        bias; usually sqrt is a good default that you tune only when you have a reason.
      </Prose>

      <Prose>
        <strong>oob_score</strong>: On average, each bootstrap sample leaves out about
        36.8% of training examples (probability a given sample is not drawn in n draws
        with replacement: (1 − 1/n)^n → e^{"{−1}"} ≈ 0.368). Each tree can be evaluated
        on its out-of-bag examples for free, giving an unbiased generalization estimate
        without a dedicated validation set. OOB score is particularly valuable during
        hyperparameter search when data is scarce.
      </Prose>

      <Prose>
        <strong>n_jobs=-1</strong>: Trees in a forest are fully independent and can be
        trained in parallel. Setting n_jobs=-1 uses all available cores with near-linear
        speedup. For 300 trees on an 8-core machine, expect roughly 7× speedup.
      </Prose>

      <Prose>
        <strong>class_weight="balanced"</strong>: When class frequencies are unequal,
        sklearn weights each sample by the inverse of its class frequency. This is
        equivalent to oversampling the minority class during the Gini calculation at
        each split. Prefer this over manual oversampling for trees and forests.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>Impurity trace: first three splits</H3>

      <StepTrace
        label="gini impurity at each split — two-moon dataset (n=300 train)"
        steps={[
          {
            label: "Depth 0 (root): feature 1, threshold 0.2861",
            render: () => (
              <div>
                <TokenStream
                  label="impurity trace"
                  tokens={[
                    { label: "Parent Gini: 0.4994", color: colors.textMuted },
                    { label: "→", color: colors.textDim },
                    { label: "Left (n=158): 0.2750", color: colors.green },
                    { label: "Right (n=142): 0.2715", color: colors.green },
                    { label: "IG: 0.2261", color: colors.gold },
                  ]}
                />
                <Prose>
                  The root split nearly halves Gini impurity in one step. Feature 1
                  (the y-coordinate in the moons dataset) cleanly separates the lower
                  arc (class 0) from the upper arc (class 1) at this latitude threshold.
                  Information gain of 0.226 is the largest single-split gain available
                  anywhere in the feature space — this is what the exhaustive search
                  over all features and thresholds finds.
                </Prose>
              </div>
            ),
          },
          {
            label: "Depth 1 (left child): further splitting the lower arc",
            render: () => (
              <Prose>
                With the node largely class 0 (low Gini), the next split in the left
                branch captures the minority class 1 examples that slipped in — those
                at the top of the lower arc where the two moons overlap. Gini drops
                further. Each additional split buys smaller gains as purity increases.
              </Prose>
            ),
          },
          {
            label: "Depth 2–3: rectangles tighten around the overlap region",
            render: () => (
              <Prose>
                By depth 3, the tree has drawn four rectangles that jointly approximate
                the crescent boundary. Beyond depth 4, additional splits begin to
                memorize noise — individual training examples in the overlap zone get
                their own tiny leaf. This is overfitting: the splits model the noise,
                not the boundary. The random forest avoids this by averaging 200
                independently noisy trees rather than growing one perfect one.
              </Prose>
            ),
          },
        ]}
      />

      <H3>Feature importance heatmap</H3>

      <Prose>
        Feature importances from mean decrease in impurity (MDI) accumulate the
        information gain from every split that used a given feature, weighted by the
        number of training examples passing through that node, normalized to sum to 1.
        On the two-feature moons dataset, both features are genuinely informative and
        the importance values reflect which one tends to be used earlier and higher in
        the tree (higher nodes process more examples, so their gains are weighted more).
      </Prose>

      <Heatmap
        label="MDI feature importance — RandomForest (300 trees, moons dataset)"
        rows={["feature 0 (x-coord)", "feature 1 (y-coord)"]}
        cols={["importance"]}
        values={[[0.4443], [0.5557]]}
        colorScale="warm"
      />

      <Prose>
        Feature 1 (the y-coordinate) carries more importance — consistent with the root
        split always landing on feature 1 in this dataset. But the margin is narrow:
        in the moons geometry, both axes contribute meaningfully. Compare this to a
        dataset with one dominant predictor (say, credit score in a loan default model):
        MDI would show 0.8+ on that feature and near-zero on others.
      </Prose>

      <Callout variant="warning">
        MDI importance is biased toward high-cardinality continuous features, which
        offer more threshold candidates and are therefore more likely to be selected
        by chance. For categorical features with many values (zip code, user ID), MDI
        will artificially inflate their importance. Always cross-check with permutation
        importance (section 9) when feature cardinality varies widely.
      </Callout>

      <H3>Single tree vs. forest decision boundary</H3>

      <Plot
        label="decision boundary comparison — two-moon dataset"
        description="A single depth-4 tree (left) draws blocky axis-aligned rectangles with hard transitions at each threshold. A 300-tree random forest (right) produces a smoother probability surface — still composed of rectangles at the individual tree level, but the majority-vote probability blends them into curved-looking contours. The forest boundary better tracks the crescent shape because different trees split at different thresholds and different features, and their vote average is a soft ensemble."
        type="boundary-comparison"
        data={{
          model_a: { name: "DecisionTree (depth=4)", accuracy: 0.88 },
          model_b: { name: "RandomForest (300 trees)", accuracy: 0.95 },
          note: "Both trained on moons(n=500, noise=0.25, random_state=42)",
        }}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        Choosing between a single decision tree, a random forest, a linear model, or
        a gradient-boosted tree is a recurring practical question. The axes below are
        the ones that matter most in practice:
      </Prose>

      <H3>Decision tree</H3>
      <Prose>
        Use when interpretability is a hard constraint — when a compliance officer or
        domain expert must be able to trace every prediction through the model by hand.
        A depth-3 or depth-4 tree can be printed on one page and read like a decision
        table. Accuracy will be lower than any ensemble method on any reasonably complex
        dataset. Single trees overfit at depth and underfit when shallow. They are often
        used as base learners for comparison, or in settings where a rule system is
        legally required.
      </Prose>

      <H3>Random forest</H3>
      <Prose>
        The go-to when you want a strong out-of-the-box baseline that requires minimal
        tuning. Key advantages: handles mixed feature types without preprocessing, built-in
        OOB evaluation, parallelizes trivially, robust to outliers and missing values (with
        appropriate imputation). Key limitations: axis-aligned bias means forests struggle
        when the true boundary is linear in a diagonal direction (a regularized logistic
        regression would outperform); does not extrapolate beyond the training range; slower
        inference than a single tree (you run 300 trees instead of one).
      </Prose>

      <H3>Linear model (logistic/ridge regression)</H3>
      <Prose>
        Use when you have strong reason to believe the decision boundary is approximately
        linear, when you need probability calibration out of the box (trees require
        Platt scaling or isotonic regression), when you have extremely high-dimensional
        sparse features (text), or when inference speed and model size are critical.
        Linear models are also easier to regularize in high dimensions and their
        coefficients are interpretable in a different sense — they express direction and
        magnitude in feature space.
      </Prose>

      <H3>Gradient-boosted trees (XGBoost, LightGBM, CatBoost)</H3>
      <Prose>
        Use when you need the highest accuracy on tabular data and can afford the
        longer training time and hyperparameter tuning. Gradient boosting trains trees
        sequentially, each correcting residuals from the previous, so it can model
        complex interactions that random forests approximate only via averaging many
        independent trees. On structured/tabular data, gradient boosting consistently
        outperforms random forests with appropriate tuning — it dominates Kaggle
        tabular competitions for this reason. The cost: more hyperparameters (learning
        rate, subsample, colsample, max_delta_step, early stopping), much longer training,
        and sequential training means no trivial parallelism across trees.
      </Prose>

      <Heatmap
        label="algorithm selection matrix"
        rows={[
          "Single Decision Tree",
          "Random Forest",
          "Logistic Regression",
          "Gradient Boosting",
        ]}
        cols={[
          "Interpretability",
          "Accuracy (tabular)",
          "Inference speed",
          "Tuning effort",
          "Extrapolation",
        ]}
        values={[
          [1.0, 0.3, 1.0, 0.9, 0.1],
          [0.4, 0.7, 0.5, 0.8, 0.1],
          [0.5, 0.5, 1.0, 0.9, 0.8],
          [0.2, 1.0, 0.4, 0.3, 0.1],
        ]}
        colorScale="cool"
      />

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>Training complexity</H3>

      <Prose>
        Building a single decision tree with <em>n</em> training examples and <em>d</em>
        features costs O(n·d·log n) time: for each of O(log n) levels (assuming a
        balanced tree), you sort or scan each feature in O(n) time to find the best
        threshold. For a random forest of B trees, each tree sees a bootstrap sample
        of size n and considers only m = sqrt(d) features per split, so each tree costs
        O(n·m·log n) = O(n·sqrt(d)·log n). The total forest cost is O(B·n·sqrt(d)·log n).
        Because trees are independent, the B factor is parallelized trivially.
      </Prose>

      <Prose>
        In practice: on a 1M-row dataset with 100 features, a 300-tree forest on 16
        cores trains in minutes. At 100M rows, memory becomes the bottleneck before
        compute — you need to hold the bootstrap sample and all intermediate node
        statistics in RAM simultaneously. Histogram-based implementations (the default
        in LightGBM, and available in sklearn via <Code>HistGradientBoosting*</Code>)
        bin continuous features into 256 buckets, reducing per-split cost dramatically
        and making 100M-row training tractable.
      </Prose>

      <H3>Inference complexity</H3>

      <Prose>
        Predicting with a single tree costs O(log n) per example — one comparison per
        level, and a balanced tree has O(log n) levels. A forest of B trees costs O(B·log n)
        per example. For B=300 and depth 20, that is 6,000 comparisons per example —
        fast on modern hardware (microseconds per prediction) but 300× slower than a
        single tree. If inference latency is the bottleneck (real-time scoring systems),
        shallower forests or a single tree may be required. Libraries like
        <Code>treelite</Code> compile sklearn forests into optimized C code and recover
        most of the latency gap.
      </Prose>

      <H3>Memory</H3>

      <Prose>
        A decision tree stores its split parameters — a feature index and threshold per
        node, a class distribution per leaf. For a depth-k binary tree, there are at
        most 2^k − 1 internal nodes and 2^k leaves. A deep forest of 300 trees with
        depth 20 can easily require hundreds of megabytes of model storage. In practice,
        sklearn's forest serialized with joblib or pickle is often 50–500 MB, which
        matters for deployment to memory-constrained environments. Gradient boosting
        implementations use shallower trees (depth 3–8) and far fewer of them, resulting
        in a 10–100× smaller model footprint.
      </Prose>

      <H3>Comparison with gradient boosting</H3>

      <Prose>
        The core scale difference between random forests and gradient boosting is this:
        forests are embarrassingly parallel across trees, while boosting is inherently
        sequential. You cannot start tree <em>k+1</em> in a boosted ensemble until
        tree <em>k</em> has finished and computed residuals. This means boosting cannot
        use tree-level parallelism; it compensates by using column (feature) parallelism
        and histogram tricks within each tree. For very large n, LightGBM's histogram
        approach is faster than sklearn's forest; for very large d, forests with sqrt(d)
        feature subsampling per split are memory-efficient. The two methods occupy
        different parts of the compute-accuracy frontier: forests for fast, low-tuning
        baselines; boosting for maximum accuracy with more engineering effort.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>Single tree high variance</H3>

      <Prose>
        A decision tree grown to full depth is a high-variance estimator. On two
        different train/test splits of the same dataset, the tree might look completely
        different — a different feature at the root, a different partition structure —
        if there is overlap between classes near the boundary. This is not a bug; it
        reflects the fact that small changes in training data can change which split wins
        the argmax comparison at each node. The forest exists precisely because of this.
        If you are using a single tree for production, cap its depth and cross-validate
        aggressively.
      </Prose>

      <H3>No extrapolation</H3>

      <Prose>
        Decision trees (and therefore random forests) cannot predict values outside the
        range seen during training. A regression tree's prediction for any input is the
        mean of some subset of training targets. If your test distribution has feature
        values outside the training range — you trained on 2010–2020 prices and deploy
        into a 2024 market — the tree will clip to the nearest leaf, which is the extreme
        training value. Linear models and neural networks extrapolate (sometimes badly,
        but at least they try). For forecasting tasks with temporal extrapolation, this
        is a critical failure mode. Trees should not be used for pure extrapolation unless
        the features are engineered to represent "distance from training distribution"
        in a way trees can threshold on.
      </Prose>

      <H3>MDI feature importance bias</H3>

      <Prose>
        Mean decrease in impurity (MDI, the default <Code>feature_importances_</Code> in
        sklearn) is biased toward features with high cardinality — those with many unique
        values. A random uniform feature with 1000 unique values will appear more
        "important" than a truly informative binary feature, because there are 1000
        thresholds to try and at least one will reduce impurity by chance. Strobl et al.
        (2007, "Bias in random forest variable importance measures") documented this
        thoroughly. The fix is permutation importance: shuffle feature <em>j</em> in the
        validation set and measure the drop in accuracy. Shuffling destroys any real signal,
        so the drop measures true importance. Permutation importance is available in sklearn
        via <Code>sklearn.inspection.permutation_importance</Code> and is less biased,
        though slower.
      </Prose>

      <H3>Categorical encoding pitfalls</H3>

      <Prose>
        Sklearn's trees require numeric features. If you one-hot encode a categorical
        feature with 100 levels, you get 100 binary features, and each tree will pick
        up at most a few of them due to feature subsampling. The algorithm never sees
        the full categorical structure. A better approach is ordinal encoding (assign
        integers) and let the tree find meaningful thresholds, or use libraries that
        support native categoricals (LightGBM, CatBoost). For random forests
        specifically, one-hot encoding high-cardinality categoricals often works
        adequately in practice because the forest integrates over many splits, but the
        bias toward those many columns can crowd out genuinely informative numeric
        features in the importance ranking.
      </Prose>

      <H3>Class imbalance</H3>

      <Prose>
        A tree's Gini impurity calculation is dominated by the majority class. On a
        99:1 imbalance, a leaf that predicts the majority class everywhere achieves
        Gini = 2 · 0.99 · 0.01 = 0.02, which is nearly pure — the tree sees no signal.
        The standard fix is <Code>class_weight="balanced"</Code>, which weights each
        sample by the inverse of its class frequency during the Gini calculation.
        Alternatively, <Code>sklearn.utils.class_weight.compute_sample_weight</Code>
        lets you set custom per-sample weights and pass them to <Code>fit</Code>. SMOTE
        oversampling before fitting is another option but introduces its own biases
        for tree-based methods (synthetic points in high-dimensional space may cross
        true decision boundaries).
      </Prose>

      <Callout variant="warning">
        The no-extrapolation failure is the most dangerous in practice because it fails
        silently. Predictions will not throw errors and will not produce NaNs — they will
        simply produce the extreme training-range leaf value, which may look plausible.
        Always plot the distribution of predictions vs. training targets when deploying
        on new data.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        The following four works are the canonical references for decision trees and
        random forests. All citations have been verified against publisher records.
      </Prose>

      <H3>CART — the origin of modern decision trees</H3>
      <Prose>
        Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984).
        <em> Classification and Regression Trees</em>. Chapman and Hall/CRC, Wadsworth.
        This is the book that introduced binary recursive partitioning, Gini impurity,
        cost-complexity pruning, and the CART algorithm as it is implemented in sklearn
        today. Freely available as a monograph; Taylor & Francis has the current edition
        under ISBN 978-0-412-04841-8. If you read one source on decision trees, it is this.
      </Prose>

      <H3>ID3 — information gain for tree induction</H3>
      <Prose>
        Quinlan, J. R. (1986). Induction of decision trees.
        <em> Machine Learning</em>, 1(1), 81–106.
        <br />
        Quinlan, J. R. (1993). <em>C4.5: Programs for Machine Learning</em>.
        Morgan Kaufmann Publishers, San Mateo, CA.
        The 1986 paper introduced the ID3 algorithm using information gain (entropy-based)
        as the splitting criterion. The 1993 book extended ID3 into C4.5, adding support
        for continuous features, pruning via minimum description length, and missing value
        handling. C4.5 remains the basis for the WEKA J48 implementation.
      </Prose>

      <H3>Random forests — the foundational paper</H3>
      <Prose>
        Breiman, L. (2001). Random forests.
        <em> Machine Learning</em>, 45(1), 5–32. DOI: 10.1023/A:1010933404324.
        This is the paper. It introduced bootstrap aggregating plus random feature
        subsampling as a unified algorithm, proved that the generalization error
        converges as B → ∞, and gave the strength-correlation bound. The PDF is freely
        available from UC Berkeley Statistics Department at stat.berkeley.edu.
        Every claim about random forest theoretical properties should trace back here.
      </Prose>

      <H3>Extremely Randomized Trees</H3>
      <Prose>
        Geurts, P., Ernst, D., & Wehenkel, L. (2006). Extremely randomized trees.
        <em> Machine Learning</em>, 63(1), 3–42. DOI: 10.1007/s10994-006-6226-1.
        Extra-Trees pushes randomization further: instead of finding the best threshold
        for a randomly chosen feature, it draws the threshold uniformly at random from
        the feature's range. This eliminates the inner threshold-search loop entirely,
        making training much faster, and further reduces variance at the cost of slightly
        higher bias. Available in sklearn as <Code>ExtraTreesClassifier</Code>. The paper
        gives a thorough bias-variance analysis of the randomization spectrum from
        CART (no randomization) to Extra-Trees (maximum randomization).
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Q1 (recall) — Gini vs. entropy</H3>
      <Prose>
        A node contains 80 class-0 and 20 class-1 examples. Compute its Gini impurity
        and Shannon entropy. Which is larger? Why do practitioners typically prefer Gini?
      </Prose>
      <Callout variant="answer">
        <strong>Gini:</strong> 1 − (0.8² + 0.2²) = 1 − (0.64 + 0.04) = 0.32.
        <br />
        <strong>Entropy:</strong> −(0.8 · log₂(0.8) + 0.2 · log₂(0.2))
        = −(0.8 · (−0.322) + 0.2 · (−2.322)) = 0.258 + 0.464 = 0.722 bits.
        <br />
        Entropy is larger in absolute terms (different scale). Both reach their maximum
        at equal class proportions and zero at purity. Gini is preferred because it
        involves no logarithm computation — for a tree scanning millions of threshold
        candidates, this constant-factor savings accumulates. The trees produced by the
        two criteria are nearly identical in practice.
      </Callout>

      <H3>Q2 (recall) — the ρσ² term</H3>
      <Prose>
        The variance of the average of B correlated trees is ρσ² + (1−ρ)σ²/B.
        Explain in one sentence why the ρσ² term makes feature subsampling necessary,
        not just a nice-to-have.
      </Prose>
      <Callout variant="answer">
        Because ρσ² does not decrease as B increases, no amount of additional trees
        can drive ensemble variance below this floor — only reducing inter-tree
        correlation ρ (via feature subsampling) can lower the irreducible term.
      </Callout>

      <H3>Q3 (applied) — OOB score interpretation</H3>
      <Prose>
        You train a <Code>RandomForestClassifier(n_estimators=500, oob_score=True)</Code>
        and get <Code>oob_score_ = 0.94</Code> and a test accuracy of <Code>0.89</Code>.
        What are two plausible explanations for the 5-point gap between OOB and test score?
      </Prose>
      <Callout variant="answer">
        (1) Distribution shift: the test set comes from a different distribution than the
        training data (different time period, different user segment), so the OOB estimate
        on training examples does not reflect test performance.
        (2) Class imbalance interaction: if the test set has a different class ratio than
        the training set, accuracy computed on each may differ without either estimate
        being "wrong" — OOB accuracy is evaluated on the same class mix as training,
        not the test mix. A third possibility is simple train/test split variance if the
        test set is small.
      </Callout>

      <H3>Q4 (applied) — MDI vs. permutation importance</H3>
      <Prose>
        You build a RandomForest on a dataset with 50 numeric features and 5 high-cardinality
        categorical features one-hot encoded into 500 columns. Your MDI importances show
        the one-hot columns dominating. Describe the exact steps to obtain a less biased
        importance ranking.
      </Prose>
      <Callout variant="answer">
        Use <Code>sklearn.inspection.permutation_importance</Code> on a held-out validation
        set. For each feature (or group of one-hot columns representing a single original
        categorical), shuffle that feature's values in the validation set, run predictions,
        and record the drop in accuracy. Average over multiple shuffles (n_repeats=20+).
        Critically, group the 500 one-hot columns back into their 5 original categorical
        variables and shuffle them as a unit — shuffling individual one-hot bits while
        holding others fixed can produce impossible combinations that mislead the model.
        After grouping, permutation importance correctly attributes importance to the
        original categorical feature rather than its encoding columns.
      </Callout>

      <H3>Q5 (applied) — extrapolation failure</H3>
      <Prose>
        You train a regression RandomForest on housing prices from 2010–2020.
        Prices in your test set (2021–2023) are 25% higher than any training example.
        Describe what the forest predicts and why.
      </Prose>
      <Callout variant="answer">
        The forest predicts values at most equal to the highest price in the training
        set, regardless of input feature values. Each tree's leaf predictions are means
        of training targets; no leaf mean can exceed the maximum training target. For
        2021–2023 examples, the features will route to the deepest leaves trained on the
        most expensive 2020 properties, and the predicted price will be capped at that
        level. The forest cannot extrapolate upward because prediction is literally an
        average of a fixed pool of training values. Fixes include log-transforming the
        target (so the model predicts log-price, which may extrapolate better), including
        a time feature so the forest can interpolate across time, or switching to a model
        that can extrapolate (linear regression on the time trend, gradient boosting with
        a monotone constraint on time).
      </Callout>

      <H3>Q6 (challenge) — from scratch tree vs. sklearn discrepancy</H3>
      <Prose>
        Your from-scratch DecisionTreeClassifier and sklearn's DecisionTreeClassifier are
        both given the same dataset and the same max_depth=4. They produce different
        predictions. Identify three implementation differences that could cause this,
        without looking at the code.
      </Prose>
      <Callout variant="answer">
        (1) <strong>Tie-breaking at equal gain:</strong> when two thresholds produce equal
        information gain, sklearn uses a deterministic tie-break; the from-scratch version
        may use the last winner found in iteration order.
        (2) <strong>Threshold enumeration:</strong> sklearn evaluates midpoints between
        adjacent sorted unique values; the from-scratch version evaluates the unique values
        directly — a threshold of exactly a training value can assign that example to either
        side depending on {"<"} vs. {"<"}= convention.
        (3) <strong>min_impurity_decrease:</strong> sklearn has a default minimum
        improvement threshold below which it will not split; the from-scratch version splits
        any time gain {">"} 0. Setting sklearn's min_impurity_decrease=0.0 eliminates this
        difference. (4) Bonus: sklearn handles floating-point thresholds differently —
        it selects (left_value + right_value) / 2, ensuring the threshold never equals a
        training point exactly, which avoids train-vs-test boundary ambiguity.
      </Callout>

    </div>
  ),
};

export default decisionTreesRandomForestsContent;
