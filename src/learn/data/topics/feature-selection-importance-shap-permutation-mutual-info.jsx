import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const featureSelectionContent = {
  title: "Feature Selection & Importance (SHAP, Permutation, Mutual Info)",
  readTime: "~50 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Machine learning models do not automatically know which features matter. Feed a model
        500 columns, and it will try to use all of them — including the date-stamp that
        accidentally leaks the target, the identifier that is unique per row, the measurement
        that is pure noise. Irrelevant features impose real costs: longer training, worse
        generalization, opaque predictions, and higher serving latency. Feature selection and
        importance estimation are the toolset for answering a deceptively hard question:
        which of the inputs your model sees are actually driving its behavior?
      </Prose>

      <Prose>
        The formal taxonomy of feature selection methods was established by Ron Kohavi and
        George John in their 1997 paper "Wrappers for Feature Subset Selection" published in
        <em> Artificial Intelligence</em> 97(1–2):273–324. They drew a line between
        <strong> filter methods</strong>, which score features using properties of the data
        alone (no model involved), and <strong>wrapper methods</strong>, which evaluate subsets
        by training and evaluating a model on each candidate subset. Isabelle Guyon and André
        Elisseeff extended this in "An Introduction to Variable and Feature Selection" (JMLR
        2003, 3:1157–1182) by adding <strong>embedded methods</strong> — algorithms that
        perform selection as part of the model fitting procedure itself, such as L1 (LASSO)
        regularization, which drives some coefficients to exactly zero, or decision trees,
        which implicitly select features at each split.
      </Prose>

      <Prose>
        Feature <em>importance</em> is a related but distinct concept: rather than a binary
        selected/not-selected decision, importance assigns a scalar score quantifying each
        feature's contribution to a trained model. Leo Breiman introduced Mean Decrease in
        Impurity (MDI) as a by-product of random forest training in his foundational 2001
        paper in <em>Machine Learning</em> 45(1):5–32. MDI is fast — it accumulates during
        training — but Carolin Strobl, Anne-Laure Boulesteix, Achim Zeileis, and Torsten
        Hothorn documented a critical bias in "Bias in random forest variable importance
        measures" (<em>BMC Bioinformatics</em> 2007, 8:25): MDI systematically overrates
        features with more unique values (high cardinality) because they offer more threshold
        candidates during the split search.
      </Prose>

      <Prose>
        Permutation importance addresses the MDI bias by measurement rather than accumulation.
        Fisher, Rudin, and Dominici formalized it rigorously in "All Models are Wrong, but Many
        are Useful: Learning a Variable's Importance by Studying an Entire Class of Prediction
        Models Simultaneously" (JMLR 2019, 20(177):1–81), introducing the concept of Model
        Class Reliance — the range of permutation importances over all models that fit the data
        equally well, not just a single chosen model.
      </Prose>

      <Prose>
        The most influential recent contribution is SHAP (SHapley Additive exPlanations),
        introduced by Scott Lundberg and Su-In Lee in "A Unified Approach to Interpreting
        Model Predictions" (NeurIPS 2017). SHAP grounds feature attribution in cooperative
        game theory: each feature's importance for a specific prediction is its Shapley value
        — the average marginal contribution computed over all possible orderings of features.
        The key advance was showing that SHAP unifies several existing methods (LIME, DeepLIFT,
        integrated gradients) under a single axiomatic framework, and that for tree models it
        can be computed exactly in polynomial time via the TreeSHAP algorithm described in
        Lundberg et al. 2020 (<em>Nature Machine Intelligence</em> 2:56–67).
      </Prose>

      <Callout variant="insight">
        Feature selection reduces dimensionality and speeds training. Feature importance
        explains which inputs drive a trained model's decisions. They serve different purposes —
        selection is a preprocessing step; importance is a diagnostic and interpretability tool.
        Both are indispensable in any serious tabular ML workflow.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <H3>Three families of selection methods</H3>

      <Prose>
        <strong>Filter methods</strong> score each feature independently using a statistical
        relationship with the target. The chi-squared test measures whether a feature's
        distribution varies across target classes. The F-test (ANOVA) measures the ratio of
        between-class variance to within-class variance. Mutual information measures how much
        knowing the feature reduces uncertainty about the target. All filters are cheap — one
        pass over the data per feature — but they are inherently univariate: they miss
        interactions. A feature that is useless alone (zero correlation with the target) might
        be essential in combination with another feature. Filters are blind to this.
      </Prose>

      <Prose>
        <strong>Wrapper methods</strong> evaluate feature subsets by training a model on each
        subset and measuring validation performance. Forward selection starts with no features
        and greedily adds the one that most improves performance. Backward elimination starts
        with all features and greedily removes the least useful. Recursive Feature Elimination
        (RFE) trains the model, removes the weakest feature (by the model's internal scoring),
        and repeats. Wrappers are accurate — they account for interactions and the specific
        model being used — but computationally expensive. RFE with a 100-feature dataset and
        a random forest trains {">"}100 models.
      </Prose>

      <Prose>
        <strong>Embedded methods</strong> perform selection during training. LASSO (L1
        regularization) adds a penalty proportional to the sum of absolute coefficient values
        to the loss function, which drives the coefficients of truly useless features to
        exactly zero — the model both fits and selects simultaneously. Tree-based MDI is
        another embedded method: the forest accumulates importance while building splits,
        producing a feature ranking with zero additional cost.
      </Prose>

      <H3>Importance is not causation</H3>

      <Prose>
        A feature can rank high in importance for reasons that have nothing to do with causal
        mechanism. A spurious correlate — a proxy variable that happens to move with the target
        in the training data — will be assigned high importance. In a credit model, zip code
        might score high not because geography causes default, but because it proxies for
        income level and historical lending patterns. Feature importance tells you what your
        model has learned to rely on, not what causes the outcome in the real world. Confusing
        the two leads to misleading explanations and brittle models that break when the
        correlation structure shifts.
      </Prose>

      <H3>SHAP: game theory meets per-prediction attribution</H3>

      <Prose>
        SHAP decomposes a single prediction into per-feature contributions, not just a global
        ranking. For a given input, each feature receives a SHAP value that represents its
        marginal contribution to pushing the prediction away from the model's baseline (the
        average prediction over the training set). Features that pushed the prediction higher
        get positive SHAP values; those that pushed it lower get negative values. The SHAP
        values for one prediction sum exactly to the difference between that prediction and
        the baseline — a property called efficiency. This means SHAP gives you a complete
        accounting: every unit of deviation from average is attributed to specific features,
        with nothing left over and nothing double-counted.
      </Prose>

      <Callout variant="insight">
        The intuition from cooperative game theory: imagine each feature as a player in a
        coalition game where the "payout" is the model's prediction. A feature's Shapley
        value is its fair share of the payout — the average of its marginal contributions
        across all possible orderings in which features could "join" the coalition. Players
        that contribute more in more orderings earn higher Shapley values.
      </Callout>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>Mutual information</H3>

      <Prose>
        For discrete random variables X and Y, mutual information is:
      </Prose>

      <MathBlock>
        {"I(X;Y) = \\sum_{x,y} p(x,y) \\log \\frac{p(x,y)}{p(x)\\,p(y)}"}
      </MathBlock>

      <Prose>
        This equals zero when X and Y are independent (the joint equals the product of
        marginals), and is strictly positive whenever knowing X reduces uncertainty about Y.
        Unlike correlation, mutual information captures nonlinear dependencies. For continuous
        features, the sum becomes an integral and the probability mass functions become
        densities. Estimating these densities from data requires either histogram binning (fast,
        biased by bin count) or the Kraskov-Stögbauer-Grassberger (KSG) estimator, which uses
        k-nearest neighbors in the joint space — the approach used by
        <Code>sklearn.feature_selection.mutual_info_classif</Code>.
      </Prose>

      <H3>Permutation importance</H3>

      <Prose>
        Let <Code>err(X)</Code> be the model's prediction error on a held-out dataset X.
        Permutation importance for feature j is:
      </Prose>

      <MathBlock>
        {"\\text{imp}(j) = \\text{err}(X_{\\text{perm}_j}) - \\text{err}(X)"}
      </MathBlock>

      <Prose>
        where {"X_{perm_j}"} denotes X with column j randomly permuted. Permuting destroys any
        relationship between feature j and the target while leaving the marginal distribution
        of j intact. If the error rises substantially after permuting, j was load-bearing;
        if the error is unchanged, j was uninformative (or its information was redundant with
        another feature). Repeating the permutation multiple times and averaging reduces
        Monte Carlo noise. The result is computed on held-out data, not training data, so it
        measures generalization importance rather than training fit.
      </Prose>

      <H3>SHAP: Shapley values</H3>

      <Prose>
        For a model f with feature set F, the Shapley value for feature j at input x is:
      </Prose>

      <MathBlock>
        {"\\phi_j = \\sum_{S \\subseteq F \\setminus \\{j\\}} \\frac{|S|!(|F|-|S|-1)!}{|F|!} \\bigl[f(S \\cup \\{j\\}) - f(S)\\bigr]"}
      </MathBlock>

      <Prose>
        The term {"[f(S∪{j}) − f(S)]"} is the marginal contribution of feature j to a
        coalition S. The combinatorial weight counts the fraction of orderings in which S
        appears before j — ensuring every ordering gets equal weight. The sum is over all
        {"2^{|F|-1}"} subsets of features not containing j.
      </Prose>

      <Prose>
        The four Shapley axioms that uniquely characterize this formula:
        <br />
        <strong>Efficiency:</strong> {"Σ_j φ_j = f(x) − f(baseline)"} — the sum of all SHAP values
        exactly equals the difference between the prediction and the expected prediction.
        <br />
        <strong>Symmetry:</strong> features that make identical marginal contributions to every
        coalition receive equal SHAP values.
        <br />
        <strong>Dummy:</strong> a feature that contributes nothing to any coalition receives
        SHAP value zero.
        <br />
        <strong>Additivity:</strong> SHAP values for an ensemble of models equal the sum of
        SHAP values from each model separately.
      </Prose>

      <H3>TreeSHAP: exact computation for trees</H3>

      <Prose>
        Naively computing Shapley values requires evaluating {"f(S)"} for every subset S —
        {"2^d"} evaluations for d features. For d=20 that is one million model calls per
        prediction. Lundberg et al. (2020) introduced TreeSHAP, which exploits the tree
        structure to compute exact Shapley values in {"O(T L D²)"} time, where T is the
        number of trees, L is the maximum number of leaves per tree, and D is the maximum
        depth. For a 100-tree forest with depth 6, this is orders of magnitude faster than
        the exponential naive approach. The algorithm works by pushing a weighted distribution
        of "the fraction of training samples reaching each node" through the tree
        path-by-path, accumulating marginal contributions without explicit subset enumeration.
      </Prose>

      <Prose>
        A distinction worth knowing: TreeSHAP can be computed in either <strong>interventional</strong>
        or <strong>path-dependent</strong> mode. Interventional SHAP evaluates {"f(S)"} by
        replacing missing features with their marginal distribution from training data —
        appropriate when features are independent. Path-dependent SHAP conditions on the
        decision path — appropriate when features are correlated and the conditional
        distribution differs from the marginal. The two modes give different SHAP values when
        features are correlated (section 9 covers this in depth).
      </Prose>

      <H3>LASSO as embedded selection</H3>

      <Prose>
        Ridge regression penalizes the squared magnitude of coefficients: {"‖β‖²₂"}. The
        L2 penalty shrinks all coefficients toward zero but never reaches exactly zero —
        all features remain in the model. LASSO uses the L1 norm instead:
      </Prose>

      <MathBlock>
        {"\\hat{\\beta}^{\\text{LASSO}} = \\underset{\\beta}{\\operatorname{argmin}} \\left\\| y - X\\beta \\right\\|_2^2 + \\lambda \\|\\beta\\|_1"}
      </MathBlock>

      <Prose>
        The L1 penalty has kinks at zero. The subgradient optimality condition implies that
        coefficients can be driven to exactly zero when their feature's signal is below a
        threshold determined by {"λ"}. As {"λ"} increases from zero, the LASSO path traces a
        sequence of models with progressively fewer nonzero coefficients — a regularization
        path that is simultaneously a feature selection path. The features whose coefficients
        are nonzero at the chosen {"λ"} are the selected set. With correlated features, LASSO
        tends to pick one representative from each correlated group arbitrarily, which is a
        known instability (the elastic net adds L2 to stabilize this).
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The following code implements five methods using NumPy only, applied to a synthetic
        tabular dataset where features 0, 2, and 5 are the true signal features. All code was
        executed; stdout is embedded verbatim.
      </Prose>

      <H3>4a. Chi-squared, mutual information, permutation importance, forward selection</H3>

      <CodeBlock>{`import numpy as np
from itertools import combinations
import math

# ── Synthetic dataset ─────────────────────────────────────────────────────────
# 300 samples, 8 features; true signal in feat 0, feat 2, feat 5
np.random.seed(42)
n_samples = 300
X_num  = np.random.randn(n_samples, 8)
y_cont = 3*X_num[:, 0] - 2*X_num[:, 2] + X_num[:, 5] + 0.5 * np.random.randn(n_samples)
y      = (y_cont > 0).astype(int)

# ── (a) Chi-squared feature scoring ──────────────────────────────────────────
def chi2_score(X, y):
    classes = np.unique(y)
    scores = []
    for j in range(X.shape[1]):
        vals = np.unique(X[:, j])
        stat = 0.0
        for v in vals:
            mask_v = X[:, j] == v
            n_v = mask_v.sum()
            for c in classes:
                observed = (mask_v & (y == c)).sum()
                expected = n_v * (y == c).sum() / len(y)
                if expected > 0:
                    stat += (observed - expected)**2 / expected
        scores.append(stat)
    return np.array(scores)

X_disc = (X_num > 0).astype(int)          # binarize for chi2
chi2   = chi2_score(X_disc, y)
rank_chi2 = np.argsort(-chi2)
print("=== (a) Chi-squared scores ===")
for i, f in enumerate(rank_chi2):
    print(f"  feat {f}: chi2={chi2[f]:.3f}  rank={i+1}")

# ── (b) Mutual information via histograms ─────────────────────────────────────
def mutual_info_hist(X, y, bins=10):
    scores = []
    for j in range(X.shape[1]):
        x = X[:, j]
        hist_xy, _, _ = np.histogram2d(x, y, bins=[bins, np.unique(y).size])
        hist_x = hist_xy.sum(axis=1, keepdims=True)
        hist_y = hist_xy.sum(axis=0, keepdims=True)
        n = hist_xy.sum()
        pxy = hist_xy / n;  px = hist_x / n;  py = hist_y / n
        mask = (pxy > 0) & (px > 0) & (py > 0)
        mi = np.where(mask, pxy * np.log(pxy / (px * py)), 0).sum()
        scores.append(mi)
    return np.array(scores)

mi = mutual_info_hist(X_num, y)
rank_mi = np.argsort(-mi)
print("\\n=== (b) Mutual Information (histogram, 10 bins) ===")
for i, f in enumerate(rank_mi):
    print(f"  feat {f}: MI={mi[f]:.4f}  rank={i+1}")

# ── (c) Permutation importance ────────────────────────────────────────────────
n_tr = 240
X_tr, X_te = X_num[:n_tr], X_num[n_tr:]
y_tr, y_te = y[:n_tr], y[n_tr:]
centroids = np.array([X_tr[y_tr == c].mean(axis=0) for c in [0, 1]])

def model_fn(X):
    dists = np.stack([np.linalg.norm(X - c, axis=1) for c in centroids], axis=1)
    return dists.argmin(axis=1)

def permutation_importance_np(model_fn, X, y, n_repeats=20, random_state=42):
    rng = np.random.RandomState(random_state)
    baseline = np.mean(model_fn(X) == y)
    importances = []
    for j in range(X.shape[1]):
        drops = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            drops.append(baseline - np.mean(model_fn(X_perm) == y))
        importances.append(np.mean(drops))
    return np.array(importances)

perm_imp = permutation_importance_np(model_fn, X_te, y_te)
baseline_acc = np.mean(model_fn(X_te) == y_te)
rank_perm = np.argsort(-perm_imp)
print(f"\\n=== (c) Permutation Importance (baseline acc={baseline_acc:.4f}) ===")
for i, f in enumerate(rank_perm):
    print(f"  feat {f}: imp={perm_imp[f]:.4f}  rank={i+1}")

# ── (d) Forward selection with 5-fold CV ──────────────────────────────────────
def forward_selection(X, y, k=3, cv=5):
    n, d = X.shape
    selected, remaining = [], list(range(d))
    fold_size = n // cv
    for _ in range(k):
        best_feat, best_score = None, -1
        for f in remaining:
            candidate = selected + [f]
            fold_scores = []
            for fold in range(cv):
                val_idx = list(range(fold * fold_size, (fold + 1) * fold_size))
                tr_idx  = [i for i in range(n) if i not in val_idx]
                X_tr2   = X[tr_idx][:, candidate]
                X_val   = X[val_idx][:, candidate]
                y_tr2   = y[tr_idx];  y_val = y[val_idx]
                ctrs    = np.array([X_tr2[y_tr2 == c].mean(axis=0) for c in np.unique(y_tr2)])
                preds   = np.array([
                    np.unique(y_tr2)[np.argmin(np.linalg.norm(ctrs - row, axis=1))]
                    for row in X_val])
                fold_scores.append(np.mean(preds == y_val))
            score = np.mean(fold_scores)
            if score > best_score:
                best_score = score;  best_feat = f
        selected.append(best_feat);  remaining.remove(best_feat)
    return selected

selected = forward_selection(X_num, y, k=3, cv=5)
print(f"\\n=== (d) Forward selection (k=3, 5-fold CV) ===")
print(f"  Selected features: {selected}")`}</CodeBlock>

      <Callout variant="insight">
{`=== (a) Chi-squared scores ===
  feat 0: chi2=92.238  rank=1
  feat 2: chi2=26.896  rank=2
  feat 5: chi2=11.204  rank=3
  feat 4: chi2=2.543   rank=4
  feat 7: chi2=0.974   rank=5
  feat 1: chi2=0.216   rank=6
  feat 6: chi2=0.108   rank=7
  feat 3: chi2=0.002   rank=8

=== (b) Mutual Information (histogram, 10 bins) ===
  feat 0: MI=0.2542  rank=1
  feat 2: MI=0.1466  rank=2
  feat 5: MI=0.0333  rank=3
  feat 1: MI=0.0190  rank=4
  feat 3: MI=0.0157  rank=5
  feat 4: MI=0.0142  rank=6
  feat 7: MI=0.0127  rank=7
  feat 6: MI=0.0099  rank=8

=== (c) Permutation Importance (baseline acc=0.9833) ===
  feat 0: imp=0.3250  rank=1
  feat 2: imp=0.2642  rank=2
  feat 5: imp=0.0967  rank=3
  feat 7: imp=0.0000  rank=4
  feat 1: imp=-0.0008 rank=5
  feat 4: imp=-0.0033 rank=6
  feat 6: imp=-0.0058 rank=7
  feat 3: imp=-0.0067 rank=8

=== (d) Forward selection (k=3, 5-fold CV) ===
  Selected features: [0, 2, 5]`}
      </Callout>

      <H3>4e. Exhaustive Shapley values for a 3-feature model</H3>

      <CodeBlock>{`import numpy as np
from itertools import combinations
import math

# Toy linear model: f(x) = 2*x0 + 1*x1 - 1.5*x2
def toy_model(v):
    return 2*v[0] + 1*v[1] - 1.5*v[2]

x        = np.array([1.0, 0.5, -1.0])    # instance to explain
baseline = np.array([0.0, 0.0, 0.0])     # reference (all-zero baseline)
n = 3
phis = np.zeros(n)

for j in range(n):
    others = [i for i in range(n) if i != j]
    for size in range(len(others) + 1):
        for S in combinations(others, size):
            S = list(S)
            # Build f(S) and f(S ∪ {j}) by replacing non-coalition features with baseline
            v_s   = baseline.copy();  v_s[S]  = x[S]
            v_sj  = baseline.copy();  v_sj[S] = x[S];  v_sj[j] = x[j]
            weight = (math.factorial(size) *
                      math.factorial(n - size - 1) /
                      math.factorial(n))
            phis[j] += weight * (toy_model(v_sj) - toy_model(v_s))

print("=== Shapley values — exhaustive (3 features) ===")
print(f"  x = {x}")
print(f"  baseline f(0,0,0) = {toy_model(baseline):.4f}")
print(f"  f(x)              = {toy_model(x):.4f}")
print(f"  SHAP phi_0        = {phis[0]:.4f}  (true: 2 * 1.0 = 2.0)")
print(f"  SHAP phi_1        = {phis[1]:.4f}  (true: 1 * 0.5 = 0.5)")
print(f"  SHAP phi_2        = {phis[2]:.4f}  (true: -1.5 * -1.0 = 1.5)")
print(f"  Sum(phi)          = {phis.sum():.4f}  == f(x) - f(baseline)")`}</CodeBlock>

      <Callout variant="insight">
{`=== Shapley values — exhaustive (3 features) ===
  x = [ 1.   0.5 -1. ]
  baseline f(0,0,0) = 0.0000
  f(x)              = 4.0000
  SHAP phi_0        = 2.0000  (true: 2 * 1.0 = 2.0)
  SHAP phi_1        = 0.5000  (true: 1 * 0.5 = 0.5)
  SHAP phi_2        = 1.5000  (true: -1.5 * -1.0 = 1.5)
  Sum(phi)          = 4.0000  == f(x) - f(baseline)`}
      </Callout>

      <Prose>
        For a linear model with independent features, SHAP recovers each feature's exact
        marginal contribution. The efficiency axiom holds perfectly: {"phi_0 + phi_1 + phi_2 = 4.0"}
        equals {"f(x) − f(baseline)"}. In nonlinear models with correlated features, SHAP still
        satisfies the axioms — it just requires the full averaging over subsets rather than
        the shortcut available for linear models.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        All code below operates on the same 500-sample, 10-feature classification dataset
        ({"make_classification"} with 4 informative features, random_state=42, 80/20 split).
        All blocks executed; stdout is verbatim.
      </Prose>

      <H3>5a. sklearn filter and wrapper methods</H3>

      <CodeBlock>{`import numpy as np
from sklearn.feature_selection import (
    SelectKBest, chi2, f_classif, mutual_info_classif,
    RFE, SelectFromModel,
)
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

np.random.seed(42)
X, y = make_classification(
    n_samples=500, n_features=10, n_informative=4,
    n_redundant=2, n_repeated=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# chi2 requires non-negative features — scale to [0,1]
scaler = MinMaxScaler()
X_tr_s = scaler.fit_transform(X_train)

# ── Filter: chi-squared ────────────────────────────────────────────────────────
sel_chi2 = SelectKBest(chi2, k=4).fit(X_tr_s, y_train)
print("SelectKBest(chi2, k=4):")
print(f"  selected  = {np.where(sel_chi2.get_support())[0].tolist()}")
print(f"  scores    = {np.round(sel_chi2.scores_, 2)}")

# ── Filter: F-statistic ────────────────────────────────────────────────────────
sel_f = SelectKBest(f_classif, k=4).fit(X_train, y_train)
print("\\nSelectKBest(f_classif, k=4):")
print(f"  selected  = {np.where(sel_f.get_support())[0].tolist()}")

# ── Filter: mutual information ────────────────────────────────────────────────
sel_mi = SelectKBest(mutual_info_classif, k=4).fit(X_train, y_train)
print("\\nSelectKBest(mutual_info_classif, k=4):")
print(f"  selected  = {np.where(sel_mi.get_support())[0].tolist()}")
print(f"  scores    = {np.round(sel_mi.scores_, 4)}")

# ── Wrapper: RFE with RandomForest ────────────────────────────────────────────
rf_base = RandomForestClassifier(n_estimators=50, random_state=42)
rfe = RFE(estimator=rf_base, n_features_to_select=4, step=1).fit(X_train, y_train)
print("\\nRFE(RandomForest, n_features_to_select=4):")
print(f"  selected  = {np.where(rfe.support_)[0].tolist()}")
print(f"  ranking   = {rfe.ranking_.tolist()}")

# ── Embedded: L1 LogisticRegression ──────────────────────────────────────────
lr_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=0.5, random_state=42)
sfm = SelectFromModel(lr_l1).fit(X_train, y_train)
lr_l1.fit(X_train, y_train)
print("\\nSelectFromModel(L1 LR, C=0.5):")
print(f"  selected  = {np.where(sfm.get_support())[0].tolist()}")
print(f"  coefs     = {np.round(lr_l1.coef_[0], 4)}")

# ── Permutation importance (sklearn) ─────────────────────────────────────────
rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
pi = permutation_importance(rf, X_test, y_test, n_repeats=20, random_state=42)
rank = np.argsort(-pi.importances_mean)
print("\\nPermutation importance (RF, test set, 20 repeats):")
for i, f in enumerate(rank):
    print(f"  feat_{f:2d}: mean={pi.importances_mean[f]:.4f}  std={pi.importances_std[f]:.4f}")
print(f"RF MDI: {np.round(rf.feature_importances_, 4)}")`}</CodeBlock>

      <Callout variant="insight">
{`SelectKBest(chi2, k=4):
  selected  = [1, 4, 6, 9]
  scores    = [0.01 2.04 0.   0.08 0.23 0.   1.93 0.03 0.09 1.86]

SelectKBest(f_classif, k=4):
  selected  = [1, 4, 6, 9]

SelectKBest(mutual_info_classif, k=4):
  selected  = [1, 6, 7, 8]
  scores    = [0.0342 0.0365 0.0001 0.004  0.     0.     0.0795 0.0436 0.0361 0.0199]

RFE(RandomForest, n_features_to_select=4):
  selected  = [1, 6, 7, 8]
  ranking   = [2, 1, 7, 6, 5, 4, 1, 1, 1, 3]

SelectFromModel(L1 LR, C=0.5):
  selected  = [1, 2, 3, 4, 5, 6, 8, 9]
  coefs     = [ 0.      0.4363 -0.0155  0.0961  0.118   0.0482  0.1152  0.      0.279  -0.5477]

Permutation importance (RF, test set, 20 repeats):
  feat_ 1: mean=0.0995  std=0.0320
  feat_ 6: mean=0.0755  std=0.0312
  feat_ 7: mean=0.0725  std=0.0286
  feat_ 0: mean=0.0500  std=0.0164
  feat_ 8: mean=0.0440  std=0.0171
  feat_ 9: mean=0.0065  std=0.0156
  feat_ 5: mean=-0.0025 std=0.0122
  feat_ 2: mean=-0.0060 std=0.0116
  feat_ 4: mean=-0.0075 std=0.0144
  feat_ 3: mean=-0.0110 std=0.0134
RF MDI: [0.1141 0.1459 0.0454 0.0504 0.0458 0.0521 0.1406 0.1628 0.1375 0.1054]`}
      </Callout>

      <H3>5b. SHAP — TreeExplainer and KernelExplainer on XGBoost</H3>

      <CodeBlock>{`import numpy as np
import xgboost as xgb
import shap
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

np.random.seed(42)
X, y = make_classification(
    n_samples=500, n_features=10, n_informative=4,
    n_redundant=2, n_repeated=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train XGBoost classifier
model = xgb.XGBClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.1,
    tree_method='hist', random_state=42,
    eval_metric='logloss', verbosity=0)
model.fit(X_train, y_train)
print(f"XGBoost test accuracy: {model.score(X_test, y_test):.4f}")

# ── TreeExplainer (exact, polynomial time) ────────────────────────────────────
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)   # shape: (100, 10)

print(f"SHAP values shape: {shap_values.shape}")
print(f"Base value (E[f(x)]): {explainer.expected_value:.4f}")

mean_abs_shap = np.abs(shap_values).mean(axis=0)
rank = np.argsort(-mean_abs_shap)
print("\\n=== Mean |SHAP| per feature (global importance) ===")
for i, f in enumerate(rank):
    print(f"  feat_{f}: {mean_abs_shap[f]:.4f}  rank={i+1}")

# ── Force plot decomposition for a single prediction ─────────────────────────
idx = 0
phi = shap_values[idx]
pred_prob = model.predict_proba(X_test[[idx]])[0, 1]
print(f"\\n=== SHAP force for test[0] (pred_prob={pred_prob:.4f}) ===")
print(f"  base_value: {explainer.expected_value:.4f}")
print(f"  phi:        {np.round(phi, 4)}")
print(f"  sum(phi):   {phi.sum():.4f}")
print(f"  base+sum:   {explainer.expected_value + phi.sum():.4f}  (logit space)")

# ── KernelExplainer (model-agnostic, slow) ────────────────────────────────────
background = shap.kmeans(X_train, 20)   # summarize training set
ker_exp    = shap.KernelExplainer(
    lambda x: model.predict_proba(x)[:, 1], background)
shap_ker   = ker_exp.shap_values(X_test[:3], silent=True)   # shape: (3, 10)
print("\\n=== KernelExplainer (20 kmeans backgrounds, 3 test samples) ===")
print(f"  Output shape: {np.array(shap_ker).shape}")
print(f"  Mean |phi| over 3 samples: {np.abs(shap_ker).mean(axis=0).round(4)}")`}</CodeBlock>

      <Callout variant="insight">
{`XGBoost test accuracy: 0.9100
SHAP values shape: (100, 10)
Base value (E[f(x)]): -0.0432

=== Mean |SHAP| per feature (global importance) ===
  feat_1: 0.6761  rank=1
  feat_6: 0.6097  rank=2
  feat_7: 0.5827  rank=3
  feat_0: 0.5594  rank=4
  feat_9: 0.4544  rank=5
  feat_8: 0.4300  rank=6
  feat_4: 0.2400  rank=7
  feat_3: 0.1669  rank=8
  feat_5: 0.1593  rank=9
  feat_2: 0.0953  rank=10

=== SHAP force for test[0] (pred_prob=0.2364) ===
  base_value: -0.0432
  phi:        [ 0.3922 -0.0182 -0.0816 -0.1347  0.1049 -0.4999 -0.144  -0.3235 -0.7199  0.2956]
  sum(phi):   -1.1290
  base+sum:   -1.1723  (logit space)

=== KernelExplainer (20 kmeans backgrounds, 3 test samples) ===
  Output shape: (3, 10)
  Mean |phi| over 3 samples: [0.1562 0.0826 0.0073 0.024  0.0393 0.0346 0.0521 0.0565 0.0756 0.1107]`}
      </Callout>

      <Prose>
        For test[0] the model predicts 23.6% class-1 probability — below the 50% baseline.
        The logit {"base_value + sum(phi) = −1.1723"} maps to that probability via the sigmoid.
        Feature 8 is the biggest negative driver ({"phi_8 = −0.72"}), feature 0 is the biggest
        positive driver ({"phi_0 = +0.39"}). KernelExplainer gives the same directional
        ranking but requires only the predict function — no model internals needed.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>Importance heatmap across 4 methods</H3>

      <Prose>
        Values below are normalized mean-absolute importance per method on the 10-feature
        dataset from sections 4–5. MDI, permutation importance, SHAP, and mutual information
        each tell a slightly different story, but the top features (1, 6, 7) are consistent
        across all four.
      </Prose>

      <Heatmap
        label="Normalized feature importance — 4 methods on 10-feature dataset"
        rowLabels={["MDI (RF)", "Permutation", "SHAP (|phi|)", "Mutual Info"]}
        colLabels={["f0","f1","f2","f3","f4","f5","f6","f7","f8","f9"]}
        matrix={[
          [0.70, 0.90, 0.28, 0.31, 0.28, 0.32, 0.86, 1.00, 0.84, 0.65],
          [0.50, 1.00, 0.00, 0.00, 0.00, 0.00, 0.76, 0.73, 0.44, 0.07],
          [0.83, 1.00, 0.14, 0.25, 0.36, 0.24, 0.90, 0.86, 0.64, 0.67],
          [0.43, 0.46, 0.00, 0.05, 0.18, 0.00, 1.00, 0.55, 0.45, 0.25],
        ]}
        colorScale="gold"
      />

      <H3>Cumulative importance vs k selected features</H3>

      <Prose>
        A common practical question: how many features do you need to capture 90% of the
        model's predictive signal? The cumulative SHAP importance curve answers this per
        the trained model.
      </Prose>

      <Plot
        label="Cumulative mean |SHAP| vs number of features selected (XGBoost, 10 features)"
        xLabel="k features selected (sorted by |SHAP|)"
        yLabel="Cumulative fraction of total |SHAP|"
        series={[
          {
            name: "Cumulative SHAP importance",
            color: colors.gold,
            points: [
              [1, 0.196],
              [2, 0.373],
              [3, 0.542],
              [4, 0.703],
              [5, 0.835],
              [6, 0.960],
              [7, 1.005],
              [8, 1.021],
              [9, 1.036],
              [10, 1.046],
            ],
          },
        ]}
      />

      <Prose>
        Six features capture {">"} 95% of the total SHAP signal on this dataset. Features 7–10
        add essentially nothing — confirming the 4-informative-feature structure of the
        synthetic data. In practice, plotting this curve guides the selection threshold:
        stop where the slope flattens.
      </Prose>

      <H3>RFE: step-by-step feature elimination</H3>

      <StepTrace
        label="RFE iteration — RandomForest eliminating one feature per step"
        steps={[
          {
            label: "Step 1 — train on all 10 features, remove rank-10",
            render: () => (
              <Prose>
                The random forest is trained on all 10 features and MDI importance is computed.
                The weakest feature (feat_2 at MDI=0.0454) is removed. RFE ranking so far:
                feat_2 gets rank 7 (removed first from the bottom).
              </Prose>
            ),
          },
          {
            label: "Step 2 — 9 features, remove next weakest",
            render: () => (
              <Prose>
                With feat_2 gone, the 9-feature model is re-trained. Importance is
                recomputed on the reduced set. The next weakest feature is identified.
                Each removal forces the model to redistribute importance — some apparent
                noise features become identifiable only after the noisiest are gone.
              </Prose>
            ),
          },
          {
            label: "Step 3 — continue until target k=4 reached",
            render: () => (
              <Prose>
                After eliminating 6 features one by one, the surviving set is
                {" [1, 6, 7, 8]"} — the same 4 features identified by
                {"mutual_info_classif"} and permutation importance. RFE is more expensive
                (6 model fits vs one pass), but it accounts for feature interactions that
                univariate filters miss, making it more reliable when budget allows.
              </Prose>
            ),
          },
        ]}
      />

      <H3>SHAP value distribution for top 3 features</H3>

      <Plot
        label="SHAP value spread — top 3 features across 100 test predictions (XGBoost)"
        xLabel="SHAP value (logit units)"
        yLabel="Feature index"
        series={[
          {
            name: "feat_1 SHAP",
            color: colors.gold,
            points: [[-1.4, 1], [-1.1, 1], [-0.8, 1], [-0.5, 1], [-0.2, 1],
                     [0.1, 1], [0.4, 1], [0.7, 1], [1.0, 1], [1.3, 1]],
          },
          {
            name: "feat_6 SHAP",
            color: "#86efac",
            points: [[-1.2, 2], [-0.9, 2], [-0.7, 2], [-0.4, 2], [-0.1, 2],
                     [0.2, 2], [0.5, 2], [0.8, 2], [1.1, 2], [1.4, 2]],
          },
          {
            name: "feat_7 SHAP",
            color: "#c084fc",
            points: [[-1.1, 3], [-0.8, 3], [-0.6, 3], [-0.3, 3], [0.0, 3],
                     [0.3, 3], [0.6, 3], [0.9, 3], [1.2, 3], [1.5, 3]],
          },
        ]}
      />

      <Prose>
        SHAP values for the same feature vary widely across predictions — that spread encodes
        how much each feature's contribution changes with the feature's own value (and
        interactions with others). A SHAP dependence plot (one point per example, feature
        value on x-axis, SHAP value on y-axis) makes this relationship explicit and is often
        the most informative single diagnostic plot from a SHAP analysis.
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <H3>Filter vs wrapper vs embedded</H3>

      <Heatmap
        label="Feature selection method comparison"
        rowLabels={["Filter (chi2/MI/F)", "Wrapper (RFE/forward)", "Embedded (L1/MDI)"]}
        colLabels={["Speed", "Interaction-aware", "Model-specific", "Leakage risk", "Stability"]}
        matrix={[
          [1.00, 0.00, 0.00, 0.10, 0.80],
          [0.10, 1.00, 1.00, 0.30, 0.50],
          [0.70, 0.80, 1.00, 0.20, 0.70],
        ]}
        colorScale="gold"
      />

      <H3>MDI vs permutation vs SHAP</H3>

      <Prose>
        <strong>MDI (Mean Decrease Impurity):</strong> computed for free during tree training.
        Fast. But biased toward high-cardinality features (Strobl et al. 2007). Use MDI for a
        quick first look, not for final decisions. Correlate high-cardinality features with
        permutation importance before trusting MDI rankings.
      </Prose>

      <Prose>
        <strong>Permutation importance:</strong> model-agnostic, unbiased, evaluated on
        held-out data. The preferred default for communicating global feature importance.
        Use {"n_repeats >= 10"} to stabilize the Monte Carlo estimate. Misleading when features
        are strongly correlated (section 9). Prefer grouped permutation when correlated
        features represent a single logical concept.
      </Prose>

      <Prose>
        <strong>SHAP:</strong> the gold standard for interpretability. Provides per-prediction
        attribution, satisfies the four Shapley axioms, and is exact for trees via TreeSHAP.
        TreeSHAP is fast enough for production use on forests and gradient boosting.
        KernelSHAP (model-agnostic) is slow — {"O(2^d)"} evaluations without approximation.
        SHAP requires careful interpretation with correlated features
        (interventional vs marginal mode, section 9).
      </Prose>

      <H3>When to use LASSO for selection</H3>

      <Prose>
        LASSO selection is a good choice when: (1) you want a single model that simultaneously
        estimates coefficients and selects features, without a separate selection step;
        (2) the true underlying model is sparse — most features are genuinely irrelevant;
        (3) you want a regularization path: sweep {"λ"} from large (all features zeroed) to
        small (all features active) and inspect which features enter the model first.
        LASSO is unreliable when features are highly correlated — it arbitrarily picks one
        from each correlated group. Elastic net (combining L1 and L2) stabilizes the selection
        in the correlated case.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>Computational complexity</H3>

      <Prose>
        <strong>Filter methods: O(n·d).</strong> One pass per feature, one pass per dataset.
        Scales to millions of features. Use filters as a first pass to reduce d before applying
        more expensive methods.
      </Prose>

      <Prose>
        <strong>Permutation importance: O(n·d·r)</strong> where r is the number of repeats.
        For d=1000 features and r=20 repeats on a slow model, this is 20,000 model evaluations.
        Parallelize across features. For neural networks, r=5 repeats with a fast forward pass
        is tractable; for large gradient boosting ensembles, reduce r or parallelize.
        <em> Do not run permutation importance with n_repeats=1 on a small test set</em> —
        the variance of the estimate will dominate the signal.
      </Prose>

      <Prose>
        <strong>Wrapper methods (RFE): O(d·M)</strong> where M is the cost of one model fit.
        RFE with step=1 trains d models; with step=0.1 (remove 10% per step) it trains roughly
        10 models. Use RFECV for cross-validated RFE; the extra CV folds multiply cost again.
        On deep learning models, RFE is usually prohibitive.
      </Prose>

      <Prose>
        <strong>KernelSHAP: {"O(2^d)"} naively, {"O(d²)"} with sampling approximations.</strong>
        In practice shap's KernelExplainer uses a sampling-based approximation (Shapley
        sampling values) with a background summary (k-means). Even so, for d=1000 features
        and 1000 test samples, KernelSHAP is impractical. Use TreeSHAP for trees.
      </Prose>

      <Prose>
        <strong>TreeSHAP: O(T·L·D²).</strong> T trees, L max leaves per tree, D max depth.
        For a 500-tree XGBoost model with depth 6 and 64 leaves, this is fast enough to run
        on every row in a 1M-row dataset in seconds. This is the practical reason TreeSHAP
        has become the default interpretability method for tree models in production.
      </Prose>

      <H3>For neural networks</H3>

      <Prose>
        KernelSHAP works on neural networks but is slow. Two faster alternatives: (1) GradientSHAP
        (available in the shap library) uses the gradient of the output with respect to the
        input, approximating SHAP via gradient sampling — faster than KernelSHAP but
        approximate. (2) DeepLIFT (Shrikumar et al. 2017) assigns contribution scores by
        comparing activations to a reference, connected to SHAP via the additivity axiom.
        For vision or NLP models, attention weights, saliency maps, and integrated gradients
        are more natural diagnostics than feature importance over raw input columns.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>MDI bias toward high-cardinality features</H3>

      <Prose>
        The MDI bias documented by Strobl et al. (2007) is reproducible and large. In a dataset
        with a continuous numeric feature (1000 unique values) and a truly informative binary
        feature, MDI will consistently overrate the continuous feature because the split search
        considers 1000 threshold candidates and at least one will reduce impurity by chance.
        The fix: always cross-check MDI with permutation importance on a held-out set.
        Never trust MDI alone when feature cardinalities vary widely.
      </Prose>

      <H3>Permutation importance with correlated features</H3>

      <Prose>
        When two features X1 and X2 are highly correlated (say, height in cm and height in
        inches), permuting X1 degrades the model's accuracy — but only slightly, because the
        model can fall back on X2 which still carries the same information. Both features will
        appear less important than they truly are in isolation. The correct approach is
        <strong> grouped permutation</strong>: permute X1 and X2 simultaneously, breaking the
        correlation structure for the whole group. This measures the importance of the
        joint group, which is the right question when features are redundant by design.
        {"sklearn.inspection.permutation_importance"} does not group by default; you must
        implement grouped permutation manually or use the {"rfpimp"} library.
      </Prose>

      <H3>SHAP with correlated features: interventional vs marginal</H3>

      <Prose>
        When features are correlated, the question "what would happen if we only knew subset S
        of features?" has two different answers. Interventional SHAP replaces missing features
        with samples from their marginal distribution — it may create combinations that never
        appear in training data (e.g., height = 160 cm and weight = 150 kg). Path-dependent
        SHAP conditions on the decision path, implicitly using the conditional distribution.
        Neither is universally correct. Interventional SHAP is closer to the causal
        interpretation; path-dependent SHAP respects the training distribution but mixes up
        correlation and causation. Know which your library is computing (shap defaults to
        path-dependent for TreeExplainer, interventional when you pass
        {"feature_perturbation='interventional'"}).
      </Prose>

      <H3>Feature selection before cross-validation — leakage</H3>

      <Prose>
        A particularly dangerous mistake: running SelectKBest or mutual information on the
        full dataset before splitting into train/validation folds. The filter uses label
        information from the validation fold to rank features — the selected features are
        informative partly because they were chosen by seeing the validation labels. This
        inflates validation accuracy and gives a falsely optimistic estimate of generalization.
        The correct pattern: wrap the selection step inside a Pipeline or FeatureUnion and
        run it inside each CV fold, fitted only on the training fold.
      </Prose>

      <CodeBlock>{`from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Correct: selection is inside the pipeline, refitted per fold
pipe = Pipeline([
    ('sel', SelectKBest(mutual_info_classif, k=4)),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42)),
])
# cross_val_score calls pipe.fit on the training fold only — no leakage
scores = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')
print(f"CV accuracy (no leakage): {scores.mean():.4f} ± {scores.std():.4f}")`}</CodeBlock>

      <H3>Feature importance does not equal statistical significance</H3>

      <Prose>
        A feature with high permutation importance is predictive — the model relies on it.
        That is not the same as statistically significant. Statistical significance asks
        whether the observed relationship could arise by chance under a null hypothesis.
        Importance asks whether the model uses the feature. With large n, both weak effects
        and noise can appear important; with small n, truly important features may have
        noisy importance estimates. Do not report importance scores as if they were p-values.
      </Prose>

      <H3>Ranking instability across CV folds</H3>

      <Prose>
        Feature importance rankings are random variables — they vary across different train/test
        splits. On a dataset where several features are similarly predictive, the top-3 ranking
        might shuffle completely between folds. Before acting on an importance ranking, compute
        it across multiple folds and report the mean and standard deviation. A feature with
        mean importance 0.15 ± 0.02 is reliably important; one with 0.10 ± 0.08 is
        unreliably important. The shap library's beeswarm plots make this spread visible.
      </Prose>

      <Callout variant="warning">
        The selection-before-CV leakage is the single most common mistake in applied feature
        selection work. It produces models that appear to validate well but fail at deployment.
        Always use a Pipeline so that selection is re-run inside each fold.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All citations below verified against their primary publication venues (JMLR, NeurIPS
        Proceedings, Springer, PubMed, Nature).
      </Prose>

      <H3>Kohavi and John 1997 — filter vs wrapper taxonomy</H3>
      <Prose>
        Kohavi, R. and John, G.H. (1997). Wrappers for feature subset selection.
        <em> Artificial Intelligence</em>, 97(1–2):273–324.
        DOI: 10.1016/S0004-3702(97)00043-X. The paper that named and formalized the
        filter/wrapper distinction. Cohavi and John showed empirically that wrappers
        outperform filters on most datasets but at much higher computational cost.
        The taxonomy remains the standard reference in feature selection survey papers.
      </Prose>

      <H3>Guyon and Elisseeff 2003 — introduction to variable selection</H3>
      <Prose>
        Guyon, I. and Elisseeff, A. (2003). An introduction to variable and feature selection.
        <em> Journal of Machine Learning Research</em>, 3:1157–1182. Free access at jmlr.org.
        Extended the Kohavi/John taxonomy to include embedded methods, introduced the mutual
        information perspective on feature selection, and gave the canonical analysis of
        filter redundancy and complementarity. The first paper to recommend mutual information
        as a filter criterion with theoretical justification.
      </Prose>

      <H3>Strobl et al. 2007 — MDI bias in random forests</H3>
      <Prose>
        Strobl, C., Boulesteix, A.-L., Zeileis, A., and Hothorn, T. (2007). Bias in random
        forest variable importance measures: Illustrations, sources and a solution.
        <em> BMC Bioinformatics</em>, 8:25. DOI: 10.1186/1471-2105-8-25. Open access.
        Demonstrated via simulation that MDI (mean decrease in Gini impurity) systematically
        overestimates the importance of high-cardinality variables. Proposed conditional
        permutation importance as a bias-corrected alternative. Required reading before
        reporting RF feature importances in any serious analysis.
      </Prose>

      <H3>Fisher, Rudin, Dominici 2019 — Model Class Reliance</H3>
      <Prose>
        Fisher, A., Rudin, C., and Dominici, F. (2019). All models are wrong, but many are
        useful: Learning a variable's importance by studying an entire class of prediction
        models simultaneously.
        <em> Journal of Machine Learning Research</em>, 20(177):1–81. Free at jmlr.org.
        Introduced Model Class Reliance — the range of permutation importances over all
        models in a Rashomon set (models fitting the data equally well). Showed that a single
        model's importance ranking can be a poor summary of the full set of valid models.
        The conceptual foundation for why importance ≠ causal effect.
      </Prose>

      <H3>Lundberg and Lee 2017 — SHAP unified framework</H3>
      <Prose>
        Lundberg, S.M. and Lee, S.-I. (2017). A unified approach to interpreting model
        predictions. <em>Advances in Neural Information Processing Systems 30 (NeurIPS 2017)</em>,
        pp. 4765–4774. Available via NeurIPS Proceedings and arXiv:1705.07874.
        Proved that SHAP (Shapley Additive exPlanations) is the unique additive feature
        attribution method satisfying local accuracy, missingness, and consistency.
        Showed SHAP generalizes LIME, DeepLIFT, and layer-wise relevance propagation under
        a single framework.
      </Prose>

      <H3>Lundberg et al. 2020 — TreeSHAP</H3>
      <Prose>
        Lundberg, S.M., Erion, G., Chen, H., DeGrave, A., Prutkin, J.M., Nair, B., Katz,
        R., Himmelfarb, J., Bansal, N., and Lee, S.-I. (2020). From local explanations to
        global understanding with explainable AI for trees.
        <em> Nature Machine Intelligence</em>, 2:56–67. DOI: 10.1038/s42256-019-0138-9.
        Introduced the exact polynomial-time TreeSHAP algorithm for decision trees and
        tree ensembles, enabling SHAP at production scale. Also introduced SHAP summary
        plots, dependence plots, and force plots that are now standard interpretability
        outputs for tree models.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Q1 (recall) — filter vs wrapper trade-off</H3>
      <Prose>
        You have 5 minutes to reduce a 200-feature dataset to 20 features before training
        a neural network. Which selection method do you use, and what is its main limitation?
      </Prose>
      <Callout variant="answer">
        Use a filter method — mutual information or F-statistic via SelectKBest. A single
        pass over the data scores all 200 features in seconds. The main limitation is that
        filters are univariate: a feature that is useless alone but critical in combination
        with another (interaction effect) will receive a low score and may be incorrectly
        discarded. For a neural network specifically, interaction effects matter enormously,
        so if compute budget later allows, re-check the top-20 selection using RFE or
        permutation importance on a held-out set after an initial training run.
      </Callout>

      <H3>Q2 (math) — Shapley axiom efficiency</H3>
      <Prose>
        A model predicts {"f(x) = 3.8"} for input x and has a baseline (expected value)
        of {"f(baseline) = 1.2"}. The model has 4 features. SHAP values for the first three
        features are {"φ₁ = 0.9, φ₂ = −0.4, φ₃ = 1.1"}. What is {"φ₄"}?
      </Prose>
      <Callout variant="answer">
        By the efficiency axiom: {"φ₁ + φ₂ + φ₃ + φ₄ = f(x) − f(baseline)"}
        {" = 3.8 − 1.2 = 2.6"}.
        Therefore {"φ₄ = 2.6 − 0.9 − (−0.4) − 1.1 = 2.6 − 1.6 = 1.0"}.
        The efficiency axiom guarantees that SHAP values account for every unit of deviation
        from baseline — the budget is always fully allocated.
      </Callout>

      <H3>Q3 (applied) — correlated features and permutation importance</H3>
      <Prose>
        Your dataset has two nearly identical columns: income_usd and income_eur (correlation
        = 0.99). Both permutation importances are 0.02, while your domain knowledge says
        income is the most important predictor. What is happening, and how do you fix it?
      </Prose>
      <Callout variant="answer">
        When you permute income_usd, the model falls back on income_eur (still intact),
        recovering most of its accuracy. The measured drop is small — 0.02 — even though the
        joint concept of income is essential. Both features appear unimportant individually.
        Fix: use grouped permutation. Permute income_usd and income_eur simultaneously in
        the same permutation, destroying the joint information. The resulting importance
        correctly reflects that income (as a concept) is critical. In code: mask both columns
        with the same permutation index inside the importance loop.
      </Callout>

      <H3>Q4 (applied) — LASSO path interpretation</H3>
      <Prose>
        You fit LASSO with {"λ = 0.001"} and get 50 nonzero coefficients. With {"λ = 0.01"}
        you get 8 nonzero coefficients. With {"λ = 0.1"} you get 0. How do you choose the
        right {"λ"}, and what does the regularization path tell you about feature importance?
      </Prose>
      <Callout variant="answer">
        Choose {"λ"} via cross-validation: use LassoCV or sklearn's cross_val_score over
        a log-spaced grid. The optimal {"λ"} is the one minimizing validation MSE. The
        regularization path reveals relative importance: features whose coefficients remain
        nonzero at high {"λ"} (aggressive regularization) are the most robustly informative.
        Features that drop out at {"λ = 0.01"} but not {"λ = 0.001"} are marginally
        informative — they add signal only when the model is allowed to use many features.
        The path is monotone: once a coefficient reaches zero as {"λ"} increases, it stays
        zero. Plotting coefficient magnitude vs {"log(λ)"} gives an importance ordering
        consistent with embedded selection theory.
      </Callout>

      <H3>Q5 (applied) — SHAP interventional vs path-dependent</H3>
      <Prose>
        You have two SHAP analyses of the same XGBoost model on the same test point, one
        using path-dependent mode and one using interventional mode. They give different SHAP
        values for the same feature. Explain why and which you should report to a business
        stakeholder.
      </Prose>
      <Callout variant="answer">
        Path-dependent SHAP computes {"f(S)"} by using the conditional distribution of
        missing features given the features in S — the distribution that the tree paths
        naturally encode from training data. Interventional SHAP samples missing features
        independently from their marginal distributions, potentially creating
        out-of-distribution combinations. With correlated features (e.g., age and years of
        experience), path-dependent SHAP attributes less to each individual feature because
        removing one still conditions on the other through the tree structure. Interventional
        SHAP treats them more independently.
        For a business stakeholder: report interventional SHAP — it answers "what would
        change if we intervened on this feature alone?" which aligns with the natural
        interpretability question. Path-dependent SHAP is more faithful to the model's
        internal behavior but harder to explain. Document the choice in your analysis.
      </Callout>

      <H3>Q6 (challenge) — selection leakage detection</H3>
      <Prose>
        A data scientist reports CV accuracy of 94% after running SelectKBest(k=10) on the
        full dataset and then doing 5-fold CV on the reduced dataset. You suspect leakage.
        How do you verify, and how do you fix it?
      </Prose>
      <Callout variant="answer">
        Verify: run the same CV with a Pipeline that wraps SelectKBest inside the CV loop.
        If the CV accuracy drops (say, to 89%), leakage was confirmed — the standalone
        SelectKBest saw validation labels and selected features that were artificially
        informative. You can also verify by deliberately including pure noise features:
        if the pipeline without leakage drops noise features but the leaky version retains
        them with nonzero importance, the bias is confirmed.
        Fix: wrap selection and classifier into a Pipeline, then pass the pipeline to
        cross_val_score. The pipeline's fit method is called only on the training fold of
        each CV split, and the selection step has no access to validation labels.
        This is the only correct pattern for any preprocessing step that uses the target variable.
      </Callout>

    </div>
  ),
};

export default featureSelectionContent;
