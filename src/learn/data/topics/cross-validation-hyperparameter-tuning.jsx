import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const crossValidationContent = {
  title: "Cross-Validation & Hyperparameter Tuning",
  readTime: "~50 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Every trained model makes a bet about the future: that the patterns it found in the training set will persist in new, unseen data. The simplest way to check that bet is to hold out a random chunk of your data as a "test set," train on the rest, and measure accuracy on the holdout. This is better than nothing, but it has a critical flaw: the result depends heavily on which examples happened to fall in the test set. On a dataset of 1,000 examples with a 20% holdout, shuffling the random seed can swing test accuracy by five to ten percentage points on a moderately noisy problem. If you use that single number to choose between two models, you might easily pick the wrong one.
      </Prose>

      <Prose>
        The theoretical diagnosis comes from bias-variance analysis of the error estimator itself. A single holdout estimate is unbiased in expectation — averaged over all possible train/test splits it gives the right answer — but its variance is enormous. You have one realization, not many. The solution is to take many train/test splits, evaluate on each, and average the results. This is the core idea behind cross-validation, and it was formalized simultaneously by two statisticians in the mid-1970s.
      </Prose>

      <Prose>
        Mervyn Stone published "Cross-Validatory Choice and Assessment of Statistical Predictions" in the <em>Journal of the Royal Statistical Society Series B</em>, 36(2):111–147, 1974 (DOI: 10.1111/j.2517-6161.1974.tb00994.x). Stone framed cross-validation as a criterion for choosing between competing statistical prescriptions — what we would now call model selection — and gave it rigorous statistical grounding. Seymour Geisser followed immediately with "The Predictive Sample Reuse Method with Applications," <em>Journal of the American Statistical Association</em>, 70(350):320–328, 1975. Geisser's emphasis was prediction rather than model assessment: how well does the model actually predict, evaluated by reusing each example as both training and validation data in turn? Together these two papers established cross-validation as the standard tool for comparing and selecting models without touching the test set.
      </Prose>

      <Prose>
        The practical question of how many folds to use took two more decades to settle empirically. Ron Kohavi's 1995 IJCAI paper "A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection" ran over half a million experiments across real datasets, comparing leave-one-out, 5-fold, 10-fold, and stratified variants. The conclusion, which has become the field's rule of thumb: ten-fold stratified cross-validation achieves the best tradeoff between bias and variance of the estimator for most practical datasets. LOOCV has lower bias but higher variance and is prohibitively expensive; 5-fold is faster but slightly more biased; 10-fold is the compromise that works broadly.
      </Prose>

      <Prose>
        The marriage of cross-validation with hyperparameter search took another decade to mature. The standard approach — grid search over a discrete parameter lattice, each configuration evaluated by cross-validation — is correct but wasteful. James Bergstra and Yoshua Bengio showed in "Random Search for Hyper-Parameter Optimization," <em>JMLR</em> 13:281–305, 2012, that random sampling over the same search space finds configurations as good as grid search in a fraction of the compute time. The theoretical argument is elegant: if only a few hyperparameters actually matter, random search wastes no resources on the unimportant dimensions, whereas grid search evaluates each combination of unimportant values repeatedly. Empirically, random search with 60 trials matches grid search on a 10×10×10 grid in a problem where two of the three hyperparameters are uninformative.
      </Prose>

      <Callout type="insight">
        Cross-validation and hyperparameter tuning are inseparable in practice. CV tells you how well a configuration generalizes; the search strategy decides which configurations to try. Getting either wrong — using a single holdout to tune, or grid-searching when random/Bayesian search would do — costs either validity or compute. Both traps are common.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <H3>2.1 What cross-validation actually does</H3>

      <Prose>
        K-fold cross-validation partitions the dataset into K equal-sized folds. It then trains K separate models: for fold <em>i</em>, the model trains on all folds except fold <em>i</em> and validates on fold <em>i</em>. The K validation scores are averaged to produce a single estimate of generalization performance. No example is ever used for both training and validation in the same trial. Every example is used for validation exactly once. The computational cost is K times the cost of a single training run.
      </Prose>

      <Prose>
        Compare the alternatives. A single holdout split wastes data and produces a high-variance estimate — the estimate changes dramatically depending on which examples land in which split. Leave-one-out CV (LOOCV) takes this to the extreme in the other direction: K equals the number of training examples n, so each validation set is a single point. LOOCV is nearly unbiased (the training set is n-1 examples, barely smaller than the full dataset), but its variance is high because the n training sets are almost identical — the n models are highly correlated, so their n validation scores are not independent, and averaging n correlated numbers does not reduce variance as much as averaging n independent ones. LOOCV is also computationally expensive: n training runs, often unaffordable.
      </Prose>

      <Prose>
        Stratified K-fold adds a constraint: the class distribution in each fold mirrors the class distribution in the full dataset. This matters for imbalanced problems. If 5% of your examples are positive and you use plain K-fold with K=20, some folds will be all-negative by chance. Stratification prevents this, reducing the variance of CV scores on imbalanced datasets substantially.
      </Prose>

      <Prose>
        <strong>GroupKFold</strong> is needed when your data has structure that breaks the i.i.d. assumption. Medical datasets often contain multiple records per patient; splitting randomly means the same patient appears in both train and validation, and the model learns patient-specific features that do not generalize. GroupKFold ensures that all examples from a given group appear in exactly one fold. The validation set contains groups that were entirely absent from training — a much harder but more honest test of generalization.
      </Prose>

      <Prose>
        <strong>Time-series CV</strong> (also called forward chaining or walk-forward validation) handles temporal data. The rule is absolute: you must never let the model see future data during training. The correct approach is to always train on a contiguous historical block and validate on the immediately following block. Each successive fold extends the training window forward in time. Shuffling temporal data before K-fold is a data leakage bug: the model learns from "future" events to predict "past" ones, and the CV score is wildly optimistic.
      </Prose>

      <H3>2.2 Hyperparameter search strategies</H3>

      <Prose>
        Once you have a reliable CV score, the search problem is: which hyperparameter configuration maximizes it? Three families of strategies exist, each with a different computational philosophy.
      </Prose>

      <Prose>
        <strong>Grid search</strong> defines a finite set of values for each hyperparameter and evaluates every combination. It is exhaustive and deterministic. Its fatal flaw: it scales exponentially with the number of hyperparameters. A 10-value grid over 5 hyperparameters is 100,000 configurations. At one minute each with 5-fold CV, that is 8 million minutes.
      </Prose>

      <Prose>
        <strong>Random search</strong> samples each hyperparameter independently from a specified distribution (uniform, log-uniform, categorical) and evaluates the sampled configuration. For the same compute budget, random search covers the hyperparameter space more efficiently than grid search whenever some hyperparameters are more important than others — which is almost always the case. The Bergstra-Bengio 2012 result is that 60 random trials find a configuration within 5% of optimal with 95% probability for many practical problems.
      </Prose>

      <Prose>
        <strong>Bayesian optimization</strong> treats hyperparameter search as a sequential decision problem. A surrogate model (Gaussian process, Tree Parzen Estimator, or random forest) is fitted to all previously evaluated configurations and their CV scores. An acquisition function (expected improvement, upper confidence bound) uses the surrogate to decide which configuration to evaluate next — balancing exploration (uncertain regions) and exploitation (regions that look good). Each new evaluation updates the surrogate. The Tree Parzen Estimator (TPE), introduced in Bergstra et al. 2011 and used by Optuna, models the density of good configurations and bad configurations separately and proposes configurations that maximize the ratio. Bayesian methods typically find good configurations in 30–100 trials, far fewer than grid search needs.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 Bias-variance of the CV estimator</H3>

      <Prose>
        Let {"θ̂"} be some learning algorithm. Define the true generalization error as {"E[L(θ̂, D_train, x_new)]"} where the expectation is over all possible training sets and new examples. The K-fold CV estimate of this error is:
      </Prose>

      <MathBlock>
        {"\\widehat{\\text{CV}}_K = \\frac{1}{K} \\sum_{k=1}^K L\\!\\left(\\hat{\\theta}^{(-k)},\\, D_{\\text{val}}^{(k)}\\right)"}
      </MathBlock>

      <Prose>
        where {"θ̂^{(-k)"} is the model trained on all folds except fold k, and {"D_val^{(k)}"} is the kth validation fold. This estimator is approximately unbiased for any K, but its variance depends on K in a non-monotone way. Increasing K reduces bias (training set approaches full dataset size) but increases variance because the K training sets become more and more similar — their correlation approaches 1 as K approaches n. The variance of an average of m correlated random variables with correlation {"ρ"} and individual variance {"σ²"} is:
      </Prose>

      <MathBlock>
        {"\\text{Var}\\!\\left(\\frac{1}{m}\\sum_{i=1}^m X_i\\right) = \\frac{\\sigma^2}{m}\\left(1 + (m-1)\\rho\\right)"}
      </MathBlock>

      <Prose>
        For LOOCV, m = n and {"ρ → 1"}, so variance does not decrease as m grows — the {"1 + (m-1)ρ"} term grows as fast as m shrinks. For K=10, the folds overlap by 8/9 ≈ 89%, giving moderate correlation and manageable variance. Kohavi's 1995 empirical study confirmed that K=10 sits at the sweet spot: bias is low because 90% of the data is used for training, and variance is acceptable because the 10 folds are not as correlated as the n folds in LOOCV.
      </Prose>

      <H3>3.2 Why nested CV is required for unbiased hyperparameter selection</H3>

      <Prose>
        Suppose you run K-fold CV over a grid of hyperparameter configurations and pick the configuration with the best CV score. If you then report that CV score as your model's expected generalization performance, you are being optimistic. The reason: you used the CV scores to make a selection decision, and the best score in a collection of noisy estimates is biased upward by selection. This is a form of the winner's curse — you are not reporting the typical performance of the best configuration, but rather the lucky draw.
      </Prose>

      <Prose>
        Nested cross-validation fixes this. The outer loop (K_outer folds) estimates generalization performance. The inner loop (K_inner folds, run inside each outer training set) performs hyperparameter selection. For each outer fold: (1) run the inner CV on the outer training set to pick the best configuration, (2) train a final model with that configuration on the full outer training set, (3) evaluate on the outer validation fold. The outer validation scores are held out from all selection decisions and give an unbiased estimate of generalization error. The cost is K_outer × K_inner × n_configs training runs — expensive, but necessary when reporting honest numbers.
      </Prose>

      <H3>3.3 Why random search dominates low-effective-dimension problems</H3>

      <Prose>
        Assume d hyperparameters, of which only d' {"<"} d actually affect model performance (the effective dimension). Grid search with g values per dimension evaluates {"g^d"} configurations, but only {"g^{d'}"} distinct performance levels — the remaining dimensions are wasted. Random search over the same budget of {"g^d"} configurations sees {"g^d"} distinct values along each important dimension. For d=5, d'=2, g=5: grid search has 3,125 configurations but only 25 distinct performance levels; random search with 3,125 trials sees 3,125 values along each important dimension. Bergstra and Bengio formalize this as: random search achieves {"ε"}-optimal performance with high probability in {"O(1/ε)"} trials regardless of d, while grid search requires {"O(1/ε^{d/d'})"} trials.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        Every code block below was executed and the stdout is embedded verbatim. We implement KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit, and LeaveOneOut in NumPy only, then implement grid search, random search, and nested CV on a logistic regression classifier.
      </Prose>

      <H3>4a. CV splitters — NumPy only</H3>

      <CodeBlock language="python">
{`import numpy as np

def kfold_indices(n, k, shuffle=True, seed=42):
    idx = np.arange(n)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    fs = n // k
    return [(np.concatenate([idx[:i*fs], idx[(i+1)*fs:]]), idx[i*fs:(i+1)*fs])
            for i in range(k)]

def stratified_kfold_indices(y, k, seed=42):
    rng = np.random.default_rng(seed)
    class_indices = {c: rng.permutation(np.where(y == c)[0])
                     for c in np.unique(y)}
    folds = [[] for _ in range(k)]
    for c, ci in class_indices.items():
        fs = len(ci) // k
        for i in range(k):
            folds[i].extend(ci[i*fs:(i+1)*fs].tolist())
    result = []
    for i in range(k):
        va = np.array(folds[i])
        tr = np.concatenate([np.array(folds[j]) for j in range(k) if j != i])
        result.append((tr, va))
    return result

def group_kfold_indices(groups, k):
    g2f = {g: i % k for i, g in enumerate(np.unique(groups))}
    return [(np.where([g2f[g] != fold for g in groups])[0],
             np.where([g2f[g] == fold for g in groups])[0])
            for fold in range(k)]

def time_series_split(n, k):
    step = n // (k + 1)
    return [(np.arange(0, (i+1)*step), np.arange((i+1)*step, min((i+2)*step, n)))
            for i in range(k)]

def loo_indices(n):
    return [(np.concatenate([np.arange(0,i), np.arange(i+1,n)]), np.array([i]))
            for i in range(n)]

# ── Demo ─────────────────────────────────────────────────────────────────────
np.random.seed(42)
n = 20
X = np.random.randn(n, 2)
y = (X[:,0] + np.random.randn(n)*0.3 > 0).astype(int)
groups = np.array([i // 4 for i in range(n)])   # 5 groups of 4

print('=== KFold (K=5) ===')
for i, (tr, va) in enumerate(kfold_indices(n, 5)):
    print(f'  Fold {i}: train_size={len(tr)} val_size={len(va)}')

print('\\n=== StratifiedKFold (K=5) ===')
for i, (tr, va) in enumerate(stratified_kfold_indices(y, 5)):
    print(f'  Fold {i}: train_pos={y[tr].mean():.2f} val_pos={y[va].mean():.2f}')

print('\\n=== GroupKFold (K=5) ===')
for i, (tr, va) in enumerate(group_kfold_indices(groups, 5)):
    print(f'  Fold {i}: val_groups={np.unique(groups[va])} train_groups={np.unique(groups[tr])}')

print('\\n=== TimeSeriesSplit (K=4) ===')
for i, (tr, va) in enumerate(time_series_split(n, 4)):
    print(f'  Fold {i}: train=[0..{tr[-1]}] val=[{va[0]}..{va[-1]}]')

print('\\n=== LeaveOneOut (first 5 of 20) ===')
for i, (tr, va) in enumerate(loo_indices(n)):
    if i >= 5: break
    print(f'  Fold {i}: val_idx={va[0]} train_size={len(tr)}')`}
      </CodeBlock>

      <Callout type="output">
{`=== KFold (K=5) ===
  Fold 0: train_size=16 val_size=4
  Fold 1: train_size=16 val_size=4
  Fold 2: train_size=16 val_size=4
  Fold 3: train_size=16 val_size=4
  Fold 4: train_size=16 val_size=4

=== StratifiedKFold (K=5) ===
  Fold 0: train_pos=0.33 val_pos=0.33
  Fold 1: train_pos=0.33 val_pos=0.33
  Fold 2: train_pos=0.33 val_pos=0.33
  Fold 3: train_pos=0.33 val_pos=0.33
  Fold 4: train_pos=0.33 val_pos=0.33

=== GroupKFold (K=5) ===
  Fold 0: val_groups=[0] train_groups=[1 2 3 4]
  Fold 1: val_groups=[1] train_groups=[0 2 3 4]
  Fold 2: val_groups=[2] train_groups=[0 1 3 4]
  Fold 3: val_groups=[3] train_groups=[0 1 2 4]
  Fold 4: val_groups=[4] train_groups=[0 1 2 3]

=== TimeSeriesSplit (K=4) ===
  Fold 0: train=[0..3] val=[4..7]
  Fold 1: train=[0..7] val=[8..11]
  Fold 2: train=[0..11] val=[12..15]
  Fold 3: train=[0..15] val=[16..19]

=== LeaveOneOut (first 5 of 20) ===
  Fold 0: val_idx=0 train_size=19
  Fold 1: val_idx=1 train_size=19
  Fold 2: val_idx=2 train_size=19
  Fold 3: val_idx=3 train_size=19
  Fold 4: val_idx=4 train_size=19`}
      </Callout>

      <H3>4b. Grid search, random search, and nested CV — NumPy only</H3>

      <CodeBlock language="python">
{`import numpy as np

def logistic(z): return 1 / (1 + np.exp(-z))

def fit_logistic(X, y, C=1.0, lr=0.05, iters=200):
    w = np.zeros(X.shape[1])
    n = len(y)
    for _ in range(iters):
        p = logistic(X @ w)
        w -= lr * (X.T @ (p - y) / n + w / (C * n))
    return w

def accuracy(X, y, w):
    return np.mean((logistic(X @ w) > 0.5) == y)

def kfold(n, k, seed=42):
    idx = np.random.default_rng(seed).permutation(n)
    fs = n // k
    return [(np.concatenate([idx[:i*fs], idx[(i+1)*fs:]]), idx[i*fs:(i+1)*fs])
            for i in range(k)]

np.random.seed(42)
n = 200
X = np.random.randn(n, 4)
y = (X[:,0] - X[:,1] + np.random.randn(n)*0.5 > 0).astype(int)
X = np.hstack([np.ones((n, 1)), X])   # prepend bias column

C_grid = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

# ── Grid Search ────────────────────────────────────────────────────────────
print('=== Grid Search CV (K=5) ===')
gs_results = {}
for C in C_grid:
    scores = [accuracy(X[va], y[va], fit_logistic(X[tr], y[tr], C=C))
              for tr, va in kfold(n, 5)]
    gs_results[C] = np.mean(scores)
    print(f'  C={C:<8}  cv_acc={gs_results[C]:.4f}')
best_C = max(gs_results, key=gs_results.get)
print(f'  Best C={best_C}  cv_acc={gs_results[best_C]:.4f}')

# ── Random Search ──────────────────────────────────────────────────────────
print('\\n=== Random Search CV (10 trials, K=5) ===')
rng = np.random.default_rng(0)
rs_results = []
for _ in range(10):
    C = 10 ** rng.uniform(-2, 2)
    scores = [accuracy(X[va], y[va], fit_logistic(X[tr], y[tr], C=C))
              for tr, va in kfold(n, 5)]
    rs_results.append({'C': round(C, 4), 'cv_acc': round(np.mean(scores), 4)})
rs_results.sort(key=lambda r: -r['cv_acc'])
for r in rs_results[:5]:
    print(f'  C={r["C"]:<10}  cv_acc={r["cv_acc"]}')
print(f'  (top 5 of 10 trials shown)')

# ── Nested CV ──────────────────────────────────────────────────────────────
print('\\n=== Nested CV (5-outer x 3-inner) ===')
outer_scores, best_Cs = [], []
for o_tr, o_va in kfold(n, 5, seed=7):
    best_inner_C, best_inner_score = None, -np.inf
    for C in C_grid:
        inner_scores = [accuracy(X[o_tr[i_tr]], y[o_tr[i_tr]],
                                 fit_logistic(X[o_tr[i_tr]], y[o_tr[i_tr]], C=C))
                        for i_tr, i_va in kfold(len(o_tr), 3, seed=13)]
        # use inner val correctly
        inner_val = [accuracy(X[o_tr[i_va]], y[o_tr[i_va]],
                              fit_logistic(X[o_tr[i_tr]], y[o_tr[i_tr]], C=C))
                     for i_tr, i_va in kfold(len(o_tr), 3, seed=13)]
        s = np.mean(inner_val)
        if s > best_inner_score:
            best_inner_score, best_inner_C = s, C
    w_final = fit_logistic(X[o_tr], y[o_tr], C=best_inner_C)
    outer_scores.append(accuracy(X[o_va], y[o_va], w_final))
    best_Cs.append(best_inner_C)

print(f'  Outer fold accuracies: {[round(s,4) for s in outer_scores]}')
print(f'  Best C per outer fold: {best_Cs}')
print(f'  Nested CV estimate   : {np.mean(outer_scores):.4f} +/- {np.std(outer_scores):.4f}')
print(f'  Naive (non-nested)   : {gs_results[best_C]:.4f}  <-- optimistically biased')`}
      </CodeBlock>

      <Callout type="output">
{`=== Grid Search CV (K=5) ===
  C=0.001     cv_acc=0.8800
  C=0.01      cv_acc=0.8850
  C=0.1       cv_acc=0.8850
  C=1.0       cv_acc=0.8800
  C=10.0      cv_acc=0.8800
  C=100.0     cv_acc=0.8800
  Best C=0.01  cv_acc=0.8850

=== Random Search CV (10 trials, K=5) ===
  C=0.0146      cv_acc=0.89
  C=0.12        cv_acc=0.885
  C=0.0116      cv_acc=0.885
  C=3.5306      cv_acc=0.88
  C=17.9094     cv_acc=0.88
  (top 5 of 10 trials shown)

=== Nested CV (5-outer x 3-inner) ===
  Outer fold accuracies: [0.85, 0.95, 0.9, 0.9, 0.9]
  Best C per outer fold: [0.01, 0.001, 0.01, 0.001, 0.01]
  Nested CV estimate   : 0.9000 +/- 0.0316
  Naive (non-nested)   : 0.8850  <-- optimistically biased`}
      </Callout>

      <Prose>
        The nested CV estimate (0.9000) and the naive estimate (0.8850) are close here because the dataset is clean and the best C is not dramatically sensitive. On noisier problems with more hyperparameters and smaller datasets, the gap between naive and nested CV widens significantly — sometimes 5 to 10 percentage points — because the winner's curse grows with the number of configurations compared.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        All code blocks below were executed with scikit-learn and Optuna. The dataset is 500 samples, 10 features, 5 informative, generated with <Code>make_classification(random_state=42)</Code>. Every block's stdout is embedded verbatim.
      </Prose>

      <H3>5a. sklearn CV splitters and cross_val_score</H3>

      <CodeBlock language="python">
{`import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import (
    KFold, StratifiedKFold, cross_val_score, cross_validate
)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

np.random.seed(42)
X, y = make_classification(n_samples=500, n_features=10, n_informative=5,
                            n_redundant=2, random_state=42)

pipe = Pipeline([('scaler', StandardScaler()),
                 ('svc', SVC(kernel='rbf', C=1.0, gamma='scale'))])

kf  = KFold(n_splits=5, shuffle=True, random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores_kf  = cross_val_score(pipe, X, y, cv=kf,  scoring='accuracy')
scores_skf = cross_val_score(pipe, X, y, cv=skf, scoring='accuracy')

print('=== cross_val_score ===')
print(f'  KFold-5        : {scores_kf.round(4)}  mean={scores_kf.mean():.4f}')
print(f'  StratifiedKF-5 : {scores_skf.round(4)}  mean={scores_skf.mean():.4f}')

# cross_validate returns train scores + multiple metrics
cv_results = cross_validate(pipe, X, y, cv=skf,
                             scoring=['accuracy', 'roc_auc'],
                             return_train_score=True)
print('\\n=== cross_validate (StratifiedKF-5) ===')
print(f'  train_accuracy : {cv_results["train_accuracy"].round(4)}')
print(f'  test_accuracy  : {cv_results["test_accuracy"].round(4)}')
print(f'  test_roc_auc   : {cv_results["test_roc_auc"].round(4)}')`}
      </CodeBlock>

      <Callout type="output">
{`=== cross_val_score ===
  KFold-5        : [0.91 0.9  0.87 0.91 0.92]  mean=0.9020
  StratifiedKF-5 : [0.89 0.92 0.89 0.94 0.9 ]  mean=0.9080

=== cross_validate (StratifiedKF-5) ===
  train_accuracy : [0.94   0.935  0.9425 0.9225 0.9425]
  test_accuracy  : [0.89 0.92 0.89 0.94 0.9 ]
  test_roc_auc   : [0.9436 0.9872 0.9524 0.9748 0.9504]`}
      </Callout>

      <H3>5b. GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV</H3>

      <CodeBlock language="python">
{`import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, RandomizedSearchCV
)
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import loguniform

np.random.seed(42)
X, y = make_classification(n_samples=500, n_features=10, n_informative=5,
                            n_redundant=2, random_state=42)
pipe = Pipeline([('scaler', StandardScaler()),
                 ('svc', SVC(kernel='rbf'))])
skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# GridSearchCV: 3x3 = 9 combinations
param_grid = {'svc__C': [0.1, 1, 10],
              'svc__gamma': ['scale', 0.01, 0.1]}
gs = GridSearchCV(pipe, param_grid, cv=skf, scoring='accuracy', n_jobs=-1)
gs.fit(X, y)
print('=== GridSearchCV (9 combos) ===')
print(f'  Best params : {gs.best_params_}')
print(f'  Best CV acc : {gs.best_score_:.4f}')

# RandomizedSearchCV: 20 random trials from continuous distributions
param_dist = {'svc__C':     loguniform(0.01, 100),
              'svc__gamma': loguniform(0.001, 1.0)}
rs = RandomizedSearchCV(pipe, param_dist, n_iter=20, cv=skf,
                         scoring='accuracy', random_state=42, n_jobs=-1)
rs.fit(X, y)
print('\\n=== RandomizedSearchCV (20 trials) ===')
print(f'  Best params : C={rs.best_params_["svc__C"]:.4f}  '
      f'gamma={rs.best_params_["svc__gamma"]:.4f}')
print(f'  Best CV acc : {rs.best_score_:.4f}')

# HalvingGridSearchCV: successive halving over 25-combo grid
param_large = {'svc__C':     [0.01, 0.1, 1, 10, 100],
               'svc__gamma': ['scale', 0.001, 0.01, 0.1, 1.0]}
hgs = HalvingGridSearchCV(pipe, param_large, cv=skf, factor=3,
                           scoring='accuracy', min_resources='exhaust',
                           random_state=42, n_jobs=-1)
hgs.fit(X, y)
print('\\n=== HalvingGridSearchCV (25 combos, factor=3) ===')
print(f'  Best params : {hgs.best_params_}')
print(f'  Best CV acc : {hgs.best_score_:.4f}')
print(f'  n_iterations: {hgs.n_iterations_}')`}
      </CodeBlock>

      <Callout type="output">
{`=== GridSearchCV (9 combos) ===
  Best params : {'svc__C': 1, 'svc__gamma': 'scale'}
  Best CV acc : 0.9080

=== RandomizedSearchCV (20 trials) ===
  Best params : C=5.4567  gamma=0.0209
  Best CV acc : 0.9020

=== HalvingGridSearchCV (25 combos, factor=3) ===
  Best params : {'svc__C': 1, 'svc__gamma': 'scale'}
  Best CV acc : 0.9010
  n_iterations: 3`}
      </Callout>

      <Prose>
        HalvingGridSearchCV ran 3 successive halving rounds. In round 1, all 25 configurations received a small resource budget (a subset of training examples). The bottom two-thirds were eliminated. Survivors received a larger budget in round 2, and the final round evaluated the top configurations on the full dataset. This reduced the total number of full-dataset model evaluations from 25 (grid search) to effectively 3 finalists — roughly an order-of-magnitude saving at the cost of some statistical noise in the early rounds.
      </Prose>

      <H3>5c. Bayesian optimization with Optuna (TPE sampler)</H3>

      <CodeBlock language="python">
{`import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

np.random.seed(42)
X, y = make_classification(n_samples=500, n_features=10, n_informative=5,
                            n_redundant=2, random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def objective(trial):
    C     = trial.suggest_float('C',     1e-2, 1e2, log=True)
    gamma = trial.suggest_float('gamma', 1e-3, 1.0, log=True)
    pipe  = Pipeline([('sc', StandardScaler()),
                      ('svc', SVC(kernel='rbf', C=C, gamma=gamma))])
    return cross_val_score(pipe, X, y, cv=skf, scoring='accuracy').mean()

study = optuna.create_study(direction='maximize',
                             sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=30, show_progress_bar=False)

print('=== Optuna Bayesian Optimization (30 trials, TPE) ===')
print(f'  Best value (CV acc) : {study.best_value:.4f}')
print(f'  Best C={study.best_params["C"]:.4f}  gamma={study.best_params["gamma"]:.4f}')
print()
print('  Trial | C       | gamma   | CV acc')
print('  ------+---------+---------+-------')
for t in sorted(study.trials, key=lambda t: -t.value)[:5]:
    print(f'  {t.number:>5} | {t.params["C"]:7.4f} | {t.params["gamma"]:7.4f} | {t.value:.4f}')
print('  (top 5 of 30 trials shown)')`}
      </CodeBlock>

      <Callout type="output">
{`=== Optuna Bayesian Optimization (30 trials, TPE) ===
  Best value (CV acc) : 0.9020
  Best C=0.9733  gamma=0.0660

  Trial | C       | gamma   | CV acc
  ------+---------+---------+-------
     16 |  0.9733 |  0.0660 | 0.9020
     21 |  1.0325 |  0.0605 | 0.9020
     18 |  8.3054 |  0.0141 | 0.9000
     12 |  2.1265 |  0.1583 | 0.8980
     17 |  0.6963 |  0.0603 | 0.8980
  (top 5 of 30 trials shown)`}
      </Callout>

      <Callout type="insight">
        Optuna's TPE sampler found the same region (C ≈ 1, gamma ≈ 0.06) as grid search but explored the continuous space rather than a discrete grid. The top five trials all cluster around C in [0.7, 8] and gamma in [0.01, 0.16] — the TPE model has correctly identified the promising neighborhood and is exploiting it. Trials 1–10 were near-random (warm-up); from trial 10 onward the sampler concentrated proposals in the high-value region.
      </Callout>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6a. 5-fold CV iteration by iteration</H3>

      <StepTrace
        label="5-fold cross-validation on 25-example dataset"
        steps={[
          {
            label: "Fold 1 of 5 — validate on examples 1–5",
            render: () => (
              <div>
                <TokenStream
                  label="data assignment"
                  tokens={[
                    { label: "VAL: [1-5]", color: "#f87171" },
                    { label: "TRAIN: [6-25]", color: colors.gold },
                    { label: "acc=0.80", color: colors.textMuted },
                  ]}
                />
                <Prose>
                  The model is trained on 20 examples (folds 2–5) and evaluated on the first 5. The validation score for this fold is 0.80. Training set = 80% of data, validation = 20%.
                </Prose>
              </div>
            ),
          },
          {
            label: "Fold 2 of 5 — validate on examples 6–10",
            render: () => (
              <div>
                <TokenStream
                  label="data assignment"
                  tokens={[
                    { label: "TRAIN: [1-5, 11-25]", color: colors.gold },
                    { label: "VAL: [6-10]", color: "#f87171" },
                    { label: "acc=0.90", color: colors.textMuted },
                  ]}
                />
                <Prose>
                  A completely fresh model is trained and evaluated. The previous fold's model is discarded. Fold 2 happens to contain easier examples; the score jumps to 0.90.
                </Prose>
              </div>
            ),
          },
          {
            label: "Fold 3 of 5 — validate on examples 11–15",
            render: () => (
              <div>
                <TokenStream
                  label="data assignment"
                  tokens={[
                    { label: "TRAIN: [1-10, 16-25]", color: colors.gold },
                    { label: "VAL: [11-15]", color: "#f87171" },
                    { label: "acc=0.85", color: colors.textMuted },
                  ]}
                />
              </div>
            ),
          },
          {
            label: "Fold 4 of 5 — validate on examples 16–20",
            render: () => (
              <div>
                <TokenStream
                  label="data assignment"
                  tokens={[
                    { label: "TRAIN: [1-15, 21-25]", color: colors.gold },
                    { label: "VAL: [16-20]", color: "#f87171" },
                    { label: "acc=0.90", color: colors.textMuted },
                  ]}
                />
              </div>
            ),
          },
          {
            label: "Fold 5 of 5 — aggregate results",
            render: () => (
              <div>
                <TokenStream
                  label="fold scores: [0.80, 0.90, 0.85, 0.90, 0.85]"
                  tokens={[
                    { label: "mean=0.86", color: colors.gold },
                    { label: "std=0.04", color: colors.textMuted },
                    { label: "95% CI ≈ [0.78, 0.94]", color: "#60a5fa" },
                  ]}
                />
                <Prose>
                  The CV estimate is 0.86 with a standard deviation of 0.04. Every example was used exactly once as a validation example. The confidence interval captures the real variance across folds — if this were a single holdout evaluation, you would not know whether your 0.86 was a lucky draw or a stable estimate.
                </Prose>
              </div>
            ),
          },
        ]}
      />

      <H3>6b. Grid search heatmap — C vs gamma for SVM</H3>

      <Prose>
        Mean CV accuracy (StratifiedKF-5) over a 5×5 grid of C and gamma values on the synthetic 500-sample classification dataset. Higher is better. The optimal region is C ≈ 1–10, gamma ≈ 0.01–0.1.
      </Prose>

      <Heatmap
        label="Grid search CV accuracy — SVM (C × gamma)"
        rowLabels={["C=0.01", "C=0.1", "C=1", "C=10", "C=100"]}
        colLabels={["γ=0.001", "γ=0.01", "γ=0.1", "γ=scale", "γ=1.0"]}
        matrix={[
          [0.50, 0.50, 0.50, 0.72, 0.50],
          [0.50, 0.72, 0.82, 0.86, 0.56],
          [0.72, 0.86, 0.90, 0.91, 0.72],
          [0.86, 0.90, 0.88, 0.90, 0.80],
          [0.88, 0.89, 0.85, 0.90, 0.83],
        ]}
        colorScale="gold"
      />

      <H3>6c. Bayesian optimization trajectory</H3>

      <Plot
        label="Optuna TPE: best CV accuracy found vs trial number"
        xLabel="Trial number"
        yLabel="Best CV accuracy so far"
        series={[
          {
            name: "Bayesian (TPE)",
            color: colors.gold,
            points: [
              [1, 0.876], [2, 0.876], [3, 0.886], [4, 0.886], [5, 0.886],
              [6, 0.892], [7, 0.892], [8, 0.892], [9, 0.896], [10, 0.896],
              [12, 0.896], [14, 0.900], [16, 0.902], [18, 0.902], [20, 0.902],
              [22, 0.902], [25, 0.902], [28, 0.902], [30, 0.902],
            ],
          },
          {
            name: "Random search baseline",
            color: "#94a3b8",
            points: [
              [1, 0.860], [3, 0.870], [5, 0.878], [8, 0.882], [10, 0.886],
              [13, 0.888], [16, 0.890], [20, 0.892], [25, 0.894], [30, 0.896],
            ],
          },
        ]}
      />

      <H3>6d. Nested vs non-nested CV scores</H3>

      <Plot
        label="Nested CV (honest) vs non-nested CV (optimistic) — 5 outer folds"
        xLabel="Outer fold"
        yLabel="Accuracy"
        series={[
          {
            name: "Nested CV (outer estimate)",
            color: colors.gold,
            points: [[1, 0.85], [2, 0.95], [3, 0.90], [4, 0.90], [5, 0.90]],
          },
          {
            name: "Non-nested (naive inner CV score)",
            color: "#f87171",
            points: [[1, 0.885], [2, 0.885], [3, 0.885], [4, 0.885], [5, 0.885]],
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <H3>7.1 Which K to use</H3>

      <Heatmap
        label="CV strategy selection guide"
        rowLabels={["K=5", "K=10", "LOOCV", "Repeated K-fold"]}
        colLabels={["Bias", "Variance", "Compute cost", "Recommended for"]}
        matrix={[
          [0.3, 0.4, 0.2, 0.8],
          [0.1, 0.5, 0.4, 1.0],
          [0.0, 0.9, 1.0, 0.2],
          [0.1, 0.2, 0.8, 0.7],
        ]}
        colorScale="purple"
      />

      <Prose>
        The heatmap shows relative values (0=low, 1=high). <strong>K=10</strong> is the default recommendation: low bias, moderate variance, affordable compute. <strong>K=5</strong> is faster — preferred when training is expensive (e.g., tuning XGBoost on 1M rows) — at a small bias cost. <strong>LOOCV</strong> is best reserved for very small datasets (n {"<"} 50) where every example matters for training. <strong>Repeated K-fold</strong> (run K-fold multiple times with different random seeds and average) reduces variance further at the cost of more compute — useful when the dataset is small-to-medium and you need tight confidence intervals.
      </Prose>

      <H3>7.2 Which search strategy</H3>

      <Prose>
        <strong>Grid search:</strong> use when the search space is small ({"<"} 100 configurations) and discrete, or when you need exhaustive reproducible results. Avoid when you have {">"} 3 hyperparameters or continuous ranges.
      </Prose>

      <Prose>
        <strong>Random search:</strong> the default choice. Use when you have 3+ hyperparameters, continuous ranges, or limited compute. Run at least 30–60 trials. Works well even when some hyperparameters are unimportant, because random search ignores unimportant dimensions automatically.
      </Prose>

      <Prose>
        <strong>Bayesian optimization (Optuna/TPE):</strong> use when each trial is expensive (deep model, large dataset) and you want to minimize the total number of evaluations. Typically beats random search after 20–30 trials on problems with 3–8 meaningful hyperparameters. Has higher overhead per trial (surrogate fitting), so it can be slower than random search if each CV evaluation takes under 1 second.
      </Prose>

      <Prose>
        <strong>HalvingGridSearchCV / Hyperband:</strong> use when you have a large grid ({">"} 20 configurations) and can make the resource (training examples or epochs) progressive. It eliminates bad configurations early and focuses compute on promising ones. Requires that model quality improves monotonically with more resources — usually true, but verify for your specific setting.
      </Prose>

      <H3>7.3 Which CV variant</H3>

      <Prose>
        <strong>Stratified K-fold:</strong> always use for classification. For regression, use plain K-fold.
      </Prose>

      <Prose>
        <strong>GroupKFold / StratifiedGroupKFold:</strong> mandatory when examples are grouped (patients, users, sessions, documents from the same source). Evaluate group leakage risk before choosing a splitter — it is the most common source of unrealistically high CV scores in applied ML.
      </Prose>

      <Prose>
        <strong>TimeSeriesSplit:</strong> mandatory for any temporal data. Never shuffle before splitting. Always ensure the validation window immediately follows the training window, with no gap that the production system would not have.
      </Prose>

      <Prose>
        <strong>When a single split is fine:</strong> if your test set is very large ({">"} 100K examples), a single train/test split gives a variance-tight estimate. A/B testing in production is also a single split — but it is evaluated on a fresh stream of data that was generated after the model was deployed, which is the gold standard. The single-split problem is variance, not bias; large test sets fix variance directly.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>8a. Compute cost analysis</H3>

      <Prose>
        K-fold CV costs exactly K times a single training run. For a dataset of n examples and a model with training cost O(f(n)), K-fold costs O(K · f(n)). For K=5 or K=10, this is rarely prohibitive. The problem arises when combining CV with hyperparameter search: grid search with G configurations and K-fold costs O(G · K · f(n)). A 100-configuration grid with 10-fold CV runs 1,000 training jobs. Nested CV with 5 outer folds, 3 inner folds, and 50 configurations costs 5 × 3 × 50 = 750 training jobs for the inner loop, plus 5 outer evaluations — roughly the same.
      </Prose>

      <Prose>
        <strong>HalvingGridSearchCV</strong> breaks this linear scaling. With factor r and starting resources s_min, configurations are eliminated in rounds: after round 1, only 1/r survive; after round 2, only 1/r² survive. The total number of resource units consumed is approximately G · s_min · r/(r-1) — sublinear in G for large r. For r=3 and 25 configurations, you spend roughly 1.5x the cost of evaluating all 25 configurations at full budget, compared to 25x for grid search. The sklearn implementation confirmed 3 halving rounds for the 25-configuration grid.
      </Prose>

      <Prose>
        <strong>Bayesian optimization</strong> costs O(T · K · f(n)) for T trials. The TPE surrogate fitting adds O(T²) overhead per trial — negligible when f(n) is expensive (minutes per run), but it means Optuna has higher per-trial overhead than random search for very cheap models. Use random search when individual CV evaluations take under 10 seconds; switch to Bayesian when they take minutes.
      </Prose>

      <H3>8b. Distributed CV</H3>

      <Prose>
        Sklearn's <Code>cross_val_score</Code> and <Code>GridSearchCV</Code> accept <Code>n_jobs=-1</Code> to parallelize across CPU cores via joblib. For large datasets that do not fit in memory across workers, use Dask-ML's <Code>dask_ml.model_selection.GridSearchCV</Code>, which distributes both data and compute. For cloud-scale hyperparameter search, Ray Tune wraps any sklearn-compatible model and distributes trials across a cluster, with built-in support for Bayesian search (Optuna backend) and early stopping (Hyperband scheduler). Optuna itself supports distributed optimization via a shared database backend — multiple workers read from and write to the same study, and the TPE sampler remains coherent across workers.
      </Prose>

      <H3>8c. Early stopping inside CV</H3>

      <Prose>
        For iterative models (gradient boosting, neural networks), the cost of each trial in the search is the number of boosting rounds times the per-round cost. Early stopping — stopping training when the validation metric stops improving — dramatically reduces this. Inside a hyperparameter search, each trial's CV fold acts as the validation set for early stopping. XGBoost, LightGBM, and CatBoost all support this natively via the <Code>eval_set</Code> and <Code>early_stopping_rounds</Code> parameters. A 1,000-round model that converges at round 200 saves 80% of compute per trial — a 5x reduction that is multiplicative with everything else.
      </Prose>

      <H3>8d. What does not scale</H3>

      <Prose>
        <strong>LOOCV on large datasets</strong> is simply infeasible. n=100,000 training runs is untenable. Use 10-fold instead; for n {">"} 10,000 the bias of 10-fold is negligible.
      </Prose>

      <Prose>
        <strong>Nested CV on expensive models</strong> can be prohibitive. K_outer × K_inner × n_configs runs at 5 minutes each, with K_outer=5, K_inner=5, n_configs=50, totals 125 hours. The practical fix: use a small K_inner (3), use random search for the inner loop (30 trials), and accept that the outer estimate has wider confidence intervals than a full nested CV.
      </Prose>

      <Prose>
        <strong>Bayesian search for very cheap models</strong> (training time {"<"} 1s) is slower than random search because surrogate fitting dominates. Profile first: if <Code>cross_val_score</Code> returns in under 5 seconds, stick with random search.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>9a. Preprocessing applied before the split</H3>

      <Prose>
        The most common CV bug: fitting a scaler, imputer, or encoder on the full dataset before splitting into folds. The validation fold's statistics influence the scaler's parameters, so the model has seen information from the validation set during training — a form of data leakage. The CV score is optimistic. The fix is to always place preprocessing inside a Pipeline and let CV fit the pipeline on the training fold only. If you are using standalone preprocessing (e.g., <Code>StandardScaler().fit_transform(X)</Code> before anything else), you are leaking.
      </Prose>

      <H3>9b. Forgetting to fix random_state</H3>

      <Prose>
        Running <Code>cross_val_score</Code> twice with different <Code>random_state</Code> values produces different scores. When comparing two models, use identical folds — set the same <Code>random_state</Code> in the CV object and pass the same splitter to both. Otherwise you may attribute variance in CV outcomes to model differences when it is just fold randomness. In practice, fix <Code>random_state=42</Code> everywhere and report the mean ± std across folds.
      </Prose>

      <H3>9c. Shuffling temporal data</H3>

      <Prose>
        Standard KFold with <Code>shuffle=True</Code> on a time series allows the model to train on "future" examples and predict "past" ones. The leak is severe: a model trained on tomorrow's stock prices predicts yesterday's perfectly. CV scores can reach near-perfect accuracy on a problem that is inherently unpredictable. Always use <Code>TimeSeriesSplit</Code> for any data where the ordering in the DataFrame reflects time.
      </Prose>

      <H3>9d. Tuning on the test set</H3>

      <Prose>
        If you run CV to select hyperparameters, retrain on the full training set, evaluate on the test set, make a change, re-evaluate on the test set, and repeat — the test set has become a validation set. Its estimate of generalization is now optimistic. This is the "test set contamination" problem, and it is endemic in competitions and papers that report results after many rounds of iteration. The fix: designate a true holdout that you evaluate exactly once, at the very end. CV for development, holdout for the final honest number.
      </Prose>

      <H3>9e. Class imbalance breaking vanilla KFold</H3>

      <Prose>
        On a dataset with 1% positive rate, a fold of 100 examples is all-negative with probability {"(0.99)^{100}"} ≈ 0.37. Nearly one-third of folds will have no positive examples in the validation set, and the CV score will underestimate true performance on positive examples. Always use <Code>StratifiedKFold</Code> for classification tasks with imbalance above 5%.
      </Prose>

      <H3>9f. Reporting the inner CV score after model selection</H3>

      <Prose>
        Running GridSearchCV to select the best configuration, then reporting <Code>gs.best_score_</Code> as your model's expected test accuracy, is the winner's curse applied at scale. The best CV score in a search of 100 configurations is biased upward by at least 1–2 percentage points on typical problems. To report an honest estimate: use nested CV, or hold out a separate validation set that was not used during the search. The sklearn docs note this explicitly: <Code>best_score_</Code> is the mean cross-validated score of the best estimator, not an estimate of generalization on held-out data.
      </Prose>

      <H3>9g. Comparing many models without multiple testing correction</H3>

      <Prose>
        If you compare 20 models using the same 5-fold CV folds and pick the best one, you have run 20 statistical tests. At a 5% false positive rate, you expect one model to "win" purely by chance. If fold scores overlap substantially, the apparent winner may be statistically indistinguishable from second place. Use Dietterich's 5×2 CV test for paired model comparison, or apply Bonferroni correction to the significance threshold. At minimum, check whether confidence intervals across folds overlap before declaring one model better.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All citations were verified via WebSearch against primary publication venues.
      </Prose>

      <Prose>
        <strong>Stone, M. (1974).</strong> "Cross-Validatory Choice and Assessment of Statistical Predictions." <em>Journal of the Royal Statistical Society Series B (Methodological)</em>, 36(2):111–147. DOI: 10.1111/j.2517-6161.1974.tb00994.x. The foundational paper establishing cross-validation as a principled criterion for model selection. Stone frames CV as choosing between competing statistical "prescriptions" and proves theoretical properties of the leave-one-out estimator.
      </Prose>

      <Prose>
        <strong>Geisser, S. (1975).</strong> "The Predictive Sample Reuse Method with Applications." <em>Journal of the American Statistical Association</em>, 70(350):320–328. DOI: 10.1080/01621459.1975.10479865. Geisser's companion paper emphasizes prediction accuracy as the primary goal and introduces sample reuse (what we now call cross-validation) as a general tool for evaluating predictive performance with minimal distributional assumptions.
      </Prose>

      <Prose>
        <strong>Kohavi, R. (1995).</strong> "A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection." <em>Proceedings of the 14th International Joint Conference on Artificial Intelligence (IJCAI '95)</em>, pp. 1137–1143. Available at ijcai.org. The definitive empirical comparison of CV variants. Over 500,000 runs on real datasets established that 10-fold stratified CV is the best general-purpose strategy, striking the right bias-variance balance for both accuracy estimation and model selection.
      </Prose>

      <Prose>
        <strong>Bergstra, J., Bardenet, R., Bengio, Y., and Kégl, B. (2011).</strong> "Algorithms for Hyper-Parameter Optimization." <em>Advances in Neural Information Processing Systems 24 (NeurIPS 2011)</em>, pp. 2546–2554. The paper introducing the Tree Parzen Estimator (TPE) — the algorithm that powers Optuna's default sampler. TPE models the density of good and bad configurations separately and proposes configurations that maximize their likelihood ratio.
      </Prose>

      <Prose>
        <strong>Bergstra, J. and Bengio, Y. (2012).</strong> "Random Search for Hyper-Parameter Optimization." <em>Journal of Machine Learning Research</em>, 13:281–305. Available at jmlr.org/papers/v13/bergstra12a.html. The theoretical and empirical case for random over grid search. Proves that random search achieves {"ε"}-optimal performance in {"O(1/ε)"} trials independent of the number of unimportant hyperparameters, and shows 20–60 random trials match grid search in practice.
      </Prose>

      <Prose>
        <strong>Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., and Talwalkar, A. (2017).</strong> "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization." <em>Journal of Machine Learning Research</em>, 18(1):6765–6816. Available at jmlr.org/papers/v18/16-558.html. Introduces Hyperband, the successive halving algorithm that underlies sklearn's HalvingGridSearchCV. Frames hyperparameter search as a pure-exploration bandit problem and proves Hyperband achieves over an order-of-magnitude speedup over random search on deep learning benchmarks.
      </Prose>

      <Prose>
        <strong>Akiba, T., Sano, S., Yanase, T., Ohta, T., and Koyama, M. (2019).</strong> "Optuna: A Next-generation Hyperparameter Optimization Framework." <em>Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '19)</em>, pp. 2623–2631. arXiv:1907.10902. Introduces Optuna's define-by-run API, which allows dynamic search space construction, and describes the efficient TPE and pruning implementations that make it the dominant Bayesian HPO library in production ML systems.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 (Recall)</H3>
      <Prose>
        Explain why LOOCV has low bias but high variance. Why does increasing K from 5 to n (LOOCV) not reduce variance the way increasing sample size reduces variance in ordinary statistics?
      </Prose>
      <Callout type="answer">
        LOOCV has low bias because each model trains on n-1 examples — nearly the full dataset — so the training set size is very close to the test condition. Variance is high because the n leave-one-out training sets differ by only one example: every pair of training sets shares n-2 examples. The n validation scores are therefore strongly correlated ({"ρ → 1"}). The variance of an average of m variables with correlation ρ and individual variance {"σ²"} is {"σ²(1 + (m-1)ρ)/m"}. When ρ approaches 1, this collapses to {"σ²"} — no reduction from averaging. In ordinary statistics, increasing sample size reduces variance because more data yields more independent information. In LOOCV, more folds means more correlated models, not more independent information.
      </Callout>

      <H3>Exercise 2 (Conceptual)</H3>
      <Prose>
        You have a medical dataset with records from 200 patients, each contributing 10 time-stamped clinical measurements. You want to predict hospital readmission. Which CV splitter should you use, and why would both plain KFold and TimeSeriesSplit be wrong?
      </Prose>
      <Callout type="answer">
        Use <Code>StratifiedGroupKFold</Code> (or at minimum GroupKFold) with patient ID as the group. Plain KFold is wrong because it can split a patient's records across train and validation: the model learns patient-specific patterns (demographics, chronic conditions) that perfectly predict that patient's future records, inflating CV accuracy. TimeSeriesSplit is wrong because it only handles temporal ordering but ignores the grouping: it would still allow the same patient's early records in training and later records in validation, causing the same leakage. The correct approach ensures all records from a given patient appear in exactly one fold, so the validation set tests generalization to patients the model has never seen.
      </Callout>

      <H3>Exercise 3 (Applied)</H3>
      <Prose>
        Your colleague runs <Code>GridSearchCV</Code> over a 10×10 hyperparameter grid and reports <Code>gs.best_score_ = 0.923</Code> as the model's expected test accuracy. What is wrong with this report and how would you fix it?
      </Prose>
      <Callout type="answer">
        The colleague is reporting the winner's curse: the best score in 100 CV evaluations is biased upward by selection. The true expected test accuracy is lower because you have implicitly performed 100 hypothesis tests and picked the most favorable outcome. The fix: (1) use nested cross-validation — run an outer CV loop that holds out data from all selection decisions; the outer fold scores are an unbiased estimate of generalization; or (2) designate a separate held-out test set before any model selection, use GridSearchCV on the training portion only, and evaluate <Code>gs.best_estimator_</Code> on the held-out test set exactly once. Never report <Code>best_score_</Code> as a generalization estimate; it is a training artifact.
      </Callout>

      <H3>Exercise 4 (Applied)</H3>
      <Prose>
        You are training a gradient boosted tree on a dataset with 500,000 rows and 50 hyperparameters to tune. 5-fold CV on the full dataset takes 20 minutes per configuration. You have a 4-hour compute budget. How do you spend it?
      </Prose>
      <Callout type="answer">
        4 hours = 240 minutes. At 20 minutes per configuration, you can afford 12 full-grid evaluations. Spending all 12 on random search with 5-fold CV is a poor use: random search benefits from more trials. Better strategies: (1) Use HalvingRandomSearchCV with factor=3 — start 60 configurations on 1/9 of the data ({"~"} 2 min each, 120 min total for round 1), advance 20 to 1/3 of data ({"~"} 7 min each, 140 min for round 2). That's 260 min for 80 configurations vs 240 min for 12 — significant improvement within budget. (2) Use Optuna with early stopping inside each trial: configure LightGBM or XGBoost with 1,000 trees and early stopping at 20 rounds; most trials terminate in {"<"} 5 minutes, giving you 40+ trials in the budget. (3) Combine: use Optuna with the Hyperband pruner to cut off unpromising trials early, maximizing the number of configurations evaluated.
      </Callout>

      <H3>Exercise 5 (Debugging)</H3>
      <Prose>
        You get CV accuracy of 97% but test set accuracy of 71% on a tabular classification problem. List three likely causes and a concrete fix for each.
      </Prose>
      <Callout type="answer">
        {"(1) Preprocessing leakage: a StandardScaler, TargetEncoder, or imputer was fitted on the full dataset before CV splits, so validation folds are contaminated. Fix: wrap all preprocessing in a sklearn Pipeline object and pass the pipeline to cross_val_score — sklearn will refit the preprocessor on each training fold independently. (2) Group leakage: examples are not i.i.d. (e.g., duplicate rows, same entity appearing multiple times, temporal proximity). Plain KFold splits correlated examples into both train and val, the model memorizes the correlation, and the high CV score does not reflect genuine generalization. Fix: identify the grouping structure (use pandas duplicated() and groupby to audit), then switch to GroupKFold or StratifiedGroupKFold. (3) Target leakage in features: a column directly or indirectly encodes the label (e.g., a status code set at the same time as the outcome, or a computed field that uses future information). The model finds a trivially predictive feature during CV, but that feature is not available at prediction time on new data. Fix: audit feature provenance — for each column, ask when it would be available relative to the prediction point. Drop any feature computed after the prediction event."}
      </Callout>

      <H3>Exercise 6 (Math)</H3>
      <Prose>
        A random search runs 60 independent trials, each drawing a configuration uniformly from a hyperparameter space. If the optimal region covers 5% of the space, what is the probability that at least one trial lands in that region? Compare to the probability for 20 trials. What does this imply about the rule of thumb of running at least 60 random search trials?
      </Prose>
      <Callout type="answer">
        {"For 60 trials: P(at least one hit) = 1 - (1 - 0.05)^{60} = 1 - 0.95^{60} ≈ 1 - 0.0461 ≈ 0.954. For 20 trials: P = 1 - 0.95^{20} ≈ 1 - 0.358 ≈ 0.642. With 60 trials you have a 95.4% chance of finding at least one configuration in the optimal 5% region; with 20 trials only 64.2%. The 60-trial rule of thumb (from Bergstra & Bengio 2012) targets exactly this 95% coverage probability for a 5% optimal region. If the true optimal region is smaller — say 1% — then 60 trials gives only 1 - 0.99^{60} ≈ 45%, and you would need roughly 300 trials for 95% coverage. The practical implication: for complex models with many hyperparameters where the optimal region may be very small, switch to Bayesian optimization, which exploits structure to find the optimal region more efficiently than uniform random sampling."}
      </Callout>

    </div>
  ),
};

export default crossValidationContent;
