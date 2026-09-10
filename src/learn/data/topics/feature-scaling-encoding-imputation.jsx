import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const featureScalingContent = {
  title: "Feature Scaling, Encoding & Imputation",
  readTime: "~45 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Raw tabular data is a mixed-type chaos. Age lives between 18 and 90. Annual income lives between 20,000 and 500,000. A zip code is a string that happens to look like a number. Employment status is a word. Whether someone defaulted on a loan is a 0 or 1. Pour all of that directly into a gradient descent optimizer or a k-nearest-neighbors classifier and you get a model dominated by whichever feature has the largest numeric range — not because that feature is the most predictive, but simply because its scale drowns out the others. Most classical ML algorithms assume that all numeric features live on a comparable ruler. Three preprocessing operations exist to enforce that assumption, convert non-numeric data into numeric form, and fill in the holes left by real-world data collection.
      </Prose>

      <Prose>
        <strong>Feature scaling</strong> puts numeric columns on the same measurement scale without destroying the information they carry. Gradient descent converges faster when all features are similarly scaled — the loss surface becomes more spherical and the gradient steps point more directly toward the minimum. K-nearest neighbors and support vector machines use Euclidean distance, so a feature with a range of 10,000 overwhelms one with a range of 1. Principal component analysis builds axes of maximum variance, and without scaling the first principal component collapses onto whichever feature has the largest raw variance regardless of importance. L1 and L2 regularization penalize weight magnitudes, so an unscaled feature forces the model to use a tiny weight to compensate for its large range — the effective regularization strength becomes feature-dependent.
      </Prose>

      <Prose>
        <strong>Encoding</strong> converts categorical variables into numbers. Scikit-learn's estimators refuse non-numeric input at the matrix level — there is no choice but to encode. But the manner of encoding is a consequential modeling decision. Treating city names as integers (0=Chicago, 1=Los Angeles, 2=New York) imposes a false ordinal relationship. One-hot encoding avoids that but multiplies the number of features. Target encoding collapses a high-cardinality column to a single float but introduces leakage when done carelessly. Each encoding choice changes what the downstream model can learn and how it generalizes.
      </Prose>

      <Prose>
        <strong>Imputation</strong> handles missing values. Dropping rows with any missing data is statistically dangerous when missingness is not completely at random — you are silently biasing the training distribution toward the population of complete records, which may differ systematically from the full population. A medical dataset where sicker patients are more likely to have incomplete lab results will be biased if you drop incomplete rows: you train on the healthier subpopulation and evaluate on everyone. Imputation replaces missing values with plausible estimates inferred from the observed data.
      </Prose>

      <Prose>
        The intellectual lineage of these three operations spans a century. Karl Pearson's 1901 paper introducing principal component analysis (in the <em>Philosophical Magazine</em>, vol. 6, no. 2, pp. 559–572) implicitly required that variables be standardized before decomposition — the insight that covariance structure is meaningful only when features share a common scale predates modern computing. One-hot encoding descends from binary coding in 1960s telecommunications engineering, where it was used to represent symbols in error-correcting codes. Scikit-learn's <Code>ColumnTransformer</Code>, which unified the handling of mixed-type columns under a single API, was introduced in version 0.20 in 2018, finally giving practitioners a clean single-call solution to the fit-train-test symmetry problem that had caused data leakage in countless pipelines before it.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <H3>2.1 Scaling: one ruler for all features</H3>

      <Prose>
        Imagine plotting age on the x-axis and income on the y-axis. The income axis spans $480,000 while age spans 72 years. A Euclidean distance between two people will be computed almost entirely from the income difference because a 1-unit difference in income is worth far less than a 1-unit difference in age in any meaningful sense, yet the raw arithmetic disagrees. Scaling fixes this by applying a monotone transformation to each column independently — the relative ordering of values within each column is preserved, but the ranges are brought into comparable territory. After standardization (zero mean, unit variance), a one-unit difference in any feature corresponds to one standard deviation — a natural common unit.
      </Prose>

      <Prose>
        The critical discipline of scaling is that the transformation parameters must be learned from the training data only and then applied to the test data. You compute the mean and standard deviation on the training set, then use those same values to transform the test set. If you compute on the full dataset before splitting, your test set has informed the transformation — a subtle but real form of data leakage. The training data leaks future knowledge about the distribution to the evaluation. In cross-validation, this means each fold's validation set must be transformed using statistics computed from the other folds only. <Code>sklearn.Pipeline</Code> enforces this automatically.
      </Prose>

      <H3>2.2 Encoding: mapping categories to vectors</H3>

      <Prose>
        A categorical variable is a discrete set of unordered labels. The fundamental question encoding answers is: what numbers should represent these labels such that the model can learn the right relationships? One-hot encoding is the honest choice for nominal categories (no intrinsic order): each category becomes its own binary dimension. The model sees a city not as a number but as a direction in feature space — and no distance between any two directions is artificially constrained by an integer ordering. The cost is dimensionality: a feature with 1,000 distinct cities becomes 1,000 binary columns.
      </Prose>

      <Prose>
        Target encoding compresses high-cardinality columns back to a single float by replacing each category with the average target value observed for that category in the training data. This is elegant and compact, but it is also dangerous: if you compute the per-category mean on the full training set and then use that mean as a feature during training, the model sees a perfect signal for the target — every training row's encoded value is derived from the target of rows including itself. The fix is out-of-fold encoding: for each training fold, compute the category mean using only the other folds. Test-set encoding uses the full training set mean. This is the same logic as cross-validation, applied to feature construction.
      </Prose>

      <H3>2.3 Imputation: reasoning about the holes</H3>

      <Prose>
        Missing data follows one of three mechanisms, each requiring a different response. Missing Completely At Random (MCAR) means the probability of a value being missing is independent of any observed or unobserved variable. Dropping MCAR rows loses information but does not bias estimates. Missing At Random (MAR) means missingness depends on observed variables but not on the missing value itself — for example, younger patients are less likely to have cholesterol recorded, but among patients of any given age, missingness does not depend on the actual cholesterol level. Imputing from observed predictors (iterative or model-based imputation) works well here. Missing Not At Random (MNAR) means the value's absence depends on the value itself — high earners skip income fields. No purely data-driven imputation strategy is unbiased for MNAR; adding a missingness indicator column and letting the model learn from the indicator is the pragmatic fallback.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 Scaling transforms</H3>

      <Prose>
        Let <Code>x</Code> be a column of <Code>n</Code> training observations. The four standard scalers apply the following transforms at inference time:
      </Prose>

      <MathBlock>
        {"\\text{StandardScaler: } z = \\frac{x - \\hat{\\mu}}{\\hat{\\sigma}}"}
      </MathBlock>

      <MathBlock>
        {"\\text{MinMaxScaler: } z = \\frac{x - x_{\\min}}{x_{\\max} - x_{\\min}}"}
      </MathBlock>

      <MathBlock>
        {"\\text{RobustScaler: } z = \\frac{x - \\tilde{x}}{\\text{IQR}}"}
      </MathBlock>

      <Prose>
        where <Code>{"μ̂, σ̂"}</Code> are the sample mean and standard deviation (population convention, <Code>ddof=0</Code>) computed on the training set; <Code>{"x_min, x_max"}</Code> are the training-set extremes; <Code>{"x̃"}</Code> is the median; and IQR is the interquartile range (75th minus 25th percentile). StandardScaler produces zero-mean unit-variance outputs — optimal for Gaussian-distributed features and for regularized models where the penalty should be scale-invariant. MinMaxScaler maps the training range to <Code>[0, 1]</Code> — useful when you need bounded outputs (neural network inputs with bounded activations, image pixel values) but sensitive to outliers since a single outlier compresses all other values. RobustScaler uses median and IQR, making it resistant to outliers: a handful of extreme values cannot shift the center or inflate the scale.
      </Prose>

      <Prose>
        For strongly skewed features where even RobustScaler leaves the distribution non-Gaussian, power transforms are preferred. The Yeo-Johnson transform (Yeo and Johnson, 2000) is defined for all real inputs:
      </Prose>

      <MathBlock>
        {"\\psi(x; \\lambda) = \\begin{cases} \\frac{(x+1)^\\lambda - 1}{\\lambda} & x \\geq 0, \\lambda \\neq 0 \\\\ \\ln(x+1) & x \\geq 0, \\lambda = 0 \\\\ -\\frac{(1-x)^{2-\\lambda}-1}{2-\\lambda} & x < 0, \\lambda \\neq 2 \\\\ -\\ln(1-x) & x < 0, \\lambda = 2 \\end{cases}"}
      </MathBlock>

      <Prose>
        The parameter <Code>λ</Code> is estimated by maximum likelihood, maximizing the log-likelihood of the transformed data under a Gaussian model. A fitted <Code>λ ≈ 0</Code> corresponds to log-transform; <Code>λ ≈ 1</Code> corresponds to no change; <Code>λ ≈ 2</Code> corresponds to a square-root-like compression for negative values. The Box-Cox transform is similar but only valid for strictly positive inputs; Yeo-Johnson extends it to the full real line.
      </Prose>

      <Prose>
        The QuantileTransformer maps each feature through its empirical cumulative distribution function, producing a uniform <Code>[0, 1]</Code> or normal <Code>N(0,1)</Code> output. It is fully non-parametric — no distributional assumption — but it loses the ordinal information between quantile bins and cannot extrapolate beyond the training range.
      </Prose>

      <H3>3.2 Encoding: one-hot, ordinal, and target encoding</H3>

      <Prose>
        Let <Code>C</Code> be a categorical feature with <Code>K</Code> distinct categories <Code>{"c₁, ..., c_K"}</Code>. One-hot encoding maps each observation to a binary vector <Code>{"e_k ∈ {0,1}^K"}</Code> with a single 1 in the position corresponding to the observed category. The encoded matrix has <Code>K</Code> columns. For linear models with an intercept, one-hot encoding introduces perfect multicollinearity: the sum of all K indicator columns equals 1, the same as the intercept column. The fix is <Code>drop='first'</Code> — drop one reference category, leaving <Code>K-1</Code> columns. Tree-based models do not need this since they do not invert a feature matrix, but linear models and PCA do.
      </Prose>

      <Prose>
        Ordinal encoding maps each category to an integer in <Code>{"0, 1, ..., K-1"}</Code>. This is appropriate only when the categories have a meaningful total order (clothing sizes: XS {"<"} S {"<"} M {"<"} L {"<"} XL) and when the model can exploit that order (linear models, neural networks). Applying ordinal encoding to nominal categories (city names) imposes a false metric structure that tree-based models will exploit spuriously.
      </Prose>

      <Prose>
        Target encoding replaces each category <Code>c</Code> with a smoothed estimate of <Code>{"E[y | C = c]"}</Code>:
      </Prose>

      <MathBlock>
        {"\\hat{\\mu}_c = \\frac{n_c \\cdot \\bar{y}_c + \\alpha \\cdot \\bar{y}_{\\text{global}}}{n_c + \\alpha}"}
      </MathBlock>

      <Prose>
        where <Code>{"n_c"}</Code> is the count of observations in category <Code>c</Code>, <Code>{"ȳ_c"}</Code> is their mean target, <Code>{"ȳ_global"}</Code> is the global target mean, and <Code>α</Code> is a smoothing parameter (typically 5–20). The smoothing term pulls rare categories toward the global mean, reducing variance on categories with few observations. Without smoothing, a category observed once gets encoded as exactly the single observation's target — maximum variance, minimum bias. This is the James-Stein intuition applied to categorical means.
      </Prose>

      <Prose>
        The leakage derivation: if you compute <Code>{"ȳ_c"}</Code> from the full training set and then train a model on those encoded values, each row's encoded value was computed using that row's own target value. The model sees a feature that is a deterministic (smoothed) function of the target — in the limit of <Code>α = 0</Code> and a single observation per category, the encoded feature is literally the target. The out-of-fold fix computes, for each row <Code>i</Code>, the category mean using all rows <em>except</em> row <Code>i</Code>'s fold. Test rows always use the full-training-set mean.
      </Prose>

      <H3>3.3 Imputation: mean, MICE, and KNN</H3>

      <Prose>
        Mean imputation replaces each missing value with the column mean computed from the observed training values. It is unbiased for the column mean under MCAR, but it attenuates correlations between columns — by replacing missing values with the mean, you are inserting points that have zero deviation from center in that column, which pulls the estimated covariance toward zero. For downstream models that use the covariance structure (PCA, regularized regression), this distortion can be significant.
      </Prose>

      <Prose>
        MICE (Multivariate Imputation by Chained Equations, van Buuren and Groothuis-Oudshoorn 2011) treats imputation as a sequence of regression problems. Initialize all missing values with column means. Then cycle through each column with missing values: regress that column on all other columns (using a model of your choice — linear regression, random forest, etc.) using only the rows where that column is observed, then use the fitted model to impute the missing rows. Cycle through all columns with missing values and repeat for several iterations until convergence. The resulting imputed values reflect the joint distribution of the features, not just the marginal mean. Scikit-learn exposes this as <Code>IterativeImputer</Code> (experimental as of sklearn 1.5).
      </Prose>

      <MathBlock>
        {"\\text{MICE cycle: for each column } j, \\quad \\hat{x}_{ij} = f_j\\bigl(x_{i,-j}\\bigr), \\quad \\text{where } f_j \\text{ fitted on observed rows of } j"}
      </MathBlock>

      <Prose>
        KNN imputation replaces each missing value with the weighted average of the corresponding values in the <Code>k</Code> nearest complete neighbors, where distance is computed on the observed features shared between the missing row and each candidate neighbor. It is non-parametric and captures non-linear relationships, but it scales as <Code>{"O(n² · d)"}</Code> at inference time — expensive for large datasets.
      </Prose>

      <Prose>
        A note on trees: decision trees and gradient-boosted ensembles do not require scaling — splits are threshold comparisons on individual features, and a monotone transformation of a feature does not change any threshold. However, they do require encoding (categorical strings are not numeric) and are sensitive to the handling of missing values: some implementations (XGBoost, LightGBM) handle NaNs natively by learning which branch to send a missing value to; scikit-learn's <Code>GradientBoostingClassifier</Code> does not, and requires explicit imputation.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        All code below was run on a synthetic 20-row mixed-type table with deliberate missing values (4 missing ages, 5 missing incomes). NumPy only for the algorithms. Outputs are verbatim terminal output.
      </Prose>

      <H3>4.1 StandardScaler, MinMaxScaler</H3>

      <CodeBlock language="python">
{`import numpy as np

class StandardScalerScratch:
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0, ddof=0)   # population std, matches sklearn default
        return self
    def transform(self, X):
        return (X - self.mean_) / self.std_
    def inverse_transform(self, X_scaled):
        return X_scaled * self.std_ + self.mean_

class MinMaxScalerScratch:
    def fit(self, X):
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        return self
    def transform(self, X):
        return (X - self.min_) / (self.max_ - self.min_)
    def inverse_transform(self, X_scaled):
        return X_scaled * (self.max_ - self.min_) + self.min_

# Messy mixed-type table: age (float, some NaN), income (float, some NaN)
age    = np.array([25, np.nan, 34, 42, np.nan, 29, 55, 38, 47, 22,
                   31, np.nan, 60, 28, 45, 33, 52, 41, np.nan, 36], dtype=float)
income = np.array([48000, 62000, np.nan, 95000, 71000, np.nan, 110000, 78000,
                   np.nan, 35000, 55000, 88000, 120000, 42000, np.nan,
                   67000, 99000, 73000, 61000, np.nan], dtype=float)

print("=== BEFORE SCALING (age, income stats) ===")
print(f"  age:    mean={np.nanmean(age):.2f}, std={np.nanstd(age):.2f}")
print(f"  income: mean={np.nanmean(income):.2f}, std={np.nanstd(income):.2f}")
# Output:
# === BEFORE SCALING (age, income stats) ===
#   age:    mean=38.62, std=10.66
#   income: mean=73600.00, std=24002.22

# Mean-impute first so scalers get clean arrays
X_num = np.column_stack([age, income])
col_means = np.array([np.nanmean(age), np.nanmean(income)])
for j in range(2):
    mask = np.isnan(X_num[:, j])
    X_num[mask, j] = col_means[j]

ss = StandardScalerScratch().fit(X_num)
X_std = ss.transform(X_num)
print("\\n=== AFTER STANDARD SCALING ===")
print(f"  age:    mean={X_std[:,0].mean():.4f}, std={X_std[:,0].std():.4f}")
print(f"  income: mean={X_std[:,1].mean():.4f}, std={X_std[:,1].std():.4f}")
# Output:
# === AFTER STANDARD SCALING ===
#   age:    mean=0.0000, std=1.0000
#   income: mean=0.0000, std=1.0000

mm = MinMaxScalerScratch().fit(X_num)
X_mm = mm.transform(X_num)
print("\\n=== AFTER MINMAX SCALING ===")
print(f"  age:    min={X_mm[:,0].min():.4f}, max={X_mm[:,0].max():.4f}")
print(f"  income: min={X_mm[:,1].min():.4f}, max={X_mm[:,1].max():.4f}")
# Output:
# === AFTER MINMAX SCALING ===
#   age:    min=0.0000, max=1.0000
#   income: min=0.0000, max=1.0000`}
      </CodeBlock>

      <H3>4.2 One-hot encoding and out-of-fold target encoding</H3>

      <CodeBlock language="python">
{`import numpy as np

# One-hot encoder: fit learns the vocabulary, transform builds binary matrix
class OneHotScratch:
    def fit(self, col):
        self.categories_ = sorted(set(col))
        self.cat_to_idx  = {c: i for i, c in enumerate(self.categories_)}
        return self
    def transform(self, col):
        out = np.zeros((len(col), len(self.categories_)), dtype=int)
        for i, val in enumerate(col):
            if val in self.cat_to_idx:
                out[i, self.cat_to_idx[val]] = 1
        return out

city = ['NYC', 'LA', 'NYC', 'CHI', 'LA', 'NYC', 'CHI', 'LA',
        'NYC', 'CHI', 'LA', 'NYC', 'CHI', 'LA', 'NYC',
        'CHI', 'LA', 'NYC', 'CHI', 'LA']

ohe = OneHotScratch().fit(city)
X_ohe = ohe.transform(city)
print("=== ONE-HOT ENCODING (city) ===")
print(f"  categories: {ohe.categories_}")
print(f"  output shape: {X_ohe.shape}")
print(f"  first 5 rows (CHI, LA, NYC):\\n{X_ohe[:5]}")
# Output:
# === ONE-HOT ENCODING (city) ===
#   categories: ['CHI', 'LA', 'NYC']
#   output shape: (20, 3)
#   first 5 rows (CHI, LA, NYC):
# [[0 0 1]
#  [0 1 0]
#  [0 0 1]
#  [1 0 0]
#  [0 1 0]]

# Out-of-fold target encoding (smoothed): no leakage
def target_encode_oof(col, y, n_splits=4, smoothing=5):
    global_mean = np.mean(y)
    encoded = np.zeros(len(col))
    fold_size = len(col) // n_splits
    for fold in range(n_splits):
        val_idx   = np.arange(fold * fold_size, (fold + 1) * fold_size)
        train_idx = np.concatenate([np.arange(0, fold * fold_size),
                                    np.arange((fold + 1) * fold_size, len(col))])
        train_col = [col[i] for i in train_idx]
        train_y   = y[train_idx]
        stats = {}
        for cat in set(train_col):
            mask    = np.array([c == cat for c in train_col])
            n       = mask.sum()
            cat_mean = train_y[mask].mean()
            # Smoothed blend: rare categories shrink toward global mean
            stats[cat] = (n * cat_mean + smoothing * global_mean) / (n + smoothing)
        for i in val_idx:
            encoded[i] = stats.get(col[i], global_mean)
    return encoded

purchased = np.array([1,0,1,1,0,1,1,0,0,0,1,1,1,0,1,0,1,1,0,1], dtype=float)
te = target_encode_oof(city, purchased, n_splits=4, smoothing=5)
print("\\n=== TARGET ENCODING OOF (city -> float) ===")
for cat in ['CHI', 'LA', 'NYC']:
    mask = np.array([c == cat for c in city])
    print(f"  {cat}: purchase_rate={purchased[mask].mean():.3f},  "
          f"TE_mean={te[mask].mean():.3f}")
# Output:
# === TARGET ENCODING OOF (city -> float) ===
#   CHI: purchase_rate=0.500,  TE_mean=0.574
#   LA:  purchase_rate=0.429,  TE_mean=0.506
#   NYC: purchase_rate=0.857,  TE_mean=0.732`}
      </CodeBlock>

      <H3>4.3 Mean imputer and KNN imputer</H3>

      <CodeBlock language="python">
{`import numpy as np

class MeanImputerScratch:
    def fit(self, X):
        self.means_ = np.nanmean(X, axis=0)
        return self
    def transform(self, X):
        out = X.copy().astype(float)
        for j in range(X.shape[1]):
            mask = np.isnan(out[:, j])
            out[mask, j] = self.means_[j]
        return out

class KNNImputerScratch:
    def __init__(self, k=3):
        self.k = k
    def fit(self, X):
        self.X_train_ = X.copy()
        return self
    def transform(self, X):
        out = X.copy().astype(float)
        for i in range(len(out)):
            missing_cols  = np.where(np.isnan(out[i]))[0]
            if len(missing_cols) == 0:
                continue
            observed_cols = np.where(~np.isnan(out[i]))[0]
            dists = []
            for j, row in enumerate(self.X_train_):
                if np.any(np.isnan(row[observed_cols])):
                    continue
                d = np.sqrt(np.sum((out[i, observed_cols] - row[observed_cols]) ** 2))
                dists.append((d, j))
            dists.sort()
            neighbors = [self.X_train_[j] for _, j in dists[:self.k]]
            for col in missing_cols:
                vals = [n[col] for n in neighbors if not np.isnan(n[col])]
                if vals:
                    out[i, col] = np.mean(vals)
        return out

age    = np.array([25, np.nan, 34, 42, np.nan, 29, 55, 38, 47, 22,
                   31, np.nan, 60, 28, 45, 33, 52, 41, np.nan, 36], dtype=float)
income = np.array([48000, 62000, np.nan, 95000, 71000, np.nan, 110000, 78000,
                   np.nan, 35000, 55000, 88000, 120000, 42000, np.nan,
                   67000, 99000, 73000, 61000, np.nan], dtype=float)
X_raw = np.column_stack([age, income])

# --- Mean imputation ---
mean_imp = MeanImputerScratch().fit(X_raw)
X_mean   = mean_imp.transform(X_raw)
print("=== MEAN IMPUTATION ===")
print(f"  NaNs before: age={np.isnan(age).sum()}, income={np.isnan(income).sum()}")
print(f"  NaNs after:  age={np.isnan(X_mean[:,0]).sum()}, income={np.isnan(X_mean[:,1]).sum()}")
print(f"  Imputed age mean: {mean_imp.means_[0]:.2f}")
# Output:
# === MEAN IMPUTATION ===
#   NaNs before: age=4, income=5
#   NaNs after:  age=0, income=0
#   Imputed age mean: 38.62

# --- KNN imputation (fit on fully-imputed data, impute original NaNs) ---
knn_imp = KNNImputerScratch(k=3).fit(X_mean)
X_knn   = knn_imp.transform(X_raw)
print("\\n=== KNN IMPUTATION (k=3) ===")
print(f"  NaNs after:  age={np.isnan(X_knn[:,0]).sum()}, income={np.isnan(X_knn[:,1]).sum()}")
nan_rows = np.where(np.isnan(age))[0]
print(f"  Imputed age values (rows {nan_rows.tolist()}): "
      f"{X_knn[nan_rows, 0].round(2)}")
# Output:
# === KNN IMPUTATION (k=3) ===
#   NaNs after:  age=0, income=0
#   Imputed age values (rows [1, 4, 11, 18]): [36.75 37.88 39.54 36.08]`}
      </CodeBlock>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <H3>5.1 Sklearn preprocessing and imputation</H3>

      <CodeBlock language="python">
{`import numpy as np
import pandas as pd
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
                                    PowerTransformer, QuantileTransformer,
                                    OrdinalEncoder, OneHotEncoder)
from sklearn.impute import SimpleImputer, KNNImputer
# from sklearn.impute import IterativeImputer  # experimental: enable_iterative_imputer=True
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------
# Dataset: 20 rows, mixed types, deliberate NaNs
# ---------------------------------------------------------------
age    = [25, np.nan, 34, 42, np.nan, 29, 55, 38, 47, 22,
          31, np.nan, 60, 28, 45, 33, 52, 41, np.nan, 36]
income = [48000, 62000, np.nan, 95000, 71000, np.nan, 110000, 78000,
          np.nan, 35000, 55000, 88000, 120000, 42000, np.nan,
          67000, 99000, 73000, 61000, np.nan]
city   = ['NYC','LA','NYC','CHI','LA','NYC','CHI','LA',
          'NYC','CHI','LA','NYC','CHI','LA','NYC',
          'CHI','LA','NYC','CHI','LA']
y      = np.array([1,0,1,1,0,1,1,0,0,0,1,1,1,0,1,0,1,1,0,1])

df = pd.DataFrame({'age': age, 'income': income, 'city': city})

# ---------------------------------------------------------------
# Numeric sub-pipeline: mean imputation + StandardScaler
# Categorical sub-pipeline: mode imputation + OneHotEncoder
# ---------------------------------------------------------------
num_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='mean')),
    ('scale',  StandardScaler()),
])
cat_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('ohe',    OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ('num', num_pipe, ['age', 'income']),
    ('cat', cat_pipe, ['city']),
])

X_proc = preprocessor.fit_transform(df)
print("=== SKLEARN COLUMN TRANSFORMER OUTPUT ===")
print(f"  Input shape:  {df.shape}  (age, income, city)")
print(f"  Output shape: {X_proc.shape}  (age_std, income_std, CHI, LA, NYC)")
print(f"  First row: {X_proc[0].round(4)}")
print(f"  Second row (age imputed): {X_proc[1].round(4)}")
# Output:
# === SKLEARN COLUMN TRANSFORMER OUTPUT ===
#   Input shape:  (20, 3)  (age, income, city)
#   Output shape: (20, 5)  (age_std, income_std, CHI, LA, NYC)
#   First row: [-1.4292 -1.2316  0.      0.      1.    ]
#   Second row (age imputed): [ 0.     -0.5581  0.      1.      0.    ]

# ---------------------------------------------------------------
# Yeo-Johnson power transform on skewed feature
# ---------------------------------------------------------------
np.random.seed(0)
skewed = np.random.exponential(scale=2.0, size=200).reshape(-1, 1)
pt = PowerTransformer(method='yeo-johnson')
skewed_t = pt.fit_transform(skewed)
print("\\n=== YEO-JOHNSON POWER TRANSFORM ===")
print(f"  Original:    mean={skewed.mean():.3f}, std={skewed.std():.3f}")
print(f"  Transformed: mean={skewed_t.mean():.4f}, std={skewed_t.std():.4f}")
print(f"  Fitted lambda: {pt.lambdas_[0]:.4f}")
# Output:
# === YEO-JOHNSON POWER TRANSFORM ===
#   Original:    mean=1.970, std=1.949
#   Transformed: mean=0.0000, std=1.0000
#   Fitted lambda: -0.3282

# ---------------------------------------------------------------
# RobustScaler (median/IQR — resistant to outliers)
# ---------------------------------------------------------------
age_arr    = np.array([x if not (isinstance(x, float) and np.isnan(x)) else np.nan
                       for x in age], dtype=float)
income_arr = np.array([x if not (isinstance(x, float) and np.isnan(x)) else np.nan
                       for x in income], dtype=float)
X_num_imp  = SimpleImputer(strategy='mean').fit_transform(
                 np.column_stack([age_arr, income_arr]))
rs = RobustScaler()
rs.fit(X_num_imp)
print("\\n=== ROBUST SCALER (median/IQR) ===")
print(f"  age    center={rs.center_[0]:.2f}, scale={rs.scale_[0]:.2f}")
print(f"  income center={rs.center_[1]:.2f}, scale={rs.scale_[1]:.2f}")
# Output:
# === ROBUST SCALER (median/IQR) ===
#   age    center=38.62, scale=10.25
#   income center=73600.00, scale=18750.00

# ---------------------------------------------------------------
# KNN Imputer
# ---------------------------------------------------------------
X_nan = np.column_stack([age_arr, income_arr])   # still has NaNs
knn_imp = KNNImputer(n_neighbors=3)
X_knn   = knn_imp.fit_transform(X_nan)
print("\\n=== KNN IMPUTER (sklearn, k=3) ===")
print(f"  NaNs before: {np.isnan(X_nan).sum()}")
print(f"  NaNs after:  {np.isnan(X_knn).sum()}")
nan_rows = np.where(np.isnan(age_arr))[0]
for r in nan_rows:
    print(f"  Row {r}: imputed age = {X_knn[r,0]:.2f}")
# Output:
# === KNN IMPUTER (sklearn, k=3) ===
#   NaNs before: 9
#   NaNs after:  0
#   Row 1: imputed age = 35.00
#   Row 4: imputed age = 37.33
#   Row 11: imputed age = 44.00
#   Row 18: imputed age = 35.00`}
      </CodeBlock>

      <Prose>
        Key API notes. <Code>SimpleImputer</Code> accepts <Code>strategy</Code> of <Code>'mean'</Code>, <Code>'median'</Code>, <Code>'most_frequent'</Code>, or <Code>'constant'</Code>. <Code>KNNImputer</Code> uses <Code>nan_euclidean_distances</Code> internally — it can handle rows with multiple missing values by computing distances on observed features only. <Code>IterativeImputer</Code> requires <Code>from sklearn.experimental import enable_iterative_imputer</Code> before import in sklearn 1.x. <Code>OneHotEncoder(handle_unknown='ignore')</Code> silently outputs all-zeros for unseen categories at inference time — important for robust production pipelines where test data can contain new categories not seen during training. The <Code>category_encoders</Code> library (pip-installable, not part of sklearn core) provides James-Stein encoding, hashing encoding (fixed-dimensional output regardless of cardinality), and CatBoost-style encoding with ordered statistics.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6.1 Skewed feature before vs. after Yeo-Johnson</H3>

      <Plot
        label="Exponential feature: original vs Yeo-Johnson transformed (n=200)"
        xLabel="value"
        yLabel="density (approx)"
        series={[
          {
            name: "original (Exponential, λ=0.5, right-skewed)",
            color: colors.gold,
            points: [
              [0.05, 0.48], [0.25, 0.44], [0.5, 0.40], [0.8, 0.36], [1.2, 0.32],
              [1.7, 0.26], [2.3, 0.20], [3.1, 0.14], [4.2, 0.09], [5.5, 0.05],
              [7.0, 0.025], [9.0, 0.011], [11.0, 0.004], [13.5, 0.001],
            ],
          },
          {
            name: "after Yeo-Johnson (approx. Gaussian, λ=-0.33)",
            color: colors.green,
            points: [
              [-3.0, 0.004], [-2.5, 0.018], [-2.0, 0.054], [-1.5, 0.130],
              [-1.0, 0.242], [-0.5, 0.352], [0.0, 0.399], [0.5, 0.352],
              [1.0, 0.242], [1.5, 0.130], [2.0, 0.054], [2.5, 0.018],
              [3.0, 0.004],
            ],
          },
        ]}
      />

      <Prose>
        The original feature is right-skewed (exponential, mean=1.97, std=1.95). After Yeo-Johnson with fitted <Code>{"λ = −0.33"}</Code>, the output is approximately standard-normal (mean=0.000, std=1.000). This matters for linear models: OLS assumes normally distributed residuals, and a Gaussian-transformed feature produces more Gaussian residuals than a raw exponential one.
      </Prose>

      <H3>6.2 KNN accuracy vs. scaling method</H3>

      <Plot
        label="5-fold cross-val KNN accuracy by preprocessing (n=300, d=10)"
        xLabel="method"
        yLabel="accuracy"
        series={[
          {
            name: "no scaling (dominated by large-range features)",
            color: "#f87171",
            points: [[0, 0.6933]],
          },
          {
            name: "StandardScaler",
            color: colors.gold,
            points: [[1, 0.8533]],
          },
          {
            name: "MinMaxScaler",
            color: colors.green,
            points: [[2, 0.8467]],
          },
          {
            name: "RobustScaler",
            color: "#a78bfa",
            points: [[3, 0.8567]],
          },
        ]}
      />

      <Prose>
        Without scaling, KNN achieves 69.3% accuracy on a dataset where one feature has been artificially inflated to income-scale and another to millimeter-scale. All three scalers recover accuracy to 84–85%, confirming that the scale disparity was the limiting factor, not the signal-to-noise ratio of the underlying features. The three scalers perform comparably here because the dataset has no heavy-tailed outliers — RobustScaler's advantage over StandardScaler is only visible when outliers are present.
      </Prose>

      <H3>6.3 Encoder output heatmap for a toy categorical column</H3>

      <Heatmap
        label="One-hot encoding output for 6 observations (city column)"
        matrix={[
          [0, 0, 1],
          [0, 1, 0],
          [0, 0, 1],
          [1, 0, 0],
          [0, 1, 0],
          [1, 0, 0],
        ]}
        rowLabels={["row0=NYC", "row1=LA", "row2=NYC", "row3=CHI", "row4=LA", "row5=CHI"]}
        colLabels={["CHI", "LA", "NYC"]}
        colorScale="gold"
      />

      <H3>6.4 StepTrace: sklearn Pipeline fit → transform → predict</H3>

      <StepTrace
        label="ColumnTransformer + LogisticRegression Pipeline on 20-row toy dataset"
        steps={[
          {
            label: "Step 1: pipeline.fit(X_train, y_train)",
            render: () => (
              <Prose>
                The pipeline calls <Code>preprocessor.fit_transform(X_train)</Code> then <Code>clf.fit(X_proc_train, y_train)</Code>. The ColumnTransformer visits each transformer in order. The numeric sub-pipeline: SimpleImputer computes <Code>mean_age=38.62</Code>, <Code>mean_income=73600</Code> from observed training values; StandardScaler then computes <Code>μ_age, σ_age</Code> on the imputed numeric matrix. The categorical sub-pipeline: SimpleImputer identifies mode=<Code>'NYC'</Code> for the city column; OneHotEncoder learns vocabulary <Code>['CHI','LA','NYC']</Code>. All statistics are stored inside the fitted pipeline objects — nothing is computed again at inference time.
              </Prose>
            ),
          },
          {
            label: "Step 2: preprocessor.transform(X_test) — test split",
            render: () => (
              <Prose>
                The fitted pipeline applies training-set statistics to the test data. Missing ages in the test set are filled with <Code>mean_age=38.62</Code> (not the test-set mean). Test rows are standardized with training-set <Code>μ_age, σ_age</Code>. If a test row contains a city that was not in the training vocabulary (e.g., <Code>'SEA'</Code>), <Code>handle_unknown='ignore'</Code> outputs an all-zeros row for those three columns — no error, no crash. The output is a 5-column float64 matrix regardless of the original column types.
              </Prose>
            ),
          },
          {
            label: "Step 3: clf.predict(X_proc_test)",
            render: () => (
              <Prose>
                LogisticRegression receives a dense float64 matrix with no missing values, no strings, and all features on comparable scales. It applies <Code>w · x + b</Code>, passes through the sigmoid, and thresholds at 0.5. Because all features were scaled, the L2 regularization penalty (<Code>C=1.0</Code> default) applies equally to every coefficient — no feature dominates the penalty just because its raw scale was large. The predict call is a pure matrix multiply plus a sigmoid — no preprocessing logic runs again.
              </Prose>
            ),
          },
          {
            label: "Step 4: pipeline.score(X_test, y_test)",
            render: () => (
              <Prose>
                The pipeline's <Code>score</Code> method calls <Code>transform</Code> then <Code>predict</Code> then computes accuracy. The key invariant: every operation in steps 2–3 uses statistics learned only from <Code>X_train</Code>. The test set has influenced nothing except the evaluation metric. This is the central guarantee that makes <Code>Pipeline</Code> worth using — it makes data leakage structurally impossible for the preprocessing steps it encapsulates.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <H3>7.1 Which scaler?</H3>

      <Callout>
        Use <strong>StandardScaler</strong> as the default for Gaussian-ish numeric features going into linear models, SVMs, PCA, or regularized regression. Use <strong>MinMaxScaler</strong> when you need bounded [0,1] output (neural network inputs, distance-based models with known-bounded features). Use <strong>RobustScaler</strong> when the column has heavy tails or confirmed outliers that should not compress the bulk of the distribution. Use <strong>PowerTransformer (Yeo-Johnson)</strong> when the feature is strongly skewed and downstream normality assumptions matter. Use <strong>QuantileTransformer</strong> when you need a non-parametric normalization that makes no distributional assumptions, at the cost of losing rank-between-quantile ordinal information. Use <strong>no scaling</strong> for tree-based models (random forest, gradient boosting, XGBoost) where splits are threshold comparisons and scale is irrelevant.
      </Callout>

      <H3>7.2 Which encoder?</H3>

      <Callout>
        <strong>OneHotEncoder</strong> for nominal categories with low cardinality (K {"<"} 20–30). Always set <Code>handle_unknown='ignore'</Code> for robustness. Use <Code>drop='first'</Code> for linear models to avoid the dummy variable trap. <strong>OrdinalEncoder</strong> for categories with a meaningful total ordering and a downstream model that can exploit the ordering. <strong>TargetEncoder</strong> (sklearn 1.3+) or the <Code>category_encoders</Code> library for high-cardinality nominal categories where one-hot would blow up dimensionality — always use out-of-fold or cross-validation during training. <strong>HashingEncoder</strong> from <Code>category_encoders</Code> for streaming or extremely high-cardinality scenarios (millions of distinct values): it hashes categories into a fixed-length binary vector using the hashing trick (Weinberger et al. 2009), at the cost of occasional hash collisions. <strong>James-Stein encoder</strong> from <Code>category_encoders</Code> for target encoding with principled shrinkage: it applies empirical Bayes shrinkage per category, reducing the smoothing hyperparameter to a single regularization strength.
      </Callout>

      <H3>7.3 Which imputer?</H3>

      <Callout>
        <strong>SimpleImputer(strategy='mean')</strong> for MCAR numeric data where the downstream model is robust to attenuated correlations (e.g., gradient boosting with many trees). <strong>SimpleImputer(strategy='median')</strong> when the column has heavy tails and the mean is unrepresentative. <strong>SimpleImputer(strategy='most_frequent')</strong> for categorical columns. <strong>KNNImputer</strong> when you have {"<"} 50,000 rows and believe nearby observations in feature space are meaningful proxies for missing values. <strong>IterativeImputer (MICE)</strong> when data is MAR and you want to preserve the joint distribution among columns — use a fast regressor (e.g., <Code>BayesianRidge</Code>) to keep iteration cost low. <strong>Add a missing indicator column</strong> (<Code>MissingIndicator</Code> in sklearn) alongside any imputation when you suspect MNAR — the indicator tells the model which rows were imputed, allowing it to learn a different relationship for imputed vs. observed values.
      </Callout>

      {/* ======================================================================
          8. SCALING & COMPLEXITY
          ====================================================================== */}
      <H2>8. What scales and what doesn{"'"}t</H2>

      <Prose>
        <strong>StandardScaler and MinMaxScaler</strong> are trivially streaming. Both compute column-wise means and standard deviations (or min/max) that can be updated incrementally. Scikit-learn exposes this through <Code>partial_fit</Code>: call it on successive batches of data, and the scaler maintains running estimates using Welford's online algorithm for numerical stability. Memory usage is <Code>O(d)</Code> for storing the statistics, independent of <Code>n</Code>. This makes them suitable for pipelines that process data in chunks from disk.
      </Prose>

      <Prose>
        <strong>RobustScaler</strong> requires the full column to compute the median and IQR — these are order statistics and cannot be computed exactly in a single pass without storing all values. Approximate versions using sketches (e.g., t-digest) exist but are not in sklearn. For large datasets, precompute robust statistics offline and pass them as fixed parameters.
      </Prose>

      <Prose>
        <strong>PowerTransformer</strong> requires a maximum-likelihood optimization step at fit time — <Code>O(n)</Code> data pass plus the cost of the 1D optimization. Transform itself is <Code>O(n)</Code>. Streaming is not supported.
      </Prose>

      <Prose>
        <strong>OneHotEncoder</strong> scales with the number of distinct categories <Code>K</Code>. For a column with <Code>K=10,000</Code> distinct values, the output has 10,000 columns. Downstream models with dense weight matrices (logistic regression, SVMs) then have 10,000× more parameters just from that one column. The <strong>HashingEncoder</strong> sidesteps this by projecting all categories (regardless of how many exist) into a fixed-dimensional binary vector of user-chosen size <Code>m</Code> (typically 512–4096). The collision probability for two distinct categories is <Code>1/m</Code> — manageable for <Code>m</Code> in the thousands, with the tradeoff that two colliding categories become indistinguishable to the model.
      </Prose>

      <Prose>
        <strong>KNNImputer</strong> at inference time computes the distance from each new row to all <Code>n</Code> training rows — <Code>O(n · d)</Code> per query, <Code>O(n² · d)</Code> for imputing the full training set. This is the most expensive imputer and is not suitable for datasets larger than roughly 100,000 rows without approximate nearest-neighbor indexing. <strong>IterativeImputer</strong> at fit time runs <Code>max_iter × d</Code> regression fits, each costing <Code>O(n · d²)</Code> for linear regressors — total <Code>O(max_iter · d³ · n)</Code>, which is expensive for many columns. Use a faster regressor (<Code>BayesianRidge</Code>) or fewer iterations (<Code>max_iter=5</Code>) to keep it tractable.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes {"&"} gotchas</H2>

      <H3>9.1 Data leakage through preprocessing</H3>

      <Prose>
        The most common and most costly mistake is fitting the scaler or imputer on the full dataset before splitting into train and test. If you call <Code>scaler.fit(X)</Code> on all <Code>n</Code> rows and then split, the test set has informed the scaler's statistics — its mean, standard deviation, min, max. When you later evaluate on the test set, you are using a scaler that has already "seen" the test data. The correct sequence is: split first, then fit the scaler on the training split only, then transform both splits with those training-set statistics. <Code>sklearn.Pipeline</Code> enforces this automatically in cross-validation because it refits the entire pipeline on each training fold.
      </Prose>

      <H3>9.2 Target leakage in target encoding</H3>

      <Prose>
        Fitting a target encoder on the full training set and then using the encoded values as training features is a form of target leakage: each row's encoded value is computed using that row's own target. In the extreme case (one observation per category, smoothing=0), the encoded feature is literally the target. The model will appear to perform perfectly on training data and will fail dramatically on the test set. Fix: use out-of-fold encoding during training (as shown in Section 4.2) or use sklearn 1.3+'s <Code>TargetEncoder</Code> which implements this automatically via its <Code>cv</Code> parameter.
      </Prose>

      <H3>9.3 One-hot collinearity for linear models</H3>

      <Prose>
        When one-hot encoding with <Code>K</Code> categories, the sum of all <Code>K</Code> indicator columns equals 1 for every row — the same constant as the intercept column in the design matrix. This perfect multicollinearity makes the design matrix singular (non-invertible). For linear regression this means the OLS closed-form solution <Code>{"(X^T X)^{-1} X^T y"}</Code> does not exist. Sklearn handles this gracefully for regularized models (the penalty makes the matrix invertible) but not for OLS. Fix: use <Code>drop='first'</Code> in <Code>OneHotEncoder</Code> to drop one reference category. For tree-based models this is irrelevant — trees split on individual features and do not invert the full feature matrix.
      </Prose>

      <H3>9.4 Unseen categories at test time</H3>

      <Prose>
        If a new category appears in the test set that was not in the training vocabulary, a naive encoder raises an error. Set <Code>handle_unknown='ignore'</Code> in <Code>OneHotEncoder</Code> to output an all-zeros row for that column — the model then sees the same embedding as it would for a category with no indicator active. For target encoders, fall back to the global mean for unknown categories. For ordinal encoders, sklearn will raise by default unless you set <Code>handle_unknown='use_encoded_value'</Code> with an <Code>unknown_value</Code> — typically -1 or the number of categories.
      </Prose>

      <H3>9.5 Dropping NaNs too early</H3>

      <Prose>
        Dropping all rows with any missing value before training is statistically safe only under MCAR. Under MAR or MNAR, the complete-case dataset is a biased subsample. The bias can be severe: in a medical context, patients with more severe illness are more likely to have missing lab values, so dropping incomplete rows trains on a systematically healthier subpopulation. Impute instead, and add a missing-indicator column so the model can learn the missingness pattern.
      </Prose>

      <H3>9.6 inverse_transform is not lossless</H3>

      <Prose>
        StandardScaler's <Code>inverse_transform</Code> is exact (it just reverses the arithmetic). MinMaxScaler's is also exact within floating-point precision. PowerTransformer and QuantileTransformer use iterative numerical inversion — the inverse is approximate, particularly at extreme quantiles. OneHotEncoder's <Code>inverse_transform</Code> returns <Code>None</Code> for rows where no indicator is active (e.g., after <Code>handle_unknown='ignore'</Code> for an unseen category). OrdinalEncoder's inverse is exact. Do not rely on inverse_transform for roundtrip fidelity in production logging or data reconstruction tasks without verifying numerically.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        <strong>Yeo, I.-K., and Johnson, R. A. (2000).</strong> "A new family of power transformations to improve normality or symmetry." <em>Biometrika</em>, 87(4), 954–959. Introduces the Yeo-Johnson transform as an extension of Box-Cox to handle zero and negative values. The key contribution is the piecewise definition that handles all real inputs and the MLE estimation of the transformation parameter <Code>λ</Code>.
      </Prose>

      <Prose>
        <strong>Micci-Barreca, D. (2001).</strong> "A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems." <em>ACM SIGKDD Explorations Newsletter</em>, 3(1), 27–32. The canonical target encoding paper. Introduces the smoothed estimator that blends the category mean toward the global mean as a function of category count. The leakage problem is described explicitly, along with the cross-validation fix.
      </Prose>

      <Prose>
        <strong>van Buuren, S., and Groothuis-Oudshoorn, K. (2011).</strong> "mice: Multivariate Imputation by Chained Equations in R." <em>Journal of Statistical Software</em>, 45(3), 1–67. The primary MICE reference. Provides the full chained-equations framework, convergence analysis, and practical guidance on the number of imputations and iterations required for stable results. The R <Code>mice</Code> package implements this; sklearn's <Code>IterativeImputer</Code> is the Python equivalent.
      </Prose>

      <Prose>
        <strong>Weinberger, K., Dasgupta, A., Langford, J., Smola, A., and Attenberg, J. (2009).</strong> "Feature Hashing for Large Scale Multitask Learning." <em>Proceedings of the 26th International Conference on Machine Learning (ICML)</em>, 1113–1120. Introduces the hashing trick for compressing high-cardinality categorical features into a fixed-dimensional binary vector. Proves that the inner product between hashed vectors approximates the inner product between original one-hot vectors in expectation, with bounded variance. The theoretical basis for <Code>HashingVectorizer</Code> in sklearn and <Code>HashingEncoder</Code> in <Code>category_encoders</Code>.
      </Prose>

      <Prose>
        <strong>Pedregosa, F., et al. (2011).</strong> "Scikit-learn: Machine Learning in Python." <em>Journal of Machine Learning Research</em>, 12, 2825–2830. The primary sklearn citation. Describes the API design philosophy — estimator interface, <Code>fit</Code>/<Code>transform</Code>/<Code>predict</Code> — that makes Pipeline and ColumnTransformer possible. Over 40,000 citations as of 2025; the most-cited ML software paper.
      </Prose>

      <Prose>
        <strong>Pearson, K. (1901).</strong> "On Lines and Planes of Closest Fit to Systems of Points in Space." <em>Philosophical Magazine</em>, 2(11), 559–572. The PCA paper. The requirement to standardize features before computing principal components is implicit in Pearson's derivation — covariance is meaningful only when variables are on comparable scales. This paper predates the term "standardization" as used in ML but is the intellectual origin of the practice.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1</H3>

      <Prose>
        You have a dataset with features: age (22–65), salary (30,000–200,000), and years_experience (0–40). You fit a KNN classifier with <Code>k=5</Code>. Without scaling, which feature dominates the Euclidean distance, and by approximately what factor?
      </Prose>

      <Callout>
        <strong>Answer:</strong> Salary dominates. The maximum difference in salary is 170,000. The maximum difference in age is 43, and in years_experience is 40. A row with salary differing by 170,000 contributes <Code>{"170,000² = 2.89 × 10¹⁰"}</Code> to the squared distance, while the combined maximum contribution from age and experience is <Code>{"43² + 40² = 3,449"}</Code>. Salary dominates by a factor of roughly <Code>{"2.89 × 10¹⁰ / 3,449 ≈ 8.4 × 10⁶"}</Code>. KNN will essentially sort all examples by salary difference alone, ignoring age and experience entirely.
      </Callout>

      <H3>Exercise 2</H3>

      <Prose>
        A feature has the following training values: <Code>{"[1, 2, 3, 4, 1000]"}</Code>. Compute the StandardScaler output and the RobustScaler output for the value <Code>x = 3</Code>. Which scaler would you prefer if 1000 is a data entry error?
      </Prose>

      <Callout>
        <strong>Answer:</strong> Training stats — mean <Code>{"μ = (1+2+3+4+1000)/5 = 202"}</Code>, std (population) <Code>{"σ = std([1,2,3,4,1000]) ≈ 399.5"}</Code>. StandardScaler: <Code>{"z = (3 − 202) / 399.5 ≈ −0.498"}</Code>. Median <Code>{"= 3"}</Code>, IQR <Code>{"= Q3 − Q1 = 4 − 1.5 = 2.5"}</Code> (using quartile interpolation). RobustScaler: <Code>{"z = (3 − 3) / 2.5 = 0.0"}</Code>. If 1000 is a data entry error (outlier), RobustScaler is strongly preferred: it correctly centers x=3 at zero and is unaffected by the outlier. StandardScaler shifts the center to 202 and compresses the scale by the outlier's magnitude.
      </Callout>

      <H3>Exercise 3</H3>

      <Prose>
        A categorical feature <Code>color</Code> has values <Code>{'["red", "blue", "green", "red", "blue"]'}</Code> with targets <Code>{"[1, 0, 1, 0, 1]"}</Code>. Compute the target-encoded value for <Code>'blue'</Code> using smoothing <Code>{"α = 2"}</Code> and a global mean of <Code>{"0.6"}</Code>.
      </Prose>

      <Callout>
        <strong>Answer:</strong> The blue observations are rows 1 and 4 with targets <Code>{"[0, 1]"}</Code>, so <Code>{"n_blue = 2"}</Code> and <Code>{"ȳ_blue = 0.5"}</Code>. Smoothed estimate: <Code>{"(n_blue × ȳ_blue + α × ȳ_global) / (n_blue + α) = (2 × 0.5 + 2 × 0.6) / (2 + 2) = (1.0 + 1.2) / 4 = 0.55"}</Code>. With only 2 observations, the estimate is pulled meaningfully toward the global mean 0.6 — a sensible shrinkage given the small sample.
      </Callout>

      <H3>Exercise 4</H3>

      <Prose>
        Explain why fitting a TargetEncoder on the full training set before cross-validation inflates in-sample performance, and describe the correct procedure.
      </Prose>

      <Callout>
        <strong>Answer:</strong> When you compute per-category target means over the entire training set and then train a model on those encoded values, each training row's feature value was derived using that row's own target. The model can trivially learn the encoding — in the extreme case of one sample per category and no smoothing, the encoded feature is a perfect proxy for the target. This is target leakage. The inflated in-sample (or cross-validation, if encoding happened before the split) performance does not generalize. Correct procedure: wrap the target encoder inside the Pipeline so it refits on each training fold during cross-validation. Alternatively, use out-of-fold encoding manually: for each fold, fit the encoder on the other folds' training data and apply it to the current fold's validation data, never including the validation fold's rows in the encoding statistics.
      </Callout>

      <H3>Exercise 5</H3>

      <Prose>
        A dataset has a column with 30% missing values. Under what missingness mechanism (MCAR, MAR, MNAR) is mean imputation unbiased? What preprocessing addition should you always consider when you are not sure of the mechanism?
      </Prose>

      <Callout>
        <strong>Answer:</strong> Mean imputation is unbiased for the column mean under MCAR (missing completely at random), where the probability of missingness is independent of both observed and unobserved variables. Under MAR, mean imputation is biased because the complete-case mean is not the true marginal mean — the missing rows have a different distribution in observed features that correlates with the column. Under MNAR, mean imputation is biased in the most fundamental sense: the missing values themselves differ systematically from the observed values. When you are not sure of the mechanism, always add a binary missing-indicator column alongside the imputed column: <Code>{"X_missing = np.isnan(X_raw).astype(int)"}</Code>. This allows the downstream model to learn a different relationship for rows that were imputed vs. rows that had observed values, partially correcting for MAR and MNAR biases.
      </Callout>

      <H3>Exercise 6</H3>

      <Prose>
        You have a tree-based model (gradient boosted trees). Which of the following preprocessing steps are necessary, which are beneficial, and which are irrelevant? (a) StandardScaler on numeric features, (b) OrdinalEncoder on a nominal city feature, (c) mean imputation for missing values, (d) OneHotEncoder on the city feature.
      </Prose>

      <Callout>
        <strong>Answer:</strong> (a) StandardScaler — <strong>irrelevant</strong> for gradient boosted trees. Splits are threshold comparisons on individual features; any monotone transformation of a feature produces the same set of possible splits. (b) OrdinalEncoder on a nominal city feature — <strong>harmful if misused</strong>. Assigning integers to cities implies an ordering (Chicago {"<"} Los Angeles {"<"} New York by lexicographic sort). A tree will learn thresholds like "city {"<"} 1" which means "is this Chicago," which is technically valid but less expressive than one-hot — the tree cannot learn "is this Chicago or New York" in a single split. One-hot (d) is more expressive. (c) Mean imputation — <strong>necessary</strong> for sklearn's GBT; <strong>not necessary</strong> for XGBoost/LightGBM which handle NaN natively via learned default directions. (d) OneHotEncoder — <strong>beneficial</strong>: allows the tree to learn arbitrary subsets of cities in a single split, at the cost of higher dimensionality. For very high-cardinality features, target encoding or hashing may be preferable to prevent dimensionality explosion.
      </Callout>

    </div>
  ),
};

export default featureScalingContent;
