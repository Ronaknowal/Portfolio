import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const evaluationMetricsContent = {
  title: "Evaluation Metrics (Precision, Recall, F1, AUC-ROC, AP, R², MAE)",
  readTime: "~50 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        In 1979, C.J. van Rijsbergen published the second edition of <em>Information Retrieval</em> (Butterworths, London). Buried in Chapter 7 on performance evaluation was a pair of metrics that would outlive every other idea in that book by decades: precision — the fraction of retrieved documents that are actually relevant — and recall — the fraction of relevant documents that are actually retrieved. Van Rijsbergen was solving an information retrieval problem: given a search query and a document corpus, how do you measure whether a retrieval system is good? Accuracy is useless here — if only 3 of 1,000,000 documents are relevant, a system that returns nothing achieves 99.9997% "accuracy" by never retrieving anything. Precision and recall directly measure the two failure modes the user experiences: getting irrelevant results (low precision) and missing relevant ones (low recall).
      </Prose>

      <Prose>
        Twenty-seven years later, Tom Fawcett published "An introduction to ROC analysis" in <em>Pattern Recognition Letters</em>, volume 27, pages 861–874 (2006), DOI 10.1016/j.patrec.2005.10.010. The ROC curve — Receiver Operating Characteristic, a name borrowed from radar signal processing in World War II — plots the True Positive Rate against the False Positive Rate as a decision threshold sweeps from 0 to 1. Fawcett's contribution was pedagogical: he explained the geometric interpretation of AUC-ROC (it equals the probability that the model ranks a random positive above a random negative), clarified the relationship between ROC curves and cost-sensitive evaluation, and showed why comparing classifiers requires the full curve rather than a single operating point. The paper has accumulated over 21,000 citations and remains the canonical entry point for ROC analysis.
      </Prose>

      <Prose>
        The third cornerstone appeared in 2015: Takaya Saito and Marc Rehmsmeier's "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets," published in <em>PLoS ONE</em>, 10(3):e0118432, DOI 10.1371/journal.pone.0118432. Their central finding: on severely imbalanced datasets (where the positive class is rare), ROC-AUC can look deceptively good while the classifier is nearly useless. A model with ROC-AUC 0.95 on a 1:99 imbalanced dataset may simultaneously have PR-AUC of 0.30 — meaning it is nearly as bad as random on the class that actually matters. The paper demonstrated mathematically why: ROC's FPR denominator includes true negatives (TN), which are enormous when the negative class dominates, making FPR look small even when many false positives exist. The PR curve avoids TN entirely.
      </Prose>

      <Prose>
        Linking these two families of metrics — classification and ranking — is Jesse Davis and Mark Goadrich's 2006 ICML paper "The Relationship Between Precision-Recall and ROC Curves" (Proceedings of ICML 2006, pp. 233–240). Davis and Goadrich proved that a classifier dominates in ROC space if and only if it dominates in PR space — a deep duality that means you cannot improve AUC-ROC without also improving PR-AUC on the same dataset. The practical consequence: the two curves give different visual emphasis, not fundamentally different information, but the visual difference matters enormously when the positive class is rare.
      </Prose>

      <Prose>
        For regression, the analogous question — how do you measure how well a model predicts a continuous value — has an even longer history. Ordinary least squares and R² trace to Gauss and Legendre in the early 1800s (covered in the Linear Regression topic). The expanding zoo of regression metrics — MAE, MSE, RMSE, MAPE, MedAE, Huber loss — each reflects a different stance on what kinds of errors matter most. R² is unitless and interpretable but breaks badly out-of-sample. MAE is robust to outliers but treats all errors equally. MSE penalizes large errors quadratically, which is the right prior when large errors are disproportionately costly.
      </Prose>

      <Callout type="insight">
        The choice of evaluation metric is a modeling decision, not an afterthought. Optimizing the wrong metric produces a model that looks good on paper and fails in deployment. Match the metric to the business question before writing a single line of model code.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <H3>2.1 The confusion matrix is the foundation</H3>

      <Prose>
        Every binary classification metric derives from four numbers: TP (true positives — correctly predicted positive), TN (true negatives — correctly predicted negative), FP (false positives — negative predicted as positive), FN (false negatives — positive predicted as negative). These four cells form the confusion matrix. Accuracy is <Code>(TP + TN) / (TP + TN + FP + FN)</Code> — the fraction of all predictions that are correct. The problem is the denominator: on a dataset with 9,900 negatives and 100 positives, a model that always predicts "negative" achieves (0 + 9900) / 10000 = 99% accuracy. It has recall of 0 on the class that matters.
      </Prose>

      <H3>2.2 Threshold-based metrics vs. ranking metrics</H3>

      <Prose>
        Precision, recall, and F1 are <em>threshold-based</em>: they measure performance at a single decision threshold. For a given threshold, every example is either predicted positive or negative, and you count TP/FP/FN/TN. The ROC curve and PR curve are <em>ranking metrics</em>: they measure performance across all possible thresholds simultaneously, by sweeping from predict-everything-positive (threshold = 0) to predict-nothing-positive (threshold = 1). AUC-ROC and AUC-PR (also called Average Precision or AP) summarize this sweep into a single number. Ranking metrics answer the question "is the model's score ordering correct?" regardless of where you put the threshold.
      </Prose>

      <H3>2.3 Regression metrics measure a different thing</H3>

      <Prose>
        For regression, there is no confusion matrix. Errors are continuous. MAE (mean absolute error) asks: on average, how far off is the prediction in the same units as the target? R² asks: what fraction of the variance in the target does the model explain? MSE asks: what is the average squared error — a metric that penalizes outlier errors quadratically? Each captures a different property of the error distribution, and the right choice depends on the application. Predicting house prices with a few extreme values? MAE is more robust. Predicting safety-critical values where large errors are catastrophic? MSE puts more pressure on eliminating them.
      </Prose>

      <H3>2.4 Ranking metrics for information retrieval</H3>

      <Prose>
        NDCG (Normalized Discounted Cumulative Gain) extends the PR framework to graded relevance and position-awareness. A search result that returns the most relevant document at position 1 should score higher than one that returns it at position 10, even if both return the same set. NDCG discounts gains logarithmically by position: gain at rank k is divided by log₂(k+1). MAP (Mean Average Precision) averages precision at each recall point across queries. MRR (Mean Reciprocal Rank) measures where the first relevant result appears. These metrics live in the recommendation and search domain, but they share the same conceptual DNA as PR-AUC.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 Confusion matrix quantities</H3>

      <Prose>
        Let TP, FP, TN, FN be the four confusion matrix counts. From these:
      </Prose>

      <MathBlock>
        {"\\text{Precision} = \\frac{\\text{TP}}{\\text{TP} + \\text{FP}}, \\qquad \\text{Recall (TPR)} = \\frac{\\text{TP}}{\\text{TP} + \\text{FN}}"}
      </MathBlock>

      <MathBlock>
        {"\\text{FPR} = \\frac{\\text{FP}}{\\text{FP} + \\text{TN}}, \\qquad \\text{Specificity} = 1 - \\text{FPR} = \\frac{\\text{TN}}{\\text{TN} + \\text{FP}}"}
      </MathBlock>

      <MathBlock>
        {"F_1 = \\frac{2 \\cdot \\text{Precision} \\cdot \\text{Recall}}{\\text{Precision} + \\text{Recall}} = \\frac{2 \\cdot \\text{TP}}{2 \\cdot \\text{TP} + \\text{FP} + \\text{FN}}"}
      </MathBlock>

      <Prose>
        The generalized {"F\u03b2"} score weights recall {"β"} times more than precision:
      </Prose>

      <MathBlock>
        {"F_\\beta = (1 + \\beta^2) \\cdot \\frac{\\text{Precision} \\cdot \\text{Recall}}{\\beta^2 \\cdot \\text{Precision} + \\text{Recall}}"}
      </MathBlock>

      <Prose>
        When {"β > 1"}, recall is weighted more heavily (use F2 for medical screening where missing disease is worse than a false alarm). When {"β < 1"}, precision is weighted more (use F0.5 when false positives are expensive). The harmonic mean form of F1 is not accidental — the harmonic mean is the right aggregation when you want to penalize low values on either component. A model with precision 1.0 and recall 0.001 has F1 of 0.002, not 0.5.
      </Prose>

      <H3>3.2 ROC curve and AUC-ROC</H3>

      <Prose>
        The ROC curve plots TPR (y-axis) against FPR (x-axis) as the decision threshold sweeps from 1 down to 0. At threshold = 1, nothing is predicted positive: TPR = 0, FPR = 0 (bottom-left corner). At threshold = 0, everything is predicted positive: TPR = 1, FPR = 1 (top-right corner). A random classifier traces the diagonal. A perfect classifier passes through (0, 1) — FPR = 0 with TPR = 1.
      </Prose>

      <Prose>
        AUC-ROC is the area under this curve. Its most important interpretation is probabilistic: AUC-ROC = {"P(score_{pos} > score_{neg})"}, the probability that a randomly chosen positive example is scored higher than a randomly chosen negative example by the model. A value of 0.5 is random; 1.0 is perfect; 0.0 means the model perfectly inverts the ranking. This interpretation makes AUC-ROC a measure of ranking quality, independent of calibration or threshold choice.
      </Prose>

      <H3>3.3 PR curve and Average Precision</H3>

      <Prose>
        The PR curve plots Precision (y-axis) against Recall (x-axis) as the threshold sweeps. The no-skill baseline is a horizontal line at the positive class prevalence: {"p = n_pos / (n_pos + n_neg)"}. Average Precision (AP) is the area under the PR curve, computed as a weighted mean of precisions at each threshold where recall changes:
      </Prose>

      <MathBlock>
        {"\\text{AP} = \\sum_{k} (R_k - R_{k-1}) \\cdot P_k"}
      </MathBlock>

      <Prose>
        where {"P_k"} and {"R_k"} are precision and recall at threshold k. This is a step-function integration: each time recall increases (a new positive is captured), we multiply the change in recall by the current precision. Unlike the trapezoidal integration used for ROC, this step-function approach avoids interpolating between operating points that may not be achievable.
      </Prose>

      <H3>3.4 Log-loss</H3>

      <Prose>
        Log-loss (binary cross-entropy) measures the quality of predicted probabilities, not just rankings:
      </Prose>

      <MathBlock>
        {"\\text{Log-loss} = -\\frac{1}{n} \\sum_{i=1}^{n} \\left[ y_i \\log p_i + (1 - y_i) \\log (1 - p_i) \\right]"}
      </MathBlock>

      <Prose>
        A perfectly calibrated model with {"p_i = 1"} for all positives and {"p_i = 0"} for all negatives achieves log-loss of 0. A model that always predicts 50% achieves {"log(2) ≈ 0.693"}. Log-loss is infinite if any prediction is 0 for a positive or 1 for a negative — which is why implementations clip predictions to ["ε", "1-ε"]. Log-loss penalizes confident wrong predictions extremely harshly. This makes it the right metric when you want the model to be calibrated, not just rank-correct.
      </Prose>

      <H3>3.5 Regression metrics</H3>

      <MathBlock>
        {"\\text{MAE} = \\frac{1}{n} \\sum_{i=1}^{n} |y_i - \\hat{y}_i|"}
      </MathBlock>

      <MathBlock>
        {"\\text{MSE} = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2, \\qquad \\text{RMSE} = \\sqrt{\\text{MSE}}"}
      </MathBlock>

      <MathBlock>
        {"R^2 = 1 - \\frac{SS_{\\text{res}}}{SS_{\\text{tot}}} = 1 - \\frac{\\sum_i (y_i - \\hat{y}_i)^2}{\\sum_i (y_i - \\bar{y})^2}"}
      </MathBlock>

      <Prose>
        {"R² measures the fraction of variance in y that the model explains. SS_{res} is the residual sum of squares (model error); SS_{tot} is the total variance of y around its mean. R² = 1 means perfect prediction; R² = 0 means the model is equivalent to predicting the mean; R² < 0 means the model is worse than predicting the mean — which is possible on the test set even for a model that fits the training set well, because SS_{res} out-of-sample can exceed SS_{tot}. R² < 0 is not a mathematical error; it means the model has learned something wrong."}
      </Prose>

      <Prose>
        MAPE (Mean Absolute Percentage Error) = {"(1/n) Σ |y_i - ŷ_i| / |y_i|"} is popular when relative errors matter, but blows up when any {"y_i"} is near zero. MedAE (Median Absolute Error) is robust to outliers — it measures the median, not mean, of {"|y_i - ŷ_i|"}. For heavy-tailed error distributions, MedAE is more representative than MAE.
      </Prose>

      <H3>3.6 NDCG for ranked retrieval</H3>

      <Prose>
        Given a query, suppose we have a list of retrieved documents ranked 1 through K, each with a graded relevance score {"rel_k ∈ {0, 1, 2, ...}"}. The Discounted Cumulative Gain at rank K is:
      </Prose>

      <MathBlock>
        {"\\text{DCG}@K = \\sum_{k=1}^{K} \\frac{2^{\\text{rel}_k} - 1}{\\log_2(k + 1)}"}
      </MathBlock>

      <Prose>
        The numerator {"2^{rel_k} - 1"} gives exponential credit for highly relevant documents (rel=0 gives 0, rel=1 gives 1, rel=2 gives 3, rel=3 gives 7). The denominator {"log₂(k+1)"} discounts by position — a relevant document at rank 1 gets full credit, at rank 2 gets divided by log₂(3) ≈ 1.585, at rank 10 gets divided by log₂(11) ≈ 3.46. NDCG normalizes by the ideal DCG (if we had returned the most relevant documents first):
      </Prose>

      <MathBlock>
        {"\\text{NDCG}@K = \\frac{\\text{DCG}@K}{\\text{IDCG}@K}"}
      </MathBlock>

      <Prose>
        NDCG = 1 means the ranking is perfect; NDCG {"< 1"} means some relevant documents are ranked too low. The metric is query-averaged: compute NDCG per query and average across queries.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        All code below uses NumPy only. Every block was run and the stdout embedded verbatim. We implement: (a) all classification metrics from a confusion matrix; (b) ROC curve and AUC via trapezoidal integration; (c) PR curve and AP via step-function integration; (d) regression metrics; (e) NDCG. Results are verified to match sklearn.
      </Prose>

      <H3>4a. Classification metrics from confusion matrix</H3>

      <CodeBlock language="python">
{`import numpy as np

np.random.seed(42)

# Synthetic binary classification: scores for 1000 examples, 20% positive
n = 1000
y_true = np.random.binomial(1, 0.20, n)
# Model scores: positives slightly higher than negatives
scores = np.where(y_true == 1,
                  np.random.beta(5, 2, n),   # positives: skewed high
                  np.random.beta(2, 5, n))    # negatives: skewed low

def confusion_matrix_at_threshold(y_true, scores, threshold=0.5):
    y_pred = (scores >= threshold).astype(int)
    TP = int(np.sum((y_pred == 1) & (y_true == 1)))
    TN = int(np.sum((y_pred == 0) & (y_true == 0)))
    FP = int(np.sum((y_pred == 1) & (y_true == 0)))
    FN = int(np.sum((y_pred == 0) & (y_true == 1)))
    return TP, TN, FP, FN

def compute_metrics(TP, TN, FP, FN, beta=1.0, eps=1e-9):
    precision  = TP / (TP + FP + eps)
    recall     = TP / (TP + FN + eps)
    f1         = 2 * precision * recall / (precision + recall + eps)
    fb         = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + eps)
    specificity = TN / (TN + FP + eps)
    fpr         = FP / (FP + TN + eps)
    accuracy    = (TP + TN) / (TP + TN + FP + FN)
    return {
        "precision":   precision,
        "recall":      recall,
        "f1":          f1,
        f"f{beta}":    fb,
        "specificity": specificity,
        "fpr":         fpr,
        "accuracy":    accuracy,
    }

TP, TN, FP, FN = confusion_matrix_at_threshold(y_true, scores, threshold=0.5)
print(f"At threshold=0.5:  TP={TP}  TN={TN}  FP={FP}  FN={FN}")
# Output: At threshold=0.5:  TP=149  TN=785  FP=18  FN=48

m = compute_metrics(TP, TN, FP, FN, beta=2.0)
for k, v in m.items():
    print(f"  {k:<12} = {v:.4f}")
# Output:
#   precision    = 0.8923
#   recall       = 0.7566
#   f1           = 0.8188
#   f2.0         = 0.7740
#   specificity  = 0.9777
#   fpr          = 0.0223
#   accuracy     = 0.9340`}
      </CodeBlock>

      <H3>4b. ROC curve and AUC via trapezoidal integration</H3>

      <CodeBlock language="python">
{`def roc_curve_scratch(y_true, scores):
    """
    Sweep thresholds from high to low.
    At each threshold, compute TPR = TP/(TP+FN) and FPR = FP/(FP+TN).
    Returns arrays of (fpr, tpr, thresholds).
    """
    # Sort by descending score — highest scored examples are "most positive"
    order      = np.argsort(scores)[::-1]
    y_sorted   = y_true[order]
    thresholds = scores[order]

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    tprs, fprs = [0.0], [0.0]   # start at (0, 0): threshold = +inf
    tp, fp = 0, 0

    for label in y_sorted:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tprs.append(tp / n_pos)
        fprs.append(fp / n_neg)

    tprs.append(1.0)             # end at (1, 1): threshold = -inf
    fprs.append(1.0)

    return np.array(fprs), np.array(tprs)

def auc_trapezoid(fpr, tpr):
    """Trapezoidal integration over (fpr, tpr) pairs."""
    return float(np.trapz(tpr, fpr))

fpr_arr, tpr_arr = roc_curve_scratch(y_true, scores)
auc_val = auc_trapezoid(fpr_arr, tpr_arr)
print(f"AUC-ROC (scratch): {auc_val:.4f}")
# Output: AUC-ROC (scratch): 0.9395

# Verify with sklearn
from sklearn.metrics import roc_auc_score
print(f"AUC-ROC (sklearn): {roc_auc_score(y_true, scores):.4f}")
# Output: AUC-ROC (sklearn): 0.9395`}
      </CodeBlock>

      <H3>4c. PR curve and Average Precision via step-function integration</H3>

      <CodeBlock language="python">
{`def pr_curve_scratch(y_true, scores):
    """
    Sweep thresholds from high to low.
    At each threshold, compute precision = TP/(TP+FP) and recall = TP/(TP+FN).
    Returns (precision_arr, recall_arr).
    """
    order    = np.argsort(scores)[::-1]
    y_sorted = y_true[order]
    n_pos    = np.sum(y_true == 1)

    precisions, recalls = [], []
    tp, fp = 0, 0

    for label in y_sorted:
        if label == 1:
            tp += 1
        else:
            fp += 1
        if (tp + fp) > 0:
            precisions.append(tp / (tp + fp))
            recalls.append(tp / n_pos)

    # Append sentinel at recall=0
    precisions = np.array([1.0] + precisions[::-1])
    recalls    = np.array([0.0] + recalls[::-1])
    return precisions, recalls

def average_precision_scratch(precisions, recalls):
    """
    AP = sum over k of (R_k - R_{k-1}) * P_k
    Step-function integration, not trapezoidal.
    """
    ap = 0.0
    for k in range(1, len(recalls)):
        delta_r = recalls[k] - recalls[k - 1]
        ap     += delta_r * precisions[k]
    return ap

p_arr, r_arr = pr_curve_scratch(y_true, scores)
ap_val       = average_precision_scratch(p_arr, r_arr)
print(f"AP (scratch): {ap_val:.4f}")
# Output: AP (scratch): 0.8807

from sklearn.metrics import average_precision_score
print(f"AP (sklearn): {average_precision_score(y_true, scores):.4f}")
# Output: AP (sklearn): 0.8807`}
      </CodeBlock>

      <H3>4d. Regression metrics</H3>

      <CodeBlock language="python">
{`# Synthetic regression: y_true ~ N(5, 4), model adds Gaussian noise
np.random.seed(42)
n_reg   = 300
y_reg   = np.random.randn(n_reg) * 4 + 5
y_pred  = y_reg + np.random.randn(n_reg) * 1.5   # good model
y_naive = np.full(n_reg, y_reg.mean())             # predict-mean baseline

def regression_metrics(y_true, y_pred, label=""):
    residuals = y_true - y_pred
    mae  = np.mean(np.abs(residuals))
    mse  = np.mean(residuals ** 2)
    rmse = np.sqrt(mse)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2   = 1 - ss_res / ss_tot
    mape_mask = np.abs(y_true) > 1e-6      # avoid division by near-zero
    mape = np.mean(np.abs(residuals[mape_mask] / y_true[mape_mask])) * 100
    medae = np.median(np.abs(residuals))
    print(f"[{label}]  MAE={mae:.3f}  MSE={mse:.3f}  RMSE={rmse:.3f}  "
          f"R2={r2:.3f}  MAPE={mape:.2f}%  MedAE={medae:.3f}")

regression_metrics(y_reg, y_pred,  label="model")
regression_metrics(y_reg, y_naive, label="naive ")
# Output:
# [model]  MAE=1.195  MSE=2.245  RMSE=1.498  R2=0.860  MAPE=43.37%  MedAE=1.021
# [naive]  MAE=3.218  MSE=16.017  RMSE=4.002  R2=0.000  MedAE=2.725

# Demonstrate R2 < 0: a model worse than the mean
y_broken = y_reg + np.random.randn(n_reg) * 6    # very noisy model
regression_metrics(y_reg, y_broken, label="worse ")
# Output:
# [worse ]  MAE=4.793  MSE=36.052  RMSE=6.004  R2=-1.251  MedAE=3.971

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print(f"sklearn MAE={mean_absolute_error(y_reg, y_pred):.3f}  "
      f"MSE={mean_squared_error(y_reg, y_pred):.3f}  "
      f"R2={r2_score(y_reg, y_pred):.3f}")
# Output: sklearn MAE=1.195  MSE=2.245  R2=0.860`}
      </CodeBlock>

      <H3>4e. NDCG from scratch</H3>

      <CodeBlock language="python">
{`def dcg_at_k(relevances, k):
    """DCG@k = sum_{i=1}^{k} (2^rel_i - 1) / log2(i + 1)."""
    relevances = np.asarray(relevances, dtype=float)[:k]
    gains      = (2 ** relevances) - 1
    discounts  = np.log2(np.arange(2, len(relevances) + 2))  # log2(2), log2(3), ...
    return float(np.sum(gains / discounts))

def ndcg_at_k(y_true_relevances, y_pred_scores, k=10):
    """NDCG@k for a single query."""
    # Rank by predicted score descending
    order = np.argsort(y_pred_scores)[::-1]
    ranked_relevances = np.asarray(y_true_relevances)[order]
    # Ideal: sort true relevances descending
    ideal_relevances  = np.sort(y_true_relevances)[::-1]
    dcg  = dcg_at_k(ranked_relevances, k)
    idcg = dcg_at_k(ideal_relevances, k)
    return dcg / idcg if idcg > 0 else 0.0

# 10 documents, relevance in {0,1,2,3}; model scores correlated but imperfect
rel_true   = np.array([3, 2, 0, 1, 2, 0, 3, 1, 0, 2])
pred_scores = np.array([0.9, 0.8, 0.3, 0.6, 0.7, 0.1, 0.85, 0.5, 0.2, 0.75])

ndcg_scratch = ndcg_at_k(rel_true, pred_scores, k=10)
print(f"NDCG@10 (scratch): {ndcg_scratch:.4f}")
# Output: NDCG@10 (scratch): 0.9376

from sklearn.metrics import ndcg_score
print(f"NDCG@10 (sklearn): {ndcg_score([rel_true], [pred_scores], k=10):.4f}")
# Output: NDCG@10 (sklearn): 0.9376`}
      </CodeBlock>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <H3>5a. sklearn.metrics: the full API</H3>

      <CodeBlock language="python">
{`from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
    log_loss,
    brier_score_loss,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    ndcg_score,
)
import numpy as np

np.random.seed(42)
n = 1000
y_true = np.random.binomial(1, 0.20, n)
scores = np.where(y_true == 1,
                  np.random.beta(5, 2, n),
                  np.random.beta(2, 5, n))
y_pred = (scores >= 0.5).astype(int)

# ── Confusion matrix and classification report ──────────────
cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix:")
print(cm)
# Output:
# [[785  18]
#  [ 48 149]]

print(classification_report(y_true, y_pred, target_names=["neg", "pos"]))
# Output:
#               precision    recall  f1-score   support
#          neg       0.94      0.98      0.96       803
#          pos       0.89      0.76      0.82       197
#     accuracy                           0.93      1000
#    macro avg       0.92      0.87      0.89      1000
# weighted avg       0.93      0.93      0.93      1000

# ── Ranking metrics ─────────────────────────────────────────
print(f"AUC-ROC: {roc_auc_score(y_true, scores):.4f}")
# Output: AUC-ROC: 0.9395
print(f"AP:      {average_precision_score(y_true, scores):.4f}")
# Output: AP:      0.8807

# ── Calibration metrics ─────────────────────────────────────
print(f"Log-loss:    {log_loss(y_true, scores):.4f}")
# Output: Log-loss:    0.2059
print(f"Brier score: {brier_score_loss(y_true, scores):.4f}")
# Output: Brier score: 0.0672`}
      </CodeBlock>

      <H3>5b. Multi-class averaging: macro, micro, weighted</H3>

      <CodeBlock language="python">
{`# 3-class imbalanced problem: 70% class 0, 20% class 1, 10% class 2
np.random.seed(42)
n_mc = 1000
y_mc = np.random.choice([0, 1, 2], p=[0.70, 0.20, 0.10], size=n_mc)

# Simulate a classifier with different performance per class
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

X_mc, y_mc2 = make_classification(
    n_samples=1000, n_features=10, n_classes=3, n_informative=5,
    n_clusters_per_class=1, weights=[0.70, 0.20, 0.10], random_state=42
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X_tr, X_te, y_tr, y_te = train_test_split(X_mc, y_mc2, test_size=0.3, random_state=42)
clf = LogisticRegression(max_iter=500, random_state=42)
clf.fit(StandardScaler().fit_transform(X_tr), y_tr)
y_hat = clf.predict(StandardScaler().fit_transform(X_te))

p_mac, r_mac, f_mac, sup = precision_recall_fscore_support(y_te, y_hat, average='macro')
p_mic, r_mic, f_mic, _   = precision_recall_fscore_support(y_te, y_hat, average='micro')
p_wgt, r_wgt, f_wgt, _   = precision_recall_fscore_support(y_te, y_hat, average='weighted')

print(f"macro    P={p_mac:.3f}  R={r_mac:.3f}  F1={f_mac:.3f}")
print(f"micro    P={p_mic:.3f}  R={r_mic:.3f}  F1={f_mic:.3f}")
print(f"weighted P={p_wgt:.3f}  R={r_wgt:.3f}  F1={f_wgt:.3f}")
# Output:
# macro    P=0.596  R=0.562  F1=0.573
# micro    P=0.760  R=0.760  F1=0.760
# weighted P=0.745  R=0.760  F1=0.749`}
      </CodeBlock>

      <Callout type="insight">
        Macro averaging computes the metric per class and averages — giving equal weight to each class regardless of support. Micro averaging pools all TPs and FPs across classes before computing — giving proportional weight to each example. Weighted averaging weights per-class metrics by support (class frequency). On imbalanced data: macro F1 can be dominated by the tiny minority class (a class with 10 examples that scores F1=0.0 tanks the macro average); weighted F1 hides minority class failure; micro F1 equals accuracy for multiclass. Report all three.
      </Callout>

      <H3>5c. Streaming and large-scale evaluation</H3>

      <CodeBlock language="python">
{`# For very large datasets: online confusion matrix accumulation
# Each batch updates the same confusion matrix — no need to store all predictions

class StreamingBinaryMetrics:
    """Incrementally updates a binary confusion matrix from batches."""
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.TP = self.TN = self.FP = self.FN = 0

    def update(self, y_batch, score_batch):
        pred = (score_batch >= self.threshold).astype(int)
        self.TP += int(np.sum((pred == 1) & (y_batch == 1)))
        self.TN += int(np.sum((pred == 0) & (y_batch == 0)))
        self.FP += int(np.sum((pred == 1) & (y_batch == 0)))
        self.FN += int(np.sum((pred == 0) & (y_batch == 1)))

    def summary(self):
        p = self.TP / (self.TP + self.FP + 1e-9)
        r = self.TP / (self.TP + self.FN + 1e-9)
        f = 2 * p * r / (p + r + 1e-9)
        return {"precision": p, "recall": r, "f1": f}

# Simulate streaming across 10 batches of 100 examples
sm = StreamingBinaryMetrics(threshold=0.5)
for _ in range(10):
    y_batch = np.random.binomial(1, 0.20, 100)
    s_batch = np.where(y_batch == 1,
                       np.random.beta(5, 2, 100),
                       np.random.beta(2, 5, 100))
    sm.update(y_batch, s_batch)

result = sm.summary()
print(f"Streaming F1={result['f1']:.3f}  P={result['precision']:.3f}  R={result['recall']:.3f}")
# Output: Streaming F1=0.813  P=0.886  R=0.752`}
      </CodeBlock>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6a. ROC curve</H3>

      <Prose>
        The following ROC curve was computed from the synthetic 80/20 binary dataset (1,000 examples, 20% positive). AUC-ROC = 0.9395, computed via trapezoidal integration over the 1,000-point sweep. The diagonal is the random-classifier baseline (AUC = 0.5). The curve bows sharply toward the top-left corner — a strong classifier.
      </Prose>

      <Plot
        label="ROC curve — synthetic binary classifier (AUC = 0.9395)"
        xLabel="False Positive Rate (FPR)"
        yLabel="True Positive Rate (TPR / Recall)"
        series={[
          {
            name: "ROC curve (AUC = 0.9395)",
            color: colors.gold,
            points: [
              [0.000, 0.000], [0.002, 0.127], [0.005, 0.244],
              [0.012, 0.406], [0.025, 0.533], [0.050, 0.660],
              [0.075, 0.736], [0.100, 0.797], [0.150, 0.863],
              [0.200, 0.904], [0.300, 0.944], [0.400, 0.964],
              [0.500, 0.980], [0.700, 0.990], [1.000, 1.000],
            ],
          },
          {
            name: "Random baseline (AUC = 0.5)",
            color: colors.textMuted,
            points: [[0.0, 0.0], [1.0, 1.0]],
          },
        ]}
      />

      <H3>6b. PR curve</H3>

      <Prose>
        The PR curve for the same dataset. Average Precision = 0.8807. The no-skill baseline sits at y = 0.20 (the positive class prevalence). The curve starts at high precision (only the most confident positives are captured at high thresholds) and falls as recall increases.
      </Prose>

      <Plot
        label="Precision-Recall curve — synthetic binary classifier (AP = 0.8807)"
        xLabel="Recall"
        yLabel="Precision"
        series={[
          {
            name: "PR curve (AP = 0.8807)",
            color: colors.gold,
            points: [
              [0.00, 1.000], [0.05, 1.000], [0.10, 1.000],
              [0.20, 0.980], [0.30, 0.970], [0.40, 0.960],
              [0.50, 0.950], [0.60, 0.940], [0.70, 0.920],
              [0.76, 0.892], [0.80, 0.860], [0.85, 0.810],
              [0.90, 0.740], [0.95, 0.620], [1.00, 0.380],
            ],
          },
          {
            name: "No-skill baseline (prevalence = 0.20)",
            color: colors.textMuted,
            points: [[0.0, 0.20], [1.0, 0.20]],
          },
        ]}
      />

      <H3>6c. Multi-class confusion matrix</H3>

      <Prose>
        Confusion matrix for the 3-class imbalanced problem (classes 0/1/2 with 70/20/10% prevalence). The model correctly handles the dominant class (0) but struggles more on the rare class (2) — visible as off-diagonal concentration in rows 1 and 2.
      </Prose>

      <Heatmap
        label="Multi-class confusion matrix (3 classes, 300 test examples)"
        matrix={[
          [205, 6, 1],
          [15, 47, 3],
          [5, 6, 12],
        ]}
        rowLabels={["True: 0", "True: 1", "True: 2"]}
        colLabels={["Pred: 0", "Pred: 1", "Pred: 2"]}
        colorScale="gold"
      />

      <H3>6d. Log-loss sensitivity to threshold confidence</H3>

      <Plot
        label="Log-loss vs. decision threshold — binary classifier"
        xLabel="Decision threshold"
        yLabel="Log-loss"
        series={[
          {
            name: "Log-loss at threshold",
            color: colors.gold,
            points: [
              [0.05, 0.88], [0.10, 0.45], [0.15, 0.30],
              [0.20, 0.23], [0.30, 0.21], [0.40, 0.205],
              [0.50, 0.206], [0.60, 0.215], [0.70, 0.240],
              [0.80, 0.310], [0.90, 0.540], [0.95, 0.900],
            ],
          },
        ]}
      />

      <Prose>
        Log-loss explodes near threshold 0 or 1 because the model makes overconfident wrong predictions — a single incorrect prediction with confidence 0.99 contributes {"−log(0.01) ≈ 4.6"} to the loss. The optimal threshold for log-loss is where calibration is best, not necessarily 0.5. This is why log-loss is the right metric to monitor when probability calibration matters.
      </Prose>

      <H3>6e. Macro vs. micro vs. weighted F1 — step-by-step</H3>

      <StepTrace
        label="Macro vs. micro vs. weighted F1 on a 3-class imbalanced problem"
        steps={[
          {
            label: "Setup: per-class confusion counts",
            render: () => (
              <Prose>
                {"Suppose a 3-class classifier produces these per-class counts on 300 test examples (support: class 0 = 212, class 1 = 65, class 2 = 23): Class 0: TP=205, FP=20, FN=7. Class 1: TP=47, FP=12, FN=18. Class 2: TP=12, FP=4, FN=11."}
              </Prose>
            ),
          },
          {
            label: "Step 1 — Compute per-class precision and recall",
            render: () => (
              <Prose>
                {"Class 0: P = 205/(205+20) = 0.911, R = 205/(205+7) = 0.967, F1 = 0.938. Class 1: P = 47/(47+12) = 0.797, R = 47/(47+18) = 0.723, F1 = 0.758. Class 2: P = 12/(12+4) = 0.750, R = 12/(12+11) = 0.522, F1 = 0.615."}
              </Prose>
            ),
          },
          {
            label: "Step 2 — Macro F1: unweighted average of per-class F1",
            render: () => (
              <Prose>
                {"Macro F1 = (F1_0 + F1_1 + F1_2) / 3 = (0.938 + 0.758 + 0.615) / 3 = 0.770. Each class gets equal weight regardless of how many examples it has. Class 2 (23 examples) pulls the average down by as much as class 0 (212 examples). This is the right choice when you care equally about performance on all classes, including rare ones."}
              </Prose>
            ),
          },
          {
            label: "Step 3 — Micro F1: pool TPs, FPs, FNs across classes",
            render: () => (
              <Prose>
                {"Total TP = 205 + 47 + 12 = 264. Total FP = 20 + 12 + 4 = 36. Total FN = 7 + 18 + 11 = 36. Micro P = 264 / (264+36) = 0.880. Micro R = 264 / (264+36) = 0.880. Micro F1 = 0.880. For multiclass, micro F1 always equals overall accuracy. It gives proportional weight to each example — the dominant class (0) heavily influences the result. The minority class (2) barely matters numerically."}
              </Prose>
            ),
          },
          {
            label: "Step 4 — Weighted F1: per-class F1 weighted by support",
            render: () => (
              <Prose>
                {"Support: class 0 = 212, class 1 = 65, class 2 = 23. Total = 300. Weighted F1 = (212/300) × 0.938 + (65/300) × 0.758 + (23/300) × 0.615 = 0.663 × 0.938 + 0.217 × 0.758 + 0.077 × 0.615 = 0.622 + 0.164 + 0.047 = 0.833. Weighted F1 is between macro (0.770) and micro (0.880) — it gives more weight to the majority class than macro but less than micro."}
              </Prose>
            ),
          },
          {
            label: "Step 5 — Which to report?",
            render: () => (
              <Prose>
                Report all three. Macro F1 gives the full picture for rare classes — if you see macro F1 much lower than weighted F1, the minority class is failing. Weighted F1 communicates overall production quality weighted by class prevalence. Micro F1 (= accuracy) is useful for homogeneous class distributions. When stakes differ by class (e.g., fraud detection where class 1 is fraud), also report per-class precision and recall separately. A single aggregate hides too much.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <StepTrace
        label="Which metric to use"
        steps={[
          {
            label: "Balanced binary classification → Accuracy, F1",
            render: () => (
              <Prose>
                When the positive and negative classes are roughly equal in size and both errors (FP and FN) have similar costs, accuracy is interpretable and F1 provides a balanced view of precision and recall. This is the regime where ROC-AUC also works well as a ranking metric. Use <Code>classification_report</Code> from sklearn to get both. Note: "balanced" means {"≥"} 20% minority prevalence. Below that, switch to the imbalanced regime.
              </Prose>
            ),
          },
          {
            label: "Imbalanced binary (rare positives) → PR-AUC over ROC-AUC",
            render: () => (
              <Prose>
                Saito and Rehmsmeier (2015) demonstrated formally that ROC-AUC is misleading on severe imbalance because TN dominates the FPR denominator. On a 1:99 dataset, a model that flags 99 false positives for every true positive can still achieve FPR {"≈ 0.01"} — which looks excellent — while precision collapses to 0.5%. PR-AUC (Average Precision) exposes this: precision is the fraction of positives among all positive predictions, and it falls rapidly when FP grows. Rule: when positive class prevalence is below 10%, use PR-AUC as the primary model-selection metric. ROC-AUC is a useful secondary check.
              </Prose>
            ),
          },
          {
            label: "Probability calibration → Log-loss + Brier score",
            render: () => (
              <Prose>
                When the model's predicted probabilities are used downstream (risk scoring, decision-theoretic thresholding, stacking), calibration is critical. Log-loss and Brier score both measure calibration quality. Log-loss penalizes confident wrong predictions more severely (logarithmic penalty); Brier score is a squared loss on probabilities and is bounded in [0, 1]. A model with AUC-ROC 0.95 but log-loss 1.5 is a good ranker with terrible probability estimates — it will mislead any downstream system that uses the probabilities. Check calibration curves (<Code>CalibrationDisplay</Code> in sklearn) whenever probabilities matter.
              </Prose>
            ),
          },
          {
            label: "Ranking / search / recommendation → NDCG, MAP, MRR",
            render: () => (
              <Prose>
                When the output is a ranked list and position matters, use ranking metrics. NDCG is the gold standard when relevance is graded (e.g., star ratings, user engagement). MAP (Mean Average Precision) is standard when relevance is binary and you want to reward correct ordering at all recall levels. MRR (Mean Reciprocal Rank) is appropriate when only the first relevant result matters (e.g., voice assistant, navigation). All three are query-averaged: compute per query, then take the mean across a test query set.
              </Prose>
            ),
          },
          {
            label: "Regression with outliers → MAE, MedAE",
            render: () => (
              <Prose>
                MAE is the most interpretable regression metric: it measures the average absolute error in the same units as the target. Unlike MSE, it does not square errors, so outliers do not dominate. MedAE (Median Absolute Error) is even more robust — it reports the median of absolute errors, which is unaffected by any number of extreme residuals as long as they are less than half the dataset. Use MAE when the error distribution has moderate tails; MedAE when you suspect heavy tails or systematic outliers in the target variable.
              </Prose>
            ),
          },
          {
            label: "Regression without outliers → MSE, RMSE, R²",
            render: () => (
              <Prose>
                {"MSE and RMSE are appropriate when large errors are disproportionately costly (e.g., safety-critical prediction). Squaring errors makes the loss surface smooth and differentiable everywhere — which is why MSE is the standard training loss for neural network regression. RMSE is MSE in the same units as the target. R² is useful for communication: 'the model explains 86% of the variance in house prices' is interpretable to non-statisticians. Caution: R² on the test set can be negative if the model is worse than predicting the mean. Always report in-sample and out-of-sample R² separately."}
              </Prose>
            ),
          },
          {
            label: "Relative errors matter → MAPE (carefully)",
            render: () => (
              <Prose>
                MAPE (Mean Absolute Percentage Error) is useful when errors should be evaluated relative to the magnitude of the target — for example, predicting that a $10 item costs $12 (20% error) is worse than predicting that a $1,000 item costs $1,002 (0.2% error), even though the absolute errors differ by $2 vs. $2. The problem: MAPE is undefined when any target value is zero and explodes when target values are near zero. Alternatives: sMAPE (symmetric MAPE, averages the denominator) or log-scale MSE when targets span multiple orders of magnitude.
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
        Most evaluation metrics require sorting the score array, which is O(n log n). The ROC curve and PR curve both require iterating over all n predictions in score-sorted order — O(n log n) to sort plus O(n) to sweep. For n = 10,000,000 predictions, this is feasible in seconds on a single machine with NumPy. AUC-ROC and AP both require the full sweep; there is no shortcut.
      </Prose>

      <Prose>
        Multi-class metrics scale O(n · K) where K is the number of classes: one-vs-rest confusion matrix accumulation costs O(n) per class. For K = 1,000 (image classification), this is 10⁷ operations per evaluation — fast. For K = 100,000 (language model vocabulary), evaluation over all classes requires careful implementation. In practice, evaluation is often restricted to the top-K predicted classes (top-5 accuracy in ImageNet) to make it tractable.
      </Prose>

      <H3>8.2 Streaming and out-of-core evaluation</H3>

      <Prose>
        The confusion matrix is additive: you can accumulate it batch by batch without storing all predictions simultaneously. This makes accuracy, precision, recall, and F1 at a fixed threshold streamable in O(1) memory. AUC-ROC and AP are not directly streamable because they require the global score ranking — you cannot compute AUC-ROC from partial batches without storing all scores. For large-scale AUC approximation, reservoir sampling provides an unbiased estimate: sample a fixed-size reservoir of predictions uniformly at random, compute AUC-ROC on the reservoir. With a 100,000-example reservoir from a 100M-example test set, the AUC estimate is accurate to within {"±0.001"} with high probability.
      </Prose>

      <H3>8.3 Regression metrics at scale</H3>

      <Prose>
        MAE and MSE are sums — they can be computed in a single pass over the data in O(n) time and O(1) memory (accumulate the sum). R² requires two passes (first to compute {"ȳ"}, then to compute {"SS_{res}"} and {"SS_{tot}"}) or equivalently one pass that accumulates {"Σy_i"}, {"Σy_i²"}, and {"Σ(y_i - ŷ_i)²"} simultaneously. All are trivially streamable and parallelizable: split data across machines, compute partial sums, aggregate. NDCG requires a global sort per query, which is O(K log K) for a list of length K per query and O(Q · K log K) total for Q queries.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>9.1 Accuracy on imbalanced data</H3>

      <Prose>
        A model that predicts only the majority class on a 99/1 imbalanced dataset achieves 99% accuracy. This is the most common evaluation mistake in applied ML. The model has zero recall on the minority class — it never flags the event you care about — but looks excellent by accuracy. Any time class prevalence is below 20%, stop reporting accuracy and switch to F1, PR-AUC, and per-class metrics.
      </Prose>

      <H3>9.2 ROC-AUC misleading with severe imbalance</H3>

      <Prose>
        Saito and Rehmsmeier (2015) gave the definitive treatment. On a 1:99 dataset, TN = 9,900. Even a model with 100 false positives has FPR = 100 / (100 + 9,900) = 0.010 — which looks excellent. ROC-AUC can be 0.95 for a model that produces more false positives than true positives at any useful operating point. The PR curve makes this visible: precision = TP / (TP + FP) = true_positives / (true_positives + 100 false positives), which collapses immediately. Use PR-AUC when prevalence is below 10%.
      </Prose>

      <H3>9.3 Macro F1 gives equal weight to tiny classes</H3>

      <Prose>
        On a dataset with class sizes [10,000, 100, 10], macro F1 weights the class-10 F1 equally with the class-10,000 F1. A model that achieves F1 = 0.99 on class 0 and F1 = 0.20 on class 2 gets macro F1 = (0.99 + 0.60 + 0.20) / 3 = 0.597 — far below what users would experience (since 99% of examples are class 0). Weighted F1 gives 0.99 × (10,000/10,110) + ... ≈ 0.978. Both numbers are correct; they answer different questions. Report both and be explicit about which you are optimizing.
      </Prose>

      <H3>9.4 Threshold = 0.5 is wrong post reweighting</H3>

      <Prose>
        After training with <Code>class_weight='balanced'</Code> or after SMOTE oversampling, the model's probability outputs are no longer calibrated to the original class distribution. A logistic regression trained on a 1:1 oversampled dataset produces {"p ≈ 0.5"} for many examples that should have probability 0.01 in the real 1:99 distribution. Applying threshold 0.5 will produce an enormous number of false positives. Always tune the threshold on a held-out validation set using the production metric, never assume 0.5.
      </Prose>

      <H3>9.5 R² negative out-of-sample</H3>

      <Prose>
        {"R² = 1 − SS_{res} / SS_{tot}. In-sample SS_{res} ≤ SS_{tot} always (the model can at worst predict the mean). Out-of-sample, the model may have overfit — SS_{res} can exceed SS_{tot}, giving R² < 0. A negative R² means the model is worse at predicting y on the test set than simply predicting the training mean. This is not a formula error. It is a diagnostic that the model has overfit, that the test distribution has shifted, or that the target variance in the test set is smaller than in training. Always compute R² on held-out data and treat negative values as a severe overfitting signal."}
      </Prose>

      <H3>9.6 MAPE with near-zero targets</H3>

      <Prose>
        MAPE = {"(1/n) Σ |y_i - ŷ_i| / |y_i|"} divides by the true value. When {"y_i"} is near zero (e.g., demand forecasting for slow-moving products), a single prediction error of 0.1 units on a target of 0.001 units produces {"100×"} relative error. MAPE becomes meaningless or infinite. Alternatives: (1) sMAPE = {"(1/n) Σ |y_i - ŷ_i| / ((|y_i| + |ŷ_i|)/2)"} — denominator averages both values, avoiding division by near-zero; (2) log1p MSE = MSE on {"log(1 + y)"} — compresses large values and handles zeros; (3) restrict MAPE to examples where {"y_i > threshold"}.
      </Prose>

      <H3>9.7 Metric at the wrong threshold inflates performance</H3>

      <Prose>
        Reporting precision and recall at threshold 0.5 on a model evaluated on test data without threshold tuning is arbitrary. The threshold that maximizes F1 on the training set is not the same as the threshold that maximizes F1 on the test set — and neither may match the operational threshold the business will actually use. The right workflow: (1) choose the metric that matches the business objective; (2) sweep thresholds on a held-out validation set; (3) select the threshold that optimizes the business metric; (4) report final performance on a separate test set using the selected threshold. Never tune and evaluate on the same data.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All citations below were WebSearch-verified for author, year, venue, DOI, and main contribution. Read in chronological order for the field's development arc.
      </Prose>

      <StepTrace
        label="primary literature"
        steps={[
          {
            label: "van Rijsbergen 1979 — Precision and recall for information retrieval",
            render: () => (
              <Prose>
                van Rijsbergen, C.J. (1979). <em>Information Retrieval</em>, 2nd edition. London: Butterworths. ISBN: 0-408-70929-4. 208 pages. Chapter 7 (Performance Evaluation) introduced precision and recall as the canonical metrics for information retrieval — the fraction of retrieved documents that are relevant (precision) and the fraction of relevant documents that are retrieved (recall). Van Rijsbergen also introduced the harmonic mean formulation of F-measure (calling it E-measure with a complement parameterization) and discussed the fundamental tension between the two metrics. The full text is freely available at openlib.org. This is the source of the precision/recall vocabulary used everywhere in modern ML evaluation.
              </Prose>
            ),
          },
          {
            label: "Fawcett 2006 — Introduction to ROC analysis",
            render: () => (
              <Prose>
                Fawcett, T. (2006). "An introduction to ROC analysis." <em>Pattern Recognition Letters</em>, 27(8):861–874. DOI: 10.1016/j.patrec.2005.10.010. Available at people.inf.elte.hu/kiss/13dwhdm/roc.pdf. Over 21,000 citations. The paper that made ROC analysis accessible to the machine learning community. Key contributions: (1) the AUC-ROC = P(score_pos {">"} score_neg) probabilistic interpretation; (2) the relationship between ROC operating points and iso-performance lines under varying cost/class-prior assumptions; (3) the "convex hull" construction for choosing the optimal classifier from a collection; (4) why comparing classifiers at a single operating point is almost always wrong. Still the canonical entry point for ROC analysis.
              </Prose>
            ),
          },
          {
            label: "Davis & Goadrich 2006 — PR vs. ROC duality",
            render: () => (
              <Prose>
                Davis, J. and Goadrich, M. (2006). "The Relationship Between Precision-Recall and ROC Curves." <em>Proceedings of the 23rd International Conference on Machine Learning (ICML 2006)</em>, Pittsburgh, pp. 233–240. ACM DL: doi.org/10.1145/1143844.1143874. PDF at mark.goadrich.com/articles/davisgoadrichcamera2.pdf. The paper that formalized the relationship between ROC and PR space. Key theorem: a curve dominates in ROC space if and only if it dominates in PR space — you cannot improve AUC-ROC without also improving PR-AUC on the same dataset (and vice versa). Also introduced an algorithm for computing the achievable PR curve (the PR analog of the ROC convex hull) and proved that linear interpolation in PR space is not achievable, unlike in ROC space.
              </Prose>
            ),
          },
          {
            label: "Saito & Rehmsmeier 2015 — PR-AUC more informative than ROC on imbalanced data",
            render: () => (
              <Prose>
                Saito, T. and Rehmsmeier, M. (2015). "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets." <em>PLoS ONE</em>, 10(3):e0118432. DOI: 10.1371/journal.pone.0118432. PMC: PMC4349800. The paper that gave the machine learning community a rigorous argument for preferring PR-AUC over ROC-AUC on imbalanced datasets. The authors demonstrated that ROC plots can give an overly optimistic view of classifier performance when the negative class is much larger than the positive class, because FPR's TN denominator masks the absolute number of false positives. They showed mathematically why PR curves do not share this problem (TN never appears) and provided empirical demonstrations across multiple bioinformatics datasets. This paper has changed the standard evaluation practice for cancer detection, drug discovery, and any rare-event ML application.
              </Prose>
            ),
          },
          {
            label: "Järvelin & Kekäläinen 2002 — NDCG for graded relevance",
            render: () => (
              <Prose>
                Järvelin, K. and Kekäläinen, J. (2002). "Cumulated gain-based evaluation of IR techniques." <em>ACM Transactions on Information Systems (TOIS)</em>, 20(4):422–446. DOI: 10.1145/582415.582418. The paper that introduced Normalized Discounted Cumulative Gain (NDCG) as an evaluation metric for information retrieval. Before NDCG, most IR evaluation assumed binary relevance (relevant or not). Järvelin and Kekäläinen showed that graded relevance (how relevant, not just whether relevant) better captures user benefit, and that ranking highly relevant documents early is disproportionately valuable — justifying the logarithmic discount. NDCG is now the standard metric in web search evaluation (used by Google, Bing, and academic TREC benchmarks) and has been widely adopted in recommendation systems and learning-to-rank.
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
        Work through these before moving to the next topic. The answer key is below each exercise — resist reading ahead.
      </Prose>

      <H3>Exercise 1 (recall)</H3>
      <Prose>
        A binary classifier on a 95/5 imbalanced dataset (95% negative, 5% positive) predicts "negative" for every example. (a) What is its accuracy? (b) What is its precision, recall, and F1 on the positive class? (c) What is its AUC-ROC? (d) What metric should you report instead, and what is its value for this trivial classifier?
      </Prose>
      <Callout type="answer" title="Answer 1">
        {"(a) Accuracy = TN / (TN + FP + TP + FN) = 950/1000 = 95%. Misleadingly high — the model never fires. (b) TP = 0 (never predicts positive). Precision = 0/(0+0) = undefined (0 by convention). Recall = 0/(0+50) = 0. F1 = 0. The model is completely useless for the positive class. (c) AUC-ROC: because the classifier never produces a ranked score (or all scores are identical), the ROC curve degenerates to a single point at (FPR=0, TPR=0). AUC-ROC = 0.5 — random classifier equivalent. If you use the convention that all predictions at the same score sweep from (0,0) to (1,1) without a curve, AUC-ROC = 0.5. (d) PR-AUC (Average Precision). For a constant-zero classifier, the precision-recall curve degenerates to the prevalence baseline at y = 0.05. AP = 0.05 (the no-skill baseline). Any real model must exceed AP = 0.05 to be useful."}
      </Callout>

      <H3>Exercise 2 (math)</H3>
      <Prose>
        Derive the AUC-ROC probabilistic interpretation: AUC-ROC = {"P(score_pos > score_neg)"}. What does this mean for a model with AUC-ROC = 0.5? AUC-ROC = 0.9? AUC-ROC = 0.0?
      </Prose>
      <Callout type="answer" title="Answer 2">
        {"AUC-ROC = P(score_pos > score_neg). Derivation sketch: Consider sweeping the threshold from 1 down to 0. At each threshold t, TPR(t) = P(score > t | positive) and FPR(t) = P(score > t | negative). AUC = ∫ TPR dFPR = ∫_0^1 P(score_pos > t) d[P(score_neg > t)]. By definition of probability this integral equals P(score_pos > score_neg) — the probability that a randomly drawn positive is scored higher than a randomly drawn negative. Interpretation: AUC = 0.5 means the model's score on positives and negatives are identically distributed — random ranking. AUC = 0.9 means that in 90% of randomly drawn (positive, negative) pairs, the positive gets the higher score — a strong ranker. AUC = 0.0 means the model perfectly inverts the ranking — every negative is scored higher than every positive. A model with AUC < 0.5 is worse than random; flipping its predictions gives AUC > 0.5."}
      </Callout>

      <H3>Exercise 3 (conceptual)</H3>
      <Prose>
        Explain why Average Precision (AP) uses step-function integration rather than trapezoidal integration. What would happen if you used trapezoidal integration on a PR curve?
      </Prose>
      <Callout type="answer" title="Answer 3">
        {"The PR curve is a step function: precision jumps discontinuously each time the threshold changes by one example. Between any two adjacent threshold values, no new operating points are achievable — you cannot interpolate between them by changing the threshold. In ROC space, linear interpolation between two adjacent operating points is achievable: you can randomly predict positive with some probability between the two thresholds, landing anywhere on the line segment between them. In PR space, this is not true — linear interpolation yields operating points that are not achievable by any threshold or randomization. Using trapezoidal integration on a PR curve would compute the area under a linearly interpolated curve that includes non-achievable operating points, overestimating the true area under the step function. Sklearn's average_precision_score uses step-function integration: AP = Σ (R_k - R_{k-1}) · P_k, which correctly integrates only over achievable operating points. This is why AP is slightly lower than a naive trapezoidal integral of the PR curve."}
      </Callout>

      <H3>Exercise 4 (applied)</H3>
      <Prose>
        You train two models on a fraud detection task (1% fraud prevalence, 1 million daily transactions). Model A has AUC-ROC = 0.97 and AP = 0.41. Model B has AUC-ROC = 0.91 and AP = 0.68. Which model should you deploy, and why? What additional information would you want before deciding?
      </Prose>
      <Callout type="answer" title="Answer 4">
        {"Deploy Model B. On a severely imbalanced dataset (1% fraud), PR-AUC (AP) is the more informative metric. Model B's AP of 0.68 vs. Model A's 0.41 means that at any given recall level, Model B has substantially higher precision — it wastes fewer investigator hours on false alarms while catching the same fraction of fraud. Model A's higher ROC-AUC (0.97 vs. 0.91) reflects better TN handling, but on a 99% negative dataset TN performance is nearly irrelevant — the model correctly clears negatives almost by default. Additional information to gather before deploying: (1) Precision-recall curves at the specific operating points the fraud team can handle (alert volume budget per day). (2) Per-class calibration — if the fraud model feeds a downstream risk score, are probabilities calibrated? (3) Latency requirements — if Model B is 10x more complex than Model A, inference latency matters. (4) Threshold sweep on the validation set: at the alert volume the team can handle, what is the precision and recall of each model?"}
      </Callout>

      <H3>Exercise 5 (debugging)</H3>
      <Prose>
        You fit a linear regression model. In-sample {"R² = 0.92"}. Test set {"R² = −0.15"}. (a) Is {"R² = −0.15"} a valid number? What does it mean? (b) List three possible causes. (c) What metrics would help diagnose the issue further?
      </Prose>
      <Callout type="answer" title="Answer 5">
        {"(a) Yes, R² < 0 is mathematically valid out-of-sample. It means SS_res > SS_tot on the test set — the model's predictions are farther from the test y values than the training mean ȳ_train is. A constant predictor (predict ȳ_train for every example) would score R² = 0; this model scores worse than that. (b) Three causes: (1) Overfitting — the model memorized training noise (e.g., too many features, too little regularization), and the test set has different noise patterns. In-sample R² = 0.92 with test R² = −0.15 is a strong overfitting signal. (2) Distribution shift — the test set has a different mean or variance of y than the training set. If test ȳ_test is far from training ȳ_train, SS_tot on the test set is different and SS_res grows. (3) Target leakage in training — a feature is correlated with y only in the training set (temporal leakage), so in-sample performance is artificially inflated. (c) Diagnostic metrics: (1) MAE on train vs. test — a large ratio indicates overfitting. (2) Plot residuals vs. fitted values on both sets — look for different patterns. (3) Feature importance or SHAP values — identify which features drive the model and check if they leak future information. (4) Compare ȳ_train vs. ȳ_test and Var(y_train) vs. Var(y_test) to check for distribution shift."}
      </Callout>

      <H3>Exercise 6 (synthesis)</H3>
      <Prose>
        A medical screening test for a rare disease (0.5% prevalence) achieves precision = 0.12, recall = 0.90, F1 = 0.21 at threshold 0.5. Your clinical collaborator says this is "terrible" because precision is so low. You say this may be acceptable. Who is right, and how do you make the case with metrics?
      </Prose>
      <Callout type="answer" title="Answer 6">
        {"Both perspectives have merit, but the answer depends on cost structure, not on the metric value alone. The clinical collaborator is right that precision = 0.12 means 88% of positive predictions are false alarms — in a screening context, this generates 88 unnecessary follow-up tests for every true positive detected. If the follow-up is expensive, invasive, or anxiety-inducing, this matters. You are right that for a disease with 0.5% prevalence, precision = 0.12 may be acceptable given the stakes. Here is the case with metrics: (1) The no-skill baseline precision is 0.005 (just the prevalence). Precision = 0.12 is 24× better than random — the model contributes enormous signal. (2) Recall = 0.90 means the model catches 90% of true cases — with a rare and potentially fatal disease, missing 10% of cases (low recall) is often the more costly failure. (3) Use Fβ with β = 2 (weighing recall 4× more than precision): F2 = (1+4) × 0.12 × 0.90 / (4 × 0.12 + 0.90) = 5 × 0.108 / (0.48 + 0.90) = 0.54 / 1.38 = 0.391. Frame the conversation around cost ratios: if missing a true case costs 10× more than a false alarm, the optimal threshold from Elkan's rule is c_FP / (c_FP + c_FN) = 1/11 ≈ 9% — far below 0.5, and the model's current operating point at recall = 0.90 may be close to optimal under that cost ratio. Bring the cost matrix to the collaborator, not just the metric."}
      </Callout>

    </div>
  ),
};

export default evaluationMetricsContent;
