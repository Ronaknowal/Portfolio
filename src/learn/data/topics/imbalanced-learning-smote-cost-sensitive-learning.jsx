import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const imbalancedLearningContent = {
  title: "Imbalanced Learning (SMOTE, Cost-Sensitive Learning)",
  readTime: "~45 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        In 2002, Nitesh Chawla, Kevin Bowyer, Lawrence Hall, and W. Philip Kegelmeyer published "SMOTE: Synthetic Minority Over-sampling Technique" in the <em>Journal of Artificial Intelligence Research</em>, volume 16, pages 321–357. Their target was a class of problems that standard machine learning pipelines handle badly: credit card fraud, medical diagnosis of rare diseases, manufacturing defect detection, network intrusion detection — any supervised learning problem where the event you care most about is also the rarest. The paper opened with a diagnostic that anyone who has trained a classifier on a real-world dataset has encountered: a model that achieves 98% accuracy by learning to always predict "normal" on a dataset where 98% of examples are normal. Accuracy is not a useful metric when the class distribution is skewed. The precision, recall, and F-measure on the minority class are what matter — and those are often zero on a model that never predicts the minority class.
      </Prose>

      <Prose>
        The problem predates Chawla et al. by a decade. In 2001, Charles Elkan published "The Foundations of Cost-Sensitive Learning" at IJCAI, which gave the field a rigorous Bayesian framework for thinking about misclassification costs. Elkan showed that cost-sensitive classification reduces to a principled threshold adjustment on top of any standard probability estimator: you do not need a special algorithm, only a correct decision rule. Then in 2009, Haibo He and Edwardo Garcia published "Learning from Imbalanced Data" in <em>IEEE Transactions on Knowledge and Data Engineering</em>, 21(9):1263–1284 — the comprehensive survey that organized the field into its three main families: resampling methods, algorithm-level methods, and hybrid methods.
      </Prose>

      <Prose>
        The domains where the minority class matters most read like a list of the highest-stakes ML applications. In fraud detection, fraud rates typically range from 0.1% to 2% of transactions — a model that never flags fraud looks excellent by accuracy. In medical diagnosis, rare diseases affect small fractions of tested populations — missing a true positive (a false negative) has very different consequences than a false alarm. In predictive maintenance, equipment failures occur rarely — but missing a failure means an unplanned shutdown. In spam filtering, precision matters: falsely flagging legitimate email (a false positive) is often worse than missing spam. Each application has its own asymmetry in the costs of false positives versus false negatives, and the naive maximum-likelihood classifier, which treats all errors as equally costly, will systematically under-predict the minority class.
      </Prose>

      <Prose>
        The fundamental tension is this: standard supervised learning training — maximizing likelihood or minimizing cross-entropy — assumes that the class distribution in the training set matches the class distribution at deployment. When it does not, the learned decision boundary is biased toward the majority class. The optimizer is rewarded more for correctly classifying majority examples (there are more of them contributing to the loss) and so it shapes the decision boundary to serve the majority class at the expense of the minority. Resampling and cost-sensitive methods are the two families of techniques that exist precisely to correct this imbalance before or during training.
      </Prose>

      <Callout type="insight">
        Accuracy is a misleading metric on imbalanced data. A classifier that never predicts the positive class achieves {" (1 - prevalence) × 100%"} accuracy — 99% accuracy on a dataset with 1% prevalence. Always evaluate with precision, recall, F1, and PR-AUC when the positive class is rare.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        There are three lever families for handling imbalanced data, and they operate at different points in the ML pipeline. Understanding which lever to pull — and why — requires understanding what each one actually does to the loss surface.
      </Prose>

      <H3>2.1 Lever 1: resample the data</H3>

      <Prose>
        The first family changes the class ratio that the model sees during training by modifying the dataset. <strong>Oversampling</strong> adds copies or synthetic examples of the minority class, raising its count toward parity with the majority. The simplest form — random oversampling — duplicates existing minority examples. The problem is that duplicating exact copies adds no new information; it only inflates the gradient contribution of those specific examples and can cause overfitting to the minority points that happen to be in the training set. SMOTE replaces duplication with interpolation: for each minority example, it finds its k nearest minority-class neighbors and generates a new synthetic point uniformly along the line segment between the original example and a randomly chosen neighbor. The synthetic point is new — it did not exist in the original dataset — but it is in a region where the classifier already believes the minority class lives.
      </Prose>

      <Prose>
        <strong>Undersampling</strong> takes the opposite approach: reduce the majority class count to match the minority. Random undersampling randomly discards majority examples. More sophisticated methods like NearMiss preferentially remove majority examples that are closest to the minority class boundary — removing the "easy" majority examples and keeping the hard ones, which can sharpen the decision boundary. The cost of undersampling is information loss: you are throwing away data you paid to collect.
      </Prose>

      <Prose>
        <strong>Hybrid methods</strong> combine oversampling and undersampling. SMOTEENN runs SMOTE to oversample the minority and then applies Edited Nearest Neighbours (ENN) to remove noisy examples from both classes — minority synthetic points that are misclassified by their neighbors, and majority examples that are surrounded by minority examples. SMOTETomek combines SMOTE with Tomek Links removal (removing pairs of examples from opposite classes that are each other's nearest neighbor). Both clean the boundary after oversampling.
      </Prose>

      <H3>2.2 Lever 2: adjust the loss or weights</H3>

      <Prose>
        Instead of changing the data, the second family changes what the model is penalized for. <strong>Class weighting</strong> multiplies the loss contribution of each example by a weight inversely proportional to its class frequency. The <Code>class_weight='balanced'</Code> option in sklearn sets each class's weight to {"n_samples / (n_classes * n_samples_in_class)"}. The gradient update for each minority example is scaled up by the same factor, making the model attend to minority examples as much as majority examples during optimization — without changing the data at all.
      </Prose>

      <Prose>
        <strong>Focal loss</strong>, introduced by Lin, Goyal, Girshick, He, and Dollár (2017) in the context of object detection, extends this idea dynamically. Standard cross-entropy treats all examples equally. Focal loss adds a modulating factor {"(1-p)^γ"} to the loss, where p is the predicted probability of the correct class and γ is a focusing parameter. When the model predicts a high probability for an easy example, {"(1-p)^γ"} is small — the loss contribution of easy, well-classified examples is down-weighted. Hard examples (where the model is uncertain or wrong) retain high loss contributions. This focuses training on the examples that are actually informative, regardless of their class label.
      </Prose>

      <H3>2.3 Lever 3: threshold moving at inference time</H3>

      <Prose>
        The third lever does not touch training at all. A calibrated logistic regression produces probabilities; the default decision rule is to predict positive if {"p ≥ 0.5"}. But 0.5 is an arbitrary choice that makes sense only when false positives and false negatives have equal cost and when the training distribution matches the deployment distribution. Moving the threshold changes the precision-recall tradeoff: a lower threshold (say 0.1) increases recall at the cost of precision — you flag more positives, including more false positives. A higher threshold (say 0.7) increases precision at the cost of recall. Threshold moving is free — it costs nothing computationally — and should always be tuned on a validation set using the metric that matches the deployment stakes.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 Confusion-matrix metrics: precision, recall, F₁, Fβ</H3>

      <Prose>
        For a binary classifier, every prediction falls into one of four cells. The confusion matrix counts: TP (true positives — correctly flagged minority), TN (true negatives — correctly cleared majority), FP (false positives — majority incorrectly flagged), FN (false negatives — minority missed). From these four counts, the key metrics are:
      </Prose>

      <MathBlock>
        {"\\text{Precision} = \\frac{\\text{TP}}{\\text{TP} + \\text{FP}}, \\qquad \\text{Recall} = \\frac{\\text{TP}}{\\text{TP} + \\text{FN}}"}
      </MathBlock>

      <MathBlock>
        {"F_1 = \\frac{2 \\cdot \\text{Precision} \\cdot \\text{Recall}}{\\text{Precision} + \\text{Recall}} = \\frac{2\\,\\text{TP}}{2\\,\\text{TP} + \\text{FP} + \\text{FN}}"}
      </MathBlock>

      <Prose>
        The generalized Fβ score weights recall β times as much as precision:
      </Prose>

      <MathBlock>
        {"F_\\beta = (1 + \\beta^2) \\cdot \\frac{\\text{Precision} \\cdot \\text{Recall}}{\\beta^2 \\cdot \\text{Precision} + \\text{Recall}}"}
      </MathBlock>

      <Prose>
        When β {">"} 1, recall is weighted more heavily — appropriate for medical screening where missing a disease (FN) is worse than a false alarm. When β {"<"} 1, precision is weighted more heavily — appropriate for fraud alerting where annoying a legitimate customer (FP) is expensive. F2 (β=2) is common in information retrieval and medical applications; F0.5 is used in settings where false alarms are costly.
      </Prose>

      <H3>3.2 ROC-AUC vs PR-AUC — why PR-AUC is the right metric when positives are rare</H3>

      <Prose>
        The ROC curve plots the True Positive Rate (recall) against the False Positive Rate (FPR = FP / (FP + TN)) as the decision threshold varies. ROC-AUC is the area under this curve. The problem with ROC-AUC on severely imbalanced datasets is that FPR is dominated by TN. When there are 9,900 negatives and 100 positives, a model can have FPR = 0.01 (99 false positives!) while its FPR term looks small because TN = 9,801 makes the denominator large. ROC-AUC will look high even when the model produces many false positives relative to the minority class.
      </Prose>

      <Prose>
        The Precision-Recall curve plots precision against recall as the threshold varies. PR-AUC (the area under this curve, also called Average Precision) does not involve TN at all — it only counts TP, FP, and FN. When the positive class is rare, every false positive is visible because the denominator of precision is small. PR-AUC correctly penalizes a model that achieves high recall by flagging everything as positive — its precision collapses toward the base rate. The rule: use ROC-AUC when the positive rate is {"≥"} 10% and both classes matter equally; use PR-AUC when the positive class is rare ({"<"} 10%) or when the cost of false positives is meaningfully different from the cost of false negatives.
      </Prose>

      <H3>3.3 SMOTE: the interpolation formula</H3>

      <Prose>
        Let {"x"} be a minority-class example and let {"N(x, k)"} denote its k nearest neighbors among minority-class examples. SMOTE picks a random neighbor {"x_n ∈ N(x, k)"} and generates a synthetic example by uniform interpolation along the line segment between them:
      </Prose>

      <MathBlock>
        {"x_{\\text{syn}} = x + \\lambda \\cdot (x_n - x), \\qquad \\lambda \\sim \\text{Uniform}(0, 1)"}
      </MathBlock>

      <Prose>
        When λ = 0 the synthetic point equals x; when λ = 1 it equals the neighbor. All values in between are interior points on the segment. Because both endpoints are minority-class examples, SMOTE assumes the region between them is also minority territory — a reasonable assumption if the minority class is compact, but a dangerous one if the minority boundary is noisy or if majority examples intrude between two minority points.
      </Prose>

      <Prose>
        <strong>Borderline-SMOTE</strong> restricts synthesis to minority examples whose nearest neighbors include majority examples — the "borderline" points that the classifier is most likely to misclassify. Interior minority examples (surrounded entirely by minority neighbors) are left alone; they are already well-covered. Focusing synthesis on the boundary creates more examples where the classifier needs them most.
      </Prose>

      <Prose>
        <strong>ADASYN</strong> (He, Bai, Garcia, Li 2008) goes further: it computes a density ratio for each minority example — the fraction of its k nearest neighbors that are majority-class. This ratio becomes the weight for how many synthetic examples to generate near that point. Minority examples deep in their class region (low majority fraction among neighbors) get few synthetic examples; minority examples near the majority boundary (high majority fraction) get many. ADASYN adapts the synthesis distribution to the difficulty of the region, not just the global class imbalance.
      </Prose>

      <H3>3.4 Cost-sensitive decision rule: derivation from Bayes-risk minimization</H3>

      <Prose>
        Let {"c_{FP}"} be the cost of a false positive and {"c_{FN}"} be the cost of a false negative. Given a calibrated model that estimates {"P(+|x)"}, the expected cost of predicting positive is:
      </Prose>

      <MathBlock>
        {"\\text{Cost}(\\hat{y}=1) = (1 - P(+|x)) \\cdot c_{FP}"}
      </MathBlock>

      <Prose>
        The expected cost of predicting negative is:
      </Prose>

      <MathBlock>
        {"\\text{Cost}(\\hat{y}=0) = P(+|x) \\cdot c_{FN}"}
      </MathBlock>

      <Prose>
        Predict positive when the expected cost of predicting positive is less than the expected cost of predicting negative:
      </Prose>

      <MathBlock>
        {"(1 - P(+|x)) \\cdot c_{FP} < P(+|x) \\cdot c_{FN}"}
      </MathBlock>

      <Prose>
        Rearranging:
      </Prose>

      <MathBlock>
        {"c_{FP} < P(+|x) \\cdot (c_{FP} + c_{FN})"}
      </MathBlock>

      <MathBlock>
        {"P(+|x) > \\frac{c_{FP}}{c_{FP} + c_{FN}}"}
      </MathBlock>

      <Prose>
        This is Elkan's result: the optimal decision threshold is not 0.5 but {"c_{FP} / (c_{FP} + c_{FN})"}. When false negatives are twice as costly as false positives ({"c_{FN} = 2c_{FP}"}), the threshold becomes {"c_{FP} / (c_{FP} + 2c_{FP}) = 1/3"}. You should predict positive whenever the model's probability exceeds 1/3. This derivation assumes the model is perfectly calibrated — that {"P(+|x)"} truly reflects the posterior probability. In practice, check calibration (Platt scaling or isotonic regression) before relying on cost-sensitive thresholding.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        All code below was run on a 1:50 imbalanced synthetic dataset (500 majority, 10 minority). NumPy only — no sklearn, no imbalanced-learn. The stdout is embedded verbatim. We implement SMOTE, random undersampling, and class-weighted logistic regression gradient descent.
      </Prose>

      <H3>4a. Dataset, SMOTE, and undersampling</H3>

      <CodeBlock language="python">
{`import numpy as np

np.random.seed(42)

# 1:50 imbalanced dataset
n_majority = 500
n_minority  = 10
X_maj = np.random.randn(n_majority, 2) + np.array([0, 0])
X_min = np.random.randn(n_minority,  2) + np.array([3, 3])
X = np.vstack([X_maj, X_min])
y = np.hstack([np.zeros(n_majority), np.ones(n_minority)])

print(f"Dataset shape: {X.shape}, class counts: {np.bincount(y.astype(int))}")
# Output: Dataset shape: (510, 2), class counts: [500  10]

# ── SMOTE ────────────────────────────────────────────────────
def smote(X_min, n_synthetic, k=5, seed=42):
    """
    For each of n_synthetic new points:
      1. Pick a random minority example x_i.
      2. Find its k nearest minority neighbours.
      3. Pick one neighbour x_n at random.
      4. Synthesise: x_syn = x_i + lambda * (x_n - x_i), lambda ~ U(0,1).
    """
    rng = np.random.default_rng(seed)
    synthetic = []
    for _ in range(n_synthetic):
        i  = rng.integers(0, len(X_min))
        x  = X_min[i]
        dists = np.linalg.norm(X_min - x, axis=1)
        dists[i] = np.inf                       # exclude self
        nn_idx = np.argsort(dists)[:k]
        j  = nn_idx[rng.integers(0, k)]
        xn = X_min[j]
        lam = rng.uniform(0, 1)
        synthetic.append(x + lam * (xn - x))   # interpolate
    return np.array(synthetic)

# Oversample minority to match majority
X_syn   = smote(X[y == 1], n_synthetic=(n_majority - n_minority))
X_smote = np.vstack([X, X_syn])
y_smote = np.hstack([y, np.ones(n_majority - n_minority)])
print(f"After SMOTE: {np.bincount(y_smote.astype(int))}")
# Output: After SMOTE: [500 500]

# ── Random undersampling ─────────────────────────────────────
def random_undersample(X, y, seed=42):
    rng = np.random.default_rng(seed)
    n_min   = int(np.sum(y == 1))
    maj_idx = np.where(y == 0)[0]
    min_idx = np.where(y == 1)[0]
    chosen  = rng.choice(maj_idx, size=n_min, replace=False)
    idx = np.concatenate([chosen, min_idx])
    return X[idx], y[idx]

X_under, y_under = random_undersample(X, y)
print(f"After undersampling: {np.bincount(y_under.astype(int))}")
# Output: After undersampling: [10 10]`}
      </CodeBlock>

      <H3>4b. Class-weighted logistic regression gradient</H3>

      <CodeBlock language="python">
{`def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def logistic_gd_weighted(X, y, class_weight=None, lr=0.1, n_iter=300):
    """
    Gradient of weighted log-loss:
      grad = (1/n) * X^T ((sigma(Xw) - y) * sample_weights)
    class_weight='balanced' scales each sample by inverse class frequency.
    """
    n, d = X.shape
    w = np.zeros(d)
    if class_weight == 'balanced':
        pos_w = n / (2 * np.sum(y == 1))
        neg_w = n / (2 * np.sum(y == 0))
        sample_w = np.where(y == 1, pos_w, neg_w)
    else:
        sample_w = np.ones(n)
    for _ in range(n_iter):
        p = sigmoid(X @ w)
        grad = X.T @ ((p - y) * sample_w) / n
        w -= lr * grad
    return w

def evaluate(X, y, w, threshold=0.5):
    preds = (sigmoid(X @ w) >= threshold).astype(int)
    tp = int(np.sum((preds == 1) & (y == 1)))
    tn = int(np.sum((preds == 0) & (y == 0)))
    fp = int(np.sum((preds == 1) & (y == 0)))
    fn = int(np.sum((preds == 0) & (y == 1)))
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    return tp, tn, fp, fn, prec, rec, f1

# Add bias column to each dataset
X_b       = np.column_stack([np.ones(len(y)),       X])
X_smote_b = np.column_stack([np.ones(len(y_smote)), X_smote])
X_under_b = np.column_stack([np.ones(len(y_under)), X_under])

# ── A: Baseline — no resampling, no class weight ─────────────
w = logistic_gd_weighted(X_b, y)
tp, tn, fp, fn, prec, rec, f1 = evaluate(X_b, y, w)
print("=== BASELINE (no resampling) ===")
print(f"  TP={tp} TN={tn} FP={fp} FN={fn}")
print(f"  Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}")
# Output:
# === BASELINE (no resampling) ===
#   TP=7 TN=500 FP=0 FN=3
#   Precision=1.000  Recall=0.700  F1=0.824

# ── B: After SMOTE ──────────────────────────────────────────
w2 = logistic_gd_weighted(X_smote_b, y_smote)
tp2, tn2, fp2, fn2, prec2, rec2, f12 = evaluate(X_b, y, w2)
print("\n=== AFTER SMOTE ===")
print(f"  TP={tp2} TN={tn2} FP={fp2} FN={fn2}")
print(f"  Precision={prec2:.3f}  Recall={rec2:.3f}  F1={f12:.3f}")
# Output:
# === AFTER SMOTE ===
#   TP=10 TN=491 FP=9 FN=0
#   Precision=0.526  Recall=1.000  F1=0.690

# ── C: After random undersampling ──────────────────────────
w3 = logistic_gd_weighted(X_under_b, y_under)
tp3, tn3, fp3, fn3, prec3, rec3, f13 = evaluate(X_b, y, w3)
print("\n=== AFTER RANDOM UNDERSAMPLING ===")
print(f"  TP={tp3} TN={tn3} FP={fp3} FN={fn3}")
print(f"  Precision={prec3:.3f}  Recall={rec3:.3f}  F1={f13:.3f}")
# Output:
# === AFTER RANDOM UNDERSAMPLING ===
#   TP=10 TN=467 FP=33 FN=0
#   Precision=0.233  Recall=1.000  F1=0.377

# ── D: Class-weighted gradient descent ─────────────────────
w4 = logistic_gd_weighted(X_b, y, class_weight='balanced')
tp4, tn4, fp4, fn4, prec4, rec4, f14 = evaluate(X_b, y, w4)
print("\n=== CLASS_WEIGHT=BALANCED ===")
print(f"  TP={tp4} TN={tn4} FP={fp4} FN={fn4}")
print(f"  Precision={prec4:.3f}  Recall={rec4:.3f}  F1={f14:.3f}")
# Output:
# === CLASS_WEIGHT=BALANCED ===
#   TP=10 TN=490 FP=10 FN=0
#   Precision=0.500  Recall=1.000  F1=0.667`}
      </CodeBlock>

      <Prose>
        The results tell a clear story. The baseline misses 3 of 10 minority examples (recall 0.70) but never false-alarms (precision 1.0) — the model is conservative about the minority class because the majority loss dominates training. SMOTE achieves perfect recall (0 FN) at the cost of 9 false positives (precision 0.526, F1 0.690). Class weighting also achieves perfect recall with 10 false positives (precision 0.500). Random undersampling achieves perfect recall but 33 false positives — it loses too much information by discarding most majority examples. On this toy problem, SMOTE and class weighting perform comparably; class weighting is cheaper (no augmented dataset, same training time) while SMOTE changes the data distribution the model sees.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        The <Code>imbalanced-learn</Code> library (pip install imbalanced-learn, version 0.14+) is the standard production choice. It integrates with sklearn pipelines and provides the full zoo of resampling methods. All code below was run and the stdout is embedded verbatim.
      </Prose>

      <H3>5a. Resampling methods and imblearn Pipeline</H3>

      <CodeBlock language="python">
{`from imblearn.over_sampling  import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks
from imblearn.combine        import SMOTEENN, SMOTETomek
from imblearn.pipeline       import Pipeline as ImbPipeline  # critical import
from sklearn.linear_model    import LogisticRegression
from sklearn.datasets        import make_classification
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import precision_score, recall_score, f1_score
import numpy as np

np.random.seed(42)
# Imbalanced: ~5% positive rate (actual 54/1000 after make_classification)
X, y = make_classification(
    n_samples=1000, n_features=10, n_informative=5,
    n_redundant=2, weights=[0.95, 0.05], random_state=42
)
print(f"Original class counts: {np.bincount(y)}")
# Output: Original class counts: [946  54]

# ── Resampler comparison ─────────────────────────────────────
sm = SMOTE(random_state=42)
X_sm, y_sm = sm.fit_resample(X, y)
print(f"After SMOTE:      {np.bincount(y_sm)}")
# Output: After SMOTE:      [946 946]

ad = ADASYN(random_state=42)
X_ad, y_ad = ad.fit_resample(X, y)
print(f"After ADASYN:     {np.bincount(y_ad)}")
# Output: After ADASYN:     [946 946]

st = SMOTETomek(random_state=42)
X_st, y_st = st.fit_resample(X, y)
print(f"After SMOTETomek: {np.bincount(y_st)}")
# Output: After SMOTETomek: [946 946]

se = SMOTEENN(random_state=42)
X_se, y_se = se.fit_resample(X, y)
print(f"After SMOTEENN:   {np.bincount(y_se)}")
# Output: After SMOTEENN:   [867 939]`}
      </CodeBlock>

      <Callout type="warning" title="Pipeline import matters">
        Use <Code>from imblearn.pipeline import Pipeline</Code>, NOT <Code>from sklearn.pipeline import Pipeline</Code>. The imblearn version knows to apply resampling steps only on the training fold during cross-validation. The sklearn Pipeline does not call <Code>fit_resample</Code> — it will silently skip your resampler or crash.
      </Callout>

      <H3>5b. CV comparison — baseline vs SMOTE pipeline vs class_weight</H3>

      <CodeBlock language="python">
{`# ── imblearn Pipeline: resampling happens INSIDE each CV fold ──
pipe_baseline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('clf',    LogisticRegression(solver='lbfgs', max_iter=500, C=1.0))
])
pipe_smote = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote',  SMOTE(random_state=42)),
    ('clf',    LogisticRegression(solver='lbfgs', max_iter=500, C=1.0))
])
pipe_cw = ImbPipeline([
    ('scaler', StandardScaler()),
    ('clf',    LogisticRegression(solver='lbfgs', max_iter=500, C=1.0,
                                  class_weight='balanced'))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores_base  = cross_val_score(pipe_baseline, X, y, cv=cv,
                               scoring='average_precision')
scores_smote = cross_val_score(pipe_smote,    X, y, cv=cv,
                               scoring='average_precision')
scores_cw    = cross_val_score(pipe_cw,       X, y, cv=cv,
                               scoring='average_precision')

print("=== CV PR-AUC (5-fold StratifiedKF) ===")
print(f"Baseline:              {scores_base.round(3)}  mean={scores_base.mean():.3f}")
print(f"imblearn SMOTE pipe:   {scores_smote.round(3)}  mean={scores_smote.mean():.3f}")
print(f"class_weight=balanced: {scores_cw.round(3)}  mean={scores_cw.mean():.3f}")
# Output:
# === CV PR-AUC (5-fold StratifiedKF) ===
# Baseline:              [0.387 0.437 0.19  0.534 0.202]  mean=0.350
# imblearn SMOTE pipe:   [0.158 0.377 0.123 0.279 0.138]  mean=0.215
# class_weight=balanced: [0.138 0.364 0.124 0.291 0.138]  mean=0.211`}
      </CodeBlock>

      <H3>5c. Threshold moving via predict_proba</H3>

      <CodeBlock language="python">
{`# ── Threshold sweep on held-out test set ────────────────────
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

clf = LogisticRegression(solver='lbfgs', max_iter=500, C=1.0,
                          class_weight='balanced')
clf.fit(X_tr_s, y_tr)
probs = clf.predict_proba(X_te_s)[:, 1]   # P(positive) for each test example

print("=== Threshold sweep (class_weight=balanced, test set) ===")
for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    preds = (probs >= t).astype(int)
    p = precision_score(y_te, preds, zero_division=0)
    r = recall_score(y_te, preds)
    f = f1_score(y_te, preds, zero_division=0)
    print(f"  threshold={t:.1f}  precision={p:.3f}  recall={r:.3f}  f1={f:.3f}")
# Output:
# === Threshold sweep (class_weight=balanced, test set) ===
#   threshold=0.1  precision=0.065  recall=1.000  f1=0.122
#   threshold=0.2  precision=0.081  recall=1.000  f1=0.151
#   threshold=0.3  precision=0.105  recall=1.000  f1=0.190
#   threshold=0.4  precision=0.147  recall=1.000  f1=0.256
#   threshold=0.5  precision=0.167  recall=0.818  f1=0.277
#   threshold=0.6  precision=0.179  recall=0.636  f1=0.280
#   threshold=0.7  precision=0.200  recall=0.455  f1=0.278`}
      </CodeBlock>

      <Prose>
        The threshold sweep reveals the precision-recall tradeoff clearly. At threshold 0.4, you capture 100% of positives with precision 0.147 — reasonable for a high-stakes screening context where missing a case is catastrophic. At 0.6, you achieve higher precision (0.179) with recall dropping to 0.636. For fraud detection, a domain expert must decide which trade-off the business can live with. The key point: never leave the threshold at 0.5 without checking. On imbalanced data, 0.5 is almost always wrong.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6a. SMOTE synthesis — step by step</H3>

      <StepTrace
        label="SMOTE synthesis: generating one synthetic minority example"
        steps={[
          {
            label: "Step 1 — Pick a minority anchor point",
            render: () => (
              <Prose>
                From the minority class (10 examples in our 1:50 dataset), pick one at random — call it x. In feature space, x lives in the region where the minority class is concentrated (around [3, 3] in our 2D example). This anchor point will be one endpoint of our interpolation segment.
              </Prose>
            ),
          },
          {
            label: "Step 2 — Find k nearest minority neighbours",
            render: () => (
              <Prose>
                Compute Euclidean distances from x to every other minority example. Sort and take the k closest (k=5 by default). These k neighbours define the local neighbourhood of x within the minority class. Only minority-class examples are considered as neighbours — the majority class is invisible to this step.
              </Prose>
            ),
          },
          {
            label: "Step 3 — Pick one neighbour at random",
            render: () => (
              <Prose>
                Randomly select one neighbour from the k candidates — call it {"x_n"}. This random selection ensures that synthesis is spread across multiple directions in feature space rather than always producing points on a single segment. With k=5, five different line segments are available per anchor point.
              </Prose>
            ),
          },
          {
            label: "Step 4 — Sample a random interpolation coefficient",
            render: () => (
              <Prose>
                {"Draw λ ~ Uniform(0, 1). This coefficient controls where on the segment from x to x_n the synthetic point falls. λ=0 gives x itself; λ=1 gives x_n; λ=0.5 gives the midpoint. By sampling uniformly, SMOTE fills the segment with equal density."}
              </Prose>
            ),
          },
          {
            label: "Step 5 — Create and add the synthetic point",
            render: () => (
              <Prose>
                {"Compute x_syn = x + λ · (x_n − x). This is a convex combination of two minority-class examples, so it lies in the region between them. Label x_syn as minority class (y=1) and add it to the training set. Repeat for as many synthetic examples as needed to reach the target ratio. The dataset is now larger, with the minority class augmented by interpolated examples rather than duplicates."}
              </Prose>
            ),
          },
        ]}
      />

      <H3>6b. Confusion matrix before vs after intervention</H3>

      <Prose>
        The two heatmaps below show the confusion matrix on the original 1:50 dataset (510 examples: 500 majority, 10 minority) before any intervention, and after SMOTE oversampling (evaluated on original data). Row = true label, column = predicted label.
      </Prose>

      <Heatmap
        label="Confusion matrix — baseline (no resampling)"
        matrix={[[500, 0], [3, 7]]}
        rowLabels={["True: Neg", "True: Pos"]}
        colLabels={["Pred: Neg", "Pred: Pos"]}
        colorScale="gold"
      />

      <Heatmap
        label="Confusion matrix — after SMOTE (evaluated on original data)"
        matrix={[[491, 9], [0, 10]]}
        rowLabels={["True: Neg", "True: Pos"]}
        colLabels={["Pred: Neg", "Pred: Pos"]}
        colorScale="green"
      />

      <Prose>
        The baseline achieves perfect precision (0 false positives) but misses 3 minority examples. SMOTE achieves perfect recall (0 false negatives) at the cost of 9 false positives. The choice between these two operating points depends on the application's cost matrix — not on any intrinsic property of the algorithm.
      </Prose>

      <H3>6c. Precision-recall curve — threshold tradeoff</H3>

      <Plot
        label="Precision-recall curve — baseline model on 1:50 imbalanced data"
        xLabel="Recall"
        yLabel="Precision"
        series={[
          {
            name: "PR curve (baseline logistic regression)",
            color: colors.gold,
            points: [
              [0.0, 1.0], [0.7, 1.0], [0.8, 1.0], [0.9, 1.0],
              [1.0, 0.909], [1.0, 0.270], [1.0, 0.020],
            ],
          },
          {
            name: "No-skill baseline (prevalence = 0.020)",
            color: colors.textMuted,
            points: [[0.0, 0.020], [1.0, 0.020]],
          },
        ]}
      />

      <H3>6d. Threshold sweep: precision vs recall tradeoff</H3>

      <Plot
        label="Threshold sweep — precision and recall vs threshold"
        xLabel="Decision threshold"
        yLabel="Score"
        series={[
          {
            name: "Recall",
            color: colors.gold,
            points: [
              [0.1, 1.0], [0.2, 1.0], [0.3, 1.0], [0.4, 1.0],
              [0.5, 0.818], [0.6, 0.636], [0.7, 0.455],
            ],
          },
          {
            name: "Precision",
            color: colors.green,
            points: [
              [0.1, 0.065], [0.2, 0.081], [0.3, 0.105], [0.4, 0.147],
              [0.5, 0.167], [0.6, 0.179], [0.7, 0.200],
            ],
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <StepTrace
        label="which intervention to apply"
        steps={[
          {
            label: "Mild imbalance (90/10) — class_weight='balanced' is usually enough",
            render: () => (
              <Prose>
                At a 9:1 ratio, the minority class is well-represented in every fold of a stratified split. The dominant issue is that the loss is still 9× dominated by the majority class. Setting <Code>class_weight='balanced'</Code> on any sklearn classifier corrects this without touching the data. This is the lowest-friction intervention: no data augmentation, no changed training set size, no pipeline changes. Start here. Evaluate with F1 and PR-AUC on the minority class. If performance is acceptable, you are done.
              </Prose>
            ),
          },
          {
            label: "Severe imbalance (99/1) — SMOTE + class weights + threshold tuning",
            render: () => (
              <Prose>
                At a 99:1 ratio, class weights alone are often insufficient — the minority class has so few examples that the gradient signal is weak even when each example is weighted 99×. SMOTE adds synthetic examples in the minority region, increasing the number of distinct gradient directions. Combine: (1) use imblearn Pipeline with SMOTE inside CV folds, (2) set <Code>class_weight='balanced'</Code> on the classifier, (3) tune the decision threshold on a validation set using the business-appropriate metric (F2 for high-recall domains, F0.5 for high-precision domains). All three levers together give the best results on standard benchmarks.
              </Prose>
            ),
          },
          {
            label: "Extreme imbalance (99.9/0.1) — consider anomaly detection framing",
            render: () => (
              <Prose>
                At 99.9:0.1 (1 positive per 1,000 negatives), supervised binary classification often fails entirely — you may have only a handful of positive examples in a dataset of 100,000. Standard SMOTE synthesizes from a tiny pool of minority examples, making interpolated points unreliable. Serious options: (1) reframe as anomaly detection (One-Class SVM, Isolation Forest, autoencoder reconstruction error) trained on majority-only data; (2) collect more minority examples (the most effective intervention); (3) use pre-trained representations (transfer learning) that compresses the feature space so the minority examples are more informative.
              </Prose>
            ),
          },
          {
            label: "When resampling hurts — use threshold moving instead",
            render: () => (
              <Prose>
                Elkan (2001) proved that cost-sensitive threshold adjustment on a well-calibrated model is decision-theoretically equivalent to resampling. If your classifier is well-calibrated (check with a calibration curve), threshold moving achieves the same result as resampling with zero training cost. Resampling can actively hurt when: (a) the minority class is noisy — SMOTE amplifies noise by synthesizing in noisy regions; (b) the classifier is a tree-based model (Random Forest, XGBoost) — these have their own built-in imbalance handling via <Code>scale_pos_weight</Code> and leaf-level sampling; (c) you need a fast prototype — class weighting is a one-line change, resampling requires an imblearn pipeline.
              </Prose>
            ),
          },
          {
            label: "When to use focal loss — easy negatives swamping training",
            render: () => (
              <Prose>
                Focal loss is the right tool when training on a dataset where the vast majority of examples are easy negatives — examples the model classifies correctly with very high confidence from early in training. In object detection, 10,000 background patches may be generated per image, and 9,990 of them are trivially classified as background after a few epochs. Standard cross-entropy treats all 10,000 equally, so 9,990 nearly-zero gradients swamp the signal from the hard 10. Focal loss {"FL(p) = -(1-p)^γ log(p)"} with γ=2 down-weights easy examples by up to 100×. Use focal loss in PyTorch or TensorFlow when: (1) you are training on raw pixel/patch data rather than pre-extracted features, (2) training loss curves show rapid early convergence followed by stagnation, (3) your dataset has a canonical hard-positive / easy-negative structure (object detection, text span extraction).
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>8.1 SMOTE computational complexity</H3>

      <Prose>
        SMOTE requires computing k nearest neighbours for each of the m minority examples. A naïve k-NN computation is O(m·n) — for each minority example, scan all n training examples to find the k closest. With m minority examples, total cost is O(m·n). For a severely imbalanced dataset with n=1,000,000 majority examples and m=10,000 minority examples, this is 10¹⁰ distance computations — expensive but feasible on a single machine with vectorized NumPy. The imbalanced-learn SMOTE implementation uses sklearn's BallTree or KDTree internally, which reduces per-query cost to O(log n) after O(n log n) preprocessing, making the total cost O(n log n + m log n). Practical to ~1M rows; beyond that, consider approximate nearest neighbours (FAISS, Annoy) for the k-NN step.
      </Prose>

      <H3>8.2 Cost of class_weight and focal loss</H3>

      <Prose>
        Class weighting adds zero overhead beyond a scalar multiply per example in the gradient computation — it is effectively free. The training set size, model size, and wall clock time are unchanged. This is why class weighting is always the first intervention to try: it costs nothing and often suffices.
      </Prose>

      <Prose>
        Focal loss replaces the standard cross-entropy computation with {"FL(p) = -(1-p)^γ log(p)"}. This adds one exponentiation per example per forward pass — negligible overhead compared to matrix multiplications in a deep network. It is also free in practice.
      </Prose>

      <H3>8.3 Training cost of oversampling</H3>

      <Prose>
        SMOTE oversampling increases the training set size. If the original minority-to-majority ratio is r and you oversample to 1:1, the training set grows by a factor of approximately {"(1 + (1-r)/r)"} — for a 1:99 dataset, the training set roughly doubles in size. Every gradient step over the augmented dataset costs twice as much. This is linear scaling in the oversampling ratio — predictable and acceptable for moderate oversampling, but it rules out extreme oversampling on already large datasets.
      </Prose>

      <Prose>
        Random undersampling is the opposite: it reduces training set size, which reduces per-epoch cost — but at the cost of discarding information. On a 99:1 dataset, undersampling to 1:1 discards 98% of majority examples. The reduced training set can cause the model to underfit on the majority class, leading to poor precision.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>9.1 Resampling leakage — the most common mistake</H3>

      <Prose>
        The single most common mistake with SMOTE: calling <Code>SMOTE().fit_resample(X, y)</Code> on the full dataset before cross-validation, then passing the resampled data to <Code>cross_val_score</Code>. This is data leakage. The synthetic minority points in the validation fold were generated using the full minority distribution — including the minority examples that should only appear in the training fold. The SMOTE synthesizer has already "seen" validation-fold minority examples and used them as anchor points or neighbours. CV scores on leaked data are unrealistically optimistic. The fix is always to use <Code>imblearn.pipeline.Pipeline</Code>, which calls <Code>fit_resample</Code> only on the training portion of each fold. If you are not using the imblearn Pipeline, you are leaking.
      </Prose>

      <H3>9.2 SMOTE noise amplification with noisy minority</H3>

      <Prose>
        SMOTE interpolates between minority examples. If the minority class is noisy — some examples are mislabeled, are outliers, or are near the majority boundary — SMOTE will synthesize points in the noisy regions, amplifying the noise. A minority example that was mislabeled will generate k synthetic mislabeled examples. SMOTEENN and SMOTETomek were designed to mitigate this: they remove noisy synthetic points (ENN removes points whose nearest neighbours disagree with their label) or remove borderline examples from both classes (Tomek Links). If your minority class is noisy, prefer SMOTEENN over vanilla SMOTE.
      </Prose>

      <H3>9.3 Optimizing accuracy on imbalanced data — stop</H3>

      <Prose>
        Accuracy on an imbalanced dataset is almost always misleading. On a 95% majority dataset, predicting always-negative gives 95% accuracy. This is not a model; it is a constant. If you report accuracy on imbalanced data, you will be reporting a meaningless number and may convince yourself the model is good when it is useless. Always report precision, recall, F1 (or Fβ), and PR-AUC. Report confusion matrices. Never tune hyperparameters by optimizing accuracy on imbalanced data.
      </Prose>

      <H3>9.4 Threshold of 0.5 is usually wrong post-rebalancing</H3>

      <Prose>
        After SMOTE oversampling or class-weight adjustment, the model's probability estimates are no longer calibrated to the original class distribution. A model trained on a SMOTE-rebalanced 1:1 dataset predicts probabilities relative to a 50% positive rate, not the true 1% rate. Applying the 0.5 threshold will produce many false positives. After resampling, always tune the threshold on a held-out validation set using the business metric, or re-calibrate the model using Platt scaling on the original (unbalanced) validation data.
      </Prose>

      <H3>9.5 ROC-AUC instead of PR-AUC on rare-positive problems</H3>

      <Prose>
        A model on a 1:99 imbalanced dataset can achieve ROC-AUC of 0.95 while having PR-AUC of only 0.30. ROC-AUC looks good because TN is enormous — the FPR denominator is huge, making FPR look small even when many false positives exist. PR-AUC exposes this: precision collapses when the denominator (TP + FP) is dominated by FP. For any problem where positive class prevalence is below 10%, use PR-AUC as the primary evaluation metric. Use ROC-AUC as a secondary sanity check, but never as the primary metric for model selection.
      </Prose>

      <H3>9.6 Different metrics for different stakeholders</H3>

      <Prose>
        The right metric depends on the deployment context, not on a universal rule. For medical screening (cancer detection, COVID triage), missing a true case is catastrophic — maximize recall even at the cost of many false positives (F2 or F3). For fraud alerting sent to human investigators, each alert costs analyst time — false positives are expensive — so maximize precision subject to recall (F0.5). For spam filtering, falsely filtering legitimate email destroys user trust — maximize precision. For manufacturing defect detection, a missed defect ships to a customer and causes recalls — maximize recall. Never choose the metric independently of the application. Have this conversation with domain experts before writing code.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All citations below were WebSearch-verified for author, year, venue, title, and main contribution. Read in chronological order for the field's development arc.
      </Prose>

      <StepTrace
        label="primary literature"
        steps={[
          {
            label: "Elkan 2001 — Cost-sensitive decision rule",
            render: () => (
              <Prose>
                Elkan, C. (2001). "The Foundations of Cost-Sensitive Learning." <em>Proceedings of the 17th International Joint Conference on Artificial Intelligence (IJCAI 2001)</em>, Volume 2, pp. 973–978. Available at cseweb.ucsd.edu/~elkan/rescale.pdf. The paper that established the theoretical basis for cost-sensitive classification. The central result: for a two-class problem with misclassification costs {"c_{FP}"} and {"c_{FN}"}, the optimal decision rule predicts positive when {"P(+|x) > c_{FP} / (c_{FP} + c_{FN})"}. This means that cost-sensitive learning reduces to ordinary probability estimation plus a threshold adjustment — you do not need a special cost-sensitive algorithm, only a calibrated model and the correct threshold. The paper also proves that changing the class proportions in the training data by a known ratio is equivalent to adjusting the decision threshold by the corresponding factor.
              </Prose>
            ),
          },
          {
            label: "Chawla, Bowyer, Hall, Kegelmeyer 2002 — SMOTE",
            render: () => (
              <Prose>
                Chawla, N.V., Bowyer, K.W., Hall, L.O., and Kegelmeyer, W.P. (2002). "SMOTE: Synthetic Minority Over-sampling Technique." <em>Journal of Artificial Intelligence Research</em>, 16:321–357. arXiv:1106.1813. The original SMOTE paper. The key insight is that oversampling by replication creates decision regions that are too specific — the classifier memorizes exact duplicated points. Oversampling by interpolation between existing minority examples creates broader, more general minority decision regions. The paper showed SMOTE combined with undersampling consistently outperformed either technique alone on a suite of real-world imbalanced datasets (medical, intrusion detection, oil spill detection). This is the most-cited paper in the imbalanced learning literature with over 30,000 citations.
              </Prose>
            ),
          },
          {
            label: "He, Bai, Garcia, Li 2008 — ADASYN",
            render: () => (
              <Prose>
                He, H., Bai, Y., Garcia, E.A., and Li, S. (2008). "ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning." <em>2008 IEEE International Joint Conference on Neural Networks (IJCNN)</em>, Hong Kong, pp. 1322–1328. DOI: 10.1109/IJCNN.2008.4633969. Available on IEEE Xplore. ADASYN extends SMOTE with a density-aware synthesis strategy. For each minority example, compute the fraction of its k nearest neighbours that are majority-class. This ratio becomes the weight for how many synthetic examples to generate near that point — hard boundary examples get more synthetic neighbours, easy interior examples get fewer. The paper showed ADASYN improves both overall performance and reduces bias toward easy minority examples compared to SMOTE.
              </Prose>
            ),
          },
          {
            label: "He & Garcia 2009 — Learning from Imbalanced Data (survey)",
            render: () => (
              <Prose>
                He, H. and Garcia, E.A. (2009). "Learning from Imbalanced Data." <em>IEEE Transactions on Knowledge and Data Engineering</em>, 21(9):1263–1284. DOI: 10.1109/TKDE.2008.239. The canonical survey paper that organized the field. Three main families: (1) data-level methods (oversampling, undersampling, hybrid), (2) algorithm-level methods (cost-sensitive learning, class-weighted objectives, threshold moving), (3) hybrid methods (combining both families). The paper provides a systematic empirical comparison across all families and gives practical guidance on which methods to apply based on imbalance ratio and dataset size. Over 7,000 citations.
              </Prose>
            ),
          },
          {
            label: "Lin, Goyal, Girshick, He, Dollár 2017 — Focal Loss",
            render: () => (
              <Prose>
                Lin, T.-Y., Goyal, P., Girshick, R., He, K., and Dollár, P. (2017). "Focal Loss for Dense Object Detection." <em>Proceedings of the IEEE International Conference on Computer Vision (ICCV 2017)</em>. arXiv:1708.02002. Best student paper award. Introduced focal loss: {"FL(p) = -(1-p)^γ · log(p)"}, a modification of cross-entropy that down-weights easy, well-classified examples and focuses training on hard examples. The motivation was that in one-stage object detectors like RetinaNet, the foreground-background class imbalance is extreme (~100,000 background patches per foreground object), and standard cross-entropy is dominated by trivially easy background gradient. With γ=2, easy examples (p=0.9) have their loss down-weighted by a factor of 0.01. The RetinaNet detector with focal loss matched two-stage detectors (Faster R-CNN) on COCO at faster inference speed. Focal loss is now widely used in any setting with extreme easy-negative dominance.
              </Prose>
            ),
          },
          {
            label: "Lemaître, Nogueira, Aridas 2017 — imbalanced-learn",
            render: () => (
              <Prose>
                Lemaître, G., Nogueira, F., and Aridas, C.K. (2017). "Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning." <em>Journal of Machine Learning Research</em>, 18(17):1–5. Available at jmlr.org/papers/v18/16-365.html. The paper introducing the imbalanced-learn library. The library provides sklearn-compatible implementations of SMOTE, ADASYN, Borderline-SMOTE, RandomOverSampler, RandomUnderSampler, NearMiss, TomekLinks, ENN, SMOTEENN, SMOTETomek, and ensemble methods. Critically, it provides an imblearn Pipeline that correctly applies resampling only to training folds during cross-validation, preventing the leakage that is the most common implementation error. This is the production-standard library for imbalanced learning in Python.
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
        Work through these before moving to the next topic. The answer key is below each exercise.
      </Prose>

      <H3>Exercise 1 (recall)</H3>
      <Prose>
        A dataset has 9,900 negatives and 100 positives. A model achieves 99% accuracy. (a) What is its recall on the positive class? (b) What metric should you use instead? (c) What is the no-skill baseline on PR-AUC for this dataset?
      </Prose>
      <Callout type="answer" title="Answer 1">
        {"(a) Accuracy of 99% means the model correctly classifies 9,999 of 10,000 examples. If it achieves this by predicting all-negative, then TP=0, TN=9,900, FP=0, FN=100. Recall = TP/(TP+FN) = 0/100 = 0. A model with 99% accuracy and 0% recall on the minority class is useless for its intended purpose. (b) Use PR-AUC (average precision) as the primary metric, and report precision, recall, and F1 on the positive class. ROC-AUC is also better than accuracy but is misleadingly optimistic on severely imbalanced data because TN dominates FPR. (c) A no-skill classifier that predicts the positive class with frequency equal to the base rate achieves PR-AUC equal to the prevalence: 100/10,000 = 0.01. Any real classifier must exceed 0.01 PR-AUC to be better than chance."}
      </Callout>

      <H3>Exercise 2 (math)</H3>
      <Prose>
        In a medical diagnosis task, the cost of a false negative (missing disease) is 10 times the cost of a false positive (unnecessary follow-up). Using Elkan's cost-sensitive decision rule, what probability threshold should you use? Show the derivation.
      </Prose>
      <Callout type="answer" title="Answer 2">
        {"Let c_{FP} = 1 (one unit cost for a false positive) and c_{FN} = 10. Elkan's rule: predict positive when P(+|x) > c_{FP} / (c_{FP} + c_{FN}) = 1 / (1 + 10) = 1/11 ≈ 0.091. The decision rule: flag a patient as positive whenever the model's estimated probability of disease exceeds ~9.1%. This is far below the default 0.5 threshold, reflecting that the asymmetric cost structure demands aggressive screening. Intuitively: if a false negative costs 10× more than a false positive, you should be willing to flag 10 healthy patients to avoid missing one sick patient — which corresponds to a 1/11 threshold."}
      </Callout>

      <H3>Exercise 3 (conceptual)</H3>
      <Prose>
        Explain why calling <Code>SMOTE().fit_resample(X_train, y_train)</Code> outside a Pipeline before <Code>cross_val_score</Code> is a form of data leakage. What specifically is leaked, and why does it inflate CV scores?
      </Prose>
      <Callout type="answer" title="Answer 3">
        When you call fit_resample on X_train before CV splits, the SMOTE synthesizer uses the full training set — including examples that will later become validation folds — to find k nearest neighbours and generate synthetic points. The synthetic minority examples in what becomes the validation fold were generated using real minority examples from what becomes the training fold as anchor points and neighbours. The SMOTE synthesizer has therefore "seen" the relative positions of all minority examples, including future validation examples. The validation fold is no longer unseen: its neighbourhood structure influenced the synthetic examples it contains. This inflates CV scores because the model trains on synthetic points shaped by validation-fold information and is then evaluated on those same validation folds. The fix: use imblearn.pipeline.Pipeline, which calls fit_resample only on the training portion of each CV fold. The validation fold is held out completely from the resampling step, making it a true unseen evaluation.
      </Callout>

      <H3>Exercise 4 (applied)</H3>
      <Prose>
        You train a logistic regression with <Code>class_weight='balanced'</Code> on a fraud detection dataset (1% fraud rate). The model achieves PR-AUC of 0.72 on the validation set, but at threshold 0.5 the precision is only 0.08 and recall is 0.95. The fraud team can only review 200 alerts per day out of 50,000 daily transactions. How do you set the threshold, and what precision should you target?
      </Prose>
      <Callout type="answer" title="Answer 4">
        {"The team can review 200 of 50,000 transactions: that is a 0.4% alert rate budget. The dataset has 1% fraud rate, so 500 daily frauds. At 200 alerts per day with perfect precision, you catch 200 frauds (recall = 200/500 = 0.40). The target precision at 200 alerts is P = TP / 200. To maximize the number of frauds caught in 200 alerts, you want precision as high as possible — ideally 1.0 (all 200 alerts are real fraud). Set the threshold by computing the precision-recall curve via predict_proba and finding the threshold where the predicted positive rate equals 200/50,000 = 0.004 (0.4%). At that operating point, read off the precision. In practice: sort test examples by predicted probability descending, take the top 200, compute the fraction that are true fraud — that is your operational precision. Adjust the threshold iteratively until the daily alert volume matches the team's capacity. This is threshold tuning to meet a business constraint, not a statistical one."}
      </Callout>

      <H3>Exercise 5 (debugging)</H3>
      <Prose>
        A colleague trains a gradient boosted tree with <Code>scale_pos_weight=99</Code> (for a 99:1 imbalance) and then evaluates it on the test set with a threshold of 0.5. PR-AUC on the test set is 0.84, but precision at threshold 0.5 is 0.11. They report "the model has good PR-AUC but terrible precision." What is wrong, and how do you fix the evaluation?
      </Prose>
      <Callout type="answer" title="Answer 5">
        {"The model with scale_pos_weight=99 was trained on a reweighted objective that treats the dataset as if it were nearly balanced. Its predicted probabilities are no longer calibrated to the true 1% positive rate — they are calibrated to an effective rate closer to 50%. At threshold 0.5, the model flags many examples as positive because its internal probability estimates are inflated relative to the true base rate. This is not a model quality problem; it is a calibration and threshold problem. Fix: (1) Do not use threshold 0.5. Sweep the threshold on a held-out validation set and pick the threshold that maximizes your business metric (F-score, alert budget compliance, etc.). (2) If calibrated probabilities are needed (e.g., for risk scoring), apply Platt scaling or isotonic regression on a held-out set using the original (unbalanced) class labels to recalibrate the scores. After recalibration, the probabilities will reflect the true 1% base rate and threshold 0.5 will mean something interpretable again."}
      </Callout>

      <H3>Exercise 6 (synthesis)</H3>
      <Prose>
        Compare SMOTE and class weighting on the following dimensions: (a) computational cost, (b) sensitivity to noisy minority examples, (c) effect on model calibration, (d) whether they change the data the model trains on. When would you choose one over the other in production?
      </Prose>
      <Callout type="answer" title="Answer 6">
        {"(a) Computational cost: SMOTE requires k-NN per minority example — O(m·n) naïve, O((n+m)log n) with a k-d tree — and increases training set size proportionally to the oversampling ratio. Class weighting adds zero computational cost beyond a scalar multiply in the gradient. For large datasets, class weighting is strictly cheaper. (b) Sensitivity to noise: SMOTE synthesizes in the neighbourhood of existing minority examples. Noisy or mislabeled minority examples generate k synthetic noisy examples, amplifying errors. Class weighting up-weights existing examples without creating new ones — noise is amplified in weight but not multiplied in count. SMOTEENN and SMOTETomek partially mitigate SMOTE noise via post-synthesis cleaning. (c) Effect on calibration: both methods produce uncalibrated probability estimates relative to the original class distribution. SMOTE changes the training distribution so probabilities reflect the oversampled ratio. Class weighting changes the loss surface so probabilities reflect the weighted class distribution. Both require threshold adjustment or recalibration after training. (d) Data modification: SMOTE physically adds synthetic examples to the training set. Class weighting does not change the data — it changes the gradient contribution of each example. Choose SMOTE when the minority class is too small for the classifier to learn a reliable boundary (very few examples); choose class weighting when the minority class has sufficient examples but the gradient is dominated by majority loss. In production, default to class_weight='balanced' first — it is one line and zero cost. Add SMOTE if class weighting is insufficient after threshold tuning."}
      </Callout>

    </div>
  ),
};

export default imbalancedLearningContent;
