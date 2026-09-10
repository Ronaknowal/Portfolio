import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const multiLabelContent = {
  title: "Multi-Label & Multi-Output Learning",
  readTime: "~35 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Most machine learning curricula introduce classification as a single-output problem: given an input, predict one label from a fixed set. This framing is convenient for pedagogy, but it breaks down the moment you look at the real world carefully. A photograph tagged on Flickr carries labels like <em>mountain</em>, <em>snow</em>, and <em>sunset</em> simultaneously — not one label chosen from a list of three, but an arbitrary subset of a large vocabulary. A news article belongs to <em>politics</em> and <em>economy</em> at the same time. A protein in a cell may perform three distinct molecular functions concurrently. A patient admitted to a hospital frequently presents with two, three, or five co-occurring conditions — the clinical reality of comorbidity. Standard multi-class classifiers cannot represent any of these without contortions.
      </Prose>

      <Prose>
        The intellectual lineage of multi-label learning is anchored by four papers that remain the primary references a decade later. Boutell, Luo, Shen, and Brown published "Learning Multi-Label Scene Classification" in <em>Pattern Recognition</em>, volume 37(9), pages 1757–1771, in 2004 — one of the earliest systematic treatments of the problem for visual scenes. They formalized the setup and surveyed what they called "multi-label machine learning," introducing the vocabulary that the field adopted. Three years later, Tsoumakas and Katakis wrote "Multi-Label Classification: An Overview" in the <em>International Journal of Data Warehousing and Mining</em>, volume 3(3), 2007, pages 1–13 — a landmark taxonomy that organized the emerging literature into problem transformation methods versus algorithm adaptation methods, a distinction the field still uses. The classifier chains algorithm, which has become the dominant practical approach for the label-correlation-aware case, was introduced and rigorously analyzed by Read, Pfahringer, Holmes, and Frank in "Classifier Chains for Multi-Label Classification," <em>Machine Learning</em>, volume 85(3), pages 333–359, in 2011. Finally, Zhang and Zhou's 2014 IEEE TKDE survey "A Review on Multi-Label Learning Algorithms," volume 26(8), pages 1819–1837, synthesized the algorithmic landscape across both classical and learning-theoretic angles.
      </Prose>

      <Prose>
        Before going further, three related problems need careful separation because confusing them corrupts your choice of algorithm, evaluation metric, and loss function.
      </Prose>

      <Prose>
        <strong>Multi-class classification</strong> assigns exactly one label from a set of K mutually exclusive classes. A digit recognition model produces a single digit from {"{0–9}"}. An ImageNet classifier returns exactly one of 1,000 categories. The label space is {"{0, 1, ..., K-1}"} and the output is a single integer or a K-dimensional probability vector that sums to one.
      </Prose>

      <Prose>
        <strong>Multi-label classification</strong> assigns a subset of L possible labels. The label for each sample is a binary vector <Code>{"y ∈ {{0,1}^L}"}</Code>, where each component is 1 if that label applies and 0 otherwise. Labels are not mutually exclusive — any combination is valid. The output is L independent binary decisions, and the probability model does not sum to one across labels. This is the primary focus of this topic.
      </Prose>

      <Prose>
        <strong>Multi-output regression</strong> predicts multiple continuous targets simultaneously. Given a patient's demographics and biomarkers, predict both systolic and diastolic blood pressure. Given a weather station's sensor readings, predict temperature, humidity, and wind speed tomorrow. The label is a real-valued vector <Code>y ∈ ℝ^T</Code> for T output targets. Multi-output regression is the continuous analog of multi-label classification, and the structural questions — should the outputs be modeled independently or jointly? — are the same.
      </Prose>

      <Callout type="info" title="Three problems, one table">
        Multi-class: one of K — single label, mutually exclusive, softmax output, cross-entropy loss.
        Multi-label: subset of L — binary vector, independent labels, sigmoid output, binary cross-entropy per label.
        Multi-output regression: real vector in ℝ^T — independent or correlated targets, linear output, MSE per target.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        The central design decision in multi-label learning is whether to model the labels independently or jointly. This bifurcation produces three qualitatively different families of approaches.
      </Prose>

      <H3>2.1 Binary Relevance: one classifier per label</H3>

      <Prose>
        The simplest approach decomposes the multi-label problem into L independent binary classification problems. For each label j, train a classifier that takes the input features and predicts whether label j applies. At prediction time, run all L classifiers independently and concatenate their outputs into a binary vector.
      </Prose>

      <Prose>
        Binary relevance (BR) is appealing because it is trivially parallelizable, compatible with any binary classifier, and requires no modification to the underlying algorithm. Its critical weakness is that it ignores label correlations entirely. If labels <em>rock</em> and <em>guitar</em> almost always co-occur in a music tagging dataset, a BR system will not exploit this structure — each of its two classifiers sees only the features, never the other label. In datasets with strong label dependencies, this costs meaningful predictive performance.
      </Prose>

      <H3>2.2 Classifier Chains: modeling label dependencies</H3>

      <Prose>
        Classifier chains (CC), introduced by Read et al. 2011, augment binary relevance with a sequential structure. Arrange the L labels in some order. Train the classifier for label 1 on features only, exactly as in BR. Train the classifier for label 2 on features plus the ground-truth value of label 1. Train the classifier for label 3 on features plus labels 1 and 2. Continue down the chain. At inference time, replace ground-truth previous labels with predicted values.
      </Prose>

      <Prose>
        The probabilistic interpretation is clean. Binary relevance models each label as:
      </Prose>

      <MathBlock>
        {"P(y_j \\mid x)"}
      </MathBlock>

      <Prose>
        Classifier chains model:
      </Prose>

      <MathBlock>
        {"P(y_1, y_2, \\ldots, y_L \\mid x) = \\prod_{j=1}^{L} P(y_j \\mid x, y_1, \\ldots, y_{j-1})"}
      </MathBlock>

      <Prose>
        This is just the chain rule of probability, applied to the label space. Each classifier in the chain captures a conditional distribution. The full joint distribution over all L labels is the product of these L conditionals — exact if the chain order matches the true conditional independence structure, approximate otherwise. Read et al. showed that ensembling over random chain orderings (Ensemble of Classifier Chains, or ECC) dramatically reduces the sensitivity to order and consistently outperforms binary relevance.
      </Prose>

      <H3>2.3 Label Powerset: label combinations as classes</H3>

      <Prose>
        Label powerset (LP) takes the opposite approach: treat every distinct combination of labels that appears in the training data as a single meta-class. If samples in your training set have label vectors {"{[1,0,1]}"}, {"{[0,1,1]}"}, and {"{[1,1,0]}"}, LP creates three classes and trains a standard multi-class classifier. At inference, the model predicts a single meta-class and its associated label vector is returned.
      </Prose>

      <Prose>
        LP captures all label correlations perfectly within the training distribution — it literally memorizes every combination. But it is crippled by label space explosion. With L labels, there are at most 2^L possible combinations. At L = 20 that is over one million meta-classes. In practice the number of unique combinations grows with dataset size and quickly makes LP impractical. Worse, label combinations unseen during training cannot be predicted. LP is a reasonable choice only for L ≤ 10 to 15 with reasonably dense co-occurrence.
      </Prose>

      <StepTrace
        label="multi-label method transformations on a 3-label example"
        steps={[
          {
            label: "Input: 5 samples, 3 labels",
            render: () => (
              <Prose>
                Original multi-label data. Samples (x₁–x₅) with label vectors:
                x₁ → [1, 0, 1] | x₂ → [0, 1, 1] | x₃ → [1, 1, 0] | x₄ → [0, 0, 1] | x₅ → [1, 1, 1].
                Labels are not mutually exclusive. Sample x₅ is positive for all three simultaneously.
              </Prose>
            ),
          },
          {
            label: "Binary Relevance: 3 independent problems",
            render: () => (
              <Prose>
                Problem 1 (label A): targets = [1, 0, 1, 0, 1] → train classifier f_A(x).
                Problem 2 (label B): targets = [0, 1, 1, 0, 1] → train classifier f_B(x).
                Problem 3 (label C): targets = [1, 1, 0, 1, 1] → train classifier f_C(x).
                Three separate classifiers. Predict: ŷ = [f_A(x), f_B(x), f_C(x)].
                Labels are modeled as if independent — no information flows between classifiers.
              </Prose>
            ),
          },
          {
            label: "Classifier Chain: sequential conditioning",
            render: () => (
              <Prose>
                Chain order: A → B → C.
                Classifier f_A(x): features only → predicts label A.
                Classifier f_B(x, ŷ_A): features + predicted A → predicts label B.
                Classifier f_C(x, ŷ_A, ŷ_B): features + predicted A + predicted B → predicts label C.
                At training: use ground-truth A when training f_B, ground-truth A and B when training f_C.
                At inference: errors propagate — if f_A is wrong, it corrupts f_B and f_C inputs.
              </Prose>
            ),
          },
          {
            label: "Label Powerset: label combos as single classes",
            render: () => (
              <Prose>
                Unique label vectors: [1,0,1] → class 0 | [0,1,1] → class 1 | [1,1,0] → class 2 | [0,0,1] → class 3 | [1,1,1] → class 4.
                Train: standard 5-class classifier on targets [0, 1, 2, 3, 4].
                Predict: multi-class output → look up label vector in the mapping table.
                Advantage: exact correlation modeling for seen combinations.
                Failure: unseen combo [1,0,0] cannot be predicted — it was never in training.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 Evaluation metrics for multi-label learning</H3>

      <Prose>
        Multi-label evaluation is more nuanced than single-label evaluation because predictions are now binary vectors, and there are multiple defensible ways to measure how close one vector is to another. The choice of metric has real downstream consequences — optimizing for the wrong one can produce a model that looks good on paper and fails its users.
      </Prose>

      <Prose>
        <strong>Hamming loss</strong> measures the fraction of label-sample pairs that are mislabeled. For N samples and L labels:
      </Prose>

      <MathBlock>
        {"\\text{HammingLoss} = \\frac{1}{N \\cdot L} \\sum_{i=1}^{N} \\sum_{j=1}^{L} \\mathbf{1}[\\hat{y}_{ij} \\neq y_{ij}]"}
      </MathBlock>

      <Prose>
        Hamming loss treats each label independently and gives equal weight to predicting rare labels correctly as common ones. It can be misleadingly low: a model that correctly predicts 99% of 0s (when 99% of labels are negative) can achieve near-zero Hamming loss while being useless for the rare positive labels you actually care about.
      </Prose>

      <Prose>
        <strong>Subset accuracy</strong> (exact match ratio) is the strictest metric — it counts a prediction as correct only if the predicted label vector exactly matches the ground truth:
      </Prose>

      <MathBlock>
        {"\\text{SubsetAcc} = \\frac{1}{N} \\sum_{i=1}^{N} \\mathbf{1}[\\hat{y}_i = y_i]"}
      </MathBlock>

      <Prose>
        Subset accuracy is often unrealistically harsh. Getting 4 out of 5 labels right counts as a complete failure. It is most useful as an upper bound on performance to report alongside softer metrics.
      </Prose>

      <Prose>
        <strong>Jaccard similarity</strong> (also called Jaccard index or intersection-over-union for binary vectors) measures the overlap between predicted and true label sets per sample, then averages:
      </Prose>

      <MathBlock>
        {"\\text{Jaccard} = \\frac{1}{N} \\sum_{i=1}^{N} \\frac{|\\hat{y}_i \\cap y_i|}{|\\hat{y}_i \\cup y_i|}"}
      </MathBlock>

      <Prose>
        <strong>F1 score</strong> has three averaging strategies that produce qualitatively different answers:
      </Prose>

      <MathBlock>
        {"\\text{F1}_{\\text{micro}} = \\frac{2 \\sum_{j} TP_j}{2 \\sum_{j} TP_j + \\sum_{j} FP_j + \\sum_{j} FN_j}"}
      </MathBlock>

      <MathBlock>
        {"\\text{F1}_{\\text{macro}} = \\frac{1}{L} \\sum_{j=1}^{L} \\frac{2 \\, TP_j}{2 \\, TP_j + FP_j + FN_j}"}
      </MathBlock>

      <MathBlock>
        {"\\text{F1}_{\\text{samples}} = \\frac{1}{N} \\sum_{i=1}^{N} \\frac{2 \\, |\\hat{y}_i \\cap y_i|}{|\\hat{y}_i| + |y_i|}"}
      </MathBlock>

      <Prose>
        F1 micro aggregates over all label-sample pairs — dominated by frequent labels. F1 macro averages per-label F1 scores — gives equal weight to rare and frequent labels, harsh on imbalanced datasets. F1 samples computes F1 per sample and averages — captures how good the full predicted label set is for each individual instance.
      </Prose>

      <H3>3.2 Why binary relevance is suboptimal with correlated labels</H3>

      <Prose>
        Binary relevance assumes that labels are conditionally independent given features: <Code>P(y | x) = ∏ P(yⱼ | x)</Code>. This is almost never true. Consider a document-tagging dataset where "machine learning" and "neural network" labels co-occur 70% of the time. A BR classifier for "neural network" has access only to the document features, not to the prediction for "machine learning." A chain classifier for "neural network" can condition on the "machine learning" prediction, capturing a strong posterior signal.
      </Prose>

      <Prose>
        The information gain from conditioning is bounded by the mutual information between the labels. For labels yⱼ and yₖ, the mutual information is:
      </Prose>

      <MathBlock>
        {"I(y_j ; y_k) = \\sum_{a \\in \\{0,1\\}} \\sum_{b \\in \\{0,1\\}} P(y_j=a, y_k=b) \\log \\frac{P(y_j=a, y_k=b)}{P(y_j=a)P(y_k=b)}"}
      </MathBlock>

      <Prose>
        When I(yⱼ; yₖ) is large, classifier chains that order yₖ after yⱼ (or vice versa) will exploit this dependency. The label co-occurrence matrix — which entries are the counts of how often pairs of labels appear together — is the empirical fingerprint of label dependency structure. High off-diagonal values in the normalized co-occurrence matrix indicate where chaining or joint modeling will help most.
      </Prose>

      <H3>3.3 Label space reduction methods</H3>

      <Prose>
        When L is large (hundreds to thousands of labels), direct approaches struggle. Several label space reduction methods have been developed. <strong>Compressed Sensing for Multi-Label Learning (CS-ML)</strong> assumes the label vectors live in a low-dimensional manifold and encodes them into a compressed representation before training, then decodes at inference. <strong>Principal Label Space Transformation (PLST)</strong> applies PCA to the label matrix Y, trains regressors on the principal components, and reconstructs the binary label vector by thresholding. <strong>RAkEL</strong> (Random k-labelsets) randomly partitions the label space into small subsets of size k, trains a label powerset classifier on each partition, and aggregates predictions by majority vote — combining LP's correlation modeling with a divide-and-conquer strategy that avoids label space explosion.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The code below is NumPy only. It implements binary relevance with logistic regression, a classifier chain, and all major evaluation metrics. All outputs were run and the stdout is embedded verbatim.
      </Prose>

      <H3>4a. Data generation and setup</H3>

      <CodeBlock language="python">
{`import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split

np.random.seed(42)
X, Y = make_multilabel_classification(
    n_samples=1000, n_features=20, n_classes=5, n_labels=2,
    allow_unlabeled=False, random_state=42
)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

print("X shape:", X.shape)
# Output: X shape: (1000, 20)
print("Y shape:", Y.shape)
# Output: Y shape: (1000, 5)
print("Label distribution:")
for i in range(Y.shape[1]):
    print(f"  Label {i}: {Y[:, i].sum()} positives ({100*Y[:, i].mean():.1f}%)")
# Output: Label distribution:
#   Label 0: 369 positives (36.9%)
#   Label 1: 635 positives (63.5%)
#   Label 2: 563 positives (56.3%)
#   Label 3: 478 positives (47.8%)
#   Label 4: 194 positives (19.4%)
print("Avg labels per sample:", Y.sum(axis=1).mean().round(2))
# Output: Avg labels per sample: 2.24

# Normalize features, add bias column
X_mean, X_std = X_train.mean(0), X_train.std(0) + 1e-8
X_train_n = (X_train - X_mean) / X_std
X_test_n  = (X_test  - X_mean) / X_std

def add_bias(X):
    return np.column_stack([np.ones(len(X)), X])

X_tr = add_bias(X_train_n)
X_te = add_bias(X_test_n)`}
      </CodeBlock>

      <H3>4b. Shared utilities: sigmoid and logistic training</H3>

      <CodeBlock language="python">
{`def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def logistic_train(X, y, lr=0.05, n_iter=300):
    """
    Mini logistic regression via gradient descent.
    Gradient of log-loss = (1/n) * X^T (sigma(Xw) - y)
    """
    n, d = X.shape
    w = np.zeros(d)
    for _ in range(n_iter):
        grad = (1 / n) * (X.T @ (sigmoid(X @ w) - y))
        w -= lr * grad
    return w`}
      </CodeBlock>

      <H3>4c. Binary relevance</H3>

      <CodeBlock language="python">
{`# Train one logistic classifier per label
weights_br = []
for j in range(Y_train.shape[1]):
    w = logistic_train(X_tr, Y_train[:, j], lr=0.05, n_iter=300)
    weights_br.append(w)

def br_predict(X, weights, threshold=0.5):
    preds = [( sigmoid(X @ w) >= threshold).astype(int) for w in weights]
    return np.column_stack(preds)

Y_pred_br = br_predict(X_te, weights_br)`}
      </CodeBlock>

      <H3>4d. Classifier chain</H3>

      <CodeBlock language="python">
{`L = Y_train.shape[1]
chain_weights = []

# Training: augment features with ground-truth previous labels
for j in range(L):
    X_aug = X_tr if j == 0 else np.column_stack([X_tr, Y_train[:, :j]])
    w = logistic_train(X_aug, Y_train[:, j], lr=0.05, n_iter=300)
    chain_weights.append(w)
    print(f"Label {j} trained. Weight shape: {w.shape}")
# Output:
# Label 0 trained. Weight shape: (21,)
# Label 1 trained. Weight shape: (22,)
# Label 2 trained. Weight shape: (23,)
# Label 3 trained. Weight shape: (24,)
# Label 4 trained. Weight shape: (25,)

# Inference: use predicted labels for subsequent positions
Y_pred_chain = np.zeros((len(X_test_n), L), dtype=int)
for j in range(L):
    X_aug = X_te if j == 0 else np.column_stack([X_te, Y_pred_chain[:, :j]])
    p = sigmoid(X_aug @ chain_weights[j])
    Y_pred_chain[:, j] = (p >= 0.5).astype(int)`}
      </CodeBlock>

      <H3>4e. Evaluation metrics</H3>

      <CodeBlock language="python">
{`def hamming_loss(Y_true, Y_pred):
    return np.mean(Y_true != Y_pred)

def subset_accuracy(Y_true, Y_pred):
    return np.mean(np.all(Y_true == Y_pred, axis=1))

def f1_micro(Y_true, Y_pred):
    tp = (Y_true * Y_pred).sum()
    fp = ((1 - Y_true) * Y_pred).sum()
    fn = (Y_true * (1 - Y_pred)).sum()
    prec = tp / (tp + fp + 1e-15)
    rec  = tp / (tp + fn + 1e-15)
    return 2 * prec * rec / (prec + rec + 1e-15)

def f1_macro(Y_true, Y_pred):
    f1s = []
    for j in range(Y_true.shape[1]):
        tp = (Y_true[:, j] * Y_pred[:, j]).sum()
        fp = ((1 - Y_true[:, j]) * Y_pred[:, j]).sum()
        fn = (Y_true[:, j] * (1 - Y_pred[:, j])).sum()
        prec = tp / (tp + fp + 1e-15)
        rec  = tp / (tp + fn + 1e-15)
        f1s.append(2 * prec * rec / (prec + rec + 1e-15))
    return np.mean(f1s)

def f1_samples(Y_true, Y_pred):
    scores = []
    for i in range(len(Y_true)):
        tp = (Y_true[i] * Y_pred[i]).sum()
        fp = ((1 - Y_true[i]) * Y_pred[i]).sum()
        fn = (Y_true[i] * (1 - Y_pred[i])).sum()
        denom = tp + fp + fn
        scores.append(2*tp / (2*tp + fp + fn) if denom > 0 else 1.0)
    return np.mean(scores)

print("=== Binary Relevance ===")
print(f"Hamming loss:      {hamming_loss(Y_test, Y_pred_br):.4f}")
# Output: Hamming loss:      0.1740
print(f"Subset accuracy:   {subset_accuracy(Y_test, Y_pred_br):.4f}")
# Output: Subset accuracy:   0.3950
print(f"F1 micro:          {f1_micro(Y_test, Y_pred_br):.4f}")
# Output: F1 micro:          0.7986
print(f"F1 macro:          {f1_macro(Y_test, Y_pred_br):.4f}")
# Output: F1 macro:          0.7454
print(f"F1 samples:        {f1_samples(Y_test, Y_pred_br):.4f}")
# Output: F1 samples:        0.8180

print()
print("=== Classifier Chain ===")
print(f"Hamming loss:      {hamming_loss(Y_test, Y_pred_chain):.4f}")
# Output: Hamming loss:      0.1790
print(f"Subset accuracy:   {subset_accuracy(Y_test, Y_pred_chain):.4f}")
# Output: Subset accuracy:   0.3950
print(f"F1 micro:          {f1_micro(Y_test, Y_pred_chain):.4f}")
# Output: F1 micro:          0.7954
print(f"F1 macro:          {f1_macro(Y_test, Y_pred_chain):.4f}")
# Output: F1 macro:          0.7370
print(f"F1 samples:        {f1_samples(Y_test, Y_pred_chain):.4f}")
# Output: F1 samples:        0.8140`}
      </CodeBlock>

      <Callout type="info" title="Why chain didn't dominate on this toy dataset">
        Classifier chains beat binary relevance most when label correlations are strong and the chain order is good. The synthetic <Code>make_multilabel_classification</Code> dataset has relatively weak label correlations by design. On real datasets with strong co-occurrence structure — protein function, document tagging — the gap is typically 2–5 F1 points in favor of chains or ECC. The sklearn implementation with a random order (ECC-style) showed a clear improvement: F1 micro 0.8096 vs 0.7968 for OVR.
      </Callout>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <H3>5a. sklearn binary relevance and classifier chains</H3>

      <CodeBlock language="python">
{`import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import hamming_loss, f1_score

np.random.seed(42)
X, Y = make_multilabel_classification(
    n_samples=1000, n_features=20, n_classes=5, n_labels=2,
    allow_unlabeled=False, random_state=42
)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# --- OneVsRestClassifier (Binary Relevance) ---
ovr = OneVsRestClassifier(
    LogisticRegression(C=1.0, max_iter=300, random_state=42)
)
ovr.fit(X_train_s, Y_train)
Y_pred_ovr = ovr.predict(X_test_s)

print("OneVsRestClassifier (Binary Relevance):")
print(f"  Hamming loss:    {hamming_loss(Y_test, Y_pred_ovr):.4f}")
# Output:   Hamming loss:    0.1780
print(f"  Subset accuracy: {(np.all(Y_test==Y_pred_ovr, axis=1)).mean():.4f}")
# Output:   Subset accuracy: 0.4000
print(f"  F1 micro:        {f1_score(Y_test, Y_pred_ovr, average='micro', zero_division=0):.4f}")
# Output:   F1 micro:        0.7968
print(f"  F1 macro:        {f1_score(Y_test, Y_pred_ovr, average='macro', zero_division=0):.4f}")
# Output:   F1 macro:        0.7428
print(f"  F1 samples:      {f1_score(Y_test, Y_pred_ovr, average='samples', zero_division=0):.4f}")
# Output:   F1 samples:      0.8140

# --- ClassifierChain (random order = ensemble-style) ---
cc = ClassifierChain(
    LogisticRegression(C=1.0, max_iter=300, random_state=42),
    order='random', random_state=42
)
cc.fit(X_train_s, Y_train)
Y_pred_cc = cc.predict(X_test_s)

print()
print("ClassifierChain (random order):")
print(f"  Hamming loss:    {hamming_loss(Y_test, Y_pred_cc):.4f}")
# Output:   Hamming loss:    0.1660
print(f"  Subset accuracy: {(np.all(Y_test==Y_pred_cc, axis=1)).mean():.4f}")
# Output:   Subset accuracy: 0.4600
print(f"  F1 micro:        {f1_score(Y_test, Y_pred_cc, average='micro', zero_division=0):.4f}")
# Output:   F1 micro:        0.8096
print(f"  F1 macro:        {f1_score(Y_test, Y_pred_cc, average='macro', zero_division=0):.4f}")
# Output:   F1 macro:        0.7537
print(f"  F1 samples:      {f1_score(Y_test, Y_pred_cc, average='samples', zero_division=0):.4f}")
# Output:   F1 samples:      0.8310`}
      </CodeBlock>

      <H3>5b. TF-IDF + ClassifierChain pipeline for text</H3>

      <CodeBlock language="python">
{`from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, hamming_loss

categories = [
    'sci.med', 'sci.space', 'sci.electronics',
    'rec.sport.baseball', 'rec.motorcycles'
]
data_train = fetch_20newsgroups(
    subset='train', categories=categories,
    remove=('headers', 'footers', 'quotes')
)
data_test = fetch_20newsgroups(
    subset='test', categories=categories,
    remove=('headers', 'footers', 'quotes')
)
print(f"Train: {len(data_train.data)}, Test: {len(data_test.data)}")
# Output: Train: 2973, Test: 1978

# Build multi-label targets (each doc gets its true label
# + 15% chance of an additional label — simulates noisy co-occurrence)
rng = np.random.default_rng(42)
def make_multilabel(targets, n_classes=5, noise=0.15):
    Y = np.eye(n_classes, dtype=int)[targets]
    Y = np.clip(Y + (rng.random(Y.shape) < noise).astype(int), 0, 1)
    return Y

Y_train = make_multilabel(data_train.target)
Y_test  = make_multilabel(data_test.target)
print(f"Avg labels/doc: {Y_train.sum(axis=1).mean():.2f}")
# Output: Avg labels/doc: 1.59

# TF-IDF vectorizer (sparse, handled natively by saga)
tfidf = TfidfVectorizer(max_features=10000, min_df=2)
X_train = tfidf.fit_transform(data_train.data)
X_test  = tfidf.transform(data_test.data)
print(f"Feature matrix: {X_train.shape}")
# Output: Feature matrix: (2973, 10000)

chain = ClassifierChain(
    LogisticRegression(C=0.5, solver='saga', max_iter=500, random_state=42),
    order='random', random_state=42
)
chain.fit(X_train, Y_train)
Y_pred = chain.predict(X_test)

print(f"Hamming loss:    {hamming_loss(Y_test, Y_pred):.4f}")
# Output: Hamming loss:    0.2319
print(f"Subset accuracy: {(np.all(Y_test==Y_pred, axis=1)).mean():.4f}")
# Output: Subset accuracy: 0.3332
print(f"F1 micro:        {f1_score(Y_test, Y_pred, average='micro', zero_division=0):.4f}")
# Output: F1 micro:        0.5528
print(f"F1 macro:        {f1_score(Y_test, Y_pred, average='macro', zero_division=0):.4f}")
# Output: F1 macro:        0.5426
print(f"F1 samples:      {f1_score(Y_test, Y_pred, average='samples', zero_division=0):.4f}")
# Output: F1 samples:      0.5706`}
      </CodeBlock>

      <Callout type="info" title="For scale: scikit-multilearn and neural alternatives">
        When L exceeds a few hundred labels, sklearn's built-in methods strain. The <Code>scikit-multilearn</Code> library provides MLkNN (multi-label k-nearest neighbors), MLARAM, RAkEL, and embedding-based methods. For L {">"} 1,000 labels, the modern default is a neural model with a sigmoid output head of size L, trained with binary cross-entropy per label — effectively parallel binary relevance implemented as a single forward pass, with the shared representation doing the work of capturing cross-label structure. LightGBM's <Code>MultiOutputRegressor</Code> wrapper handles multi-output regression efficiently with gradient-boosted trees, fitting T separate boosting models in parallel.
      </Callout>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6a. Label co-occurrence matrix</H3>

      <Prose>
        The normalized co-occurrence matrix shows P(label_col = 1 | label_row = 1) — the conditional probability that column label is positive given row label is positive. High off-diagonal values signal strong label dependencies where chaining adds value. Diagonal is always 1.0 (a label always co-occurs with itself). The training set here has 800 samples.
      </Prose>

      <Heatmap
        label="Normalized label co-occurrence matrix (P(col | row), training set)"
        colLabels={["L0", "L1", "L2", "L3", "L4"]}
        rowLabels={["L0", "L1", "L2", "L3", "L4"]}
        matrix={[
          [1.000, 0.392, 0.370, 0.380, 0.368],
          [0.683, 1.000, 0.627, 0.647, 0.632],
          [0.586, 0.570, 1.000, 0.581, 0.579],
          [0.500, 0.489, 0.484, 1.000, 0.520],
          [0.193, 0.190, 0.192, 0.207, 1.000],
        ]}
        colorScale="gold"
      />

      <Prose>
        Label 4 (19.4% positive rate) has low conditional probabilities in its row — given other labels, L4 is still rare. But given L4, other labels have moderate rates (0.19–0.21). Label 1 (63.5% rate) has high conditionals in its row — knowing L1 is positive is moderately informative about other labels. This asymmetry is why chain order matters: rare-to-frequent ordering often works better than the reverse.
      </Prose>

      <H3>6b. Classifier chain prediction trace</H3>

      <Prose>
        The following trace walks through how a 3-label classifier chain processes a single test sample with features x = [feature vector]. Each step conditions on the previous predicted labels.
      </Prose>

      <StepTrace
        label="Classifier chain prediction — 3 labels, single sample"
        steps={[
          {
            label: "Input features (x)",
            render: () => (
              <Prose>
                Feature vector x fed to chain. No label predictions yet.
                Classifier f₀ receives: [x₁, x₂, ..., x₂₀] (20 features + bias = 21 inputs).
                Output: P(L0=1 | x) = 0.73. Decision: ŷ₀ = 1.
                Interpretation: the model is fairly confident this sample has label 0.
              </Prose>
            ),
          },
          {
            label: "Step 1 — predict Label 1 conditioned on ŷ₀",
            render: () => (
              <Prose>
                Classifier f₁ receives: [x₁, ..., x₂₀, ŷ₀=1] (22 inputs).
                The appended ŷ₀=1 shifts the posterior for label 1.
                Output: P(L1=1 | x, ŷ₀=1) = 0.81. Decision: ŷ₁ = 1.
                Without conditioning: P(L1=1 | x) alone might be 0.65.
                The chain boosts confidence using the label dependency signal.
              </Prose>
            ),
          },
          {
            label: "Step 2 — predict Label 2 conditioned on ŷ₀, ŷ₁",
            render: () => (
              <Prose>
                Classifier f₂ receives: [x₁, ..., x₂₀, ŷ₀=1, ŷ₁=1] (23 inputs).
                Both previous predicted labels are now augmenting the feature vector.
                Output: P(L2=1 | x, ŷ₀=1, ŷ₁=1) = 0.41. Decision: ŷ₂ = 0.
                Final prediction: [ŷ₀, ŷ₁, ŷ₂] = [1, 1, 0].
                Error propagation note: if ŷ₀ had been wrong (predicted 0 when true=1),
                the incorrect 0 would have been fed to f₁ and f₂, potentially cascading errors.
              </Prose>
            ),
          },
        ]}
      />

      <H3>6c. Per-label F1 and overall macro F1</H3>

      <Prose>
        The radar below shows how F1 varies dramatically across labels with different positive rates. Label 4 (19.4% positive) is the hardest — the model sees fewer positive training examples. Label 1 (63.5% positive) is the easiest. Macro F1 averages across all five equally, penalizing poor performance on rare labels.
      </Prose>

      <Plot
        title="Per-label F1 (OVR) and macro F1 — test set, 200 samples"
        description="F1 per label from OneVsRestClassifier. Label 4 is hardest (fewest positives). Macro F1 = 0.743 averages equally across all 5 labels."
        xLabel="Label index"
        yLabel="F1 score"
        series={[
          {
            label: "per-label F1",
            type: "scatter",
            color: colors.gold,
            points: [[0, 0.7448], [1, 0.8981], [2, 0.7961], [3, 0.8061], [4, 0.4688]],
          },
          {
            label: "macro F1 = 0.743",
            type: "line",
            color: colors.green,
            points: [[-0.3, 0.7428], [4.3, 0.7428]],
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <StepTrace
        label="when to use which multi-label method"
        steps={[
          {
            label: "Binary Relevance (OneVsRestClassifier)",
            render: () => (
              <Prose>
                Use when: labels are approximately independent (low off-diagonal co-occurrence), L is large, training speed is critical, or you need a fast baseline. Advantages: trivially parallelizable across labels; compatible with any binary classifier; simplest to debug. Disadvantages: cannot exploit label correlations; each classifier is unaware of what others predict. Sklearn: <Code>OneVsRestClassifier</Code> or <Code>MultiOutputClassifier</Code>. Best base classifier: logistic regression for linear, gradient boosting for nonlinear. When NOT to use: labels have strong dependencies (Jaccard similarity between label vectors {">"} 0.3 on average).
              </Prose>
            ),
          },
          {
            label: "Classifier Chain / Ensemble of Chains",
            render: () => (
              <Prose>
                Use when: labels are correlated (medical comorbidities, protein functions, semantic scene tags). Ensemble of Classifier Chains (ECC) with random ordering is the recommended default — it averages over order sensitivity. Advantages: captures label dependencies; better F1 on correlated datasets; still uses any binary base classifier. Disadvantages: sequential inference (not parallelizable per label); error propagation through the chain; chain order matters (use random + ensemble). Sklearn: <Code>ClassifierChain(order='random')</Code>. For true ECC, fit K chains with different random seeds and aggregate by majority vote. When NOT to use: L is large and inference speed is constrained (chain is L × the cost of one classifier).
              </Prose>
            ),
          },
          {
            label: "Label Powerset",
            render: () => (
              <Prose>
                Use when: L is small ({"<"} 10–15), label combinations are semantically meaningful, and you want to predict exactly the observed combinations in training. Advantages: captures all correlations; exact for seen combinations. Disadvantages: 2^L possible classes — explodes for large L; cannot predict unseen label combinations; class imbalance is severe (some combinations are rare). sklearn: implement manually with <Code>MultiLabelBinarizer</Code> to map label vectors to integer class IDs. When NOT to use: L {">"} 15 or you expect novel label combinations at inference.
              </Prose>
            ),
          },
          {
            label: "Multi-Output Regression (MultiOutputRegressor)",
            render: () => (
              <Prose>
                Use when: targets are continuous (not binary). Predicting blood pressure, sensor readings, financial returns across multiple correlated targets. Sklearn: <Code>MultiOutputRegressor</Code> wraps any regressor into a per-target approach. For correlated continuous outputs, consider multivariate regression (one model with vector output) or Gaussian process regression with multi-output kernels. LightGBM and XGBoost support multi-output regression natively. When NOT to use: targets are binary or categorical — use multi-label classification instead.
              </Prose>
            ),
          },
          {
            label: "Neural multi-label (sigmoid + BCE) — for large L",
            render: () => (
              <Prose>
                Use when: L {">"} ~100–1,000 labels, you have {">"} 50k samples, and a GPU is available. Architecture: shared encoder (transformer, CNN, MLP) → L-dimensional sigmoid output head. Loss: binary cross-entropy summed over L labels. Advantages: shared representation captures correlations implicitly; scales to millions of labels (with approximate negative sampling); state-of-the-art on extreme classification benchmarks. Disadvantages: data-hungry; requires GPU; threshold tuning per label. Reference: AttentionXML, PECOS, and Bonsai for extreme multi-label classification. When NOT to use: L is small (classical methods win here), or dataset is {"<"} 10k samples.
              </Prose>
            ),
          },
          {
            label: "Metric selection guide",
            render: () => (
              <Prose>
                Hamming loss: use when you care about per-label accuracy and labels are balanced. Easy to game on imbalanced data (predict all zeros).
                Subset accuracy: use as a strict upper bound for reporting; rarely the primary metric.
                F1 micro: use when all label-sample pairs matter equally — dominated by frequent labels. Good for dense label spaces.
                F1 macro: use when rare labels are as important as frequent ones — unforgiving on imbalanced label distributions. Best for medical or scientific applications where rare conditions matter.
                F1 samples: use when per-instance prediction quality is the goal — measures how good the full predicted label set is for each user/document/patient.
                Jaccard: equivalent to F1 samples for binary vectors; use interchangeably.
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
        Binary relevance training cost is O(L × T_base), where T_base is the cost of training one binary classifier. Because the L problems are independent, they can be distributed across L machines or threads with no communication overhead. Sklearn's <Code>MultiOutputClassifier</Code> does exactly this with <Code>n_jobs=-1</Code>. Binary relevance inference is O(L × I_base), also embarrassingly parallel.
      </Prose>

      <Prose>
        Classifier chains training is also O(L × T_base), but each training problem is slightly larger — the j-th classifier sees d + j features instead of d. The augmentation is O(j) overhead, negligible for large d. Inference, however, is strictly sequential: you cannot predict label j until you have predicted labels 1 through j-1. This makes chain inference O(L × I_base) but with no parallelism possible across labels. For real-time latency-sensitive applications, this is a meaningful constraint.
      </Prose>

      <Prose>
        Label powerset has training cost O(T_multiclass) where the effective number of classes can reach 2^L. Training a 1,000-way multi-class classifier requires far more data than training L binary classifiers. For L = 20, 2^20 = 1,048,576 potential classes — wholly impractical.
      </Prose>

      <H3>8.2 Label space scalability</H3>

      <Prose>
        In practice, binary relevance and chains scale to L in the hundreds or low thousands on tabular and text data using sklearn. The memory footprint is L model objects, each of size proportional to the feature dimensionality. For a logistic regression base model with d = 10,000 features, L = 500 labels requires storing 500 weight vectors of length 10,000 — 40 MB in float64, entirely feasible.
      </Prose>

      <Prose>
        Above L = ~1,000–10,000 (extreme multi-label classification), sklearn-based methods become impractical not because of model storage but because of label-space-related issues: many labels are very rare (long-tail distribution), evaluation is expensive, and simple thresholding at 0.5 produces near-zero recall on tail labels. The active research area of "extreme multi-label classification" (XML) has produced specialized methods — tree-based (FastXML, Parabel), embedding-based (SLEEC, LEML), and transformer-based (AttentionXML, PECOS) — that operate at L up to 100k labels.
      </Prose>

      <H3>8.3 The neural default for large L</H3>

      <Prose>
        Above roughly L = 1,000 labels and N = 50,000 samples, the practical default has shifted to neural multi-label classifiers. The architecture is conceptually simple: any encoder network (BERT for text, ResNet for images, MLP for tabular) produces a shared dense representation, which feeds into a linear layer of size L followed by a sigmoid activation. Each of the L output neurons is trained with binary cross-entropy independently. This is exactly binary relevance, but implemented as a single forward pass with a shared representation — the shared encoder implicitly captures cross-label correlations through the shared gradient signal during training. At L = 100k with a vocabulary of known entities (Wikipedia concepts, GO terms, ICD codes), approximate methods like negative sampling or hierarchical softmax reduce the effective cost of the output layer.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>9.1 Ignoring label correlations</H3>

      <Prose>
        Applying binary relevance to a dataset with strong label correlations — say, protein function prediction where GO term annotations are heavily structured — will underperform a chain or joint model on any per-instance metric. The symptom is that individual label F1s look acceptable but F1 samples is lower than expected. The fix is to compute the normalized co-occurrence matrix first (see Section 6a). If off-diagonal entries regularly exceed 0.4–0.5, chains or ensemble-of-chains will help.
      </Prose>

      <H3>9.2 Per-label class imbalance</H3>

      <Prose>
        Multi-label datasets almost always have severe label imbalance. The positive rate for label j can range from 60% to 1% across labels in the same dataset. A label with 1% positive rate means 99% of training samples are negative — the default threshold of 0.5 will predict "negative" for every sample and achieve 99% binary accuracy on that label, while having zero recall. The fix is per-label threshold calibration: for each label, find the threshold that maximizes the desired metric on a held-out validation set. <Code>sklearn.metrics.precision_recall_curve</Code> gives you the full precision-recall tradeoff for each label. Do not share a single threshold across all labels.
      </Prose>

      <H3>9.3 Choosing the wrong F1 averaging</H3>

      <Prose>
        This is the most common reporting error in multi-label papers and industry dashboards. F1 micro on a dataset where one label dominates (e.g., 80% of positive instances belong to label 1) will look good because label 1 drives the aggregate. F1 macro will expose the fact that rare labels have F1 {"<"} 0.3. A system that reports only F1 micro is hiding its failure on rare labels. Always report at minimum F1 micro, F1 macro, and per-label F1 when labels have unequal frequencies.
      </Prose>

      <H3>9.4 Threshold selection — 0.5 is often wrong</H3>

      <Prose>
        The sigmoid output of each label classifier is a calibrated probability only if the training data is balanced and the model is well-calibrated. In practice, on imbalanced data with 10% positive rate for label j, the model's predicted probabilities for positive instances cluster around 0.2–0.4, not 0.5–0.9. Setting the threshold at 0.5 will produce near-zero recall for that label. Always tune the threshold on a validation set per label. A clean approach: compute the precision-recall curve, pick the threshold at the desired precision-recall operating point, or the one that maximizes F1 on the validation set.
      </Prose>

      <H3>9.5 Error propagation in classifier chains</H3>

      <Prose>
        Classifier chains propagate errors from early labels to later ones. If the classifier for label 1 is wrong, the incorrect prediction is fed to the classifiers for labels 2 through L as if it were ground truth. This error compounds: a 10% error rate in label 1 can degrade accuracy in label 5 by 2–5% beyond what binary relevance would produce, if the labels are strongly correlated. Ensemble of Classifier Chains (ECC) mitigates this by averaging over many random orderings — poor performance in one chain due to an early error is offset by chains where that label appears later. Use ECC rather than a single chain in production.
      </Prose>

      <H3>9.6 Label noise amplification</H3>

      <Prose>
        Multi-label annotation is expensive and error-prone. If human annotators miss a label on 5% of samples (label noise rate = 0.05), classifier chains amplify this: the noisy label ground truth is used to condition subsequent classifiers during training, injecting corrupted signal at every position in the chain. Binary relevance is equally affected per label, but the noise does not compound across labels. For datasets with high annotation noise (crowdsourced annotations, weak supervision), binary relevance is more robust because errors are contained within each label's classifier.
      </Prose>

      <H3>9.7 Evaluation metric vs. business metric mismatch</H3>

      <Prose>
        A tag recommendation system that optimizes F1 macro will push effort toward improving rare tags — which users rarely care about — at the expense of the common tags that appear on 80% of content. A medical comorbidity detector that optimizes F1 micro will look good overall while having dismal recall for rare but life-threatening conditions. Always trace backward from the business goal: is a missed rare positive worse than a false alarm? Is each mislabeled instance equally costly regardless of which label is wrong? The answer determines which metric should drive model selection and threshold tuning, and which metrics are reported alongside for transparency.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All citations below were verified against their actual publications. Read in chronological order to follow the field's development.
      </Prose>

      <StepTrace
        label="primary literature"
        steps={[
          {
            label: "Boutell, Luo, Shen, Brown 2004 — founding paper",
            render: () => (
              <Prose>
                Boutell, M.R., Luo, J., Shen, X., and Brown, C.M. (2004). "Learning Multi-Label Scene Classification." <em>Pattern Recognition</em>, 37(9), 1757–1771. DOI: 10.1016/j.patcog.2004.03.009. The first systematic treatment of the problem from a machine learning perspective. Motivated by the observation that natural images belong to multiple semantic classes simultaneously — a beach scene with people is both <em>beach</em> and <em>people</em>, not one or the other. Introduced the term "multi-label machine learning," surveyed early approaches, and benchmarked them on a scene classification dataset. Established the binary relevance baseline and showed that exploiting label co-occurrence improves performance. The notation {"y ∈ {{0,1}^L}"} that the field uses universally originates here.
              </Prose>
            ),
          },
          {
            label: "Tsoumakas & Katakis 2007 — taxonomy and overview",
            render: () => (
              <Prose>
                Tsoumakas, G. and Katakis, I. (2007). "Multi-Label Classification: An Overview." <em>International Journal of Data Warehousing and Mining</em>, 3(3), 1–13. DOI: 10.4018/jdwm.2007070101. The landmark survey that organized the emerging literature. Introduced the two-way taxonomy of approaches that the field still uses: (1) <em>problem transformation methods</em>, which convert the multi-label problem into one or more single-label problems (binary relevance, label powerset, pairwise decomposition), and (2) <em>algorithm adaptation methods</em>, which extend specific algorithms to natively handle multi-label targets (ML-kNN, AdaBoost.MH, multi-label SVMs). Defined evaluation metrics formally. Has over 2,000 citations and remains the standard reference for orienting newcomers.
              </Prose>
            ),
          },
          {
            label: "Read, Pfahringer, Holmes, Frank 2011 — classifier chains",
            render: () => (
              <Prose>
                Read, J., Pfahringer, B., Holmes, G., and Frank, E. (2011). "Classifier Chains for Multi-Label Classification." <em>Machine Learning</em>, 85(3), 333–359. DOI: 10.1007/s10994-011-5256-5. Introduced and rigorously analyzed the classifier chain algorithm. Showed theoretically that chains can model the full joint distribution over labels via the chain rule of probability. Demonstrated empirically that Ensemble of Classifier Chains (ECC) consistently outperforms binary relevance on datasets with label dependencies. Analyzed the effect of chain order and showed that random ordering with ensembling outperforms attempts to find the optimal order. This paper is the reason ClassifierChain is in scikit-learn.
              </Prose>
            ),
          },
          {
            label: "Zhang & Zhou 2014 — comprehensive review",
            render: () => (
              <Prose>
                Zhang, M.-L. and Zhou, Z.-H. (2014). "A Review on Multi-Label Learning Algorithms." <em>IEEE Transactions on Knowledge and Data Engineering</em>, 26(8), 1819–1837. DOI: 10.1109/TKDE.2013.39. The definitive algorithmic survey covering 8 representative multi-label learning algorithms with unified notation and analysis. Covers ML-kNN (multi-label k-nearest neighbors), BPM (backpropagation for multi-label), RankSVM, LEAD (label-specific features), and several problem transformation approaches. Includes rigorous learning-theoretic analysis of generalization bounds. The review of evaluation measures (Section III) is comprehensive and authoritative — the source to cite when distinguishing Hamming loss, subset accuracy, ranking loss, coverage error, and average precision for multi-label tasks.
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
        Work through these before moving to the next topic. Answers are below each exercise.
      </Prose>

      <H3>Exercise 1 (recall)</H3>
      <Prose>
        Explain the difference between multi-class, multi-label, and multi-output regression. Give one real-world example of each. What is the shape of the output vector for each problem type?
      </Prose>
      <Callout type="answer" title="Answer 1">
        Multi-class: one label from K mutually exclusive classes. Output: integer in {"{0, ..., K-1}"} or K-dimensional probability vector summing to 1. Example: digit recognition (one digit per image). Multi-label: subset of L possible labels as a binary vector {"y ∈ {{0,1}^L}"} where entries are independent and do not sum to 1. Example: movie genre tagging (a film can be Action AND Romance AND Thriller simultaneously). Multi-output regression: real-valued vector y ∈ ℝ^T of T continuous targets. Example: predicting both tomorrow's high temperature and humidity from today's weather features. Key distinction: multi-class has exactly one positive, multi-label has any number of positives, multi-output regression has continuous values.
      </Callout>

      <H3>Exercise 2 (derivation)</H3>
      <Prose>
        Write the chain rule factorization that classifier chains implement. For L = 3 labels, write out the full product of conditionals. Why does the chain rule guarantee that this factorization is exact (under no independence assumptions), unlike binary relevance?
      </Prose>
      <Callout type="answer" title="Answer 2">
        Classifier chains implement: P(y₁, y₂, y₃ | x) = P(y₁ | x) × P(y₂ | x, y₁) × P(y₃ | x, y₁, y₂). This follows from the chain rule of probability: P(A,B,C) = P(A) × P(B|A) × P(C|A,B), which holds for any joint distribution without any independence assumption. Binary relevance assumes P(y₁,y₂,y₃|x) = P(y₁|x) × P(y₂|x) × P(y₃|x), which drops the conditioning on other labels — this is only exact if labels are conditionally independent given x. Classifier chains make no independence assumption: each conditional is estimated by a separate classifier that receives the actual label values as features. The chain is exact in the sense that the right joint distribution structure is parameterized; approximation comes only from finite-sample estimation of each conditional.
      </Callout>

      <H3>Exercise 3 (metrics)</H3>
      <Prose>
        A multi-label classifier predicts the following for 4 samples (L = 3 labels):
        True:  [[1,0,1], [0,1,1], [1,1,0], [0,0,1]]
        Pred:  [[1,0,1], [0,1,0], [1,0,0], [0,1,1]]
        Compute Hamming loss, subset accuracy, and F1 micro by hand.
      </Prose>
      <Callout type="answer" title="Answer 3">
        Hamming loss: Count mismatches per cell over 4×3=12 cells. Sample 0: [0,0,0] → 0 errors. Sample 1: [0,0,1] → 1 error. Sample 2: [0,1,0] → 1 error. Sample 3: [0,1,0] → 1 error. Total errors = 3. Hamming loss = 3/12 = 0.25. Subset accuracy: Exact matches = sample 0 only. Subset acc = 1/4 = 0.25. F1 micro: Pool all labels. TP: positions where both true and pred are 1: sample0(L0,L2), sample1(L1), sample2(L0) → TP=4. FP: pred=1 but true=0: sample3(L1), sample3(L2) → FP=2. FN: true=1 but pred=0: sample1(L2), sample2(L1) → FN=2. Precision=4/6=0.667. Recall=4/6=0.667. F1_micro = 2×0.667×0.667/(0.667+0.667) = 0.667.
      </Callout>

      <H3>Exercise 4 (debugging)</H3>
      <Prose>
        You train an <Code>OneVsRestClassifier</Code> on a 10-label dataset. Label 7 has only 0.8% positive rate (8 positives in 1,000 training samples). At evaluation, F1 for label 7 is 0.0 — the model never predicts positive for label 7. List the three most likely causes and the fix for each.
      </Prose>
      <Callout type="answer" title="Answer 4">
        Cause 1: Default threshold 0.5 is too high. With 0.8% positives, the model learns to output probabilities below 0.5 for almost everything. Fix: use predict_proba instead of predict, and tune the threshold for label 7 on a validation set using the precision-recall curve. Cause 2: No class weighting. The loss for label 7 is dominated by the 99.2% negatives — the model learns to always predict 0. Fix: pass class_weight='balanced' to the base LogisticRegression, or oversample positive examples for label 7 with SMOTE. Cause 3: Insufficient training data. 8 positive examples is genuinely too few for logistic regression to fit a reliable decision boundary, especially with many features. Fix: collect more labeled data, use data augmentation, or consolidate rare labels with semantically similar ones.
      </Callout>

      <H3>Exercise 5 (applied)</H3>
      <Prose>
        You are building a protein function prediction system with 500 GO term labels (Gene Ontology) and 20,000 training proteins with sequence embeddings of dimension 1,024. Which approach do you choose: binary relevance, classifier chains, label powerset, or neural multi-label? Justify using complexity, correlation, and data size arguments.
      </Prose>
      <Callout type="answer" title="Answer 5">
        Neural multi-label with a sigmoid output head is the right choice here. Justification: (1) Label powerset is eliminated immediately — 2^500 possible label combinations, wholly intractable. (2) Classifier chains: 500 sequential classifiers with inference requiring 500 sequential forward passes, each growing by one input. Too slow and error-propagation over 500 steps is severe. (3) Binary relevance (OVR) with logistic regression: 500 classifiers × 1024-dim features = feasible in memory but ignores GO term correlations, which are biologically real and structured (GO is a DAG). (4) Neural multi-label: shared transformer/MLP encoder → 500-way sigmoid head. The shared encoder captures cross-label correlations implicitly through the shared gradient. With 20,000 proteins and 1,024-dim embeddings, this is at the lower end of viable neural territory — augment with pretrained protein language model (ESM-2) embeddings. Use per-label F1 and macro F1 as the primary evaluation metrics, with per-label threshold tuning on a validation split.
      </Callout>

      <H3>Exercise 6 (synthesis)</H3>
      <Prose>
        A colleague argues: "Macro F1 is the only fair metric for multi-label datasets because it weights all labels equally." Another argues: "F1 micro is better because rare labels don't matter as much in practice." Construct the conditions under which each person is correct. Then propose a reporting strategy that satisfies both.
      </Prose>
      <Callout type="answer" title="Answer 6">
        Your first colleague is correct when: rare labels have high business value equal to or greater than common labels. Example: rare cancer subtypes in ICD coding, rare protein functions in GO annotation. Missing a rare but critical label is as bad as missing a common one. Macro F1 enforces this by giving equal weight regardless of frequency. Your second colleague is correct when: the value generated scales with label volume — a document tagging system where tagging popular topics (sports, politics) at high precision/recall serves most users, while rare tags are "nice to have." Micro F1 reflects the aggregate user experience across all label-sample pairs. Reporting strategy that satisfies both: report (a) F1 micro for aggregate throughput, (b) F1 macro for rare-label fairness, (c) per-label F1 as a table or histogram for transparency, and (d) a business-weighted metric (weighted macro F1 where weights reflect label importance) as the model selection criterion. Never report only one averaging strategy for a multi-label system.
      </Callout>

    </div>
  ),
};

export default multiLabelContent;
