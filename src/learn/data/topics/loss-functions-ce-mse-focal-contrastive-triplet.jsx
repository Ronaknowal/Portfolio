import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const lossFunctionsContent = {
  title: "Loss Functions (CE, MSE, Focal, Contrastive, Triplet)",
  readTime: "~42 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Every neural network is a parameterized function, and every training procedure is really a search for parameters that minimize a loss. The loss function is where all the problem's structure lives: what you are trying to predict, how you measure error, which mistakes you punish, and what you are willing to tolerate. Change the loss and you change the entire character of what the model learns, sometimes more than changing the architecture. The five losses in this topic — MSE, cross-entropy, focal, contrastive, triplet (with InfoNCE as a close cousin of contrastive) — trace a 200-year intellectual lineage from celestial mechanics to modern self-supervised learning.
      </Prose>

      <Prose>
        The oldest is mean squared error. Carl Friedrich Gauss published "Theoria Motus Corporum Coelestium in Sectionibus Conicis Solem Ambientium" in 1809, a treatise on computing the orbits of celestial bodies from noisy observations. To justify the method of least squares — which Gauss had been using since 1795 and Legendre had published in 1805 — Gauss proved what we now call the Gauss-Markov theorem and showed something stronger: if observation errors are independent, zero-mean, and Gaussian, then the least-squares estimator is the maximum likelihood estimator. Minimizing squared error is equivalent to assuming Gaussian noise. This single insight is why MSE is the default regression loss two centuries later: it is not an arbitrary choice, it is the correct choice whenever your noise model is Gaussian.
      </Prose>

      <Prose>
        Cross-entropy arrived in 1948 from a completely different direction. Claude Shannon's "A Mathematical Theory of Communication" (Bell System Technical Journal, 27, 379–423 and 623–656) founded information theory and introduced entropy as the expected number of bits needed to encode samples from a distribution. Cross-entropy is the expected number of bits needed to encode samples from distribution p when you use the optimal code for distribution q. Minimizing cross-entropy over q pulls q toward p — it is the natural loss for probability matching. In the 1980s and 1990s, as feedforward networks were used for classification, cross-entropy (equivalent to negative log-likelihood under Bernoulli/categorical assumptions) replaced MSE for classification because it produces stronger, well-scaled gradients whenever predictions are confidently wrong.
      </Prose>

      <Prose>
        Focal loss is a 2017 answer to a very practical problem. Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár published "Focal Loss for Dense Object Detection" (arXiv:1708.02002, ICCV 2017) as part of the RetinaNet paper. Dense object detectors evaluate ~100k candidate boxes per image, of which maybe 10 contain objects. Plain cross-entropy is drowned by the 99,990 easy negatives — each contributes a small but non-zero loss, and their sum dwarfs the loss on the handful of informative examples. Lin et al. modulated standard CE by a factor <Code>{"(1 - p_t)^γ"}</Code> that smoothly down-weights easy examples. With <Code>γ=2</Code>, a sample classified at <Code>p=0.9</Code> contributes 100x less gradient than it would under plain CE. The trick made one-stage detectors competitive with two-stage detectors and has since been adopted across medical imaging, fraud detection, and every other extreme-imbalance setting.
      </Prose>

      <Prose>
        Contrastive loss predates the deep-learning boom. Raia Hadsell, Sumit Chopra, and Yann LeCun published "Dimensionality Reduction by Learning an Invariant Mapping" at CVPR 2006. Their setup: take pairs of points with a similarity label, learn an embedding where similar pairs are close in Euclidean distance and dissimilar pairs are at least <Code>m</Code> apart. The loss is explicitly metric-shaping rather than class-predicting — it cares about relative geometry, not labels. This paper is the ancestor of every modern representation-learning loss, from triplet to SimCLR.
      </Prose>

      <Prose>
        The triplet variant followed in 2015. Florian Schroff, Dmitry Kalenichenko, and James Philbin at Google published "FaceNet: A Unified Embedding for Face Recognition and Clustering" (arXiv:1503.03832, CVPR 2015). Instead of pairs, they used triples: an anchor face, a positive (same identity), and a negative (different identity). The loss required the anchor-positive distance to be smaller than the anchor-negative distance by at least a margin. FaceNet hit 99.63% on Labeled Faces in the Wild — the first superhuman face recognition system — and made triplet loss the standard for face verification, person re-identification, and speaker verification.
      </Prose>

      <Prose>
        Contrastive learning then got a reformulation that scaled. Aaron van den Oord, Yazhe Li, and Oriol Vinyals introduced InfoNCE in "Representation Learning with Contrastive Predictive Coding" (arXiv:1807.03748, 2018). InfoNCE is a categorical cross-entropy over similarity scores: out of a set of <Code>{"K+1"}</Code> candidates, one is the true positive, and you must identify it. Ting Chen et al. pushed this approach on images in "SimCLR" (arXiv:2002.05709, ICML 2020), showing that with large batches and strong augmentation, InfoNCE on self-supervised pairs matches supervised pretraining. Prannay Khosla et al. extended it to supervised settings in "Supervised Contrastive Learning" (arXiv:2004.11362, 2020). In parallel, the face-recognition community developed angular-margin losses: Jiankang Deng et al.'s ArcFace (arXiv:1801.07698, CVPR 2019) replaced Euclidean margins with a geodesic margin on the hypersphere, giving better class separation with less hyperparameter tuning.
      </Prose>

      <Callout type="insight">
        Every loss in this family answers the same question — "what should the gradient say?" — from a different angle. MSE asks for distance minimization under Gaussian noise. Cross-entropy asks for probability matching. Focal loss asks for probability matching that ignores easy examples. Contrastive and triplet losses abandon probabilities entirely and ask for geometric structure in an embedding space. InfoNCE merges the two worlds by reformulating contrastive learning as a cross-entropy over similarity scores.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        A loss function does one job: it tells the optimizer what direction to move the parameters. Its shape determines which examples drive learning. Understanding each loss means understanding which examples it amplifies, which it suppresses, and what geometry it is trying to sculpt.
      </Prose>

      <H3>2.1 MSE — distance minimization</H3>

      <Prose>
        MSE treats predictions as points and labels as points. The loss is the squared Euclidean distance between them: <Code>{"(y − ŷ)²"}</Code>. Its gradient with respect to the prediction is <Code>{"2(ŷ − y)"}</Code> — linear in the error, zero at the target, and symmetric in over- vs under-prediction. Squaring means large errors dominate: a prediction off by 10 contributes 100x more gradient than one off by 1. This is the right behavior if your errors are Gaussian (truly rare large errors matter more), but the wrong behavior if your errors are heavy-tailed (a rare outlier drags the whole fit).
      </Prose>

      <H3>2.2 Cross-entropy — probability matching</H3>

      <Prose>
        Cross-entropy treats predictions as a probability distribution <Code>q</Code> and labels as a target distribution <Code>p</Code> (usually one-hot). The loss is <Code>{"−Σ p log q"}</Code> — the number of extra bits you pay for using <Code>q</Code> instead of the optimal code for <Code>p</Code>. The gradient with respect to the logit of class <Code>i</Code> is <Code>{"q_i − p_i"}</Code> — the difference between what the model says and what the label says. Cross-entropy punishes confident wrong predictions violently (<Code>{"−log(0.01) ≈ 4.6"}</Code>) and confident right predictions almost not at all (<Code>{"−log(0.99) ≈ 0.01"}</Code>), which is exactly the gradient signal you want for classification.
      </Prose>

      <H3>2.3 Focal — focus on hard examples</H3>

      <Prose>
        Focal loss rescales cross-entropy by <Code>{"(1 − p_t)^γ"}</Code>, where <Code>p_t</Code> is the probability assigned to the true class. When <Code>p_t</Code> is near 1 (easy), the factor is near 0 — the gradient is almost gone. When <Code>p_t</Code> is near 0 (hard), the factor is near 1 — the gradient is preserved. The parameter <Code>γ</Code> (typically 2) controls the sharpness of this down-weighting. The effect on extreme-imbalance training is striking: easy negatives that used to contribute 99% of the total loss now contribute a few percent, and the optimizer's gradient budget is spent on the rare hard examples where learning actually happens.
      </Prose>

      <H3>2.4 Contrastive — pull positives, push negatives</H3>

      <Prose>
        Contrastive loss abandons the prediction-vs-label frame. Instead, it takes pairs and enforces a metric: similar pairs should have small embedding distance; dissimilar pairs should have distance at least <Code>m</Code>. The loss is <Code>{"y·D² + (1−y)·max(0, m − D)²"}</Code>. Similar pairs pay a cost proportional to squared distance — always pulling them closer. Dissimilar pairs only pay when their distance is less than the margin — once they are far enough apart, they are ignored. This asymmetry is the key to the method: positives are always pulled, negatives are only pushed until they reach the margin, then released.
      </Prose>

      <H3>2.5 Triplet — relative distance with a margin</H3>

      <Prose>
        Triplet loss takes an anchor, a positive, and a negative. It does not care about absolute distances — it only requires that the anchor-positive distance be smaller than the anchor-negative distance by at least a margin <Code>α</Code>:
      </Prose>

      <MathBlock>{"\\mathcal{L} = \\max\\!\\left(0,\\; D(a, p)^2 - D(a, n)^2 + \\alpha\\right)"}</MathBlock>

      <Prose>
        If the negative is already <Code>α</Code> farther than the positive, the triplet is "easy" and contributes zero gradient. If the negative is closer than the positive (a "hard" triplet) or within the margin (a "semi-hard" triplet), the triplet contributes gradient that pulls anchor and positive together and pushes anchor and negative apart simultaneously. Good triplet training depends almost entirely on mining: easy triplets are useless, fully-hard triplets are unstable, semi-hard triplets are the sweet spot.
      </Prose>

      <H3>2.6 InfoNCE — contrastive as classification</H3>

      <Prose>
        InfoNCE unifies contrastive and cross-entropy learning. Given a query <Code>q</Code>, one positive key <Code>{"k⁺"}</Code>, and <Code>{"K"}</Code> negative keys, the loss is the categorical cross-entropy of picking the positive out of the <Code>{"K+1"}</Code> candidates, where each candidate's logit is its similarity to the query divided by a temperature <Code>τ</Code>:
      </Prose>

      <MathBlock>{"\\mathcal{L}_{\\text{InfoNCE}} = -\\log \\frac{\\exp(\\text{sim}(q, k^+)/\\tau)}{\\sum_{i=0}^{K} \\exp(\\text{sim}(q, k_i)/\\tau)}"}</MathBlock>

      <Prose>
        InfoNCE benefits from large batch sizes because every non-positive sample in the batch is a free negative. With a batch of 4096, every example is classified against 4095 implicit negatives — no explicit triplet mining needed. The temperature <Code>τ</Code> controls how sharply the softmax distinguishes neighbors: small <Code>τ</Code> makes hard negatives dominate, large <Code>τ</Code> smooths the loss.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Math foundation</H2>

      <H3>3.1 MSE from Gaussian likelihood</H3>

      <Prose>
        Assume observations are generated by <Code>{"y = f(x) + ε"}</Code> where <Code>{"ε ∼ N(0, σ²)"}</Code>. The likelihood of observing <Code>y</Code> given a model prediction <Code>ŷ = f(x; θ)</Code> is:
      </Prose>

      <MathBlock>{"p(y \\mid x, \\theta) = \\frac{1}{\\sqrt{2\\pi\\sigma^2}} \\exp\\!\\left(-\\frac{(y - \\hat{y})^2}{2\\sigma^2}\\right)"}</MathBlock>

      <Prose>
        Maximizing log-likelihood over a dataset of N independent samples:
      </Prose>

      <MathBlock>{"\\log p(\\mathbf{y} \\mid X, \\theta) = -\\frac{N}{2}\\log(2\\pi\\sigma^2) - \\frac{1}{2\\sigma^2}\\sum_{i=1}^N (y_i - \\hat{y}_i)^2"}</MathBlock>

      <Prose>
        The first term does not depend on <Code>θ</Code>. Maximizing the log-likelihood in <Code>θ</Code> is exactly minimizing <Code>{"Σ(y_i − ŷ_i)²"}</Code>. Minimizing MSE is maximum likelihood estimation under Gaussian noise — and the optimal constant predictor is the mean of <Code>y</Code>. If errors are Laplace-distributed instead, the same derivation yields MAE (mean absolute error) and the optimal constant predictor is the median. Loss shape is a statement about noise.
      </Prose>

      <H3>3.2 BCE from Bernoulli likelihood</H3>

      <Prose>
        For binary classification, the label <Code>{"y ∈ {0, 1}"}</Code> is a Bernoulli sample with success probability <Code>{"p = σ(z)"}</Code> where <Code>z</Code> is the logit. The likelihood of one observation is:
      </Prose>

      <MathBlock>{"p(y \\mid z) = \\sigma(z)^y (1 - \\sigma(z))^{1-y}"}</MathBlock>

      <Prose>
        Negative log-likelihood over N samples gives the binary cross-entropy loss:
      </Prose>

      <MathBlock>{"\\mathcal{L}_{\\text{BCE}} = -\\frac{1}{N}\\sum_{i=1}^{N}\\big[y_i \\log \\sigma(z_i) + (1 - y_i)\\log(1 - \\sigma(z_i))\\big]"}</MathBlock>

      <Prose>
        The gradient with respect to the logit simplifies dramatically: <Code>{"∂L/∂z = σ(z) − y"}</Code>. The sigmoid derivative cancels the logarithm's reciprocal. This is why numerically stable BCE is always implemented in logit space — computing <Code>σ(z)</Code> and then <Code>{"log σ(z)"}</Code> can underflow when <Code>z</Code> is a large negative number, but the fused <Code>BCEWithLogitsLoss</Code> computes <Code>{"max(z, 0) − z·y + log(1 + exp(−|z|))"}</Code>, which is stable everywhere.
      </Prose>

      <H3>3.3 Categorical cross-entropy and log-sum-exp</H3>

      <Prose>
        For multiclass classification with C classes, labels are one-hot and predictions are softmax probabilities. For a single example with logit vector <Code>z</Code> and true class <Code>y</Code>:
      </Prose>

      <MathBlock>{"\\mathcal{L}_{\\text{CE}} = -\\log \\frac{\\exp(z_y)}{\\sum_{c=1}^{C}\\exp(z_c)} = -z_y + \\log \\sum_{c=1}^{C}\\exp(z_c)"}</MathBlock>

      <Prose>
        The second form — <Code>{"−z_y + logsumexp(z)"}</Code> — is the one every numerical library implements. Computing <Code>{"exp(z_c)"}</Code> directly overflows when logits exceed ~700 (float64) or ~88 (float32). The stable trick subtracts the max before exponentiating:
      </Prose>

      <MathBlock>{"\\text{logsumexp}(z) = m + \\log \\sum_{c}\\exp(z_c - m),\\quad m = \\max_c z_c"}</MathBlock>

      <Prose>
        After subtracting <Code>m</Code>, the largest exponential is <Code>{"exp(0) = 1"}</Code>; the others are all <Code>{"≤ 1"}</Code>. No overflow, and underflow is harmless. The gradient with respect to the logit of class <Code>c</Code> is <Code>{"p_c − y_c"}</Code> where <Code>{"p = softmax(z)"}</Code> and <Code>{"y"}</Code> is the one-hot target — the same clean form as BCE.
      </Prose>

      <H3>3.4 Focal loss</H3>

      <Prose>
        Let <Code>{"p_t"}</Code> denote the probability assigned to the true class (for binary: <Code>{"p_t = p"}</Code> if <Code>y=1</Code>, else <Code>{"p_t = 1−p"}</Code>). The focal loss is:
      </Prose>

      <MathBlock>{"\\mathcal{L}_{\\text{FL}} = -\\alpha_t (1 - p_t)^{\\gamma} \\log p_t"}</MathBlock>

      <Prose>
        The <Code>{"(1 − p_t)^γ"}</Code> term is the focusing factor. When <Code>{"p_t → 1"}</Code> (well-classified), the factor <Code>{"→ 0"}</Code>. When <Code>{"p_t → 0"}</Code> (misclassified), the factor <Code>{"→ 1"}</Code> and focal reduces to standard CE. The <Code>{"α_t"}</Code> term is a class-balancing weight: typically <Code>α</Code> for positives and <Code>{"1 − α"}</Code> for negatives. For <Code>{"γ=0, α=1"}</Code>, focal equals plain CE. For RetinaNet, Lin et al. recommend <Code>{"γ=2, α=0.25"}</Code>.
      </Prose>

      <H3>3.5 Contrastive (margin-based)</H3>

      <Prose>
        With embedding distance <Code>{"D(x₁, x₂) = ‖f(x₁) − f(x₂)‖₂"}</Code> and similarity label <Code>{"y ∈ {0, 1}"}</Code>:
      </Prose>

      <MathBlock>{"\\mathcal{L}_{\\text{contrastive}} = y \\cdot D^2 + (1 - y) \\cdot \\max(0, m - D)^2"}</MathBlock>

      <Prose>
        The cosine version replaces <Code>D</Code> with <Code>{"1 − \\cos θ"}</Code> and typically uses normalized embeddings. The squared hinge form is Hadsell et al.'s original; later work uses <Code>{"max(0, m − D)"}</Code> without squaring, which gives a linear push on near-margin negatives rather than quadratic.
      </Prose>

      <H3>3.6 Triplet loss</H3>

      <Prose>
        With anchor <Code>a</Code>, positive <Code>p</Code>, negative <Code>n</Code>, and margin <Code>α</Code>:
      </Prose>

      <MathBlock>{"\\mathcal{L}_{\\text{triplet}} = \\max\\!\\left(0,\\; \\|f(a) - f(p)\\|^2 - \\|f(a) - f(n)\\|^2 + \\alpha\\right)"}</MathBlock>

      <Prose>
        The loss is zero whenever the negative is farther than the positive by at least <Code>α</Code>. Schroff et al. showed that training on all triplets is wasteful — most are easy — and proposed semi-hard mining: within each batch, for each anchor-positive pair, select the negative that satisfies <Code>{"D(a,n) > D(a,p)"}</Code> but <Code>{"D(a,n) < D(a,p) + α"}</Code>. These are triplets where the negative is on the wrong side of the margin but not catastrophically close, giving stable gradients.
      </Prose>

      <H3>3.7 InfoNCE</H3>

      <Prose>
        With query <Code>q</Code>, positive key <Code>{"k⁺"}</Code>, and negatives <Code>{"{k_i}_{i=1}^K"}</Code>:
      </Prose>

      <MathBlock>{"\\mathcal{L}_{\\text{InfoNCE}} = -\\log \\frac{\\exp(\\text{sim}(q, k^+)/\\tau)}{\\exp(\\text{sim}(q, k^+)/\\tau) + \\sum_{i=1}^{K}\\exp(\\text{sim}(q, k_i)/\\tau)}"}</MathBlock>

      <Prose>
        Typical choices: cosine similarity with L2-normalized embeddings, temperature <Code>{"τ ∈ [0.05, 0.2]"}</Code>. Van den Oord et al. proved InfoNCE is a lower bound on mutual information <Code>{"I(q; k⁺)"}</Code> up to <Code>{"log K"}</Code>, which is why more negatives (larger batches or memory banks) tighten the bound and improve representation quality. Implementation-wise, InfoNCE is just <Code>F.cross_entropy</Code> applied to the <Code>{"B × B"}</Code> similarity matrix with diagonal labels — every modern contrastive framework uses this trick.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        NumPy implementations for every loss. All code was executed; outputs shown are verbatim stdout.
      </Prose>

      <H3>4a. MSE, BCE, and softmax CE</H3>

      <CodeBlock language="python">
{`import numpy as np
np.random.seed(42)

# --- MSE ---
def mse(y, yhat):
    return np.mean((y - yhat) ** 2)

def mse_grad(y, yhat):
    return 2 * (yhat - y) / y.shape[0]  # dL/dyhat

y    = np.array([1.0, 2.0, 3.0, 4.0])
yhat = np.array([1.1, 1.9, 3.2, 3.7])
print("=== MSE ===")
print(f"MSE: {mse(y, yhat):.6f}")
print(f"Gradient wrt yhat: {mse_grad(y, yhat).round(4)}")

# --- BCE from logits (numerically stable) ---
def bce_with_logits(y, z):
    # L = max(z, 0) - z*y + log(1 + exp(-|z|))   — stable for any z
    return np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z))))

def bce_grad(y, z):
    p = 1.0 / (1.0 + np.exp(-z))
    return (p - y) / y.shape[0]       # dL/dz = sigmoid(z) - y

y = np.array([0.0, 1.0, 1.0, 0.0])
z = np.array([-2.0, 1.5, 0.2, -0.4])
print("\\n=== BCE with logits ===")
print(f"Loss: {bce_with_logits(y, z):.6f}")
print(f"Gradient wrt logits: {bce_grad(y, z).round(4)}")

# --- Softmax CE with log-sum-exp ---
def log_softmax(z):
    m = np.max(z, axis=-1, keepdims=True)
    return z - m - np.log(np.sum(np.exp(z - m), axis=-1, keepdims=True))

def cross_entropy(z, y_int):
    ls = log_softmax(z)
    return -np.mean(ls[np.arange(len(y_int)), y_int])

z = np.array([[ 2.0, 0.5, -1.0],
              [ 0.1, 0.2,  0.3],
              [-0.5,-1.0,  2.0]])
y_int = np.array([0, 2, 2])
print("\\n=== Softmax Cross-Entropy ===")
print(f"Loss: {cross_entropy(z, y_int):.6f}")

# Stability check: enormous logits
z_big = np.array([[1000.0, 1001.0, 999.0]])
print(f"log_softmax with logits ~1000 (stable): {log_softmax(z_big).round(4)}")
# Output:
# === MSE ===
# MSE: 0.037500
# Gradient wrt yhat: [ 0.05 -0.05  0.1  -0.15]
#
# === BCE with logits ===
# Loss: 0.359874
# Gradient wrt logits: [ 0.0298 -0.0456 -0.1125  0.1003]
#
# === Softmax Cross-Entropy ===
# Loss: 0.455709
# log_softmax with logits ~1000 (stable): [[-1.4076 -0.4076 -2.4076]]`}
      </CodeBlock>

      <Prose>
        Two things to verify. First, the clean gradient forms: MSE's gradient is linear in the residual, BCE's gradient is <Code>{"sigmoid(z) − y"}</Code> — no gradient explosion even at large negative logits because the stable log-sum-exp handles <Code>{"exp(1000)"}</Code> correctly. Second, the BCE gradient is largest for the most confidently wrong example (index 2: <Code>{"z=0.2, y=1"}</Code>) at <Code>{"−0.1125"}</Code> and smallest for the most confidently right (index 0: <Code>{"z=−2, y=0"}</Code>) at <Code>{"+0.0298"}</Code>. This is exactly the gradient structure that drives classification learning.
      </Prose>

      <H3>4b. Focal loss — gradient behavior under 1:99 imbalance</H3>

      <CodeBlock language="python">
{`import numpy as np
np.random.seed(0)

def focal_loss_binary(y, p, alpha=0.25, gamma=2.0, eps=1e-8):
    p = np.clip(p, eps, 1 - eps)
    p_t     = np.where(y == 1, p, 1 - p)
    alpha_t = np.where(y == 1, alpha, 1 - alpha)
    return -np.mean(alpha_t * (1 - p_t) ** gamma * np.log(p_t))

def bce_binary(y, p, eps=1e-8):
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

# 1:99 imbalance — only 1% positives. A degenerate classifier outputs 0.05 everywhere.
N = 10000
y = np.zeros(N); y[:100] = 1.0
np.random.shuffle(y)
p_poor = np.full(N, 0.05)
print("=== 1:99 imbalance, predicting 0.05 everywhere (majority collapse) ===")
print(f"BCE                           : {bce_binary(y, p_poor):.6f}")
print(f"Focal (gamma=2, alpha=0.25)   : {focal_loss_binary(y, p_poor, 0.25, 2.0):.6f}")

def focal_contrib(y, p, alpha=0.25, gamma=2.0):
    p = np.clip(p, 1e-8, 1-1e-8)
    p_t     = np.where(y == 1, p, 1 - p)
    alpha_t = np.where(y == 1, alpha, 1 - alpha)
    return alpha_t * (1 - p_t) ** gamma * (-np.log(p_t))

# Contributions of one hard positive vs one easy negative
hp    = focal_contrib(np.array([1.0]), np.array([0.1]))[0]        # y=1, p=0.1  (hard)
en    = focal_contrib(np.array([0.0]), np.array([0.02]))[0]       # y=0, p=0.02 (easy)
hp_ce = -np.log(0.1)
en_ce = -np.log(1 - 0.02)
print(f"\\nHard positive contribution  — BCE: {hp_ce:.4f}  Focal: {hp:.4f}")
print(f"Easy negative contribution  — BCE: {en_ce:.4f}  Focal: {en:.6f}")
print(f"Ratio hard/easy             — BCE: {hp_ce/en_ce:.2f}x  Focal: {hp/en:.2f}x")
# Output:
# === 1:99 imbalance, predicting 0.05 everywhere (majority collapse) ===
# BCE                           : 0.080738
# Focal (gamma=2, alpha=0.25)   : 0.006854
#
# Hard positive contribution  -- BCE: 2.3026  Focal: 0.4663
# Easy negative contribution  -- BCE: 0.0202  Focal: 0.000006
# Ratio hard/easy             -- BCE: 113.97x  Focal: 76932.51x`}
      </CodeBlock>

      <Prose>
        The critical number is the ratio. Under plain BCE, the hard positive contributes ~114x more than one easy negative. That sounds like a lot — until you remember there are 99x more negatives than positives, so in aggregate the easy negatives dominate the gradient. Under focal loss, the per-example ratio jumps to 76,932x — large enough that even a sea of easy negatives cannot outweigh a few hard positives. That is the mechanism: focal does not change the sign of the gradient, it just changes the <em>weighting</em> so the gradient direction is dominated by the hard examples.
      </Prose>

      <H3>4c. Contrastive and triplet (Euclidean, with semi-hard mining)</H3>

      <CodeBlock language="python">
{`import numpy as np
np.random.seed(1)

# --- Contrastive (Euclidean, Hadsell/Chopra/LeCun 2006) ---
def contrastive_loss(x1, x2, y, margin=1.0):
    D = np.linalg.norm(x1 - x2, axis=1)
    pos = y * (D ** 2)
    neg = (1 - y) * (np.maximum(0, margin - D) ** 2)
    return np.mean(pos + neg), D

x1 = np.array([[0.0, 0.0]] * 3)
x2 = np.array([[0.1, 0.0],   # close, y=1 (good)
               [2.0, 0.0],   # far,   y=1 (bad — should be close)
               [2.0, 0.0]])  # far,   y=0 (good)
y_c = np.array([1, 1, 0])
loss, D = contrastive_loss(x1, x2, y_c, margin=1.0)
print("=== Contrastive (margin=1.0) ===")
for i in range(3):
    term = (D[i]**2) if y_c[i] == 1 else (max(0, 1.0 - D[i]) ** 2)
    print(f"  pair {i}  y={y_c[i]}  D={D[i]:.2f}  loss term={term:.4f}")
print(f"Mean loss: {loss:.4f}")

# --- Triplet (FaceNet 2015) ---
def triplet_loss(a, p, n, margin=0.2):
    d_ap = np.sum((a - p) ** 2, axis=1)
    d_an = np.sum((a - n) ** 2, axis=1)
    losses = np.maximum(0.0, d_ap - d_an + margin)
    return losses.mean(), d_ap, d_an, losses

a = np.array([[0.0, 0.0]] * 3)
p = np.array([[0.3, 0.0], [0.9, 0.0], [0.1, 0.0]])
n = np.array([[1.0, 0.0], [0.5, 0.0], [0.2, 0.0]])
L, dap, dan, per = triplet_loss(a, p, n, margin=0.2)
print("\\n=== Triplet (margin=0.2) ===")
for i in range(3):
    if   dan[i] > dap[i] + 0.2: kind = "easy"
    elif dan[i] > dap[i]:       kind = "semi-hard"
    else:                        kind = "hard"
    print(f"  i={i}  D(a,p)^2={dap[i]:.3f}  D(a,n)^2={dan[i]:.3f}  loss={per[i]:.3f}  [{kind}]")
print(f"Mean loss: {L:.4f}")

# --- Semi-hard negative mining ---
def semi_hard_mine(a, d_ap, neg_bank, margin=0.2):
    dists = np.sum((a - neg_bank) ** 2, axis=1)
    mask = (dists > d_ap) & (dists < d_ap + margin)
    if mask.any():
        idx = np.argmin(dists[mask])
        return np.where(mask)[0][idx], "semi-hard"
    # Fallback to hardest violator (closest negative still within margin)
    viol = dists < d_ap + margin
    if viol.any():
        idx = np.argmax(dists[viol])
        return np.where(viol)[0][idx], "hard-fallback"
    return int(np.argmin(dists)), "no-violation"

anchor = np.array([0.0, 0.0])
pos    = np.array([0.3, 0.0])
d_ap   = np.sum((anchor - pos) ** 2)
neg_bank = np.array([[0.1, 0.0],  # closer than pos (hard)
                     [0.4, 0.0],  # semi-hard
                     [0.5, 0.0],  # semi-hard
                     [2.0, 0.0]]) # easy
print("\\n=== Semi-hard mining ===")
print(f"Anchor-positive D^2 = {d_ap:.3f}")
picked, kind = semi_hard_mine(anchor, d_ap, neg_bank, margin=0.2)
print(f"Picked negative: candidate {picked}  ({kind})")
# Output:
# === Contrastive (margin=1.0) ===
#   pair 0  y=1  D=0.10  loss term=0.0100
#   pair 1  y=1  D=2.00  loss term=4.0000
#   pair 2  y=0  D=2.00  loss term=0.0000
# Mean loss: 1.3367
#
# === Triplet (margin=0.2) ===
#   i=0  D(a,p)^2=0.090  D(a,n)^2=1.000  loss=0.000  [easy]
#   i=1  D(a,p)^2=0.810  D(a,n)^2=0.250  loss=0.760  [hard]
#   i=2  D(a,p)^2=0.010  D(a,n)^2=0.040  loss=0.170  [semi-hard]
# Mean loss: 0.3100
#
# === Semi-hard mining ===
# Anchor-positive D^2 = 0.090
# Picked negative: candidate 1  (semi-hard)`}
      </CodeBlock>

      <Prose>
        Pair 1 in the contrastive example shows the loss doing its job: it is a label-similar pair whose embeddings are far apart (<Code>{"D=2"}</Code>), so the loss is a hefty <Code>{"D² = 4"}</Code>, producing a large gradient that will pull them together. Pair 2 is label-dissimilar and already beyond the margin (<Code>{"D=2 > 1 = m"}</Code>), so its loss is zero — once negatives are far enough, contrastive leaves them alone, which prevents the collapse you would get from always-on repulsion.
      </Prose>

      <Prose>
        In the triplet example, the hard triplet (i=1) has <Code>{"D(a,n)² &lt; D(a,p)²"}</Code> — the negative is closer than the positive — and contributes a large loss of 0.76. The semi-hard triplet (i=2) has the negative slightly farther than the positive but not by the full margin, contributing 0.17. The easy triplet (i=0) is zero. In the mining step, candidate 1 at distance 0.16 is picked because it satisfies <Code>{"D² > D_ap² = 0.09"}</Code> and <Code>{"D² &lt; D_ap² + 0.2 = 0.29"}</Code> — the semi-hard regime Schroff et al. found optimal for FaceNet.
      </Prose>

      <H3>4d. InfoNCE with in-batch negatives</H3>

      <CodeBlock language="python">
{`import numpy as np
np.random.seed(7)

def l2_normalize(x, eps=1e-12):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)

def infonce(queries, keys, tau=0.1):
    """
    queries: (B, D). Assume queries[i] and keys[i] are positive pairs.
    Every other key in the batch is a negative (SimCLR-style in-batch negatives).
    """
    q = l2_normalize(queries)
    k = l2_normalize(keys)
    logits = (q @ k.T) / tau                    # (B, B) cosine similarities / tau
    m = np.max(logits, axis=1, keepdims=True)   # numerically stable logsumexp
    lse = m.squeeze(1) + np.log(np.sum(np.exp(logits - m), axis=1))
    pos = np.diag(logits)                       # positive logit per row
    return np.mean(lse - pos)

B, D = 4, 8
base = np.random.randn(B, D)
queries = base + 0.05 * np.random.randn(B, D)
keys    = base + 0.05 * np.random.randn(B, D)
print("=== InfoNCE (in-batch negatives) ===")
for tau in [1.0, 0.5, 0.1, 0.07]:
    print(f"tau={tau:<5}  loss={infonce(queries, keys, tau=tau):.4f}")
print(f"\\nRandom q/k at tau=0.1: {infonce(np.random.randn(B, D), np.random.randn(B, D), 0.1):.4f}")
print(f"Uniform-guess upper bound -log(1/B) = {np.log(B):.4f}")
# Output:
# === InfoNCE (in-batch negatives) ===
# tau=1.0    loss=0.8204
# tau=0.5    loss=0.5223
# tau=0.1    loss=0.0463
# tau=0.07   loss=0.0133
#
# Random q/k at tau=0.1: 2.5234
# Uniform-guess upper bound -log(1/B) = 1.3863`}
      </CodeBlock>

      <Prose>
        Two things. First, smaller <Code>τ</Code> makes the loss much smaller when the positive pair is genuinely the most similar in the batch — the softmax becomes a spike on the correct index. Second, when the signal is random (no positive structure) and <Code>τ</Code> is small, the loss can exceed the uniform-guess bound: the softmax confidently picks the wrong index, which costs more than guessing uniformly. This is the temperature tradeoff in practice: small <Code>τ</Code> is powerful when your positives are well-structured but punishing when they are not.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <H3>5a. PyTorch canonical losses</H3>

      <CodeBlock language="python">
{`import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)

# --- MSE ---
pred = torch.tensor([1.1, 1.9, 3.2, 3.7])
tgt  = torch.tensor([1.0, 2.0, 3.0, 4.0])
print(f"MSELoss:                      {nn.MSELoss()(pred, tgt).item():.6f}")

# --- BCEWithLogitsLoss with pos_weight (imbalance) ---
logits  = torch.tensor([[-2.0], [1.5], [0.2], [-0.4]])
targets = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
bce_plain    = nn.BCEWithLogitsLoss()
bce_weighted = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([99.0]))   # 1:99 imbalance
print(f"BCEWithLogits plain:          {bce_plain(logits, targets).item():.6f}")
print(f"BCEWithLogits pos_weight=99:  {bce_weighted(logits, targets).item():.6f}")

# --- CrossEntropyLoss with label smoothing ---
logits  = torch.tensor([[ 2.0, 0.5, -1.0],
                        [ 0.1, 0.2,  0.3],
                        [-0.5,-1.0,  2.0]])
targets = torch.tensor([0, 2, 2])
print(f"CE plain:                     {nn.CrossEntropyLoss()(logits, targets).item():.6f}")
print(f"CE label_smoothing=0.1:       {nn.CrossEntropyLoss(label_smoothing=0.1)(logits, targets).item():.6f}")

# --- Focal loss from torchvision ---
from torchvision.ops import sigmoid_focal_loss
fl = sigmoid_focal_loss(torch.tensor([[-2.0], [1.5], [0.2], [-0.4]]),
                        torch.tensor([[0.0], [1.0], [1.0], [0.0]]),
                        alpha=0.25, gamma=2.0, reduction='mean')
print(f"sigmoid_focal_loss a=0.25 g=2: {fl.item():.6f}")
# Output:
# MSELoss:                      0.037500
# BCEWithLogits plain:          0.359874
# BCEWithLogits pos_weight=99:  19.948902
# CE plain:                     0.455709
# CE label_smoothing=0.1:       0.570154
# sigmoid_focal_loss a=0.25 g=2: 0.023824`}
      </CodeBlock>

      <Callout type="info" title="Why BCEWithLogits instead of Sigmoid + BCE">
        Computing <Code>sigmoid(z)</Code> first and then <Code>log(sigmoid(z))</Code> can underflow to <Code>{"log(0) = −∞"}</Code> when <Code>z</Code> is a large negative number — the sigmoid saturates to 0 in float32. The fused form <Code>{"max(z, 0) − z·y + log(1 + exp(−|z|))"}</Code> implemented by <Code>BCEWithLogitsLoss</Code> is stable at all logit magnitudes. Always pass logits, never sigmoid outputs, to your binary loss in production. The same rule holds for <Code>CrossEntropyLoss</Code>: pass logits, not softmax outputs — PyTorch fuses the log-softmax and NLL into a single numerically stable kernel.
      </Callout>

      <H3>5b. Triplet and InfoNCE in PyTorch</H3>

      <CodeBlock language="python">
{`import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(42)

# --- Triplet margin loss ---
# Construct triplets where the negative is close enough to violate margin.
anchor   = torch.randn(4, 8)
positive = anchor + 0.5  * torch.randn(4, 8)
negative = anchor + 0.25 * torch.randn(4, 8)     # negative somewhat close to anchor
tm = nn.TripletMarginLoss(margin=0.2)
print(f"TripletMarginLoss (margin=0.2):  {tm(anchor, positive, negative).item():.6f}")

# --- InfoNCE via F.cross_entropy on a similarity matrix ---
B, D = 4, 16
q = F.normalize(torch.randn(B, D), dim=-1)
k = F.normalize(torch.randn(B, D), dim=-1)
k = F.normalize(k + 0.5 * q, dim=-1)             # inject positive structure
tau = 0.1
logits = (q @ k.T) / tau                         # (B, B)
labels = torch.arange(B)                         # diag is positive
loss   = F.cross_entropy(logits, labels)
print(f"InfoNCE via cross_entropy (tau=0.1): {loss.item():.6f}")
# Output:
# TripletMarginLoss (margin=0.2):  0.933596
# InfoNCE via cross_entropy (tau=0.1): 0.084603`}
      </CodeBlock>

      <Prose>
        The InfoNCE one-liner is the single most important idiom in modern contrastive learning. Take normalized query and key batches, compute their pairwise similarity matrix, divide by temperature, and pass the result to <Code>F.cross_entropy</Code> with diagonal labels. That is SimCLR, MoCo, CLIP, and every major contrastive method — all variants of this four-line pattern with different strategies for generating <Code>q</Code> and <Code>k</Code> and for expanding the pool of negatives.
      </Prose>

      <H3>5c. SentenceTransformers and specialized losses</H3>

      <CodeBlock language="python">
{`# SentenceTransformers wraps all the common embedding losses for fine-tuning.
# (Commentary — API calls shown for reference.)

from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# 1. TripletLoss — anchor, positive, negative triples with a margin
train_examples = [
    InputExample(texts=['anchor text', 'positive text', 'negative text']),
    # ...
]
loss_triplet = losses.TripletLoss(model=model, triplet_margin=0.5)

# 2. ContrastiveLoss — pair-based, margin-based, Hadsell-style
train_pairs = [
    InputExample(texts=['sent A', 'sent B'], label=1.0),   # similar
    InputExample(texts=['sent A', 'sent C'], label=0.0),   # dissimilar
]
loss_contrastive = losses.ContrastiveLoss(model=model)

# 3. MultipleNegativesRankingLoss — SimCLR/InfoNCE with in-batch negatives.
#    THE workhorse for modern bi-encoder training — no explicit negatives needed.
#    Every non-paired item in the batch serves as a negative.
train_pairs_positive_only = [
    InputExample(texts=['question 1', 'relevant doc 1']),
    InputExample(texts=['question 2', 'relevant doc 2']),
]
loss_mnr = losses.MultipleNegativesRankingLoss(model=model)
# Larger batches => more negatives => better representations.
# Typical: batch_size=32-256 for GPU memory, scale_factor=20 (= 1/tau with tau=0.05)

# 4. CosineSimilarityLoss — MSE on cosine(sent1, sent2) vs continuous label in [0,1]
loss_cosine = losses.CosineSimilarityLoss(model=model)

# 5. Angular margin losses for classification on embedding models
# ArcFace / CosFace are typically implemented as custom heads rather than drop-in losses;
# face_recognition_pytorch and pytorch-metric-learning provide them as modules.`}
      </CodeBlock>

      <Callout type="info" title="Picking the right SentenceTransformers loss">
        For retrieval fine-tuning with positive pairs only and no labeled negatives, use <Code>MultipleNegativesRankingLoss</Code> — it is InfoNCE with in-batch negatives and is the default recommendation from the library authors. For labeled pair datasets (with explicit similar/dissimilar labels), <Code>ContrastiveLoss</Code> is appropriate. Reserve <Code>TripletLoss</Code> for datasets already shaped as anchor-positive-negative triples (e.g., traditional face datasets) — the in-batch InfoNCE variant will typically outperform it when you have the choice.
      </Callout>

      <H3>5d. Class imbalance: pos_weight vs sample_weight vs focal</H3>

      <CodeBlock language="python">
{`# Three ways to handle imbalance in PyTorch — pick based on problem shape.

import torch, torch.nn as nn
from torchvision.ops import sigmoid_focal_loss

# Setup: 1:99 imbalance
N = 1000
logits = torch.randn(N, 1)
y = torch.zeros(N, 1); y[:10] = 1.0

# Option 1 — pos_weight: reweight positive class in the loss formula.
#   - Pro: exact, deterministic, no batch-composition dependence.
#   - Con: does not down-weight easy negatives.
#   - Use when: class imbalance is your main issue, examples within each class are similar.
pos_count = y.sum().item()
neg_count = N - pos_count
pos_weight = torch.tensor([neg_count / pos_count])     # = 99.0
loss_pw = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(logits, y)

# Option 2 — WeightedRandomSampler: resample minority class more often.
#   - Pro: easy to compose with any loss; changes batch composition not loss formula.
#   - Con: can overfit to minority (same positives seen many times per epoch).
#   - Use when: you want the same effect but with a different loss (e.g., focal).

# Option 3 — focal loss: down-weight easy examples regardless of class.
#   - Pro: handles both class imbalance AND example difficulty.
#   - Con: two hyperparameters (alpha, gamma), can over-focus on outliers if noise is high.
#   - Use when: dense prediction, extreme imbalance, or noisy labels are ruled out.
loss_focal = sigmoid_focal_loss(logits, y, alpha=0.25, gamma=2.0, reduction='mean')

print(f"pos_weight={pos_weight.item()}")
print(f"BCE with pos_weight : {loss_pw.item():.6f}")
print(f"Focal (a=0.25, g=2) : {loss_focal.item():.6f}")`}
      </CodeBlock>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6a. MSE vs MAE vs Huber on outlier-dense data</H3>

      <Prose>
        Given 7 observations of a scalar — six near zero plus one outlier at 10 — we compute the loss for each constant predictor <Code>c</Code>. MSE is minimized at the mean (1.43, dragged toward the outlier). MAE is minimized at the median (0.10, ignores the outlier). Huber lies between them.
      </Prose>

      <Plot
        label="Loss vs constant predictor c — six points near zero + one outlier at 10"
        xLabel="Constant predictor c"
        yLabel="Loss"
        series={[
          {
            name: "MSE",
            color: colors.gold,
            points: [
              [-1.0, 18.200], [-0.5, 16.021], [0.0, 14.343], [0.5, 13.164],
              [1.0, 12.486], [1.5, 12.307], [2.0, 12.629], [2.5, 13.450],
              [3.0, 14.771], [3.5, 16.593], [4.0, 18.914], [5.0, 25.057],
            ],
          },
          {
            name: "MAE",
            color: colors.green,
            points: [
              [-1.0, 2.429], [-0.5, 1.929], [0.0, 1.600], [0.5, 1.786],
              [1.0, 2.143], [1.5, 2.500], [2.0, 2.857], [2.5, 3.214],
              [3.0, 3.571], [3.5, 3.929], [4.0, 4.286], [5.0, 5.000],
            ],
          },
          {
            name: "Huber (delta=1)",
            color: "#c084fc",
            points: [
              [-1.0, 1.947], [-0.5, 1.564], [0.0, 1.386], [0.5, 1.421],
              [1.0, 1.653], [1.5, 2.000], [2.0, 2.357], [2.5, 2.714],
              [3.0, 3.071], [3.5, 3.429], [4.0, 3.786], [5.0, 4.500],
            ],
          },
        ]}
      />

      <Prose>
        MSE's quadratic penalty on the outlier (magnitude-squared matters) drags its minimum toward <Code>{"c = 1.5"}</Code>. MAE's linear penalty treats the outlier as one point among seven; its minimum at <Code>{"c = 0.0"}</Code> is the median. Huber is linear beyond <Code>|r|&gt;1</Code> (so outliers do not dominate) and quadratic near zero (so it is differentiable and well-behaved for gradient methods). The practical rule: if your error distribution is Gaussian, MSE is optimal; if you have heavy tails or suspected outliers, Huber gives you MSE's smooth gradients near zero without MSE's outlier sensitivity.
      </Prose>

      <H3>6b. Cross-entropy vs focal as p_t varies</H3>

      <Prose>
        The focusing factor <Code>{"(1 − p_t)^γ"}</Code> collapses the loss for well-classified examples. As <Code>γ</Code> grows, the loss becomes a spike on the tail of hard examples. Plotted on a common axis.
      </Prose>

      <Plot
        label="CE vs Focal loss as p_t varies (0.01 to 0.99)"
        xLabel="p_t — probability assigned to true class"
        yLabel="Loss contribution"
        series={[
          {
            name: "CE (gamma=0)",
            color: colors.gold,
            points: [
              [0.01, 4.605], [0.08, 2.556], [0.15, 1.897], [0.25, 1.386],
              [0.35, 1.050], [0.45, 0.799], [0.55, 0.598], [0.65, 0.431],
              [0.75, 0.288], [0.85, 0.163], [0.92, 0.084], [0.99, 0.010],
            ],
          },
          {
            name: "Focal gamma=1",
            color: colors.green,
            points: [
              [0.01, 4.559], [0.08, 2.358], [0.15, 1.612], [0.25, 1.040],
              [0.35, 0.683], [0.45, 0.440], [0.55, 0.269], [0.65, 0.151],
              [0.75, 0.072], [0.85, 0.025], [0.92, 0.007], [0.99, 0.000],
            ],
          },
          {
            name: "Focal gamma=2",
            color: "#c084fc",
            points: [
              [0.01, 4.513], [0.08, 2.175], [0.15, 1.370], [0.25, 0.780],
              [0.35, 0.444], [0.45, 0.242], [0.55, 0.121], [0.65, 0.053],
              [0.75, 0.018], [0.85, 0.004], [0.92, 0.001], [0.99, 0.000],
            ],
          },
          {
            name: "Focal gamma=5",
            color: "#f472b6",
            points: [
              [0.01, 4.380], [0.08, 1.707], [0.15, 0.859], [0.25, 0.326],
              [0.35, 0.121], [0.45, 0.040], [0.55, 0.012], [0.65, 0.003],
              [0.75, 0.001], [0.85, 0.000], [0.92, 0.000], [0.99, 0.000],
            ],
          },
        ]}
      />

      <Prose>
        For <Code>{"p_t = 0.9"}</Code> (an easy example), plain CE contributes 0.105 while focal at <Code>{"γ=2"}</Code> contributes only 0.001 — a 100x reduction. For <Code>{"p_t = 0.1"}</Code> (hard), CE contributes 2.30 and focal 1.87 — only a 1.2x reduction. This is exactly the behavior Lin et al. designed: aggressive suppression of easy examples, near-full retention of hard examples. The <Code>γ=5</Code> curve is extreme and can over-focus on a handful of outliers or noisy labels; <Code>γ=2</Code> is the standard sweet spot validated across dense detection, segmentation, and medical imaging.
      </Prose>

      <H3>6c. Contrastive loss as distance varies</H3>

      <Prose>
        The two branches of contrastive loss, plotted against embedding distance <Code>D</Code>. For positive pairs (<Code>y=1</Code>), the loss is <Code>{"D²"}</Code> — always pulling. For negative pairs (<Code>y=0</Code>, <Code>{"m=1"}</Code>), the loss is <Code>{"max(0, m−D)²"}</Code> — pushes apart until <Code>{"D ≥ m"}</Code>, then goes to zero.
      </Prose>

      <Plot
        label="Contrastive loss as D varies — positive pair (y=1) vs negative pair (y=0, margin=1)"
        xLabel="Embedding distance D"
        yLabel="Loss"
        series={[
          {
            name: "Positive (y=1): D^2",
            color: colors.gold,
            points: [
              [0.0, 0.0000], [0.2, 0.0443], [0.4, 0.1773], [0.6, 0.3989],
              [0.8, 0.7091], [1.0, 1.0000], [1.2, 1.3407], [1.4, 1.8726],
              [1.6, 2.5600], [1.8, 3.2400], [2.0, 4.0000],
            ],
          },
          {
            name: "Negative (y=0, m=1): max(0, m-D)^2",
            color: colors.green,
            points: [
              [0.0, 1.0000], [0.2, 0.6233], [0.4, 0.3352], [0.6, 0.1357],
              [0.8, 0.0249], [1.0, 0.0000], [1.2, 0.0000], [1.4, 0.0000],
              [1.6, 0.0000], [1.8, 0.0000], [2.0, 0.0000],
            ],
          },
        ]}
      />

      <Prose>
        The asymmetry is the whole point. Positives are pulled everywhere — the gradient never disappears. Negatives are only pushed until they are margin-separated; once a negative has been pushed beyond <Code>m</Code>, the loss releases it and no more gradient flows. Without this release, the loss would keep pushing all negatives to infinity, which is degenerate (embeddings could not be finite) and would waste gradient budget on already-separated pairs. The margin is the regularizer that lets the model focus on the pairs that still matter.
      </Prose>

      <H3>6d. Focal vs CE gradient contribution under 1:99 imbalance</H3>

      <Prose>
        Heatmap of per-example loss contribution at different prediction confidence levels, for a hard positive (<Code>y=1</Code>) and an easy negative (<Code>y=0</Code>). Darker means larger contribution to the total loss.
      </Prose>

      <Heatmap
        label="Loss contribution — per-example (CE vs Focal gamma=2)"
        rowLabels={["CE", "Focal γ=2"]}
        colLabels={["y=1, p=0.05 (hard+)", "y=1, p=0.20", "y=1, p=0.50", "y=0, p=0.02 (easy-)", "y=0, p=0.10", "y=0, p=0.30"]}
        matrix={[
          [2.996, 1.609, 0.693, 0.020, 0.105, 0.357],
          [0.676, 0.206, 0.043, 0.000006, 0.000789, 0.024],
        ]}
        colorScale="warm"
      />

      <Prose>
        Under CE, the easy negative (<Code>y=0, p=0.02</Code>) contributes 0.020 and the hard positive (<Code>y=1, p=0.05</Code>) contributes 2.996 — ratio of 150x. Multiply by the 99:1 imbalance and easy negatives dominate the total gradient. Under focal, easy-negative contribution collapses to 6e-6; ratio explodes to 113,000x. Even with 99x more easy negatives than hard positives in the batch, the focal-loss gradient is now dominated by the handful of hard positives. This is the single visualization that explains why RetinaNet worked.
      </Prose>

      <H3>6e. Triplet mining strategies</H3>

      <Prose>
        FaceNet's core insight: mine semi-hard triplets, not random or fully-hard ones. Trace through one batch at convergence-time (margin <Code>α=0.2</Code>).
      </Prose>

      <StepTrace
        label="Triplet mining: random vs semi-hard vs hardest"
        steps={[
          {
            label: "Random mining — most triplets are 'easy'",
            render: () => (
              <Prose>
                Anchor-positive distance squared: 0.09. Randomly sampled negative distances squared: 1.00, 0.80, 1.50, 2.10, 0.70. For margin 0.2, only triplets where <Code>{"D(a,n)² < D(a,p)² + 0.2 = 0.29"}</Code> produce gradient. None of these do — loss is zero. The gradient signal is empty, training stalls. Random mining wastes most of the batch.
              </Prose>
            ),
          },
          {
            label: "Fully-hard mining — choose the closest negative",
            render: () => (
              <Prose>
                Closest negative to anchor has distance squared 0.05 — closer than the positive. Loss = <Code>{"max(0, 0.09 − 0.05 + 0.2) = 0.24"}</Code>. Large gradient. But the negative is "hard" for a reason: often it is a label-noise example or an outlier that happens to look like the anchor. Always picking the hardest negative makes training unstable; Schroff et al. saw embeddings collapse with fully-hard mining because one noisy triplet could yank the whole embedding in a bad direction.
              </Prose>
            ),
          },
          {
            label: "Semi-hard mining — the FaceNet sweet spot",
            render: () => (
              <Prose>
                Pick a negative with <Code>{"D(a,n)² > D(a,p)²"}</Code> (the margin is still active but the negative is not closer than the positive). From the same batch, candidates satisfying <Code>{"0.09 < D² < 0.29"}</Code> are at distances 0.12 and 0.18. Either produces a bounded-but-meaningful loss (0.17 and 0.11 respectively). Training is stable, the gradient is informative, and the embedding geometry improves monotonically.
              </Prose>
            ),
          },
          {
            label: "Batch-hard mining (TriNet, 2017)",
            render: () => (
              <Prose>
                Within each identity in the batch, pick the <em>hardest positive</em> (farthest same-identity example) and the <em>hardest negative</em> (closest different-identity example). Hermans, Beyer, and Leibe's "In Defense of the Triplet Loss for Person Re-Identification" (arXiv:1703.07737, 2017) showed this outperforms semi-hard when batches contain multiple examples per identity. It has since become the standard for ReID.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        The right loss is determined by the problem shape: regression vs classification vs retrieval, label structure, imbalance level, and scale.
      </Prose>

      <StepTrace
        label="Which loss to use"
        steps={[
          {
            label: "Regression — MSE, MAE, or Huber",
            render: () => (
              <Prose>
                Use MSE when errors are Gaussian and you have few outliers — it is the maximum likelihood estimator, its gradients are smooth, it has a closed-form solution for linear models, and it is computationally trivial. Use MAE (L1) when errors are heavy-tailed or outliers are expected — the median-based minimizer is outlier-robust. Use Huber when you want both: quadratic near zero (for smooth gradients and clean convergence) and linear far from zero (for outlier tolerance). The crossover parameter <Code>δ</Code> controls where the switch happens; <Code>{"δ = 1.0"}</Code> is a safe default if your targets are scaled to unit variance. For quantile regression (predict the 90th percentile, not the mean), use pinball/quantile loss.
              </Prose>
            ),
          },
          {
            label: "Binary classification — BCEWithLogits, possibly with pos_weight or focal",
            render: () => (
              <Prose>
                Default to <Code>BCEWithLogitsLoss</Code> — it is numerically stable, correct, and the reference implementation. For moderate imbalance (1:5 to 1:20), add <Code>pos_weight=N_neg/N_pos</Code>. For severe imbalance (1:100+), use focal loss from <Code>torchvision.ops.sigmoid_focal_loss</Code> with <Code>{"alpha=0.25, gamma=2.0"}</Code>. For multi-label classification (each example can have multiple positive labels), <Code>BCEWithLogitsLoss</Code> with per-class output is still correct; label-smoothing on BCE (sometimes called "soft BCE") has been reported to help with label noise.
              </Prose>
            ),
          },
          {
            label: "Multiclass classification — CrossEntropy, label smoothing for noisy labels",
            render: () => (
              <Prose>
                Default to <Code>nn.CrossEntropyLoss</Code>. Add <Code>{"label_smoothing=0.1"}</Code> when training on large datasets where label noise is a concern (ImageNet-scale, web-scraped data, translation corpora). Label smoothing produces slightly calibrated probabilities and consistently improves validation accuracy by 0.2–0.8 percentage points on image classification. For extreme multi-class (1M+ classes), use sampled softmax, hierarchical softmax, or noise-contrastive estimation — computing the full softmax is prohibitive at that scale.
              </Prose>
            ),
          },
          {
            label: "Extreme imbalance or dense prediction — Focal",
            render: () => (
              <Prose>
                Use focal loss for: one-stage object detection (RetinaNet, YOLOv4+), semantic segmentation with rare classes (medical images, satellite imagery), binary classification with 1:100+ imbalance, and any setting where easy negatives overwhelm the gradient. Use <Code>{"γ=2"}</Code> unless evidence suggests otherwise. Lower <Code>γ</Code> (1.0) if your labels have significant noise; higher <Code>γ</Code> (5) risks overweighting outliers. Pair focal with <Code>α</Code> class-balancing if you also have class imbalance; usually <Code>{"α=0.25"}</Code> works.
              </Prose>
            ),
          },
          {
            label: "Embedding/retrieval (small data) — Triplet",
            render: () => (
              <Prose>
                Use triplet loss when: you have anchor-positive-negative triples (face recognition, speaker verification, signature matching), your dataset is small-to-medium (10k–1M triples), and you can implement semi-hard mining. Margin <Code>{"α ∈ [0.1, 0.5]"}</Code> for normalized embeddings, larger if unnormalized. Triplet with batch-hard mining is still state of the art for person re-identification and works well for speaker embeddings. Drawback: triplet mining doubles the code complexity of the training loop and is hyperparameter-sensitive.
              </Prose>
            ),
          },
          {
            label: "Embedding/retrieval (large data) — InfoNCE / SupCon / MultipleNegativesRanking",
            render: () => (
              <Prose>
                Use InfoNCE/SimCLR-style losses when: you can use large batches (512+), your positives come for free (augmentations, paired data, click logs), and you want to avoid the complexity of triplet mining. For supervised settings with class labels, use SupCon (Khosla 2020) — it extends InfoNCE to multiple positives per query within a batch. For bi-encoder retrieval fine-tuning (BEIR-style), <Code>MultipleNegativesRankingLoss</Code> from SentenceTransformers is the proven default. Temperature: <Code>{"τ ∈ [0.05, 0.1]"}</Code>.
              </Prose>
            ),
          },
          {
            label: "Face recognition or fine-grained verification — ArcFace / CosFace",
            render: () => (
              <Prose>
                For classification-via-embedding where you have identity labels and want maximally discriminative embeddings, use ArcFace. It replaces the standard softmax logit <Code>{"cos θ"}</Code> with <Code>{"cos(θ + m)"}</Code>, adding an angular margin <Code>m</Code> in the geodesic sense. ArcFace (<Code>{"m = 0.5"}</Code>, <Code>{"s = 64"}</Code>) is the industry standard for face recognition and has become the default for whale identification, plant species ID, and many other fine-grained classification problems where test classes may overlap with training classes. CosFace uses a subtractive margin <Code>{"cos θ − m"}</Code>; both work well. SphereFace is the earlier, less-stable ancestor.
              </Prose>
            ),
          },
          {
            label: "Ranking and learning to rank — listwise losses",
            render: () => (
              <Prose>
                For search ranking and recommendation ordering tasks, pairwise losses (RankNet) and listwise losses (ListNet, LambdaRank, LambdaLoss) outperform pointwise approaches like MSE or CE. LambdaMART is still the gradient-boosting standard. In neural retrieval, ApproxNDCG losses and Plackett-Luce losses are used but pure InfoNCE with appropriate negatives usually suffices. If exact ordering of top-k matters, optimize a smoothed version of NDCG or MRR directly.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          8. WHAT SCALES
          ====================================================================== */}
      <H2>8. What scales</H2>

      <H3>8.1 Cross-entropy at large vocabulary</H3>

      <Prose>
        Cross-entropy over <Code>C</Code> classes costs <Code>{"O(n · C)"}</Code> per step: compute <Code>C</Code> logits, do log-sum-exp over <Code>C</Code> terms, pick the correct one. For <Code>{"C = 10"}</Code> (CIFAR-10) this is free; for <Code>{"C = 1000"}</Code> (ImageNet) it is cheap; for <Code>{"C = 32000"}</Code> (GPT-2 vocabulary) it is measurable but manageable; for <Code>{"C = 50,000"}</Code>–<Code>{"200,000"}</Code> (T5, mT5, Llama) the final linear layer and the log-sum-exp begin to dominate runtime. When <Code>{"C > 1M"}</Code> (word-level LMs, recommendation systems with millions of items), the full softmax becomes infeasible and the standard mitigations kick in.
      </Prose>

      <Prose>
        Sampled softmax (Jean et al., 2015) computes the numerator for the true class plus the softmax over a random sample of <Code>k</Code> negatives, with importance weights to correct for the sampling bias. Noise-contrastive estimation (Gutmann & Hyvärinen, 2010) reformulates as binary classification: distinguish the true target from noise samples. Both reduce the per-step cost from <Code>{"O(C)"}</Code> to <Code>{"O(k)"}</Code> where <Code>{"k ≪ C"}</Code> (typically 25–500). Hierarchical softmax groups classes in a tree and pays <Code>{"O(log C)"}</Code> per step; it has mostly been replaced by sampled softmax in modern practice because sampling is simpler and gives better probability estimates.
      </Prose>

      <H3>8.2 InfoNCE and batch size</H3>

      <Prose>
        InfoNCE's quality depends on the number of negatives — mathematically, the loss is a lower bound on mutual information up to <Code>{"log K"}</Code>, so doubling <Code>K</Code> gives about <Code>{"log 2 ≈ 0.69"}</Code> nats of tighter bound. SimCLR's breakthrough result on ImageNet required batches of 4096 or 8192, each giving ~8191 free negatives. Smaller batches (256) work but represent a real quality drop. This batch-size dependence is a scaling problem: 8192-image batches of high-resolution images need 32–128 GPUs with model parallelism or gradient accumulation.
      </Prose>

      <Prose>
        MoCo (He et al., 2020) sidesteps this via a memory bank: maintain a queue of recent keys from past batches (65k–131k entries), use them all as negatives. A momentum-updated encoder keeps the queue's key representations consistent with the current model. This lets you have hundreds of thousands of negatives on a single GPU. SimCLR's direct approach and MoCo's queue approach converge to similar performance at sufficient compute; MoCo is more memory-efficient, SimCLR is simpler.
      </Prose>

      <H3>8.3 Triplet mining at scale</H3>

      <Prose>
        Naively, the number of possible triplets is <Code>{"O(N³)"}</Code> — for a dataset of 1M images, that is <Code>{"10^{18}"}</Code> triplets, infeasible to enumerate. Two scalable strategies exist. Batch-hard mining (Hermans et al., 2017): form batches containing <Code>P</Code> identities with <Code>K</Code> images each (typical <Code>{"P=16, K=4"}</Code>), then for each anchor pick the hardest positive and hardest negative within the batch — <Code>{"O(P·K·N)"}</Code> per batch. Offline mining: every few epochs, embed the full dataset, use a nearest-neighbor index (Faiss, ScaNN) to find the top-K hardest negatives for each anchor, train on those triplets for a while, then re-mine. Offline mining is the gold standard for face recognition but adds engineering complexity; batch-hard mining is the pragmatic default.
      </Prose>

      <H3>8.4 Loss kernel cost in modern training</H3>

      <Prose>
        For a typical LLM training step (7B parameters, sequence length 4096, batch size 16), the forward/backward pass through the transformer is ~95% of runtime, while the cross-entropy kernel on <Code>{"50000 × 16 × 4096"}</Code> logits is ~3%. This is why loss choice rarely dominates compute for supervised training of transformers; the dominant cost is the model, not the loss. However, for encoder-only contrastive training with large batch sizes, the similarity matrix is <Code>{"O(B²)"}</Code> and the loss becomes a non-trivial fraction of compute — optimized fused kernels (xformers, FlashAttention-style fused softmax) help here.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES
          ====================================================================== */}
      <H2>9. Failure modes</H2>

      <H3>9.1 MSE pulled by a single outlier</H3>

      <Prose>
        A single mislabeled <Code>{"y = 10^6"}</Code> in a dataset of otherwise modest targets will pull the MSE-minimizing predictor toward it quadratically. In a linear regression on <Code>n=1000</Code> samples with true scale of 10, one outlier at 1e6 contributes <Code>{"(10^6)^2 = 10^{12}"}</Code> to the sum of squares — 10 million times more than a typical squared error. The fit becomes essentially "predict the outlier." The symptom: validation MSE is dominated by one or two examples and removing them improves the fit dramatically. The fix: detect and remove outliers, use MAE or Huber, or transform the target (e.g., log-transform for multiplicative noise).
      </Prose>

      <H3>9.2 Cross-entropy and majority-class collapse</H3>

      <Prose>
        Under severe class imbalance (1:99 or worse), the optimal CE prediction for every example is the majority class probability. A model that outputs <Code>{"p=0.01"}</Code> for every example gets CE <Code>{"≈ 0.056"}</Code>, which is better than any model that tries to distinguish classes and gets even a few wrong with high confidence. The symptom: training loss decreases nicely, but the model predicts zero positives. Accuracy looks great (99% — matching the baseline) but recall is zero. The fix: use <Code>pos_weight</Code> in BCE, use focal loss, use class-balanced sampling, or evaluate on AUC/F1/PR-AUC rather than accuracy.
      </Prose>

      <H3>9.3 Focal gamma too high</H3>

      <Prose>
        Focal loss with <Code>{"γ ≥ 5"}</Code> over-focuses on the hardest examples. If your labels have any noise — which real-world labels always do — the hardest examples are often the noisy ones. Training on them aggressively pushes the model toward memorizing label noise. Symptoms: training loss drops rapidly, validation loss increases after a few epochs, validation accuracy decreases. Fix: reduce <Code>γ</Code> to 2.0 (the Lin et al. default), add label smoothing, or ensemble focal-trained models with a standard CE model to average out noise memorization.
      </Prose>

      <H3>9.4 Triplet collapse when margin is too small</H3>

      <Prose>
        If margin <Code>α</Code> is zero or negative, the loss is satisfied by the trivial solution <Code>{"f(x) = 0"}</Code> for all <Code>x</Code> — all distances are zero, the "D(a,p) less than D(a,n) + 0" condition is vacuously true, the loss is zero. The embeddings collapse to a single point. The symptom: training converges suspiciously fast, embeddings cluster at the origin, downstream accuracy is chance level. Fix: use <Code>{"α > 0"}</Code> (typically 0.1–0.5), L2-normalize embeddings so they live on the unit hypersphere (which prevents collapse to zero), and monitor the average pairwise distance during training — if it goes to zero, stop and investigate.
      </Prose>

      <H3>9.5 Contrastive without temperature or normalization</H3>

      <Prose>
        InfoNCE and modern contrastive losses require L2-normalized embeddings (so similarity is bounded in <Code>{"[−1, 1]"}</Code>) and a temperature parameter <Code>τ</Code>. Without normalization, the logit magnitudes are controlled by embedding norms, not by angular similarity — the model can achieve low loss by simply making one embedding have huge norm, which has nothing to do with representation quality. Without temperature, the softmax is insensitive to small similarity differences (logits are all between 0 and 1, so exp(0) and exp(1) differ by only e≈2.7x). Symptoms: loss plateaus, representations are low-rank, downstream accuracy is poor. Fix: always apply <Code>F.normalize(x, dim=-1)</Code> before similarity, always divide similarity by a temperature (0.05–0.2 typical).
      </Prose>

      <H3>9.6 Label smoothing hurts calibration on some tasks</H3>

      <Prose>
        Label smoothing <Code>{"ε=0.1"}</Code> redistributes target probability from the true class (now <Code>{"1 − ε"}</Code>) to the other classes (each <Code>{"ε/(C−1)"}</Code>). It generally improves top-1 accuracy on large datasets but damages the model's probability calibration: the model learns to avoid probabilities close to 1.0, so confidence estimates are systematically pushed toward the interior of the simplex. For downstream tasks that use predicted probabilities (calibrated thresholds, uncertainty estimation, active learning), label smoothing can hurt. Müller et al. ("When Does Label Smoothing Help?", 2019) analyzed when it helps and when it does not. Fix: use label smoothing for accuracy-driven metrics, remove it when calibration matters, or apply post-hoc temperature scaling to recover calibration.
      </Prose>

      <H3>9.7 Pos_weight being set to the wrong ratio</H3>

      <Prose>
        <Code>pos_weight</Code> in <Code>BCEWithLogitsLoss</Code> is the weight multiplier for the positive-class term, not a probability. The correct value for exact rebalancing is <Code>{"N_neg / N_pos"}</Code> (how many times rarer positives are). Setting it to <Code>{"N_pos / N_neg"}</Code> (the reverse) amplifies the majority class and makes imbalance worse. Setting it to the raw ratio (like 0.01 for 1:99) also makes things worse. Symptom: model still collapses to majority class despite <Code>pos_weight</Code>. Fix: double-check the formula and compute from class counts explicitly: <Code>{"pos_weight = torch.tensor([n_neg / n_pos])"}</Code>.
      </Prose>

      <H3>9.8 Forgetting reduction='mean' assumption</H3>

      <Prose>
        PyTorch losses default to <Code>{"reduction='mean'"}</Code> — they average across the batch. If you mix losses with different reductions, the per-example loss magnitudes silently scale differently, and one loss can dominate by an order of magnitude without any warning. Common bug: using <Code>{"F.mse_loss(..., reduction='sum')"}</Code> for one term and <Code>{"F.cross_entropy(...)"}</Code> (mean) for another in a multi-task objective. The sum-reduced MSE scales with batch size while the mean-reduced CE does not, so larger batches silently re-weight the losses. Fix: always use consistent reductions across losses in a combined objective, and verify scales at initialization by logging each loss term separately.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All citations verified against primary venues. Read roughly in chronological order for the intellectual lineage.
      </Prose>

      <StepTrace
        label="Primary literature"
        steps={[
          {
            label: "Gauss 1809 — MSE from Gaussian likelihood",
            render: () => (
              <Prose>
                Gauss, C.F. (1809). <em>Theoria Motus Corporum Coelestium in Sectionibus Conicis Solem Ambientium</em>. Hamburg: Friedrich Perthes and I.H. Besser. Book II, Section III gives the first rigorous derivation that the method of least squares is the maximum likelihood estimator under Gaussian noise — Gauss's motivation was computing planetary orbits from Piazzi's 1801 observations of Ceres. The English translation by Charles Henry Davis (1857) is available via the HathiTrust Digital Library. Stigler's <em>The History of Statistics</em> (1986) covers the Gauss-Legendre priority dispute over least squares and the subsequent development of the Gauss-Markov theorem. The modern form of MSE as a training loss for neural networks carries the same mathematical justification Gauss laid down two centuries ago.
              </Prose>
            ),
          },
          {
            label: "Shannon 1948 — Information theory and cross-entropy",
            render: () => (
              <Prose>
                Shannon, C.E. (1948). "A Mathematical Theory of Communication." <em>Bell System Technical Journal</em>, 27(3):379–423 and 27(4):623–656. This is the founding document of information theory. Shannon defines entropy, mutual information, channel capacity, and implicitly the concept of cross-entropy (via relative entropy / KL divergence). Available as a PDF via Harvard's <Code>math.harvard.edu/~ctm/home/text/others/shannon</Code>. The connection to machine learning losses was made much later: in the 1980s, Rumelhart-Hinton-Williams's backpropagation paper (1986) used squared error; the shift to cross-entropy for classification was consolidated through the 1990s with papers like Solla et al.'s "Accelerated Learning in Layered Neural Networks" and Bishop's <em>Neural Networks for Pattern Recognition</em> (1995), which explicitly framed classification as maximum likelihood under Bernoulli/categorical outputs.
              </Prose>
            ),
          },
          {
            label: "Hadsell, Chopra, LeCun 2006 — Contrastive loss",
            render: () => (
              <Prose>
                Hadsell, R., Chopra, S., and LeCun, Y. (2006). "Dimensionality Reduction by Learning an Invariant Mapping." In <em>Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR)</em>, volume 2, pages 1735–1742. DOI: 10.1109/CVPR.2006.100. Available via NYU at <Code>yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf</Code>. The foundational paper for pair-based metric learning. Hadsell et al. introduced the margin-based contrastive loss, applied it to a siamese convolutional network, and showed that learned embeddings could invert geometric transformations of MNIST and NORB images. This paper is the common ancestor of every modern representation-learning loss.
              </Prose>
            ),
          },
          {
            label: "Chopra, Hadsell, LeCun 2005 — Siamese network precursor",
            render: () => (
              <Prose>
                An immediate precursor worth noting: Chopra, S., Hadsell, R., and LeCun, Y. (2005). "Learning a similarity metric discriminatively, with application to face verification." <em>CVPR</em>, 1:539–546. The siamese network architecture and a closely-related contrastive-style loss for face verification. This 2005 paper established the engineering template that FaceNet and triplet loss would later refine.
              </Prose>
            ),
          },
          {
            label: "Schroff, Kalenichenko, Philbin 2015 — FaceNet / triplet loss",
            render: () => (
              <Prose>
                Schroff, F., Kalenichenko, D., and Philbin, J. (2015). "FaceNet: A Unified Embedding for Face Recognition and Clustering." arXiv:1503.03832. <em>CVPR 2015</em>, pages 815–823. The paper that established triplet loss with semi-hard mining as the standard for face recognition. FaceNet hit 99.63% on Labeled Faces in the Wild — a new SOTA — and 95.12% on YouTube Faces DB. Key engineering contributions: (1) formulate face recognition as metric learning on 128-dim L2-normalized embeddings, (2) train end-to-end on 200M images with a single triplet loss, (3) mine semi-hard negatives within each batch rather than offline. The paper also contributed the observation that harder-than-semi-hard triplets destabilize training — a finding replicated across many metric-learning domains.
              </Prose>
            ),
          },
          {
            label: "Lin et al. 2017 — Focal loss",
            render: () => (
              <Prose>
                Lin, T.-Y., Goyal, P., Girshick, R., He, K., and Dollár, P. (2017). "Focal Loss for Dense Object Detection." arXiv:1708.02002. <em>ICCV 2017</em>, pages 2980–2988. The paper introduced focal loss as the key innovation enabling RetinaNet, a one-stage detector that matched two-stage accuracy. Ablations in the paper tested <Code>{"γ ∈ {0, 0.5, 1, 2, 5}"}</Code> and <Code>{"α ∈ {0.25, 0.5, 0.75}"}</Code> on COCO; the sweet spot was <Code>{"γ=2, α=0.25"}</Code>. The paper also explicitly compared focal to online hard example mining (OHEM) and showed focal superior with less engineering complexity. Dollár's Facebook AI Research page hosts the authoritative PDF; the Detectron2 and MMDetection implementations are the reference for practitioners.
              </Prose>
            ),
          },
          {
            label: "Van den Oord, Li, Vinyals 2018 — InfoNCE (CPC)",
            render: () => (
              <Prose>
                Van den Oord, A., Li, Y., and Vinyals, O. (2018). "Representation Learning with Contrastive Predictive Coding." arXiv:1807.03748. The paper introduced InfoNCE and proved it is a lower bound on mutual information between the query and positive. CPC applied this to speech, images, and text — the unified formulation of "noise-contrastive estimation + cross-entropy classification over candidates" that became the foundation of SimCLR, MoCo, CLIP, and essentially every subsequent self-supervised method.
              </Prose>
            ),
          },
          {
            label: "Deng et al. 2019 — ArcFace",
            render: () => (
              <Prose>
                Deng, J., Guo, J., Xue, N., and Zafeiriou, S. (2019). "ArcFace: Additive Angular Margin Loss for Deep Face Recognition." arXiv:1801.07698 (posted 2018, published <em>CVPR 2019</em>, pages 4690–4699). ArcFace adds an angular margin <Code>m</Code> directly to the angle between the feature and its class weight vector in the softmax logit: <Code>{"cos θ → cos(θ + m)"}</Code>. Unlike SphereFace's multiplicative margin, the additive angular margin is more stable. The paper demonstrates state-of-the-art performance on MS1M, VGGFace2, MegaFace, and IJB-B/C benchmarks with the canonical <Code>{"m = 0.5, s = 64"}</Code>. ArcFace became the de facto standard for face recognition after 2019 and underpins nearly every commercial face recognition system's embedding model.
              </Prose>
            ),
          },
          {
            label: "Chen, Kornblith, Norouzi, Hinton 2020 — SimCLR",
            render: () => (
              <Prose>
                Chen, T., Kornblith, S., Norouzi, M., and Hinton, G. (2020). "A Simple Framework for Contrastive Learning of Visual Representations." arXiv:2002.05709. <em>ICML 2020</em>. SimCLR showed that self-supervised representation learning could match supervised pretraining on ImageNet if you used (a) strong data augmentations (random crop + color jitter is the key pair), (b) a projection head between the encoder and the InfoNCE loss, (c) very large batch sizes (4096–8192) or a memory bank, (d) the cosine similarity in L2-normalized space with temperature ~0.1. The paper's batch-size ablation is definitive: larger batches strictly better. SimCLR v2 (Chen et al., 2020b) extended this to semi-supervised distillation pipelines.
              </Prose>
            ),
          },
          {
            label: "Khosla et al. 2020 — Supervised Contrastive Learning",
            render: () => (
              <Prose>
                Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P., Maschinot, A., Liu, C., and Krishnan, D. (2020). "Supervised Contrastive Learning." arXiv:2004.11362. <em>NeurIPS 2020</em>. Extends InfoNCE to the supervised setting where a batch contains multiple positives per anchor (all examples sharing the same class label). The loss averages over all positive pairs for each anchor rather than picking one. SupCon beats cross-entropy on ImageNet top-1 by ~1 percentage point and is more robust to label noise and adversarial examples. The paper also provides the theoretical connection between SupCon and the "class-collapse" phenomenon: minimizing SupCon asymptotically pulls all examples of a class to a single point on the hypersphere.
              </Prose>
            ),
          },
          {
            label: "Hermans, Beyer, Leibe 2017 — In Defense of the Triplet Loss",
            render: () => (
              <Prose>
                Hermans, A., Beyer, L., and Leibe, B. (2017). "In Defense of the Triplet Loss for Person Re-Identification." arXiv:1703.07737. Introduced batch-hard triplet mining (within-batch hardest positive + hardest negative per identity) and showed that, with proper mining, triplet loss is competitive with the then-popular classification-based approaches for person ReID. This paper and its mining strategy remain the practitioner's default for ReID and fine-grained retrieval tasks with per-identity labels.
              </Prose>
            ),
          },
          {
            label: "Huber 1964 — Robust regression",
            render: () => (
              <Prose>
                Huber, P.J. (1964). "Robust Estimation of a Location Parameter." <em>Annals of Mathematical Statistics</em>, 35(1):73–101. The foundational paper on robust regression — introduces the Huber loss and proves it is minimax-optimal among certain classes of contamination models. Still the standard reference for understanding why and when to replace MSE with a bounded-influence loss.
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
        Work through each before reading the answer. The exercises test derivation, implementation, decision-making, and debugging.
      </Prose>

      <H3>Exercise 1 (derivation)</H3>
      <Prose>
        Show that minimizing MSE is equivalent to maximum likelihood estimation under Gaussian noise. Then show that minimizing MAE is equivalent to MLE under Laplace noise. What does this imply about which loss you should use if your empirical residuals look heavy-tailed?
      </Prose>
      <Callout type="answer" title="Answer 1">
        Under <Code>{"y = f(x;θ) + ε"}</Code> with <Code>{"ε ∼ N(0, σ²)"}</Code>, the log-likelihood of a dataset is <Code>{"Σ log p(y_i | x_i, θ) = −(1/2σ²) Σ(y_i − ŷ_i)² + const"}</Code>. Maximizing this over θ is minimizing the sum of squared errors — MSE.
        Under Laplace noise <Code>{"p(ε) = (1/2b) exp(−|ε|/b)"}</Code>, the log-likelihood is <Code>{"−(1/b) Σ|y_i − ŷ_i| + const"}</Code>. Maximizing is minimizing sum of absolute errors — MAE.
        The Laplace distribution has heavier tails than Gaussian — rare large deviations are more probable under Laplace. If your residuals show heavy tails (kurtosis much greater than 3, or visible outliers in a Q-Q plot against normal), MAE is a more principled choice than MSE — it will not be dominated by outliers. In practice, Huber loss is often preferred: it behaves like MSE in the bulk (smooth gradients) and like MAE in the tails (outlier-robust).
      </Callout>

      <H3>Exercise 2 (derivation)</H3>
      <Prose>
        Derive the gradient of softmax cross-entropy loss <Code>{"L = −log(exp(z_y) / Σ exp(z_c))"}</Code> with respect to the logit <Code>{"z_c"}</Code>. Why does this derivation justify using the fused <Code>CrossEntropyLoss</Code> in PyTorch rather than applying softmax and then NLL separately?
      </Prose>
      <Callout type="answer" title="Answer 2">
        Write <Code>{"L = −z_y + log Σ_c exp(z_c)"}</Code>. Take <Code>{"∂L/∂z_c"}</Code>:
        <ul>
          <li>If <Code>c = y</Code>: <Code>{"∂L/∂z_y = −1 + exp(z_y)/Σ exp(z_c) = p_y − 1"}</Code>.</li>
          <li>If <Code>c ≠ y</Code>: <Code>{"∂L/∂z_c = exp(z_c)/Σ exp(z_c) = p_c"}</Code>.</li>
        </ul>
        Unified: <Code>{"∂L/∂z_c = p_c − 1{c = y} = p_c − y_c"}</Code> (where <Code>{"y_c"}</Code> is the one-hot indicator).
        The fused kernel justification: the forward computation uses <Code>{"log Σ exp(z_c)"}</Code> (log-sum-exp), which is numerically stable via max-subtraction. The backward computation needs only <Code>{"softmax(z) − y_one_hot"}</Code>. Neither the forward nor the backward pass requires explicitly computing <Code>{"softmax(z)"}</Code> and then its log — doing so can underflow (the softmax of a large logit vector can have components as small as <Code>{"exp(−700) = 0"}</Code> in float64, and <Code>{"log(0) = −∞"}</Code>). The fused <Code>F.cross_entropy</Code> kernel combines log-softmax and NLL into one numerically stable computation.
      </Callout>

      <H3>Exercise 3 (applied)</H3>
      <Prose>
        You are training a binary tumor classifier on a medical imaging dataset. There are 50,000 images, of which 200 contain tumors. Plain BCE gives you 99.6% accuracy and 0% recall. Explain why, then give three alternative training objectives and pick the one you would try first. Provide the PyTorch code.
      </Prose>
      <Callout type="answer" title="Answer 3">
        The ratio is 1:249 (200 positives vs 49,800 negatives). Under plain BCE, the model that always predicts <Code>{"p ≈ 0.004"}</Code> minimizes the loss — it gets 99.6% accuracy on the majority class and essentially never activates. Accuracy is a degenerate metric here; you need recall, precision, and PR-AUC.
        Three alternatives:
        <ol>
          <li><strong>BCEWithLogits with pos_weight = 249</strong>: makes each positive's loss count 249x more, exactly rebalancing.</li>
          <li><strong>Focal loss</strong>: downweights easy negatives (which dominate), focuses gradient on hard examples.</li>
          <li><strong>Oversampling + plain BCE</strong>: use a WeightedRandomSampler to balance batch composition; the loss then sees a balanced problem.</li>
        </ol>
        First try: <strong>BCEWithLogitsLoss with pos_weight</strong>. It is exact, deterministic, has no extra hyperparameters to tune, and will not lose signal to stochastic sampling. If it underperforms, escalate to focal.
        <Code>{"pos_weight = torch.tensor([49800/200])  # = 249.0"}</Code>
        <Code>{"criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)"}</Code>
        Evaluate using PR-AUC and recall at fixed precision (e.g., 95% precision), not raw accuracy.
      </Callout>

      <H3>Exercise 4 (debugging)</H3>
      <Prose>
        You are training a sentence embedding model with <Code>MultipleNegativesRankingLoss</Code> (InfoNCE). Your loss drops to near 0 in 100 steps and stays there. Your downstream retrieval accuracy is chance level. What is the most likely bug, and what is the fix?
      </Prose>
      <Callout type="answer" title="Answer 4">
        Most likely: your "negatives" are not actually negatives — the embedding model has found a shortcut that makes positives trivially identifiable. Two common causes:
        <ol>
          <li><strong>Duplicate positives in the batch</strong>: if two of the "positive pairs" in a batch are near-duplicates of each other (e.g., paraphrases, or worse, outright copies), then each serves as a false negative for the other — the "negative" is actually a positive. The model learns to distinguish based on surface form.</li>
          <li><strong>Leaked features</strong>: your positive pairs share some surface artifact that is not shared with negatives (a timestamp, an ID, a category token). The model learns the shortcut, not the semantics.</li>
        </ol>
        Fix: (a) Deduplicate your positive pairs — check that no two batch members have the same text. (b) Log the top-5 most-similar in-batch items for a query during training; if they are all paraphrases of the query (but none are the intended positive), you have leakage. (c) Add a no-duplicate sampler or use hard-negative mining to force the model past the shortcut. (d) Verify downstream accuracy using a held-out test set with no training leakage.
      </Callout>

      <H3>Exercise 5 (conceptual + math)</H3>
      <Prose>
        A colleague claims: "Triplet loss and InfoNCE are fundamentally different — one uses Euclidean distance, the other uses dot products." Is this claim correct? Show that InfoNCE with cosine similarity and triplet loss with normalized embeddings have a deep mathematical connection. Give the connection and the practical reason InfoNCE is now preferred over triplet for most new work.
      </Prose>
      <Callout type="answer" title="Answer 5">
        The claim is superficial. With L2-normalized embeddings, squared Euclidean distance and cosine similarity are related by <Code>{"‖a − b‖² = 2 − 2·cos(a, b)"}</Code>. So "minimize <Code>{"‖a − p‖²"}</Code> and maximize <Code>{"‖a − n‖²"}</Code>" with normalized embeddings is equivalent to "maximize <Code>{"cos(a, p)"}</Code> and minimize <Code>{"cos(a, n)"}</Code>" — the same geometric objective.
        Triplet loss with margin <Code>α</Code> enforces a hard constraint: <Code>{"D(a,p)² − D(a,n)² + α ≤ 0"}</Code>, clipped at zero. InfoNCE with temperature <Code>τ</Code> is a soft probabilistic version: <Code>{"−log(exp(sim(a,p)/τ) / Σ exp(sim(a,k)/τ))"}</Code>. As <Code>{"τ → 0"}</Code>, the softmax becomes a hard argmax and InfoNCE approaches a hinge-like objective similar to triplet. As <Code>{"τ → ∞"}</Code>, it becomes uniform and the gradient vanishes. The temperature plays the role of a soft margin.
        Why InfoNCE has won most new work:
        <ol>
          <li><strong>No mining</strong>: every non-positive in the batch is a negative automatically. Triplet requires dedicated mining code that is hyperparameter-sensitive.</li>
          <li><strong>Scaling</strong>: InfoNCE improves monotonically with batch size (more negatives = tighter MI bound). Triplet needs explicit mining to get the same benefit.</li>
          <li><strong>Stability</strong>: softmax gradients are smooth; triplet's max(0, ·) is piecewise linear with non-differentiable kinks.</li>
          <li><strong>Theory</strong>: InfoNCE is a mutual information lower bound. Triplet does not have a clean probabilistic interpretation.</li>
        </ol>
        Triplet is still preferred when: you have explicit anchor-positive-negative triples (no way to get in-batch negatives), or small data where one triplet per step is enough compute. Otherwise, InfoNCE / MultipleNegativesRankingLoss is the modern default.
      </Callout>

    </div>
  ),
};

export default lossFunctionsContent;
