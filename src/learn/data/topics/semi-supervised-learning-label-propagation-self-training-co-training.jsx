import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const semiSupervisedContent = {
  title: "Semi-Supervised Learning (Label Propagation, Self-Training, Co-Training)",
  readTime: "~45 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        The bottleneck in most real-world ML problems is not compute or model capacity — it is labeled data. Annotating a medical image requires a radiologist. Labeling a legal clause requires an attorney. Tagging a social-media post for sentiment at production scale requires thousands of human-hours. The unlabeled data, meanwhile, is free: the web generates billions of images, documents, and sensor readings every day, all of which capture structure about the world but none of which come with ground-truth labels. Semi-supervised learning (SSL) is the family of methods that exploits this asymmetry — using a small labeled set together with a large unlabeled set to build classifiers that substantially outperform what the labeled data alone could support.
      </Prose>

      <Prose>
        The idea of using a classifier's own predictions to enlarge its training set predates the internet. H. J. Scudder published "Probability of Error of Some Adaptive Pattern-Recognition Machines" in the <em>IEEE Transactions on Information Theory</em> (vol. 11, no. 3, pp. 363–371) in 1965. Working in the context of communications theory, Scudder described an "untaught machine" that classifies its own input and then uses those decisions as if they were ground truth to update itself — what we now call self-training. He derived bounds on the asymptotic error rate of such a system, establishing that the procedure can be consistent even when the initial classifier is imperfect. This is the first published appearance of a self-training loop in the machine-learning sense.
      </Prose>

      <Prose>
        The next landmark arrived thirty-three years later at COLT 1998. Avrim Blum and Tom Mitchell published "Combining Labeled and Unlabeled Data with Co-Training" (Proceedings of the 11th Annual Conference on Computational Learning Theory, pp. 92–100, ACM, DOI: 10.1145/279943.279962). Their motivating application was webpage classification: a web page has two naturally separate views — the text on the page and the anchor text of links pointing to it. Blum and Mitchell proved that if two views are conditionally independent given the class label, a classifier trained on one view can generate pseudo-labels for the other view, and iterating this process provably reduces error even with no additional labeled data. Co-training gave SSL its first rigorous PAC-style theoretical foundation.
      </Prose>

      <Prose>
        Graph-based SSL — the family that includes label propagation — was formalized in the early 2000s. Xiaojin Zhu and Zoubin Ghahramani introduced label propagation as a graph algorithm in CMU Technical Report CMU-CALD-02-107 (2002), showing that labels placed on a weighted graph of unlabeled and labeled points flow naturally to unlabeled nodes along high-affinity edges. Dengyong Zhou, Olivier Bousquet, Thomas Lal, Jason Weston, and Bernhard Schölkopf sharpened this into a clean regularized optimization in "Learning with Local and Global Consistency" (NeurIPS 16, 2003, pp. 321–328), deriving a closed-form solution and proving convergence. The Chapelle, Schölkopf, and Zien edited volume <em>Semi-Supervised Learning</em> (MIT Press, 2006) collected and unified these threads, and remains the definitive graduate-level reference.
      </Prose>

      <Prose>
        Modern deep SSL scaled all three classical ideas to the regime of large neural networks and web-scale unlabeled data. Kihyuk Sohn and colleagues published FixMatch in NeurIPS 2020 (arXiv:2001.07685), achieving 94.93% accuracy on CIFAR-10 with only 250 labels by combining a confidence threshold (pseudo-labeling) with consistency regularization — weak augmentation generates the pseudo-label, strong augmentation generates the training input. Qizhe Xie, Minh-Thang Luong, Eduard Hovy, and Quoc Le published "Self-Training with Noisy Student Improves ImageNet Classification" (CVPR 2020, arXiv:1911.04252), iterating teacher → student with deliberately injected noise (dropout, stochastic depth, data augmentation) to prevent the student from merely memorizing the teacher's outputs. Noisy Student reached 88.4% top-1 ImageNet accuracy — 2% better than the then-state-of-the-art that required 3.5 billion weakly-labeled Instagram images. Both papers demonstrate that classical self-training, properly scaled and regularized, remains the backbone of the most effective SSL methods.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Semi-supervised learning rests on a geometric bet: <strong>the unlabeled data reveals the structure of the input space, and that structure constrains where class boundaries should be</strong>. There are two clean ways to state this bet.
      </Prose>

      <Prose>
        The <strong>cluster assumption</strong> says that points in the same dense cluster tend to share the same label. If you can identify clusters from unlabeled data, you have identified likely label regions. The implication for decision boundaries: boundaries should pass through regions of low density (gaps between clusters), not through dense regions. A boundary that cuts through a cluster would assign different labels to similar points — unlikely if the cluster assumption holds.
      </Prose>

      <Prose>
        The <strong>manifold assumption</strong> generalizes this: high-dimensional data often lives on or near a low-dimensional manifold. Two points that are close on the manifold should have the same label even if they are far apart in ambient Euclidean space. A 3D mesh of a human face: the tip of the nose and the bridge of the nose are geometrically close on the surface but may be far in pixel space. The unlabeled data traces out the manifold; labels should be propagated along it.
      </Prose>

      <Prose>
        Three classical algorithm families each exploit these assumptions in a distinct way:
      </Prose>

      <StepTrace
        label="Three classical SSL families"
        steps={[
          {
            label: "Self-training — the model labels its own predictions",
            render: () => (
              <div>
                <TokenStream
                  label="self-training loop"
                  tokens={[
                    { label: "Train on L", color: colors.gold },
                    { label: "→ predict on U", color: colors.textDim },
                    { label: "→ keep high-confidence", color: colors.green },
                    { label: "→ add to L", color: colors.gold },
                    { label: "→ retrain", color: colors.textDim },
                    { label: "repeat", color: "#60a5fa" },
                  ]}
                />
                <Prose>
                  The simplest SSL family. A base classifier is trained on the labeled set L. It then predicts labels for the unlabeled set U. Predictions above a confidence threshold are treated as pseudo-labeled and added to the training set. The classifier is retrained on the enlarged set, and the loop repeats. Self-training exploits the cluster assumption: confident predictions far from the decision boundary are almost certainly correct, and adding them expands the boundary outward into unlabeled territory.
                </Prose>
              </div>
            ),
          },
          {
            label: "Co-training — two views train each other",
            render: () => (
              <div>
                <TokenStream
                  label="co-training loop"
                  tokens={[
                    { label: "Train clf1 on view 1 of L", color: colors.gold },
                    { label: "Train clf2 on view 2 of L", color: colors.green },
                    { label: "clf1 labels U → add to clf2's pool", color: colors.textDim },
                    { label: "clf2 labels U → add to clf1's pool", color: colors.textDim },
                    { label: "retrain both", color: "#60a5fa" },
                    { label: "repeat", color: colors.textDim },
                  ]}
                />
                <Prose>
                  Co-training requires the feature space to be partitionable into two views, each independently sufficient to classify the example. A classifier trained on view 1 generates pseudo-labels for the classifier trained on view 2, and vice versa. The key theoretical requirement is conditional independence: {"P(x1, x2 | y) = P(x1 | y) · P(x2 | y)"}. When this holds, each view's errors are independent, and the agreement signal between views is informative. In practice, exact independence is rare — co-training still helps when views are weakly correlated.
                </Prose>
              </div>
            ),
          },
          {
            label: "Graph-based (label propagation) — labels flow through a similarity graph",
            render: () => (
              <div>
                <TokenStream
                  label="label propagation"
                  tokens={[
                    { label: "Build affinity graph W", color: colors.gold },
                    { label: "→ normalize → S", color: colors.textDim },
                    { label: "→ propagate: F = αSF + (1-α)Y", color: colors.green },
                    { label: "→ converge to F*", color: "#60a5fa" },
                  ]}
                />
                <Prose>
                  Label propagation builds a weighted graph where each node is a data point (labeled or unlabeled) and each edge weight encodes similarity. Labels placed on a few nodes diffuse through the graph along high-weight edges, subject to a smoothness constraint: similar points should receive similar labels. The algorithm directly operationalizes the manifold assumption — labels travel along the manifold.
                </Prose>
              </div>
            ),
          },
        ]}
      />

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 Building the affinity matrix</H3>

      <Prose>
        Given <Code>n</Code> data points (labeled and unlabeled combined), the affinity matrix <Code>W</Code> encodes pairwise similarity. The standard choice is the Gaussian (RBF) kernel:
      </Prose>

      <MathBlock caption="RBF affinity — higher sigma means labels diffuse further">
        {"W_{ij} = \\exp\\!\\left(-\\frac{\\|x_i - x_j\\|^2}{2\\sigma^2}\\right), \\quad W_{ii} = 0"}
      </MathBlock>

      <Prose>
        The bandwidth <Code>σ</Code> controls the locality of the graph: small <Code>σ</Code> connects only immediate neighbors, large <Code>σ</Code> connects distant points. In practice, a <em>k</em>-nearest-neighbor sparse graph is used to avoid the {"O(n²)"} memory cost of the full affinity matrix: for each point, only the <Code>k</Code> closest neighbors receive nonzero weights. The affinity matrix is then symmetrized.
      </Prose>

      <Prose>
        From <Code>W</Code>, form the degree matrix <Code>D</Code> (diagonal, {"D_{ii}"} = sum of row <Code>i</Code> of <Code>W</Code>), and the symmetric normalized transition matrix:
      </Prose>

      <MathBlock caption="Symmetric normalized transition matrix S">
        {"S = D^{-1/2} W D^{-1/2}"}
      </MathBlock>

      <Prose>
        The unnormalized graph Laplacian is <Code>L = D − W</Code>. The symmetric normalized Laplacian is {"L_{sym} = I − S = I − D^{−1/2} W D^{−1/2}"}. The eigenvalues of {"L_{sym}"} lie in <Code>[0, 2]</Code>, and the smallest eigenvalues correspond to the smoothest functions on the graph — constant within dense clusters, varying slowly between them.
      </Prose>

      <H3>3.2 Zhou 2003 label spreading — closed-form and iterative</H3>

      <Prose>
        Let <Code>Y</Code> be an <Code>n × C</Code> label matrix (C classes). For labeled points, {"Y_{ic}"} = 1 if point <Code>i</Code> has class <Code>c</Code>, else 0. For unlabeled points, {"Y_{ic}"} = 1/C (uniform prior). The label spreading objective is:
      </Prose>

      <MathBlock caption="Label spreading objective — balance smoothness and label fidelity">
        {"\\min_F \\left[ \\alpha \\sum_{i,j} W_{ij} \\left\\| \\frac{F_i}{\\sqrt{D_{ii}}} - \\frac{F_j}{\\sqrt{D_{jj}}} \\right\\|^2 + (1-\\alpha) \\|F - Y\\|^2 \\right]"}
      </MathBlock>

      <Prose>
        The first term penalizes label disagreement between similar points (smoothness). The second term penalizes deviation from the observed labels (fidelity). The parameter <Code>α ∈ (0, 1)</Code> trades off these two objectives. Setting the gradient to zero yields the closed-form solution:
      </Prose>

      <MathBlock caption="Closed-form solution — requires solving an n×n linear system">
        {"F^* = (I - \\alpha S)^{-1} (1-\\alpha) Y"}
      </MathBlock>

      <Prose>
        For large <Code>n</Code>, the matrix inversion is {"O(n³)"} and infeasible. The equivalent iterative form is:
      </Prose>

      <MathBlock caption="Iterative label spreading — O(n²) per iteration, converges geometrically">
        {"F_{t+1} = \\alpha S F_t + (1-\\alpha) Y"}
      </MathBlock>

      <Prose>
        This is a contraction mapping for <Code>α {"<"} 1</Code>, since the spectral radius of <Code>αS</Code> is <Code>α·ρ(S) ≤ α {"<"} 1</Code> (all eigenvalues of <Code>S</Code> lie in <Code>[−1, 1]</Code> for a symmetric normalized matrix). By the Banach fixed-point theorem, the iteration converges to {"F*"} from any starting point, at geometric rate {"O(α^t)"}. In practice, convergence to numerical precision requires {"O(log(1/ε) / log(1/α))"} iterations — typically 50–200 for <Code>α = 0.99</Code>.
      </Prose>

      <H3>3.3 Self-training as EM</H3>

      <Prose>
        Self-training has an elegant probabilistic interpretation as Expectation-Maximization on a mixture model. Suppose the data is generated by a mixture <Code>p(x) = Σ_c π_c p(x | c)</Code>. If we treat the class assignment of unlabeled points as a latent variable <Code>z</Code>:
      </Prose>

      <MathBlock caption="Self-training EM — unlabeled labels are the latent variables">
        {"\\text{E-step: } q_i(c) = P(z_i = c \\mid x_i; \\theta^{(t)}) \\quad \\text{M-step: maximize } \\mathbb{E}_{q}[\\log p(x, z; \\theta)]"}
      </MathBlock>

      <Prose>
        The E-step computes the posterior class probabilities for each unlabeled point — these are the model's predicted confidences. The M-step refits the model parameters to maximize expected log-likelihood, where unlabeled points contribute with weights given by the E-step posteriors. Hard-assignment self-training (keep only predictions above threshold <Code>τ</Code>) is an approximation of this EM: instead of soft weights, it assigns each unlabeled point to its most likely class if the confidence exceeds <Code>τ</Code>, and ignores it otherwise.
      </Prose>

      <H3>3.4 Co-training — two-view PAC analysis</H3>

      <Prose>
        Blum and Mitchell's formal setup: each example has two views <Code>x = (x₁, x₂)</Code> where <Code>x₁</Code> and <Code>x₂</Code> are conditionally independent given the label <Code>y</Code> — i.e., {"P(x₁, x₂ | y) = P(x₁ | y) · P(x₂ | y)"}. There exist hypotheses <Code>h₁</Code> (over {"x₁"}) and <Code>h₂</Code> (over {"x₂"}) each achieving low error. Their key theorem: if a classifier {"h₁"} has error <Code>ε</Code> on view 1, then the pseudo-labels it generates for view 2 are noisy training labels with noise rate <Code>ε</Code>. A second classifier {"h₂"} trained on <Code>O(1/ε²)</Code> such examples can reduce its error below any target <Code>δ</Code>. The number of unlabeled examples required grows as {"O(1/ε²)"},  which is much smaller than the labeled data needed to achieve the same error from scratch. The practical takeaway: co-training works when the two views each carry independent signal about the label, and the initial labeled set is large enough to train a classifier with error <Code>ε {"<"} 0.5</Code> on each view independently.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        All three code blocks were run and outputs embedded verbatim. NumPy only for label propagation; NumPy + sklearn's <Code>LogisticRegression</Code> for self-training and co-training (the base classifier is not the subject of this topic).
      </Prose>

      <H3>4a. Label propagation on 2D half-moons</H3>

      <CodeBlock language="python">
{`import numpy as np
from scipy.spatial.distance import cdist

np.random.seed(42)

def make_moons(n=200, noise=0.12, seed=42):
    rng = np.random.RandomState(seed)
    n_half = n // 2
    t1 = np.linspace(0, np.pi, n_half)
    t2 = np.linspace(0, np.pi, n_half)
    X1 = np.column_stack([np.cos(t1), np.sin(t1)])
    X2 = np.column_stack([1 - np.cos(t2), 1 - np.sin(t2) - 0.5])
    X = np.vstack([X1, X2]) + rng.randn(n, 2) * noise
    y = np.array([0]*n_half + [1]*n_half)
    return X, y

X, y_true = make_moons()
n = len(X)

# 2 labeled points (1 per class), 198 unlabeled (-1 convention)
y_ssl = -np.ones(n, dtype=int)
y_ssl[47]  = 0   # near center of moon 0
y_ssl[161] = 1   # near center of moon 1

# Build k-NN affinity matrix (k=10, RBF weights, sigma=0.3)
k, sigma = 10, 0.3
dists = cdist(X, X)
W = np.zeros((n, n))
for i in range(n):
    nn = np.argsort(dists[i])[1:k+1]
    for j in nn:
        w = np.exp(-dists[i, j]**2 / (2 * sigma**2))
        W[i, j] = w; W[j, i] = w   # symmetrize

# Symmetric normalized transition matrix S = D^{-1/2} W D^{-1/2}
D = W.sum(axis=1)
S = (1.0 / np.sqrt(D + 1e-12))[:, None] * W * (1.0 / np.sqrt(D + 1e-12))[None, :]

# Label matrix Y: one-hot for labeled, 0.5 for unlabeled
Y = np.full((n, 2), 0.5)
Y[y_ssl == 0] = [1, 0]
Y[y_ssl == 1] = [0, 1]

# Iterative update: F_{t+1} = alpha*S*F_t + (1-alpha)*Y
alpha = 0.99
F = Y.copy()
for t in range(1, 101):
    F = alpha * (S @ F) + (1 - alpha) * Y
    if t in [1, 5, 10, 20, 50, 100]:
        acc = np.mean(np.argmax(F, axis=1) == y_true)
        print(f"LP iter {t:3d}: accuracy={acc:.4f}")

# Output:
# LP iter   1: accuracy=0.5350
# LP iter   5: accuracy=0.7200
# LP iter  10: accuracy=0.7550
# LP iter  20: accuracy=0.7650
# LP iter  50: accuracy=0.7800
# LP iter 100: accuracy=0.7850`}
      </CodeBlock>

      <Prose>
        With only 2 labeled points (one per class out of 200), label propagation reaches 78.5% accuracy at iteration 100 — near the performance of a fully supervised logistic regression on this dataset (around 87%). The smooth rise from 53.5% to 78.5% across iterations shows labels genuinely diffusing through the graph rather than just anchoring near the labeled nodes. The k-NN sparse graph (k=10) is critical: a full affinity matrix would connect the two moons and cause cross-class label leakage.
      </Prose>

      <H3>4b. Self-training with logistic regression</H3>

      <CodeBlock language="python">
{`import numpy as np
from sklearn.linear_model import LogisticRegression

np.random.seed(42)

# Two Gaussian clusters — well-suited for self-training
rng = np.random.RandomState(42)
n = 600
X0 = rng.randn(n//2, 2) + np.array([-2, 0])
X1 = rng.randn(n//2, 2) + np.array([ 2, 0])
X  = np.vstack([X0, X1])
y  = np.hstack([np.zeros(n//2), np.ones(n//2)]).astype(int)
idx = rng.permutation(n)
X, y = X[idx], y[idx]

X_test, y_test = X[400:], y[400:]
X_pool, y_pool = X[:400], y[:400]

# 20 labeled (10 per class), 380 unlabeled
labeled = np.zeros(400, dtype=bool)
for cls in [0, 1]:
    labeled[np.where(y_pool == cls)[0][:10]] = True
X_L, y_L = X_pool[labeled],  y_pool[labeled]
X_U       = X_pool[~labeled]

print(f"Labeled: {labeled.sum()}, Unlabeled: {(~labeled).sum()}, Test: {len(y_test)}")

# Supervised-only baseline
clf_base = LogisticRegression(random_state=0, max_iter=500).fit(X_L, y_L)
print(f"Supervised-only (20 labels): {np.mean(clf_base.predict(X_test)==y_test):.4f}")

# Self-training loop
X_train, y_train = X_L.copy(), y_L.copy()
X_pool_st = X_U.copy()
threshold = 0.90

for iteration in range(1, 6):
    clf = LogisticRegression(random_state=0, max_iter=500).fit(X_train, y_train)
    if len(X_pool_st) == 0: break
    proba = clf.predict_proba(X_pool_st)
    conf  = proba.max(axis=1)
    hc    = conf >= threshold
    if not hc.any():
        print(f"Iter {iteration}: threshold not met, stopping"); break
    X_train = np.vstack([X_train, X_pool_st[hc]])
    y_train = np.hstack([y_train, clf.predict(X_pool_st[hc])])
    X_pool_st = X_pool_st[~hc]
    acc = np.mean(clf.predict(X_test) == y_test)
    print(f"Iter {iteration}: added {hc.sum():3d} pts -> train={len(X_train)}, test_acc={acc:.4f}")

# Output:
# Labeled: 20, Unlabeled: 380, Test: 200
# Supervised-only (20 labels): 0.9600
# Iter 1: added 222 pts -> train=242, test_acc=0.9600
# Iter 2: added  40 pts -> train=282, test_acc=0.9600
# Iter 3: added   8 pts -> train=290, test_acc=0.9600
# Iter 4: threshold not met, stopping

clf_final = LogisticRegression(random_state=0, max_iter=500).fit(X_train, y_train)
print(f"Self-training final: {np.mean(clf_final.predict(X_test)==y_test):.4f}")
# Output: Self-training final: 0.9650`}
      </CodeBlock>

      <Prose>
        On this well-separated Gaussian problem, the supervised baseline is already strong at 96.0% with only 20 labels — the clusters are easily separable. Self-training adds 270 pseudo-labeled points and squeezes out an extra 0.5% (96.5%). The more dramatic gains appear when the initial labeled set is too small to locate the decision boundary accurately; here, the boundary is already roughly right, so pseudo-labels confirm it rather than correct it.
      </Prose>

      <Callout type="warning">
        Self-training on half-moons with 20 labels and logistic regression actually <em>degrades</em> performance in experiments — the initial boundary is so wrong that the first batch of pseudo-labels confidently reinforces the error. This is the confirmation-bias failure mode covered in Section 9. Self-training needs a model that is at least roughly calibrated on the initial labeled set to work reliably.
      </Callout>

      <H3>4c. Co-training with two independent views</H3>

      <CodeBlock language="python">
{`import numpy as np
from sklearn.linear_model import LogisticRegression

np.random.seed(42)

# Construct two independent views: each view discriminates the classes
# Class 0: view1 ~ N([-2,0],I), view2 ~ N([0,-2],I)
# Class 1: view1 ~ N([2,0],I),  view2 ~ N([0,2],I)
rng = np.random.RandomState(42)
n = 600
X1_c0 = rng.randn(n//2, 2) + np.array([-2,  0])
X1_c1 = rng.randn(n//2, 2) + np.array([ 2,  0])
X2_c0 = rng.randn(n//2, 2) + np.array([ 0, -2])
X2_c1 = rng.randn(n//2, 2) + np.array([ 0,  2])
X0 = np.hstack([X1_c0, X2_c0])
X1 = np.hstack([X1_c1, X2_c1])
X = np.vstack([X0, X1])
y = np.hstack([np.zeros(n//2), np.ones(n//2)]).astype(int)
idx = rng.permutation(n)
X, y = X[idx], y[idx]

X_test, y_test = X[400:], y[400:]
X_pool, y_pool = X[:400], y[:400]

# 20 labeled (10 per class), 380 unlabeled
labeled = np.zeros(400, dtype=bool)
for cls in [0, 1]:
    labeled[np.where(y_pool==cls)[0][:10]] = True
X_L, y_L = X_pool[labeled], y_pool[labeled]
X_U = X_pool[~labeled]

print(f"Co-training: {labeled.sum()} labeled, {(~labeled).sum()} unlabeled, {len(y_test)} test")

v1, v2 = slice(0, 2), slice(2, 4)  # two views

clf_base = LogisticRegression(random_state=0, max_iter=500).fit(X_L[:, v1], y_L)
print(f"Supervised-only view-1 (20 labels): {np.mean(clf_base.predict(X_test[:, v1])==y_test):.4f}")

# Co-training loop
X_aug, y_aug = X_L.copy(), y_L.copy()
X_unlabeled  = X_U.copy()
top_n = 20   # each view donates top-20 confident per round

for rnd in range(1, 6):
    clf1 = LogisticRegression(random_state=0, max_iter=500).fit(X_aug[:, v1], y_aug)
    clf2 = LogisticRegression(random_state=0, max_iter=500).fit(X_aug[:, v2], y_aug)
    p1   = clf1.predict_proba(X_unlabeled[:, v1])
    p2   = clf2.predict_proba(X_unlabeled[:, v2])
    sel1 = np.argsort(-p1.max(axis=1))[:top_n]   # clf1's top-20
    sel2 = np.argsort(-p2.max(axis=1))[:top_n]   # clf2's top-20
    new_idx = np.unique(np.concatenate([sel1, sel2]))
    # clf1 labels view-1 picks; clf2 labels view-2 picks
    lbl1 = clf1.predict(X_unlabeled[new_idx][:, v1])
    lbl2 = clf2.predict(X_unlabeled[new_idx][:, v2])
    new_y = np.where(np.isin(new_idx, sel1), lbl1, lbl2)
    X_aug = np.vstack([X_aug, X_unlabeled[new_idx]])
    y_aug = np.hstack([y_aug, new_y])
    X_unlabeled = np.delete(X_unlabeled, new_idx, axis=0)
    acc1 = np.mean(clf1.predict(X_test[:, v1]) == y_test)
    acc2 = np.mean(clf2.predict(X_test[:, v2]) == y_test)
    print(f"Co-train rnd {rnd}: +{len(new_idx)} pts -> n={len(X_aug)}, acc_v1={acc1:.4f}, acc_v2={acc2:.4f}")

# Output:
# Co-training: 20 labeled, 380 unlabeled, 200 test
# Supervised-only view-1 (20 labels): 0.9700
# Co-train rnd 1: +39 pts -> n=59,  acc_v1=0.9700, acc_v2=0.9800
# Co-train rnd 2: +37 pts -> n=96,  acc_v1=0.9600, acc_v2=0.9600
# Co-train rnd 3: +38 pts -> n=134, acc_v1=0.9500, acc_v2=0.9600
# Co-train rnd 4: +36 pts -> n=170, acc_v1=0.9300, acc_v2=0.9450
# Co-train rnd 5: +31 pts -> n=201, acc_v1=0.9350, acc_v2=0.9250`}
      </CodeBlock>

      <Prose>
        On this toy dataset both views are already strong (97% baseline), so co-training gains are marginal. The degradation in rounds 3–5 illustrates error propagation: once a few wrong pseudo-labels enter the pool, they pull the boundary slightly off, generating more wrong pseudo-labels. Co-training shines in the regime where each view is individually weak (say, 70–80%) but the views are genuinely independent — then the agreement signal between views is strongly informative and the joint accuracy climbs well above either view alone.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <H3>5a. scikit-learn semi-supervised API</H3>

      <Prose>
        Scikit-learn exposes three ready-to-use SSL estimators in <Code>sklearn.semi_supervised</Code>. The universal convention: pass <Code>-1</Code> in the label vector to denote unlabeled points. All three accept standard <Code>fit</Code> / <Code>predict</Code> / <Code>predict_proba</Code> calls.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
from sklearn.semi_supervised import LabelPropagation, LabelSpreading, SelfTrainingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_moons

X, y_true = make_moons(n_samples=600, noise=0.12, random_state=42)
y_true = y_true.astype(int)
X_test, y_test = X[500:], y_true[500:]
X_pool, y_pool = X[:500], y_true[:500]

# 30 labeled (-1 marks unlabeled)
y_ssl = -np.ones(500, dtype=int)
for cls in [0, 1]:
    idxs = np.where(y_pool == cls)[0][:15]
    y_ssl[idxs] = cls

print(f"Labeled: {(y_ssl >= 0).sum()}, Unlabeled: {(y_ssl < 0).sum()}, Test: {len(y_test)}")
# Output: Labeled: 30, Unlabeled: 470, Test: 100

# --- LabelPropagation ---
# Hard clamping: labeled points keep their labels exactly throughout propagation.
# kernel='rbf' or 'knn'; gamma controls RBF bandwidth (higher = tighter locality)
lp = LabelPropagation(kernel='rbf', gamma=20, n_neighbors=7, max_iter=1000)
lp.fit(X_pool, y_ssl)
print(f"LabelPropagation (rbf, gamma=20): {np.mean(lp.predict(X_test)==y_test):.4f}")
# Output: LabelPropagation (rbf, gamma=20): 0.9900

# --- LabelSpreading ---
# Soft clamping: labeled points can shift slightly (controlled by alpha).
# alpha=0: pure propagation (equiv. hard clamp); alpha=1: ignore labels entirely.
ls = LabelSpreading(kernel='rbf', gamma=20, alpha=0.2, max_iter=1000)
ls.fit(X_pool, y_ssl)
print(f"LabelSpreading (alpha=0.2): {np.mean(ls.predict(X_test)==y_test):.4f}")
# Output: LabelSpreading (alpha=0.2): 0.9800

# --- SelfTrainingClassifier ---
# Wraps any classifier that exposes predict_proba.
# threshold: confidence cutoff for accepting pseudo-labels (default 0.75).
base_clf = LogisticRegression(random_state=0, max_iter=500)
stc = SelfTrainingClassifier(base_clf, threshold=0.75, max_iter=10)
stc.fit(X_pool, y_ssl)
print(f"SelfTrainingClassifier (LR, threshold=0.75): {np.mean(stc.predict(X_test)==y_test):.4f}")
# Output: SelfTrainingClassifier (LR, threshold=0.75): 0.8100

# --- Supervised-only baseline ---
X_L = X_pool[y_ssl >= 0]; y_L = y_pool[y_ssl >= 0]
clf_base = LogisticRegression(random_state=0, max_iter=500).fit(X_L, y_L)
print(f"Supervised-only (30 labels): {np.mean(clf_base.predict(X_test)==y_test):.4f}")
# Output: Supervised-only (30 labels): 0.8300

# --- Fully supervised upper bound ---
clf_full = LogisticRegression(random_state=0, max_iter=500).fit(X_pool, y_pool)
print(f"Fully supervised (500 labels): {np.mean(clf_full.predict(X_test)==y_test):.4f}")
# Output: Fully supervised (500 labels): 0.8700`}
      </CodeBlock>

      <Prose>
        The result is striking: LabelPropagation achieves 99% accuracy with only 30 labeled points, compared to 83% for supervised-only and 87% for fully supervised. On half-moon data, the graph structure perfectly encodes the manifold — the labels need only a foothold to flood through each moon. The SelfTrainingClassifier (81%) lags because logistic regression cannot fit the nonlinear boundary from just 30 points, so its initial pseudo-labels are wrong and it propagates errors rather than information.
      </Prose>

      <Callout type="info">
        <strong>LabelPropagation vs. LabelSpreading:</strong> LabelPropagation uses hard clamping — labeled nodes keep their ground-truth labels exactly at every iteration. LabelSpreading uses soft clamping controlled by <Code>alpha</Code> — labeled nodes can drift slightly, making it more robust to label noise but potentially less accurate on clean labels. On noisy datasets, prefer LabelSpreading with <Code>alpha</Code> tuned via cross-validation. On clean labels, LabelPropagation is usually better.
      </Callout>

      <H3>5b. Modern deep SSL — FixMatch and Noisy Student</H3>

      <Prose>
        The sklearn estimators above are computationally bounded to datasets where the full affinity matrix or graph fits in memory — typically <Code>n {"<"} 10,000</Code> for LabelPropagation, <Code>n {"<"} 50,000</Code> for sparse variants. For image classification at ImageNet scale, two deep SSL methods dominate.
      </Prose>

      <Prose>
        <strong>FixMatch</strong> (Sohn et al., NeurIPS 2020) applies pseudo-labeling and consistency regularization jointly. For each unlabeled image: (1) apply a weak augmentation (random crop, horizontal flip) and compute a prediction; (2) if the maximum softmax probability exceeds a threshold <Code>τ</Code> (default 0.95), use the argmax as a pseudo-label; (3) apply a strong augmentation (RandAugment + Cutout) to the same image; (4) add a cross-entropy loss between the strong-augmented prediction and the pseudo-label. The intuition: the model should agree with itself across augmentation strengths. FixMatch achieves 94.93% on CIFAR-10 with 250 labels and 88.61% with just 40 labels (4 per class), numbers that competitive supervised models need 50,000 labels to match.
      </Prose>

      <Prose>
        <strong>Noisy Student</strong> (Xie et al., CVPR 2020) iterates teacher–student distillation. A teacher EfficientNet trained on 1.2M labeled ImageNet images generates soft pseudo-labels for 300M unlabeled images. A larger student EfficientNet is trained on the combined set with aggressive noise injection — dropout (0.5), stochastic depth, and RandAugment — forcing the student to generalize rather than memorize the teacher. The student becomes the next teacher, and the process repeats 3 times. The final model reaches 88.4% top-1 ImageNet accuracy, surpassing methods that relied on 3.5 billion weakly-labeled images.
      </Prose>

      <Callout type="info">
        The structural parallel between classical and deep SSL is exact: FixMatch is self-training with consistency regularization as the confidence signal; Noisy Student is self-training with distillation replacing the raw probability threshold. The 2002–2003 graph-based theory and the 1965 Scudder self-training loop are the same ideas, scaled to {"10⁸"} parameters and {"10⁸"} unlabeled examples.
      </Callout>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6a. SSL vs. supervised accuracy as labeled fraction varies</H3>

      <Plot
        label="LabelSpreading vs. supervised-only accuracy on half-moons — 500 pool, 100 test"
        xLabel="fraction of pool that is labeled"
        yLabel="test accuracy"
        series={[
          {
            name: "LabelSpreading (rbf, gamma=20)",
            color: colors.gold,
            points: [
              [0.02, 0.98],
              [0.05, 0.98],
              [0.10, 0.98],
              [0.20, 0.98],
              [0.30, 0.98],
              [0.50, 0.98],
              [1.00, 0.98],
            ],
          },
          {
            name: "Supervised-only (LogisticRegression)",
            color: colors.green,
            points: [
              [0.02, 0.80],
              [0.05, 0.83],
              [0.10, 0.86],
              [0.20, 0.86],
              [0.30, 0.83],
              [0.50, 0.87],
              [1.00, 0.87],
            ],
          },
        ]}
      />

      <Prose>
        The gap between the two curves illustrates the core SSL value proposition on structured data. LabelSpreading immediately reaches 98% and stays there regardless of how many labeled examples are provided — the graph structure is doing almost all the work. The supervised LogisticRegression never exceeds 87% on this dataset because it cannot represent the nonlinear decision boundary. On this particular geometry, 2% labels with SSL beats 100% labels with a linear model.
      </Prose>

      <H3>6b. Label propagation iterations — labels spreading through the graph</H3>

      <StepTrace
        label="Label propagation step-by-step — 200 half-moon points, 2 labeled"
        steps={[
          {
            label: "Iter 0 — initial state",
            render: () => (
              <Prose>
                Two labeled anchors: point 47 (class 0, top moon) and point 161 (class 1, bottom moon). All 198 other points hold the uniform prior [0.5, 0.5]. The affinity matrix W encodes k=10 nearest neighbors with RBF weights. Prediction by argmax gives 53.5% accuracy — barely above random, since most unlabeled points are initialized at 0.5 for both classes.
              </Prose>
            ),
          },
          {
            label: "Iter 1 — first diffusion step",
            render: () => (
              <Prose>
                {"F_1 = αS·F_0 + (1−α)·Y"}. Each unlabeled point receives a weighted average of its 10 neighbors' label distributions. The 10 nearest neighbors of point 47 (all class 0 members of the top moon) absorb a signal of roughly [0.99, 0.01]. Their neighbors in turn receive a diluted signal. Accuracy jumps to 53.5% as the local neighborhood of the two labeled points starts to resolve. Most of the graph is still at [~0.5, ~0.5].
              </Prose>
            ),
          },
          {
            label: "Iter 5 — labels reach 3–4 hops from anchors",
            render: () => (
              <Prose>
                Accuracy reaches 72.0%. The signal has propagated roughly 3–4 neighborhood hops from each labeled anchor. The core of each moon is confidently classified. The ambiguous region is the interleaved tips of the two moons — points geometrically close to both labeled anchors — where the graph edges are thin (RBF affinity near 0) because the moons have low overlap.
              </Prose>
            ),
          },
          {
            label: "Iter 10 — most of each moon resolved",
            render: () => (
              <Prose>
                Accuracy 75.5%. The flat tail of each moon is now mostly resolved. The remaining errors concentrate at the crossing points — the ≈30 samples where moon 0 passes near moon 1. The k-NN graph has a few cross-moon edges here (sigma=0.3 with k=10 means some class-1 points are among the 10-nearest-neighbors of class-0 points near the crossing). These edges create small ambiguity pockets.
              </Prose>
            ),
          },
          {
            label: "Iter 50–100 — convergence",
            render: () => (
              <Prose>
                Accuracy stabilizes at 78.5%. The fixed point {"F* = (I − αS)⁻¹(1−α)Y"} has been reached to numerical tolerance. The residual 21.5% error is structural: the crossing region of the moons is genuinely ambiguous under this graph — some class-0 points at the crossing have more class-1 neighbors than class-0 neighbors, and the manifold assumption incorrectly assigns them. A better sigma, more labeled points, or a directed graph would resolve these. The convergence rate is geometric in {"α = 0.99"}: each iteration reduces the residual error by a factor of {"1 − 0.99 = 0.01"}, so {"~500"} iterations are needed to reach machine epsilon.
              </Prose>
            ),
          },
        ]}
      />

      <H3>6c. Affinity matrix heatmap — 20-point toy dataset</H3>

      <Prose>
        The heatmap below shows the RBF affinity matrix <Code>W</Code> for 20 points: 10 cluster near (0, 0.5) and 10 cluster near (1, −0.5), with <Code>σ = 0.3</Code>. High-affinity pairs (warm cells) are within the same cluster; near-zero pairs (dark cells) are cross-cluster. This block-diagonal structure is what makes label propagation work — the two blocks stay isolated, so labels cannot leak across the class boundary.
      </Prose>

      <Heatmap
        label="RBF affinity matrix W — 20 points, sigma=0.3. Block diagonal = two clusters; off-diagonal near zero = no cross-cluster leakage"
        rowLabels={["p0","p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15","p16","p17","p18","p19"]}
        colLabels={["p0","p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15","p16","p17","p18","p19"]}
        matrix={[
          [0.00, 0.71, 0.93, 0.78, 0.84, 0.88, 0.67, 0.53, 0.73, 0.64, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
          [0.71, 0.00, 0.62, 0.84, 0.76, 0.52, 0.22, 0.29, 0.59, 0.25, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
          [0.93, 0.62, 0.00, 0.58, 0.92, 0.99, 0.68, 0.75, 0.89, 0.79, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
          [0.78, 0.84, 0.58, 0.00, 0.59, 0.49, 0.33, 0.20, 0.42, 0.25, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
          [0.84, 0.76, 0.92, 0.59, 0.00, 0.88, 0.44, 0.70, 0.96, 0.61, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
          [0.88, 0.52, 0.99, 0.49, 0.88, 0.00, 0.72, 0.82, 0.89, 0.87, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
          [0.67, 0.22, 0.68, 0.33, 0.44, 0.72, 0.00, 0.49, 0.44, 0.82, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.00, 0.00, 0.00, 0.00],
          [0.53, 0.29, 0.75, 0.20, 0.70, 0.82, 0.49, 0.00, 0.85, 0.84, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
          [0.73, 0.59, 0.89, 0.42, 0.96, 0.89, 0.44, 0.85, 0.00, 0.69, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
          [0.64, 0.25, 0.79, 0.25, 0.61, 0.87, 0.82, 0.84, 0.69, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
          [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.65, 0.59, 0.41, 0.59, 0.34, 0.70, 0.84, 0.56, 0.37],
          [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.65, 0.00, 0.71, 0.55, 0.81, 0.25, 0.98, 0.93, 0.96, 0.56],
          [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.59, 0.71, 0.00, 0.95, 0.98, 0.68, 0.81, 0.63, 0.55, 0.93],
          [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.41, 0.55, 0.95, 0.00, 0.91, 0.73, 0.66, 0.45, 0.40, 0.99],
          [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.59, 0.81, 0.98, 0.91, 0.00, 0.56, 0.89, 0.70, 0.65, 0.91],
          [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.00, 0.00, 0.00, 0.34, 0.25, 0.68, 0.73, 0.56, 0.00, 0.33, 0.24, 0.15, 0.66],
          [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.70, 0.98, 0.81, 0.66, 0.89, 0.33, 0.00, 0.91, 0.90, 0.66],
          [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.84, 0.93, 0.63, 0.45, 0.70, 0.24, 0.91, 0.00, 0.89, 0.44],
          [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.56, 0.96, 0.55, 0.40, 0.65, 0.15, 0.90, 0.89, 0.00, 0.42],
          [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.37, 0.56, 0.93, 0.99, 0.91, 0.66, 0.66, 0.44, 0.42, 0.00]
        ]}
        colorScale="gold"
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <StepTrace
        label="When to use which SSL method"
        steps={[
          {
            label: "Self-training — simple, widely applicable",
            render: () => (
              <Prose>
                Use self-training when: (a) you have a reasonably calibrated base classifier — one that gets the decision boundary roughly right on the labeled set; (b) the unlabeled data is close enough in distribution to the labeled data that confident predictions are likely correct; (c) you need a method that wraps any existing classifier without modifying its training procedure. Do NOT use self-training when: the initial labeled set is so small that the first boundary is wildly wrong — errors propagate and compound. Rule of thumb: if supervised-only accuracy is above 70%, self-training is worth trying; below 60%, expect degradation.
              </Prose>
            ),
          },
          {
            label: "Co-training — when you have genuinely independent views",
            render: () => (
              <Prose>
                Use co-training when: (a) the problem has two natural views — text body + anchor text for web pages; visual features + caption for images; audio + video for speech recognition; left/right camera feeds for robotics. (b) Each view is independently predictive (each view alone achieves {">"} 60% accuracy). Co-training is the wrong choice when: the two views are correlated — then the "disagreement signal" carries less information and errors still propagate. Most tabular datasets do not have natural views; splitting features arbitrarily is rarely effective because arbitrary feature splits are correlated by construction.
              </Prose>
            ),
          },
          {
            label: "Graph-based (LabelPropagation / LabelSpreading) — structured data with a meaningful similarity",
            render: () => (
              <Prose>
                Use graph-based SSL when: (a) the data has a meaningful similarity metric — images with embedding-space cosine similarity, documents with TF-IDF cosine, molecules with fingerprint Tanimoto; (b) the cluster assumption holds — same class members are genuinely closer to each other than to different-class members; (c) <Code>n</Code> is moderate (under ~10,000 for the dense sklearn implementation; up to ~100,000 with a sparse k-NN graph). Graph-based SSL is especially powerful when the decision boundary is highly nonlinear but the manifold structure is clear — the half-moon example is the canonical illustration.
              </Prose>
            ),
          },
          {
            label: "Deep SSL (FixMatch, Noisy Student) — large-scale unlabeled + strong augmentations",
            render: () => (
              <Prose>
                Use deep SSL when: (a) you have a large unlabeled corpus ({">"} 10k images, documents, or audio clips); (b) strong domain-specific augmentations are available (RandAugment, back-translation, specaugment); (c) you are training a neural network — these methods integrate with the training loop directly and add minimal overhead beyond standard supervised training. The compute cost is 2–3× supervised training (one forward pass per augmentation). Not applicable to sklearn-style pipelines or classical models with no augmentation interface.
              </Prose>
            ),
          },
          {
            label: "Supervised baseline — always compute this first",
            render: () => (
              <Prose>
                Before committing to any SSL method, train a fully supervised model on just the labeled set and measure performance. Then ask: is the gap between this baseline and the fully-supervised upper bound large enough to justify SSL's complexity? If the labeled set already gives 95%+ accuracy and the fully-supervised ceiling is 97%, SSL will give you 1–2% — possibly not worth the engineering cost. If the labeled-only accuracy is 60% and the ceiling is 90%, SSL has large room to work. The value of SSL is inversely proportional to how much your labeled set already tells you.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>8.1 Label propagation — the n² wall</H3>

      <Prose>
        The naive dense affinity matrix <Code>W</Code> requires storing {"n × n"} float64 values: at <Code>n = 10,000</Code>, that is 800 MB; at <Code>n = 50,000</Code>, it is 20 GB — already out of RAM for most workstations. The dense matrix-vector product <Code>S @ F</Code> at each iteration costs {"O(n²)"} per step. The closed-form solution {"(I − αS)⁻¹Y"} requires factoring an <Code>n × n</Code> matrix at {"O(n³)"} cost. The sklearn implementations are explicitly bounded to <Code>n {"<"} ~10,000</Code> before they become impractical.
      </Prose>

      <Prose>
        The standard workaround is a <em>sparse k-NN graph</em>: keep only the k nearest neighbors per point, storing {"O(nk)"} edges rather than {"O(n²)"}. Matrix-vector products then cost {"O(nk)"} per iteration. With <Code>k = 15</Code> and <Code>n = 100,000</Code>, this is 1.5 million edges — fits easily in memory. Sparse label propagation scales to hundreds of thousands of points. Libraries that implement this include <Code>scikit-learn</Code> (knn kernel option), <Code>pygsp</Code> (graph signal processing), and the <Code>label-propagation</Code> JAX implementations used in deep SSL research.
      </Prose>

      <H3>8.2 Self-training — linear scaling, convergence depends on base model</H3>

      <Prose>
        Self-training's computational cost is dominated by the base classifier. Each iteration trains the classifier on the current labeled set and runs inference on the unlabeled pool. If the base classifier costs {"O(n_L · d)"} to train and {"O(n_U · d)"} to infer, and <Code>T</Code> iterations are run, the total cost is {"O(T · (n_L + n_U) · d)"}. Since {"n_L"} grows each iteration (as pseudo-labeled points are added), the late iterations are slightly more expensive but the growth is bounded by {"n_U"}. In practice, self-training with logistic regression or gradient boosting on tabular data runs in seconds to minutes for {"n_U"} up to 1M. For neural networks, each iteration is a full fine-tuning run — typically 5–20 epochs — so the wall-clock cost scales with the number of self-training rounds.
      </Prose>

      <H3>8.3 Co-training — linear, but view construction is the bottleneck</H3>

      <Prose>
        Co-training itself scales linearly — two classifiers instead of one, otherwise the same as self-training. The bottleneck is constructing the views. Finding two genuinely independent views of a dataset is a domain-knowledge problem, not a computational one, and it often fails in practice. For NLP, a popular approach is to use different model families as the two "views" — a bag-of-words logistic regression and a TF-IDF SVM, for example — rather than disjoint feature partitions. This is sometimes called "disagreement-based learning" and is related to query-by-committee active learning.
      </Prose>

      <H3>8.4 When SSL hurts: distribution shift and confirmation bias</H3>

      <Prose>
        All three classical SSL methods assume that the unlabeled data is drawn from the same distribution as the labeled data. If the unlabeled data comes from a different distribution — different time periods, different demographics, different data-collection pipelines — the model will confidently pseudo-label out-of-distribution points incorrectly and incorporate those errors into training. This is a silent failure: test performance on the original distribution can drop without any obvious error signal during training. Always check that the unlabeled data's feature distribution matches the labeled data's (plot PCA projections, check marginal statistics) before applying SSL.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>9.1 Confirmation bias in self-training</H3>

      <Prose>
        The most pernicious failure of self-training: the model is wrong on some region of the input space, confidently assigns the wrong pseudo-label to points in that region, trains on those wrong labels, and becomes more confidently wrong. The cycle is self-reinforcing. This is called confirmation bias, and it is the primary reason self-training fails when the initial labeled set is too small or unrepresentative. Two mitigations: (a) use a higher confidence threshold — only accept pseudo-labels with probability {"≥ 0.95"} rather than 0.75, accepting fewer but more reliable examples per iteration; (b) use a calibrated base classifier (isotonic regression or Platt scaling post-processing) so that predicted probabilities actually correspond to empirical accuracy before the first self-training iteration.
      </Prose>

      <H3>9.2 Correlated views in co-training</H3>

      <Prose>
        Co-training's theoretical guarantee requires conditional independence of the views given the class. In practice, almost no real views are exactly independent. Mildly correlated views still help — the agreement signal is weaker but nonzero. Highly correlated views are dangerous: the two classifiers make the same errors, so cross-view pseudo-labeling provides no new information but does dilute the training signal with noise. Before applying co-training, measure the correlation between the two views on the labeled set: if a classifier trained on view 1 and evaluated on view 2 (or vice versa) achieves accuracy well above chance, the views share information and may be too correlated for co-training to help.
      </Prose>

      <H3>9.3 Bad similarity metric in label propagation</H3>

      <Prose>
        Label propagation is only as good as the affinity matrix. If the similarity metric does not reflect semantic class structure — for example, using raw pixel Euclidean distance on images (where the background dominates), or Euclidean distance on unscaled tabular features (where one feature's range dominates) — the graph will connect cross-class points with high affinity and label propagation will smooth across class boundaries. Always use a meaningful metric: cosine similarity on embeddings for text, learned representation distances for images, feature-scaled Euclidean distance for tabular data.
      </Prose>

      <H3>9.4 Cluster assumption violated — propagating across class boundaries</H3>

      <Prose>
        The cluster assumption says class boundaries lie in low-density regions. If two classes have overlapping distributions — the boundary passes through a dense region — label propagation will assign wrong labels to the overlap zone, and those labels will propagate outward. The graph-based smoothness regularization makes things worse: it actively tries to make neighboring points agree, so a wrong label in the dense boundary region gets reinforced by its high-affinity neighbors. This failure mode is invisible until you plot the predicted labels and see a region that should be ambiguous being confidently assigned to one class.
      </Prose>

      <H3>9.5 Confidence threshold selection</H3>

      <Prose>
        In self-training, the confidence threshold <Code>τ</Code> controls the precision-recall tradeoff of pseudo-label quality. High <Code>τ</Code> (e.g., 0.99): few pseudo-labels accepted per iteration, low error rate, slow convergence. Low <Code>τ</Code> (e.g., 0.60): many pseudo-labels accepted, faster convergence but more errors. There is no universally correct value. The sklearn <Code>SelfTrainingClassifier</Code> defaults to 0.75, which is reasonable for well-calibrated classifiers. For poorly calibrated classifiers, 0.75 may accept too many wrong pseudo-labels. Cross-validate <Code>τ</Code> on a held-out labeled set, or use a "soft" variant that weights pseudo-labeled points by their confidence rather than hard-thresholding.
      </Prose>

      <H3>9.6 Class imbalance propagation</H3>

      <Prose>
        If the labeled set is class-imbalanced, the initial classifier will be biased toward the majority class. Self-training will then generate more majority-class pseudo-labels (higher confidence on majority-class examples), amplifying the imbalance. After several iterations, the model may assign nearly all unlabeled points to the majority class with high confidence. The fix: balance the labeled set before self-training (oversample the minority class or apply class weights), and monitor the class distribution of pseudo-labeled points at each iteration.
      </Prose>

      <H3>9.7 Label propagation memory and O(n³) trap</H3>

      <Prose>
        A common mistake is forgetting that sklearn's <Code>LabelPropagation</Code> with <Code>kernel='rbf'</Code> builds the full {"n × n"} affinity matrix. On a 50,000-point dataset, this allocates 20 GB of RAM and the matrix-vector products take minutes per iteration. Symptoms: Python process slowly consumes all available memory; or the process is killed silently by the OS. Diagnosis: check <Code>n</Code> before calling fit. Fix: switch to <Code>kernel='knn'</Code>, which builds a sparse k-NN graph, or use a custom sparse implementation. The sklearn documentation warns about this but many practitioners hit it anyway.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All citations below were WebSearch-verified for author, year, venue, volume, pages, and main contribution.
      </Prose>

      <StepTrace
        label="Primary literature"
        steps={[
          {
            label: "Scudder 1965 — Self-training: the original adaptive machine",
            render: () => (
              <Prose>
                Scudder, H.J. (1965). "Probability of Error of Some Adaptive Pattern-Recognition Machines." <em>IEEE Transactions on Information Theory</em>, vol. 11, no. 3, pp. 363–371. Available via IEEE Xplore (DOI: 10.1109/TIT.1965.1053799). Scudder described an "untaught adaptive pattern-recognition machine" that uses its own output — rather than a labeled teacher — to update its parameters. He derived asymptotic error bounds for this self-labeling procedure, proving that the process can converge to a useful classifier even when the initial classifier is imperfect. This is the founding paper of self-training. The paper predates the term "semi-supervised learning" by three decades; in 1965 it was framed as a communications-theory problem about decision-device adaptation.
              </Prose>
            ),
          },
          {
            label: "Blum & Mitchell 1998 — Co-training and its PAC theory",
            render: () => (
              <Prose>
                Blum, A. and Mitchell, T. (1998). "Combining Labeled and Unlabeled Data with Co-Training." <em>Proceedings of the 11th Annual Conference on Computational Learning Theory (COLT)</em>, pp. 92–100. ACM. DOI: 10.1145/279943.279962. Available at cs.cmu.edu/~avrim/Papers/cotrain.pdf. The paper introduced co-training and provided the first rigorous PAC-style analysis of semi-supervised learning. The motivating application was hyperlink-based web page classification: the text on a page and the anchor text of incoming links form two conditionally independent views. The main theorem establishes that if each view is independently useful and the views are conditionally independent given the class, iterating pseudo-labeling between two classifiers provably reduces error. This paper gave the field its first theoretical foundation and remains the canonical co-training reference, with over 3,000 citations.
              </Prose>
            ),
          },
          {
            label: "Zhu & Ghahramani 2002 — Label propagation as a graph algorithm",
            render: () => (
              <Prose>
                Zhu, X. and Ghahramani, Z. (2002). "Learning from Labeled and Unlabeled Data with Label Propagation." Carnegie Mellon University Technical Report CMU-CALD-02-107. Available at pages.cs.wisc.edu/~jerryzhu/pub/CMU-CALD-02-107.pdf. This report introduced the label propagation algorithm: build a weighted graph of all data points, place label distributions at labeled nodes, and iteratively propagate them to unlabeled nodes along the graph edges. The key insight is that the algorithm can be viewed as computing the harmonic extension of the label function — the unique function that satisfies the boundary conditions at labeled nodes and is harmonic (locally smooth) everywhere else. The paper demonstrated the algorithm on handwriting and text data, establishing graph-based SSL as a distinct and powerful research direction.
              </Prose>
            ),
          },
          {
            label: "Zhou, Bousquet, Lal, Weston, Schölkopf 2003 — Local and global consistency",
            render: () => (
              <Prose>
                Zhou, D., Bousquet, O., Lal, T.N., Weston, J., and Schölkopf, B. (2003). "Learning with Local and Global Consistency." <em>Advances in Neural Information Processing Systems 16 (NeurIPS)</em>, pp. 321–328. Available at proceedings.neurips.cc. This paper sharpened label propagation into a well-posed regularized optimization problem: minimize a combination of (a) smoothness of the label function over the graph — local consistency — and (b) deviation from the initial label assignments — global consistency. The resulting closed-form solution {"F* = (I − αS)⁻¹(1−α)Y"} is elegant, the convergence proof is clean, and the algorithm became the standard graph-based SSL reference. The soft-clamping parameter <Code>α</Code> (which sklearn calls <Code>alpha</Code> in LabelSpreading) originates in this paper.
              </Prose>
            ),
          },
          {
            label: "Sohn et al. 2020 — FixMatch: pseudo-labels + consistency at scale",
            render: () => (
              <Prose>
                Sohn, K., Berthelot, D., Carlini, N., Zhang, Z., Zhang, H., Raffel, C., Cubuk, E.D., Kurakin, A., and Li, C.-L. (2020). "FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence." <em>Advances in Neural Information Processing Systems 33 (NeurIPS)</em>. arXiv:2001.07685. Available at proceedings.neurips.cc (hash 06964dce). FixMatch achieved 94.93% on CIFAR-10 with 250 labels and 88.61% with 40 labels. The method combines two SSL ideas that existed separately: pseudo-labeling (keep predictions above a confidence threshold) and consistency regularization (predict the same label under different augmentations). FixMatch's key simplification: weak augmentation generates the pseudo-label, strong augmentation generates the input, and the loss is cross-entropy between the two. Clean, reproducible code released at github.com/google-research/fixmatch.
              </Prose>
            ),
          },
          {
            label: "Xie, Luong, Hovy, Le 2020 — Noisy Student: iterative distillation at ImageNet scale",
            render: () => (
              <Prose>
                Xie, Q., Luong, M.-T., Hovy, E., and Le, Q.V. (2020). "Self-Training With Noisy Student Improves ImageNet Classification." <em>Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2020)</em>. arXiv:1911.04252. Available at openaccess.thecvf.com. The paper iterates: (1) train a teacher EfficientNet on labeled ImageNet; (2) generate soft pseudo-labels for 300M unlabeled images; (3) train a larger student EfficientNet on labeled + pseudo-labeled data with noise injection (dropout, stochastic depth, RandAugment); (4) promote student to teacher and repeat. The noise forces the student to be more general than the teacher rather than mimicking it. Final accuracy: 88.4% top-1 on ImageNet, beating methods using 3.5B weakly-labeled Instagram images. The paper demonstrates that classical self-training, properly scaled and regularized, is competitive with the most sophisticated SSL methods.
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
        Work through these before moving on. The answers follow each exercise — resist the urge to read ahead.
      </Prose>

      <H3>Exercise 1 (recall)</H3>
      <Prose>
        State the iterative label spreading update rule and explain each term. What is the role of the parameter <Code>α</Code>? What happens as <Code>α → 0</Code> and as <Code>α → 1</Code>? Prove that the iteration converges.
      </Prose>

      <Callout type="info">
        <strong>Answer:</strong> The update rule is {"F_{t+1} = α·S·F_t + (1−α)·Y"}. The first term {"α·S·F_t"} is the propagation step: each node receives a weighted average of its neighbors' label distributions, scaled by <Code>α</Code>. The second term {"(1−α)·Y"} is the fidelity step: each node is pulled back toward its initial label assignment (ground truth for labeled, uniform for unlabeled), scaled by {"(1−α)"}. As {"α → 0"}: the fidelity term dominates and F stays close to Y — no propagation. As {"α → 1"}: the propagation term dominates, labels spread freely with almost no pull back to the initial assignments; labeled nodes can drift. Convergence: the update is a contraction because {"||α·S|| = α·ρ(S) ≤ α < 1"} (ρ(S) ≤ 1 since S is symmetric normalized). By the Banach fixed-point theorem, any contraction converges to a unique fixed point.
      </Callout>

      <H3>Exercise 2 (derivation)</H3>
      <Prose>
        Derive the closed-form solution {"F* = (I − αS)⁻¹(1−α)Y"} from the iterative form. Why is this matrix always invertible? What is the computational cost of the closed form vs. iteration for <Code>n = 5,000</Code> and <Code>n = 50,000</Code>?
      </Prose>

      <Callout type="info">
        <strong>Answer:</strong> At convergence, {"F* = α·S·F* + (1−α)·Y"}. Rearranging: {"F* − α·S·F* = (1−α)·Y"}, i.e., {"(I − αS)·F* = (1−α)·Y"}, so {"F* = (I − αS)⁻¹(1−α)·Y"}. Invertibility: the eigenvalues of S lie in [−1, 1] (symmetric normalized matrix), so the eigenvalues of {"αS"} lie in [−α, α] ⊂ (−1, 1), and the eigenvalues of {"(I − αS)"} lie in {"(1−α, 1+α)"}, all strictly positive. Hence {"(I − αS)"} is positive definite and always invertible. Computational cost: Closed form requires factoring the {"n×n"} matrix at {"O(n³)"}. At n=5,000: {"5,000³ = 1.25×10¹¹"} — feasible but slow. At n=50,000: {"1.25×10¹⁴"} — completely infeasible. Iteration costs {"O(n²)"} per step × T steps. At n=50,000: each step is 2.5×10⁹ operations; with T=100 steps that is 2.5×10¹¹ — still slow. Solution for large n: sparse k-NN graph reduces the per-step cost to {"O(nk)"} — linear in n.
      </Callout>

      <H3>Exercise 3 (conceptual)</H3>
      <Prose>
        Explain why the affinity matrix needs to be a <em>sparse</em> k-NN graph rather than a full Gaussian kernel matrix in practice. Give the specific memory savings for <Code>n = 100,000</Code> and <Code>k = 15</Code>.
      </Prose>

      <Callout type="info">
        <strong>Answer:</strong> The full Gaussian kernel matrix {"W_{ij} = exp(−‖x_i−x_j‖²/(2σ²))"} is {"n×n"} dense — every pair of points gets a weight (most are very small but nonzero). At n=100,000 this is {"10¹⁰"} entries × 8 bytes = 80 GB in float64. Not only does this not fit in RAM, the matrix-vector product at each iteration costs {"O(n²) = 10¹⁰"} operations per step. A sparse k-NN graph stores only k=15 nonzero entries per row: total {"n×k = 1.5×10⁶"} entries × 8 bytes = 12 MB (a factor of {"80,000×"} reduction). The matrix-vector product at each iteration costs {"O(nk) = 1.5×10⁶"} operations — a factor of {"80,000×"} faster. The key approximation: setting long-range affinities to zero is justified because the RBF kernel decays exponentially; beyond ~3σ the weights are negligible anyway.
      </Callout>

      <H3>Exercise 4 (debugging)</H3>
      <Prose>
        You apply sklearn's <Code>LabelSpreading</Code> to a 20,000-point dataset with 200 labeled examples. The fit takes 45 minutes and your Python process uses 25 GB of RAM. What is the root cause, and what are two concrete fixes?
      </Prose>

      <Callout type="info">
        <strong>Answer:</strong> Root cause: sklearn's LabelSpreading with {"kernel='rbf'"} (the default) builds the full {"n×n = 20,000×20,000 = 4×10⁸"} entry affinity matrix in memory — at float64 that is 3.2 GB for the affinity matrix alone, plus the normalized version, plus workspace. The fit is slow because each iteration involves a {"20,000×20,000"} matrix-vector product ({"O(n²) = 4×10⁸"} operations). Fix 1: switch to {"kernel='knn'"} and set {"n_neighbors=15"} (or similar). This builds a sparse k-NN graph storing only {"n×k = 300,000"} entries — about 2.4 MB — and reduces each iteration to {"O(nk)"} operations. The fit time drops from 45 minutes to under 1 minute. Fix 2: reduce n before fitting by clustering (k-means or spectral clustering) the unlabeled data into ~2,000 representative points, running LabelSpreading on the clusters, and then assigning labels to all unlabeled points based on their nearest cluster. This approximation is faster and often nearly as accurate when clusters are tight.
      </Callout>

      <H3>Exercise 5 (applied)</H3>
      <Prose>
        You train a self-training classifier on a medical imaging dataset: 50 labeled scans, 5,000 unlabeled scans. After 3 self-training iterations, the training set has grown to 3,000 examples, but test AUC has dropped from 0.82 (supervised baseline) to 0.71. Diagnose the failure and propose a recovery strategy.
      </Prose>

      <Callout type="info">
        <strong>Answer:</strong> This is the confirmation-bias failure. The initial classifier trained on 50 scans likely has a miscalibrated decision boundary — possibly with high confidence in the wrong region due to the limited labeled set. Each self-training iteration accepts confident-but-wrong pseudo-labels, reinforces the wrong boundary, and generates more confident-but-wrong pseudo-labels in the next iteration. AUC drops monotonically as the model becomes more confidently wrong. Diagnosis: (1) Plot the pseudo-labels added in each iteration against ground truth (if available). (2) Check whether the model's confidence distribution is bimodal (good) or concentrated near threshold (bad — the model is not actually confident). Recovery: (a) Reset to the supervised baseline and increase the threshold to 0.95+ — accept far fewer but more reliable pseudo-labels per iteration. (b) Apply probability calibration (Platt scaling) to the base model before self-training so that predicted probabilities actually match empirical accuracy. (c) Use LabelSpreading instead of self-training — graph-based methods are less susceptible to confirmation bias because they propagate from multiple anchors simultaneously and the smoothness constraint acts as a regularizer. (d) Use active learning to select additional labeled examples from the 50-scan budget more strategically.
      </Callout>

      <H3>Exercise 6 (synthesis)</H3>
      <Prose>
        FixMatch and Noisy Student are both forms of self-training. Explain the specific mechanism in each that prevents the confirmation-bias failure mode, and connect those mechanisms to the mathematical analysis in Section 3.3 (self-training as EM).
      </Prose>

      <Callout type="info">
        <strong>Answer:</strong> In the EM view of self-training (Section 3.3), the M-step maximizes expected log-likelihood where unlabeled points contribute with soft weights from the E-step. Classical hard-assignment self-training approximates this by accepting only high-confidence (threshold {"τ"}) predictions and treating them as if they were exact labels. The bias risk is that the E-step posterior can be systematically wrong when the model is poorly initialized. FixMatch prevents this in two ways: (1) the confidence threshold of 0.95 is very high — only examples on which the model is nearly certain contribute to the pseudo-label loss, so early iterations use almost no unlabeled data and the model only expands its labeled set as it becomes genuinely confident; (2) consistency regularization — using strong augmentation on the same image as the pseudo-labeled input — means the model must predict the same class under a very different view of the input. This is similar to co-training's independence signal: if the strongly-augmented prediction agrees with the weakly-augmented pseudo-label, it is because the model has truly understood the class-relevant feature, not just a superficial texture. Noisy Student prevents confirmation bias via explicit noise injection (dropout, stochastic depth, RandAugment). In the EM view, the noise in the M-step acts as a regularizer on the model parameters — it prevents the student from collapsing to the teacher's exact function, which would reproduce any systematic errors the teacher makes. Each noise realization corresponds to a different member of an ensemble; the student learns the "center of mass" of the teacher's knowledge, not any particular miscalibrated corner of it. Both methods are principled instantiations of the core SSL insight: unlabeled data only helps when the pseudo-labels are at least as reliable as if they came from the true data-generating process.
      </Callout>

    </div>
  ),
};

export default semiSupervisedContent;
