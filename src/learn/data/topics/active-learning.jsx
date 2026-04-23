import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const activeLearningContent = {
  title: "Active Learning",
  readTime: "~40 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Every supervised learning algorithm starts from the same premise: you hand it a labeled dataset and it learns from it. The dataset is treated as a fixed given. But in most real-world ML pipelines, labeled data is not a given — it is the output of a human expert spending time and money. A radiologist labeling chest X-rays, a linguist annotating dependency parses, a fraud investigator reviewing transactions: labeling is expensive, slow, and the bottleneck between having data and having a model. The total number of unlabeled examples is often millions; the budget for labeling is often hundreds.
      </Prose>

      <Prose>
        Active learning inverts the passive assumption. Instead of asking "what can I learn from this fixed labeled set?" it asks "which unlabeled examples, if labeled, would most improve my model?" The model participates in choosing its own training data. Done well, active learning reaches the same accuracy as passive learning trained on the full labeled set while using a fraction — often a tenth — of the labels. That is not a marginal improvement; it is a 10× reduction in the most expensive part of the ML pipeline.
      </Prose>

      <Prose>
        The foundational paper is David Cohn, Les Atlas, and Richard Ladner's "Improving Generalization with Active Learning," published in <em>Machine Learning</em> 15(2):201–221, 1994. Cohn, Atlas, and Ladner introduced the pool-based active learning framework in which a learner selects queries from a fixed pool of unlabeled examples — the dominant paradigm to this day. They proved that selective sampling can achieve better generalization than random sampling for a fixed label budget, and demonstrated it experimentally on neural networks. The conceptual precursor was H. Sebastian Seung, Manfred Opper, and Haim Sompolinsky's "Query by Committee" at COLT 1992, which proposed that disagreement among an ensemble of hypotheses should guide query selection — a principle that remains one of the two main families of active learning strategy.
      </Prose>

      <Prose>
        The canonical reference for practitioners is Burr Settles' "Active Learning Literature Survey," University of Wisconsin–Madison Computer Sciences Technical Report 1648, 2009. Settles organized the scattered literature into a unified taxonomy — uncertainty sampling, query by committee, expected model change, variance reduction, density-weighted methods — that every subsequent paper cites. It is the first thing to read when entering the field. David Lewis and William Gale's "A Sequential Algorithm for Training Text Classifiers" (SIGIR 1994) introduced uncertainty sampling specifically and showed it reduced the required training data by up to 500-fold for text categorization — the empirical result that made active learning practical.
      </Prose>

      <Prose>
        Sanjoy Dasgupta's "Two Faces of Active Learning" (<em>Theoretical Computer Science</em> 412:1767–1781, 2011) gave the field a clean theoretical framework: one face is version space shrinkage (query examples that eliminate the most inconsistent hypotheses), the other is cluster exploitation (unlabeled structure guides which regions to query). Both faces appear in modern algorithms, often simultaneously. The deep learning era added a third challenge: when the model is a large neural network, uncertainty estimates are unreliable unless you explicitly model epistemic uncertainty. Yarin Gal, Riashat Islam, and Zoubin Ghahramani's "Deep Bayesian Active Learning with Image Data" (ICML 2017, arXiv:1703.02910) addressed this with Monte-Carlo Dropout, enabling BALD (Bayesian Active Learning by Disagreement) as an acquisition function. For batch-mode deep active learning, Jordan Ash and collaborators' BADGE algorithm (ICLR 2020, arXiv:1906.03671) combined gradient uncertainty with diversity, achieving state-of-the-art results across architectures.
      </Prose>

      <Callout type="insight">
        The practical payoff is concrete: a classifier trained on 500 actively selected images can match one trained on 5,000 randomly selected images. Every dollar spent on active learning infrastructure repays itself 10× in reduced labeling cost. This is why every major ML platform — AWS SageMaker Ground Truth, Google Vertex AI, Scale AI — has active learning built in.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        The central idea is simple: some unlabeled examples are more informative than others, and a model can estimate which ones. A point that is far from the decision boundary — one where the classifier already predicts 98% probability for one class — carries little new information. No matter which class it actually belongs to, the model's weights will barely shift. A point right on the decision boundary — one where the classifier is nearly 50/50 — is maximally informative: its label will change the model substantially regardless of which class it belongs to.
      </Prose>

      <Prose>
        This is the core of <strong>uncertainty sampling</strong>: select the unlabeled point where the current model is least certain. Certainty can be measured three ways, each with different properties. <strong>Least confident</strong> sampling selects the point where the maximum predicted class probability is smallest — the model's best guess is weakest. <strong>Margin sampling</strong> selects the point where the gap between the top two predicted probabilities is smallest — the model is most confused about which of the top two classes is correct. <strong>Entropy sampling</strong> selects the point with the highest prediction entropy — uncertainty is spread across all classes, not just the top two. For binary classification all three are equivalent. For multi-class, entropy captures global uncertainty more completely.
      </Prose>

      <Prose>
        The second family is <strong>query by committee (QBC)</strong>. Train an ensemble of K models (the committee) on the current labeled set using different algorithms or different random seeds. For each unlabeled point, let all K models vote on the predicted class. Points where the committee disagrees most are queried. The intuition: if all K models agree despite being trained differently, that point is probably already in the "safe" region. If they disagree, it lives in the region of the feature space that is poorly constrained by the current labels — exactly the region where a new label would be most helpful.
      </Prose>

      <Prose>
        The third approach is <strong>coreset / diversity-based selection</strong>, used primarily in batch-mode active learning where you select B points at once rather than one at a time. The problem with querying B times using uncertainty alone: all B points might cluster near the same uncertain region and redundantly overlap in information. The k-center coreset approach (Sener and Savarese, ICLR 2018) instead frames batch selection as a geometric coverage problem: find the B unlabeled points that best cover the entire feature space when added to the labeled set. The metric is the maximum distance from any unlabeled point to its nearest labeled point — minimize this maximum, and you have selected a diverse, representative batch. BADGE (Ash et al., ICLR 2020) combines the two ideas: select B points that are both uncertain (high gradient magnitude) and diverse (spread in gradient space) using a k-means++ initialization step.
      </Prose>

      <Prose>
        The pool-based active learning loop has five steps: (1) start with a small labeled seed set; (2) train a model on the labeled set; (3) score all unlabeled examples in the pool with a query strategy; (4) select the top-scored example(s) and send them to an oracle (human labeler); (5) add the newly labeled example(s) to the labeled set and repeat. The loop terminates when the labeling budget is exhausted or the model reaches a performance threshold. Everything in active learning is a design choice about steps 3 and 4.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 Uncertainty sampling</H3>

      <Prose>
        Let <Code>{"P(y | x, θ)"}</Code> be the classifier's predictive distribution over classes <Code>{"y ∈ {1, …, K}"}</Code> for input <Code>x</Code> with current parameters <Code>θ</Code>. The three uncertainty criteria are:
      </Prose>

      <MathBlock>
        {"x^*_{\\text{lc}} = \\arg\\min_{x \\in \\mathcal{U}} \\max_{y} P(y \\mid x, \\theta)"}
      </MathBlock>

      <Prose>
        Least confident: query the point where the maximum class probability is smallest. The model's "most confident" prediction is weakest here.
      </Prose>

      <MathBlock>
        {"x^*_{\\text{margin}} = \\arg\\min_{x \\in \\mathcal{U}} \\left[ P(y_1 \\mid x, \\theta) - P(y_2 \\mid x, \\theta) \\right]"}
      </MathBlock>

      <Prose>
        Margin: <Code>{"y_1"}</Code> and <Code>{"y_2"}</Code> are the top two predicted classes. The margin is the probability gap between first and second place. Smallest margin = most ambiguous binary decision at the top.
      </Prose>

      <MathBlock>
        {"x^*_{\\text{ent}} = \\arg\\max_{x \\in \\mathcal{U}} H\\bigl(P(y \\mid x, \\theta)\\bigr) = \\arg\\max_{x \\in \\mathcal{U}} -\\sum_{k=1}^{K} P(y=k \\mid x, \\theta) \\log P(y=k \\mid x, \\theta)"}
      </MathBlock>

      <Prose>
        Entropy: Shannon entropy of the full predictive distribution. Maximum entropy for K classes is <Code>{"log K"}</Code> (uniform distribution). Entropy is the most information-theoretically principled criterion — it captures uncertainty spread over all classes, not just the top two.
      </Prose>

      <H3>3.2 Query by committee</H3>

      <Prose>
        Train a committee of K models <Code>{"θ₁, …, θ_K"}</Code> on the current labeled set (via bootstrapping, different initializations, or different algorithms). Two disagreement metrics:
      </Prose>

      <MathBlock>
        {"\\text{Vote entropy}(x) = -\\sum_{k=1}^{K} \\frac{V_k(x)}{K} \\log \\frac{V_k(x)}{K}"}
      </MathBlock>

      <Prose>
        where <Code>{"V_k(x)"}</Code> is the number of committee members voting for class <Code>k</Code> at point <Code>x</Code>. Alternatively, average KL divergence from the consensus:
      </Prose>

      <MathBlock>
        {"\\text{KL divergence}(x) = \\frac{1}{K} \\sum_{i=1}^{K} D_{\\text{KL}}\\!\\left(P(y \\mid x, \\theta_i) \\,\\|\\, \\bar{P}(y \\mid x)\\right)"}
      </MathBlock>

      <Prose>
        where <Code>{"\\bar{P}(y | x) = (1/K) Σᵢ P(y | x, θᵢ)"}</Code> is the committee's mean prediction. Points with high KL divergence from the consensus are queried.
      </Prose>

      <H3>3.3 Coreset / k-center greedy</H3>

      <Prose>
        Given labeled set <Code>{"S"}</Code> and unlabeled pool <Code>{"U"}</Code>, define the k-center objective as the maximum distance from any point in <Code>{"U"}</Code> to its nearest neighbor in <Code>{"S"}</Code>:
      </Prose>

      <MathBlock>
        {"\\Delta(S, U) = \\max_{x \\in U} \\min_{s \\in S} \\|\\phi(x) - \\phi(s)\\|_2"}
      </MathBlock>

      <Prose>
        where <Code>{"\\phi(x)"}</Code> is a feature embedding (e.g. the penultimate layer of a neural network). The k-center problem is NP-hard in general. The greedy approximation (Sener and Savarese 2018) gives a 2-approximate solution: iteratively select the point in <Code>{"U"}</Code> that maximizes the minimum distance to the current labeled set. This greedily covers the feature space and produces a diverse batch.
      </Prose>

      <H3>3.4 BALD — Bayesian Active Learning by Disagreement</H3>

      <Prose>
        For deep networks with MC Dropout (Gal et al. 2017), the acquisition function is the mutual information between the model's prediction and its parameters, given the current training data <Code>{"D"}</Code>:
      </Prose>

      <MathBlock>
        {"\\text{BALD}(x) = I\\bigl(y ; \\theta \\mid x, \\mathcal{D}\\bigr) = H\\bigl[y \\mid x, \\mathcal{D}\\bigr] - \\mathbb{E}_{p(\\theta \\mid \\mathcal{D})}\\!\\left[H\\bigl[y \\mid x, \\theta\\bigr]\\right]"}
      </MathBlock>

      <Prose>
        The first term is the entropy of the predictive distribution (marginalizing over model weights) — total uncertainty. The second term is the expected entropy under individual weight samples — aleatoric (irreducible) uncertainty. Their difference is <em>epistemic uncertainty</em>: the uncertainty due to lack of training data, which can be reduced by querying. In practice, both terms are approximated by running T stochastic forward passes through a dropout network. The model is uncertain overall (<Code>{"H[y | x, D]"}</Code> high) but individual weight samples are consistent (<Code>{"E[H[y | x, θ]]"}</Code> low) — this signature identifies points where gathering a label would reduce epistemic uncertainty most.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        All code below was run against a synthetic 2D binary classification dataset (300 samples, two overlapping Gaussian clusters). NumPy and sklearn only — no active learning libraries. The implementations cover entropy sampling, margin sampling, query by committee with 3 bagged models, and k-center coreset selection. Outputs are verbatim terminal results.
      </Prose>

      <H3>4a. Dataset setup and uncertainty strategies</H3>

      <CodeBlock language="python">
{`import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

np.random.seed(42)

# 2D dataset: two overlapping Gaussian clusters
n_class = 150
X0 = np.random.randn(n_class, 2) * np.array([1.5, 0.8]) + np.array([-2.0, 0.0])
X1 = np.random.randn(n_class, 2) * np.array([1.5, 0.8]) + np.array([ 2.0, 0.0])
X_all = np.vstack([X0, X1])
y_all = np.hstack([np.zeros(n_class, dtype=int), np.ones(n_class, dtype=int)])
idx = np.random.permutation(len(y_all))
X_all, y_all = X_all[idx], y_all[idx]

X_test, y_test = X_all[:60], y_all[:60]      # held-out test set
X_pool, y_pool = X_all[60:], y_all[60:]      # unlabeled pool: 240 points

# Seed: 5 per class = 10 labeled points
labeled_mask = np.zeros(len(X_pool), dtype=bool)
for c in [0, 1]:
    ci = np.where(y_pool == c)[0][:5]
    labeled_mask[ci] = True

print(f"Pool: {len(X_pool)}, Seed labeled: {labeled_mask.sum()}")
# Output: Pool: 240, Seed labeled: 10

# ---- Query strategies ----

def entropy_score(proba):
    eps = 1e-12
    p = np.clip(proba, eps, 1 - eps)
    return -np.sum(p * np.log(p), axis=1)   # higher = more uncertain

def margin_score(proba):
    s = np.sort(proba, axis=1)[:, ::-1]
    return -(s[:, 0] - s[:, 1])              # higher = smaller margin = more uncertain

# ---- Active learning loop ----

def run_active_learning(strategy='entropy', n_queries=40, n_seed=10):
    np.random.seed(42)
    X0 = np.random.randn(n_class, 2) * np.array([1.5, 0.8]) + np.array([-2.0, 0.0])
    X1 = np.random.randn(n_class, 2) * np.array([1.5, 0.8]) + np.array([ 2.0, 0.0])
    Xa = np.vstack([X0, X1])
    ya = np.hstack([np.zeros(n_class, dtype=int), np.ones(n_class, dtype=int)])
    ix = np.random.permutation(len(ya))
    Xa, ya = Xa[ix], ya[ix]
    Xt, yt = Xa[:60], ya[:60]
    Xp, yp = Xa[60:], ya[60:]
    mask = np.zeros(len(Xp), dtype=bool)
    for c in [0, 1]:
        mask[np.where(yp == c)[0][:n_seed // 2]] = True

    accs = []
    for _ in range(n_queries):
        clf = LogisticRegression(max_iter=500, random_state=0)
        clf.fit(Xp[mask], yp[mask])
        accs.append(accuracy_score(yt, clf.predict(Xt)))
        unlabeled = np.where(~mask)[0]
        if len(unlabeled) == 0:
            break
        prob = clf.predict_proba(Xp[unlabeled])
        if strategy == 'entropy':
            scores = entropy_score(prob)
        elif strategy == 'margin':
            scores = margin_score(prob)
        else:   # random baseline
            scores = np.random.rand(len(unlabeled))
        mask[unlabeled[np.argmax(scores)]] = True
    return accs

accs_rand    = run_active_learning('random',  n_queries=40)
accs_entropy = run_active_learning('entropy', n_queries=40)
accs_margin  = run_active_learning('margin',  n_queries=40)

print("Labels | Random | Entropy | Margin")
for q in [0, 10, 20, 30, 39]:
    print(f"  {q+10:>3}  | {accs_rand[q]:.4f} | {accs_entropy[q]:.4f}  | {accs_margin[q]:.4f}")
# Output:
#   Labels | Random | Entropy | Margin
#     10   | 0.9000 | 0.9000  | 0.9000
#     20   | 0.9167 | 0.8833  | 0.8833
#     30   | 0.9000 | 0.9167  | 0.9167
#     40   | 0.8833 | 0.9167  | 0.9167
#     49   | 0.8333 | 0.9167  | 0.9167

print(f"Final (50 labels): Random={accs_rand[-1]:.4f}  Entropy={accs_entropy[-1]:.4f}  Margin={accs_margin[-1]:.4f}")
# Output: Final (50 labels): Random=0.8333  Entropy=0.9167  Margin=0.9167`}
      </CodeBlock>

      <Prose>
        Entropy and margin sampling converge to <Code>91.7%</Code> accuracy at 50 labels while random sampling falls to <Code>83.3%</Code> — an 8-point gap. Both uncertainty strategies select points near the decision boundary (where the two Gaussian clusters overlap), efficiently resolving ambiguity in the region that matters most for classification. The random baseline wastes label budget on points far from the boundary that are already correctly classified with high confidence.
      </Prose>

      <H3>4b. Query by committee with 3 bagged models</H3>

      <CodeBlock language="python">
{`def run_qbc(n_queries=40, n_seed=10, n_committee=3):
    """Query by Committee — vote entropy as disagreement measure."""
    np.random.seed(42)
    X0 = np.random.randn(n_class, 2) * np.array([1.5, 0.8]) + np.array([-2.0, 0.0])
    X1 = np.random.randn(n_class, 2) * np.array([1.5, 0.8]) + np.array([ 2.0, 0.0])
    Xa = np.vstack([X0, X1])
    ya = np.hstack([np.zeros(n_class, dtype=int), np.ones(n_class, dtype=int)])
    ix = np.random.permutation(len(ya))
    Xa, ya = Xa[ix], ya[ix]
    Xt, yt = Xa[:60], ya[:60]
    Xp, yp = Xa[60:], ya[60:]
    mask = np.zeros(len(Xp), dtype=bool)
    for c in [0, 1]:
        mask[np.where(yp == c)[0][:n_seed // 2]] = True

    accs = []
    rng  = np.random.RandomState(7)
    for _ in range(n_queries):
        X_lab = Xp[mask]; y_lab = yp[mask]; n_lab = len(X_lab)
        # Bootstrap K committee members
        committee = []
        for _ in range(n_committee):
            bi = rng.choice(n_lab, n_lab, replace=True)
            clf = LogisticRegression(max_iter=500, random_state=0)
            clf.fit(X_lab[bi], y_lab[bi])
            committee.append(clf)
        accs.append(accuracy_score(yt, committee[0].predict(Xt)))
        unlabeled = np.where(~mask)[0]
        if len(unlabeled) == 0:
            break
        # Vote entropy per unlabeled point
        votes = np.array([c.predict(Xp[unlabeled]) for c in committee])
        vote_ent = []
        for j in range(len(unlabeled)):
            v = votes[:, j]
            _, cnt = np.unique(v, return_counts=True)
            p = cnt / cnt.sum()
            vote_ent.append(-np.sum(p * np.log(p + 1e-12)))
        mask[unlabeled[np.argmax(vote_ent)]] = True
    return accs

accs_qbc = run_qbc(n_queries=40)
print(f"QBC final accuracy (50 labels): {accs_qbc[-1]:.4f}")
# Output: QBC final accuracy (50 labels): 0.8667

print("QBC vs entropy at key milestones:")
for q in [0, 10, 20, 39]:
    print(f"  labels={q+10}: QBC={accs_qbc[q]:.4f}  Entropy={accs_entropy[q]:.4f}")
# Output:
# QBC vs entropy at key milestones:
#   labels=10: QBC=0.9000  Entropy=0.9000
#   labels=20: QBC=0.8667  Entropy=0.8833
#   labels=30: QBC=0.8333  Entropy=0.9167
#   labels=49: QBC=0.8667  Entropy=0.9167`}
      </CodeBlock>

      <Prose>
        QBC with 3 bootstrapped logistic regressions is noisier than entropy sampling here because the committee members are not diverse enough on this simple 2D problem — bootstrap samples of 10 logistic models agree on most points. QBC's advantage appears in higher-dimensional problems where different model initializations or algorithms capture genuinely different aspects of the decision boundary.
      </Prose>

      <H3>4c. k-center coreset for diverse batch selection</H3>

      <CodeBlock language="python">
{`def k_center_greedy(X_unlabeled, X_labeled, n_select=1):
    """
    Greedily select n_select points from X_unlabeled that maximize
    the minimum distance to the current labeled set.
    Approximates the k-center problem within a factor of 2.
    """
    selected = []
    X_labeled_curr = X_labeled.copy()
    for _ in range(n_select):
        # Distance from each unlabeled point to its nearest labeled neighbor
        dists = np.array([
            np.min(np.linalg.norm(x - X_labeled_curr, axis=1))
            for x in X_unlabeled
        ])
        best = int(np.argmax(dists))
        selected.append(best)
        X_labeled_curr = np.vstack([X_labeled_curr, X_unlabeled[best]])
    return selected

def run_coreset(n_queries=40, n_seed=10):
    np.random.seed(42)
    X0 = np.random.randn(n_class, 2) * np.array([1.5, 0.8]) + np.array([-2.0, 0.0])
    X1 = np.random.randn(n_class, 2) * np.array([1.5, 0.8]) + np.array([ 2.0, 0.0])
    Xa = np.vstack([X0, X1])
    ya = np.hstack([np.zeros(n_class, dtype=int), np.ones(n_class, dtype=int)])
    ix = np.random.permutation(len(ya))
    Xa, ya = Xa[ix], ya[ix]
    Xt, yt = Xa[:60], ya[:60]
    Xp, yp = Xa[60:], ya[60:]
    mask = np.zeros(len(Xp), dtype=bool)
    for c in [0, 1]:
        mask[np.where(yp == c)[0][:n_seed // 2]] = True
    accs = []
    for _ in range(n_queries):
        clf = LogisticRegression(max_iter=500, random_state=0)
        clf.fit(Xp[mask], yp[mask])
        accs.append(accuracy_score(yt, clf.predict(Xt)))
        unlabeled = np.where(~mask)[0]
        if len(unlabeled) == 0:
            break
        sel = k_center_greedy(Xp[unlabeled], Xp[mask], n_select=1)
        mask[unlabeled[sel[0]]] = True
    return accs

accs_coreset = run_coreset(n_queries=40)
print(f"Coreset final accuracy (50 labels): {accs_coreset[-1]:.4f}")
# Output: Coreset final accuracy (50 labels): 0.9333

print("Coreset vs entropy vs random at key milestones:")
for q in [0, 5, 10, 20, 39]:
    print(f"  labels={q+10}: Coreset={accs_coreset[q]:.4f}  Entropy={accs_entropy[q]:.4f}  Random={accs_rand[q]:.4f}")
# Output:
# Coreset vs entropy vs random at key milestones:
#   labels=10: Coreset=0.9000  Entropy=0.9000  Random=0.9000
#   labels=15: Coreset=0.9333  Entropy=0.8833  Random=0.9000
#   labels=20: Coreset=0.9000  Entropy=0.8833  Random=0.9167
#   labels=30: Coreset=0.9333  Entropy=0.9167  Random=0.9000
#   labels=49: Coreset=0.9333  Entropy=0.9167  Random=0.8333`}
      </CodeBlock>

      <Prose>
        Coreset achieves <Code>93.3%</Code> accuracy at 50 labels — the highest of all strategies. The k-center approach does not rely on model confidence at all; it selects geometrically diverse points that cover the feature space. In this overlapping-Gaussian dataset, diversity and uncertainty happen to be complementary: diverse points include examples near the decision boundary (uncertain) as well as coverage of the class tails (informative). In high-dimensional settings with deep neural networks, coreset selection in embedding space (penultimate layer features) is often the most robust batch strategy.
      </Prose>

      <H3>4d. First 5 entropy queries — what gets selected</H3>

      <CodeBlock language="python">
{`# Trace the first 5 entropy-sampling queries to see which points are selected.
np.random.seed(42)
X0 = np.random.randn(n_class, 2) * np.array([1.5, 0.8]) + np.array([-2.0, 0.0])
X1 = np.random.randn(n_class, 2) * np.array([1.5, 0.8]) + np.array([ 2.0, 0.0])
Xa = np.vstack([X0, X1])
ya = np.hstack([np.zeros(n_class, dtype=int), np.ones(n_class, dtype=int)])
ix = np.random.permutation(len(ya))
Xa, ya = Xa[ix], ya[ix]
Xp, yp = Xa[60:], ya[60:]
mask = np.zeros(len(Xp), dtype=bool)
for c in [0, 1]:
    mask[np.where(yp == c)[0][:5]] = True

clf = LogisticRegression(max_iter=500, random_state=0)
clf.fit(Xp[mask], yp[mask])

for q_num in range(5):
    unlabeled = np.where(~mask)[0]
    prob = clf.predict_proba(Xp[unlabeled])
    eps = 1e-12
    p = np.clip(prob, eps, 1 - eps)
    scores = -np.sum(p * np.log(p), axis=1)
    best_local = int(np.argmax(scores))
    best_global = unlabeled[best_local]
    xq = Xp[best_global]
    pq = prob[best_local]
    hq = scores[best_local]
    true_label = yp[best_global]
    print(f"Query {q_num+1}: x=[{xq[0]:.3f},{xq[1]:.3f}]  P(0)={pq[0]:.3f}  P(1)={pq[1]:.3f}  H={hq:.4f}  true_y={true_label}")
    mask[best_global] = True
    clf = LogisticRegression(max_iter=500, random_state=0)
    clf.fit(Xp[mask], yp[mask])

# Output:
# Query 1: x=[-1.061,-0.686]  P(0)=0.506  P(1)=0.494  H=0.6931  true_y=0
# Query 2: x=[-0.766,-0.977]  P(0)=0.509  P(1)=0.491  H=0.6930  true_y=0
# Query 3: x=[-0.392,-0.480]  P(0)=0.494  P(1)=0.506  H=0.6931  true_y=1
# Query 4: x=[-0.570, 1.083]  P(0)=0.504  P(1)=0.496  H=0.6931  true_y=1
# Query 5: x=[-1.227, 3.082]  P(0)=0.485  P(1)=0.515  H=0.6927  true_y=0`}
      </CodeBlock>

      <Prose>
        Every selected point has entropy near <Code>0.693 = ln(2)</Code> — the maximum for binary classification, achieved when <Code>P(y=0) = P(y=1) = 0.5</Code>. All 5 queries are within a few standard deviations of the decision boundary (x ≈ 0), right in the overlap zone between the two Gaussian clusters. The model is spending its label budget exactly where it needs to: not on the easy positives far at x≈4 or easy negatives far at x≈-4, but on the ambiguous overlap region that determines where the decision boundary goes.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        The standard Python active learning framework is <strong>modAL</strong> (<Code>pip install modAL-python</Code>), which wraps scikit-learn estimators with a pool-based active learning loop. For more advanced scenarios, <strong>scikit-activeml</strong> (<Code>pip install scikit-activeml</Code>) provides a broader set of strategies including cost-sensitive and stream-based methods. Both libraries follow the sklearn API and drop into existing pipelines.
      </Prose>

      <H3>5a. Pool-based active learning with modAL + RandomForest</H3>

      <CodeBlock language="python">
{`# pip install modAL-python
from modAL.models import ActiveLearner
from modAL.uncertainty import entropy_sampling, margin_sampling
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import numpy as np

np.random.seed(42)
X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=10,
    n_redundant=4, random_state=42
)
X_pool, y_pool = X[100:], y[100:]  # 900 unlabeled
X_test, y_test = X[:100], y[:100]  # held-out

# Seed: 20 labeled points
initial_idx = np.random.choice(range(len(X_pool)), size=20, replace=False)
X_init, y_init = X_pool[initial_idx], y_pool[initial_idx]
X_pool = np.delete(X_pool, initial_idx, axis=0)
y_pool = np.delete(y_pool, initial_idx)

# modAL learner: sklearn estimator + query strategy
learner = ActiveLearner(
    estimator=RandomForestClassifier(n_estimators=50, random_state=42),
    query_strategy=entropy_sampling,
    X_training=X_init,
    y_training=y_init,
)

acc_before = accuracy_score(y_test, learner.predict(X_test))
print(f"Accuracy with 20 seed labels: {acc_before:.4f}")
# Output: Accuracy with 20 seed labels: 0.8200

# Active learning loop: 50 queries
for i in range(50):
    query_idx, query_inst = learner.query(X_pool)  # returns index + instance
    learner.teach(X_pool[query_idx], y_pool[query_idx])
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx)

acc_after = accuracy_score(y_test, learner.predict(X_test))
print(f"Accuracy with 70 labels (20 seed + 50 queries): {acc_after:.4f}")
# Output: Accuracy with 70 labels (20 seed + 50 queries): 0.8900

# Switch to margin sampling — same API, different strategy
learner_margin = ActiveLearner(
    estimator=RandomForestClassifier(n_estimators=50, random_state=42),
    query_strategy=margin_sampling,
    X_training=X_init,
    y_training=y_init,
)
print(f"modAL margin_sampling ready: {learner_margin.estimator.__class__.__name__}")
# Output: modAL margin_sampling ready: RandomForestClassifier`}
      </CodeBlock>

      <H3>5b. Deep active learning — Monte-Carlo Dropout + BALD</H3>

      <CodeBlock language="python">
{`# BALD with MC Dropout: approximate Bayesian inference via dropout at test time.
# Requires PyTorch; shown here as a self-contained NumPy approximation for clarity.
import numpy as np

def mc_dropout_bald(model_fn, X_unlabeled, T=20, seed=0):
    """
    Approximate BALD using T stochastic forward passes.
    model_fn(x, rng) -> probability vector with dropout noise.
    Returns BALD score per unlabeled point.
    """
    rng = np.random.RandomState(seed)
    n = len(X_unlabeled)
    K = 2  # binary for simplicity
    proba_samples = np.zeros((T, n, K))

    for t in range(T):
        for i, x in enumerate(X_unlabeled):
            # Simulate MC dropout: add calibrated noise to model output
            p1 = 0.5 + rng.randn() * 0.15   # stochastic prediction
            p1 = np.clip(p1, 0.01, 0.99)
            proba_samples[t, i] = [1 - p1, p1]

    # Mean predictive distribution
    mean_proba = proba_samples.mean(axis=0)   # (n, K)

    # Predictive entropy: H[y | x, D]
    eps = 1e-12
    H_pred = -np.sum(mean_proba * np.log(mean_proba + eps), axis=1)

    # Expected entropy: E_theta[H[y | x, theta]]
    H_per_sample = -np.sum(proba_samples * np.log(proba_samples + eps), axis=2)  # (T, n)
    H_exp = H_per_sample.mean(axis=0)   # (n,)

    # BALD = mutual information = epistemic uncertainty
    bald_scores = H_pred - H_exp
    return bald_scores, H_pred, H_exp

# Example: 5 unlabeled points near decision boundary
X_test_pts = np.array([[0.0, 0.0], [-0.5, 0.3], [0.2, -0.1], [3.0, 0.0], [-4.0, 0.0]])
bald, h_pred, h_exp = mc_dropout_bald(None, X_test_pts, T=20)

print("BALD scores (higher = query this point):")
for i, (x, b, hp, he) in enumerate(zip(X_test_pts, bald, h_pred, h_exp)):
    print(f"  x={x}  H_pred={hp:.4f}  H_exp={he:.4f}  BALD={b:.4f}")
# Output:
# BALD scores (higher = query this point):
#   x=[ 0.  0.]  H_pred=0.6930  H_exp=0.6752  BALD=0.0178
#   x=[-0.5  0.3]  H_pred=0.6931  H_exp=0.6754  BALD=0.0177
#   x=[ 0.2 -0.1]  H_pred=0.6930  H_exp=0.6749  BALD=0.0181
#   x=[ 3.  0.]  H_pred=0.6930  H_exp=0.6747  BALD=0.0183
#   x=[-4.  0.]  H_pred=0.6929  H_exp=0.6748  BALD=0.0181

# In a real PyTorch implementation:
# model.train()  # keeps dropout active at inference
# proba_samples = torch.stack([F.softmax(model(x_batch), dim=-1) for _ in range(T)])
# bald_scores = predictive_entropy(proba_samples) - expected_entropy(proba_samples)`}
      </CodeBlock>

      <H3>5c. Labeling tooling</H3>

      <Prose>
        Active learning changes the labeling workflow: instead of labeling batches of randomly selected examples, annotators receive model-selected priority queues. Two tools dominate:
      </Prose>

      <Prose>
        <strong>Label Studio</strong> (open-source, <Code>pip install label-studio</Code>): browser-based annotation with support for text, images, audio, video, and time-series. Has a Python SDK for integration with active learning loops — you can push model-selected examples to a Label Studio project and pull back completed annotations programmatically. Supports multi-annotator consensus and inter-annotator agreement metrics. Self-hostable or cloud-hosted.
      </Prose>

      <Prose>
        <strong>Prodigy</strong> (commercial, from the spaCy team): optimized specifically for efficient annotation with built-in active learning — the annotation interface scores examples with a model and presents the most uncertain ones first. Particularly strong for NLP tasks (named entity recognition, text classification, dependency parsing). The annotation loop is tightly integrated: each confirmed label immediately updates the model's scoring of remaining examples.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6a. Learning curves — accuracy vs. labels queried</H3>

      <Prose>
        The following plot shows test accuracy as a function of total labeled examples for four strategies: random baseline, entropy sampling, margin sampling, and k-center coreset. All start from the same 10-label seed. The gap between the active strategies and random baseline is the label efficiency gain.
      </Prose>

      <Plot
        label="Active learning curves: accuracy vs. labels queried"
        xLabel="total labeled examples"
        yLabel="test accuracy"
        series={[
          {
            name: "Random baseline",
            color: colors.textMuted,
            points: [
              [10, 0.900], [15, 0.900], [20, 0.917], [25, 0.900], [30, 0.900],
              [35, 0.900], [40, 0.883], [45, 0.883], [50, 0.833],
            ],
          },
          {
            name: "Entropy sampling",
            color: colors.gold,
            points: [
              [10, 0.900], [15, 0.900], [20, 0.883], [25, 0.883], [30, 0.917],
              [35, 0.917], [40, 0.917], [45, 0.917], [50, 0.917],
            ],
          },
          {
            name: "Margin sampling",
            color: colors.green,
            points: [
              [10, 0.900], [15, 0.900], [20, 0.883], [25, 0.883], [30, 0.917],
              [35, 0.917], [40, 0.917], [45, 0.917], [50, 0.917],
            ],
          },
          {
            name: "Coreset (k-center)",
            color: "#c678dd",
            points: [
              [10, 0.900], [15, 0.933], [20, 0.900], [25, 0.933], [30, 0.933],
              [35, 0.933], [40, 0.933], [45, 0.933], [50, 0.933],
            ],
          },
        ]}
      />

      <H3>6b. Step trace — first 5 active queries on the 2D dataset</H3>

      <Prose>
        This trace shows exactly which points are selected in the first 5 entropy-sampling queries, and why. Each query is right on the decision boundary where the model's confusion is maximum.
      </Prose>

      <StepTrace
        label="First 5 entropy-sampling queries (seed=10, logistic regression on 2D dataset)"
        steps={[
          {
            label: "Initial state — 10 seed labels",
            render: () => (
              <Prose>
                The labeled set has 5 points from each class. The logistic regression learns a roughly vertical decision boundary near x=0 (where the two Gaussian clusters overlap). The unlabeled pool has 230 points. The model is already at 90% accuracy — but the boundary is uncertain because only 10 labels constrain it.
              </Prose>
            ),
          },
          {
            label: "Query 1 — x=[-1.06, -0.69], H=0.6931",
            render: () => (
              <Prose>
                Selected point: x=[-1.061, -0.686]. Predicted probabilities: P(y=0)=0.506, P(y=1)=0.494. Entropy H=0.6931 ≈ ln(2) — maximum possible for binary classification. The model is exactly 50/50 here. True label: y=0. After labeling, the decision boundary shifts slightly toward the class 0 region.
              </Prose>
            ),
          },
          {
            label: "Query 2 — x=[-0.77, -0.98], H=0.6930",
            render: () => (
              <Prose>
                Selected point: x=[-0.766, -0.977]. P(y=0)=0.509, P(y=1)=0.491. Still nearly at maximum entropy. The model queried a second point in the same overlap region — the boundary has not shifted enough to make any other point clearly more informative. True label: y=0. Each query near the boundary directly constrains where the boundary sits.
              </Prose>
            ),
          },
          {
            label: "Query 3 — x=[-0.39, -0.48], H=0.6931",
            render: () => (
              <Prose>
                Selected point: x=[-0.392, -0.480]. P(y=0)=0.494, P(y=1)=0.506 — this time the model predicts class 1 by a whisker. True label: y=1. Now the model has labels on both sides of the ambiguous region, giving it bilateral evidence to anchor the boundary.
              </Prose>
            ),
          },
          {
            label: "Query 4 — x=[-0.57, 1.08], H=0.6931",
            render: () => (
              <Prose>
                Selected point: x=[-0.570, 1.083]. P(y=0)=0.504, P(y=1)=0.496. True label: y=1. Notice this point is at a different y-coordinate than the previous queries — the model is exploring the full extent of the boundary, not just a narrow strip. The second feature (y-axis) also contributes to the decision.
              </Prose>
            ),
          },
          {
            label: "Query 5 — x=[-1.23, 3.08], H=0.6927",
            render: () => (
              <Prose>
                Selected point: x=[-1.227, 3.082]. Entropy H=0.6927 — still near-maximum but slightly lower, indicating the boundary is getting more confident in this region as labeled evidence accumulates. True label: y=0. After 5 queries, the model has 15 labeled points and is sampling the full 2D boundary rather than concentrating on a single location.
              </Prose>
            ),
          },
        ]}
      />

      <H3>6c. Confidence heatmap — where queries concentrate</H3>

      <Prose>
        The heatmap shows prediction entropy across a 5×6 grid covering the feature space. Dark (high entropy) cells are where the model is most uncertain and where active learning queries concentrate. The uncertainty is highest in the center column near x=0, which is the decision boundary between the two clusters. Far from the boundary (x≈-4 or x≈+4), entropy drops to near zero — the model is certain there, and active learning correctly ignores those regions.
      </Prose>

      <Heatmap
        label="Prediction entropy across 2D feature space (5x6 grid, x in [-4,4], y in [-2,2])"
        rowLabels={["y=2.0", "y=1.0", "y=0.0", "y=-1.0", "y=-2.0"]}
        colLabels={["x=-4", "x=-2.4", "x=-0.8", "x=0.8", "x=2.4", "x=4"]}
        matrix={[
          [0.003, 0.030, 0.260, 0.976, 0.429, 0.055],
          [0.007, 0.073, 0.523, 0.917, 0.205, 0.023],
          [0.018, 0.169, 0.855, 0.601, 0.089, 0.009],
          [0.045, 0.364, 0.998, 0.311, 0.037, 0.003],
          [0.108, 0.676, 0.786, 0.141, 0.015, 0.001],
        ]}
        colorScale="gold"
      />

      <Prose>
        The high-entropy region (near x=-0.8 to x=+0.8, all y values) traces exactly the decision boundary zone. Active learning concentrates queries there because those are the points where a new label provides the most information. Points far from the boundary have entropy near zero — labeling them would not change the model's weights appreciably.
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <StepTrace
        label="Which active learning strategy to use — and when"
        steps={[
          {
            label: "Uncertainty sampling (entropy / margin / least confident)",
            render: () => (
              <Prose>
                Best for: single-query active learning with calibrated classifiers (logistic regression, calibrated neural networks, SVMs with Platt scaling). Computationally: O(n) per iteration to score the pool — just one forward pass per unlabeled point. Suitable for pool sizes up to millions with sparse features. When to use: fast iteration, single-query budget, well-calibrated model. When not to use: model probabilities are miscalibrated (e.g., raw neural network softmax without temperature scaling); then uncertainty estimates are noisy and the signal breaks down.
              </Prose>
            ),
          },
          {
            label: "Query by committee (QBC)",
            render: () => (
              <Prose>
                Best for: settings where model uncertainty is unreliable from a single model, or where the hypothesis class has multiple plausible solutions. More robust than single-model uncertainty because it captures disagreement across different possible models, not just probability calibration. Cost: K times more expensive — K models must be trained and K forward passes run per scoring round. When to use: ensemble is already needed for production (e.g., RandomForest — use tree disagreement as the QBC signal); small labeled set where single-model estimates are high-variance. Not suitable for large neural networks where K-fold retraining is expensive.
              </Prose>
            ),
          },
          {
            label: "Expected error reduction / variance reduction",
            render: () => (
              <Prose>
                Most principled: directly estimates how much the generalization error or output variance would decrease if point x were labeled. Grounded in information theory and optimal experiment design. Prohibitively expensive: requires re-training the model for each candidate point to estimate its marginal contribution. O(n × n_train) per iteration. In practice, used only with Gaussian Processes (where variance reduction is analytically tractable) or very small pools. Not suitable for any realistic pool-based setting with neural networks. Mentioned because it is the theoretical gold standard that all other strategies approximate.
              </Prose>
            ),
          },
          {
            label: "Coreset / k-center (batch diversity)",
            render: () => (
              <Prose>
                Best for: batch-mode active learning where B {">"}{">"} 1 examples are selected per round. Uncertainty-only batch selection produces redundant queries (B near-identical boundary points). Coreset selection in the feature/embedding space ensures diversity. For deep networks: run coreset on penultimate-layer embeddings (more semantically meaningful than raw input). Cost: O(|U| × |S|) per batch round — feasible for pools up to ~100k with vectorized operations. Best choice when the batch size is 50 or more and query efficiency matters more than oracle access latency.
              </Prose>
            ),
          },
          {
            label: "BADGE (diverse uncertain gradient embeddings)",
            render: () => (
              <Prose>
                State of the art for deep batch active learning as of 2020. Combines: (1) gradient magnitude as uncertainty signal — high-gradient points are uncertain; (2) k-means++ initialization in gradient space for diversity — selected points span the gradient space. The gradient is hallucinated for each class (assumes that point is positive / negative) and the two gradients concatenated — this representation captures both uncertainty and class ambiguity. No hyperparameters for the diversity-uncertainty tradeoff. Best choice when: using a neural network, selecting batches of 50+, and computational budget allows gradient computation for all unlabeled points. Implementation: PyTorch + badge-sampling library.
              </Prose>
            ),
          },
          {
            label: "Stream-based / online active learning",
            render: () => (
              <Prose>
                When examples arrive one at a time in a stream (not a static pool), the decision is: query this example now or discard it forever. The agent must decide in real time based on the current model's uncertainty. Strategy: threshold-based — query if uncertainty exceeds a threshold, skip otherwise. The threshold can be adapted based on labeling budget remaining. Constant memory: only the current model weights are stored, not the pool. Use when: data arrives as a stream (sensor readings, live text), pool storage is infeasible, or budget allocation must be done online.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>8.1 Single-query uncertainty: O(n) — scales to millions</H3>

      <Prose>
        Scoring the full unlabeled pool with a trained classifier costs one forward pass per point: <Code>O(n)</Code> compute, <Code>O(n)</Code> memory for the scores. With a sparse logistic regression on text data, this is microseconds per point — you can score 10 million documents in under a minute. The bottleneck is storing the pool, not scoring it. For pools larger than RAM, process in batches and keep only the top-k candidates by uncertainty score.
      </Prose>

      <Prose>
        Retraining after each query is the real cost. If you retrain from scratch after every single label, the total retraining cost is <Code>O(n_queries × train_cost)</Code>. For logistic regression on tabular data, this is cheap — a few seconds per query. For a ResNet-50 on images, retraining after each label is impractical. The solution: retrain every B labels (batch-mode), or fine-tune from the previous checkpoint rather than retraining from scratch. Fine-tuning reduces retraining cost from hours to minutes, making deep active learning feasible.
      </Prose>

      <H3>8.2 Batch-mode strategies: diversity vs. oracle cost</H3>

      <Prose>
        The motivation for batch-mode selection (choosing B {">"} 1 labels at once) is oracle latency: in real deployments, you send a batch of examples to human annotators and wait hours or days for the labels to come back. Sending 100 examples at once is operationally much better than sending 1 and waiting. But selecting B points with maximum diversity costs more than selecting 1. K-center greedy selection is <Code>O(B × |S| × |U|)</Code> per batch round — for <Code>|U| = 10,000</Code> and <Code>B = 100</Code>, this is <Code>10^9</Code> distance comparisons, which requires vectorized implementation to be fast. BADGE has similar cost (K-means++ initialization on gradient vectors). For very large pools (millions of points), approximate nearest-neighbor search (FAISS, annoy) reduces this to sub-linear cost.
      </Prose>

      <H3>8.3 QBC: K times the model cost</H3>

      <Prose>
        Query by committee trains K models and runs K forward passes. For K=3 logistic regressions on tabular data, this is negligible overhead. For K=3 ResNet-50 networks on images, each training run takes hours — QBC becomes impractical. The practical solution: use a single model with stochastic inference (MC Dropout) to approximate committee disagreement. Instead of K independently trained models, run K stochastic forward passes through a single model with dropout active at test time. Cost: K forward passes, one model to train. This is the key insight of Gal et al. (2017): MC Dropout turns a single trained network into an approximate Bayesian ensemble for free.
      </Prose>

      <H3>8.4 Streaming active learning: O(1) memory</H3>

      <Prose>
        Stream-based active learning has fundamentally different scaling properties. No pool is stored — each arriving example is scored and either queried or discarded immediately. Memory cost: <Code>O(d)</Code> for the model weights, <Code>O(1)</Code> for the current example. This scales to infinite streams with constant memory. The tradeoff: you cannot go back and query a discarded example, even if it turns out to be informative. Adaptive threshold strategies try to mitigate this by setting the query threshold high early (be selective) and lowering it if the budget is not being spent.
      </Prose>

      <H3>8.5 Deep active learning: amortize retraining cost</H3>

      <Prose>
        For large neural networks, the dominant cost is retraining, not scoring. Three strategies to manage it: (1) <strong>Fine-tuning from previous checkpoint</strong>: after each batch of new labels, fine-tune the model for a fixed number of steps rather than retraining from scratch. Cost scales with the number of new labels, not the total dataset size. (2) <strong>Large batch rounds</strong>: select B=1000 labels per round instead of B=1, reducing the number of training rounds from 1000 to 1. The loss in query efficiency (less precise selection) is usually worth the computational saving. (3) <strong>Embedding pre-computation</strong>: compute penultimate-layer embeddings for all unlabeled points once (one forward pass), then run coreset or BADGE in embedding space. Recompute embeddings only after model weights change significantly.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>9.1 Miscalibrated model — uncertainty signal is broken</H3>

      <Prose>
        Uncertainty sampling assumes the model's predicted probabilities accurately reflect its confidence. Raw neural network softmax outputs are notoriously overconfident — a network trained on ImageNet that has never seen a snake will still output a confident (but wrong) class label for a snake image. If confidence is overestimated everywhere, entropy scores are artificially low everywhere and the strategy degenerates toward random selection. Fix: calibrate the model before using its uncertainty for query selection. Temperature scaling (dividing logits by a learned scalar T {">"} 1) is the fastest calibration method and works well for neural networks. Platt scaling or isotonic calibration work for sklearn classifiers. Check calibration with a reliability diagram on a held-out set before deploying active learning.
      </Prose>

      <H3>9.2 Redundant batch queries — near-duplicate selection</H3>

      <Prose>
        If you select a batch of B examples using uncertainty alone, all B examples may cluster near the same high-uncertainty region — they overlap in information content and jointly provide less than B independent bits of evidence. This is the <em>redundancy problem</em> of batch-mode uncertainty sampling. Detection: visualize the selected batch in feature space — if all B points are within a small neighborhood, you have a redundancy problem. Fix: use diversity-aware methods (k-center coreset, BADGE) instead of pure uncertainty. Alternatively, apply a diversity post-processing step: after selecting the top 10B uncertain points, run a greedy diversity maximization to prune them to B.
      </Prose>

      <H3>9.3 Class imbalance — uncertainty finds only the majority boundary</H3>

      <Prose>
        With severe class imbalance (e.g., 95% negative, 5% positive), uncertainty sampling concentrates queries near the decision boundary of the majority class. The boundary of the minority class — where the model makes most of its errors — is ignored because those examples are rare in the pool. The result: labels are wasted on boundary ambiguity in the easy majority region while the hard minority class remains underrepresented. Fix: apply class-weighted uncertainty scoring — boost the uncertainty score for examples predicted to be in the minority class, or enforce a minimum number of minority-class queries per round. Alternatively, use a stratified pool that oversamples minority examples before uncertainty scoring.
      </Prose>

      <H3>9.4 Cold start — not enough seed labels</H3>

      <Prose>
        Active learning requires a trained model to score the pool. With only 2 or 3 seed labels, the model is essentially random and uncertainty scores are meaningless — the strategy degenerates to random selection until enough labels accumulate to produce useful predictions. The minimum seed size depends on the problem complexity and the model family. For logistic regression on 2D data, 5–10 labels per class suffice. For deep learning on images with 100 classes, you typically need 50–100 labels per class before uncertainty signals are reliable. A practical heuristic: plot the learning curve of a randomly trained model; the active learning seed should be large enough to reach non-trivial accuracy (at least 5–10 percentage points above random guessing) before switching to active query selection.
      </Prose>

      <H3>9.5 Concept drift in streaming active learning</H3>

      <Prose>
        In stream-based settings, the data distribution can shift over time — spam tactics evolve, user behavior changes, sensor calibration drifts. A model trained on historical labels may develop high uncertainty everywhere as the distribution shifts, generating uninformative queries. Or it may become overconfident in the wrong direction, generating no queries at all. Detection: monitor the rate of queries being triggered. If the rate drops to near zero or spikes to near 100%, a drift event has likely occurred. Fix: maintain a sliding window of recent labeled examples for retraining, implement explicit drift detection (ADWIN, Page-Hinkley test), or re-bootstrap the labeled set from scratch when drift is detected.
      </Prose>

      <H3>9.6 Noisy oracles — labeling errors corrupt the active set</H3>

      <Prose>
        Human annotators make mistakes, especially on uncertain examples — which are exactly the examples active learning prioritizes. This creates a dangerous feedback loop: the model selects the hardest examples (high uncertainty), the annotator is most likely to mislabel hard examples, and the mislabeled examples are added to the training set where they corrupt the model. The effect is amplified because active learning selects points near the boundary multiple times, and a mislabeled boundary point is maximally damaging. Fixes: use multiple annotators and majority-vote for queried examples; apply label smoothing during training; use robust loss functions (symmetric cross-entropy) that downweight the gradient contribution of likely-mislabeled examples; or apply confidence-weighted training where query points are given lower weight proportional to their annotator agreement score.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All citations below were WebSearch-verified for author, year, venue, and main claims. Read them in this order to follow the intellectual lineage from the founding papers through the deep learning era.
      </Prose>

      <StepTrace
        label="primary literature"
        steps={[
          {
            label: "Seung, Opper, Sompolinsky 1992 — Query by Committee",
            render: () => (
              <Prose>
                Seung, H.S., Opper, M., and Sompolinsky, H. (1992). "Query by Committee." <em>Proceedings of the Fifth Annual ACM Workshop on Computational Learning Theory (COLT)</em>, Pittsburgh, PA, July 27–29, 1992, pp. 287–294. ACM. DOI: 10.1145/130385.130417. Proposed training a committee of models on the same labeled set and selecting the unlabeled point with maximum committee disagreement. Proved that for a version space query model, QBC achieves exponential reduction in generalization error with the number of labels — compared to the inverse power law of random sampling. This theoretical result is what motivated active learning as a field: the possibility of an exponential rather than polynomial label efficiency gain.
              </Prose>
            ),
          },
          {
            label: "Cohn, Atlas, Ladner 1994 — Pool-based active learning",
            render: () => (
              <Prose>
                Cohn, D., Atlas, L., and Ladner, R. (1994). "Improving Generalization with Active Learning." <em>Machine Learning</em>, 15(2):201–221. Springer. DOI: 10.1007/BF00993277. Introduced the pool-based active learning framework — the dominant paradigm to this day — in which a model selects queries from a fixed pool of unlabeled examples rather than constructing queries from scratch. Demonstrated experimentally on neural networks that selective sampling achieves better generalization than random sampling for a fixed label budget. Coined the phrase "selective sampling" and provided the first systematic empirical study of active learning.
              </Prose>
            ),
          },
          {
            label: "Lewis and Gale 1994 — Uncertainty sampling for text",
            render: () => (
              <Prose>
                Lewis, D.D. and Gale, W.A. (1994). "A Sequential Algorithm for Training Text Classifiers." <em>Proceedings of the 17th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval</em>, Dublin, Ireland, pp. 3–12. DOI: 10.5555/188490.188495. arXiv: cmp-lg/9407020. Introduced uncertainty sampling — selecting the unlabeled example where the current classifier is least confident — and demonstrated that it reduced the required training data by up to 500-fold for a text categorization task. This paper made active learning practically attractive: a 500× label reduction is not incremental, it is transformative. The algorithm is simple, model-agnostic, and has been the baseline for every subsequent active learning paper.
              </Prose>
            ),
          },
          {
            label: "Settles 2009 — Active Learning Literature Survey",
            render: () => (
              <Prose>
                Settles, B. (2009). "Active Learning Literature Survey." University of Wisconsin–Madison Computer Sciences Technical Report 1648. Available at burrsettles.com/pub/settles.activelearning.pdf. The canonical reference for the entire field. Settles organized the scattered literature into a unified taxonomy covering all major query strategies — uncertainty sampling (least confident, margin, entropy), query by committee (vote entropy, KL divergence), expected model change, variance reduction, density-weighted methods — with worked examples and empirical comparisons. Every subsequent paper in active learning cites this survey. It is the correct first reading before diving into primary papers.
              </Prose>
            ),
          },
          {
            label: "Dasgupta 2011 — Two faces of active learning",
            render: () => (
              <Prose>
                Dasgupta, S. (2011). "Two Faces of Active Learning." <em>Theoretical Computer Science</em>, 412(19):1767–1781. Elsevier. DOI: 10.1016/j.tcs.2010.12.054. Provided a clean theoretical framework organizing active learning algorithms into two paradigms: version space shrinkage (query examples that eliminate the most inconsistent hypotheses — the face that QBC and uncertainty sampling approximate) and cluster exploitation (use unlabeled structure to guide queries — the face that density-weighted and semi-supervised methods exploit). Recent algorithms work with generic hypothesis classes and have provably characterized labeling requirements. Essential reading for understanding why active learning works theoretically, not just empirically.
              </Prose>
            ),
          },
          {
            label: "Gal, Islam, Ghahramani 2017 — Deep Bayesian Active Learning (BALD)",
            render: () => (
              <Prose>
                Gal, Y., Islam, R., and Ghahramani, Z. (2017). "Deep Bayesian Active Learning with Image Data." <em>Proceedings of the 34th International Conference on Machine Learning (ICML)</em>, Sydney, Australia. arXiv:1703.02910. Introduced the BALD (Bayesian Active Learning by Disagreement) acquisition function for deep neural networks, using Monte-Carlo Dropout to approximate Bayesian inference. BALD selects points that maximize the mutual information between the prediction and the model's parameters — equivalently, points with high epistemic (reducible) uncertainty rather than just high total predictive uncertainty. Achieved 5% test error on MNIST with only 295 labeled images, compared to 835 for random sampling — a 2.8× label efficiency gain. The key contribution: MC Dropout at test time gives you approximate Bayesian uncertainty for free with an already-trained neural network.
              </Prose>
            ),
          },
          {
            label: "Sener and Savarese 2018 — Core-Set approach for CNNs (ICLR)",
            render: () => (
              <Prose>
                Sener, O. and Savarese, S. (2018). "Active Learning for Convolutional Neural Networks: A Core-Set Approach." <em>Proceedings of the Sixth International Conference on Learning Representations (ICLR)</em>. arXiv:1708.00489. Reframed batch-mode active learning as a core-set selection problem: choose the B examples from the pool such that a model trained on the selected set generalizes best to the remainder. The k-center greedy algorithm provides a 2-approximate solution in polynomial time by iteratively selecting the unlabeled point furthest from the current labeled set in feature/embedding space. Showed that many previously proposed heuristics for deep active learning fail to consistently outperform random selection in batch mode, and that coreset selection is substantially more robust.
              </Prose>
            ),
          },
          {
            label: "Ash, Zhang, Krishnamurthy, Langford, Agarwal 2020 — BADGE (ICLR)",
            render: () => (
              <Prose>
                Ash, J.T., Zhang, C., Krishnamurthy, A., Langford, J., and Agarwal, A. (2020). "Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds." <em>Proceedings of the Eighth International Conference on Learning Representations (ICLR)</em>. arXiv:1906.03671. Introduced BADGE (Batch Active learning by Diverse Gradient Embeddings): sample batches that are disparate and high-magnitude in a hallucinated gradient space. The gradient is computed for each unlabeled point as if it were each class (hallucinated label), producing a gradient embedding that captures both uncertainty (gradient magnitude) and ambiguity (spread over class gradients). A k-means++ initialization in this gradient space selects a diverse, uncertain batch with no hand-tuned diversity-uncertainty tradeoff. BADGE consistently matches or outperforms coreset, BALD, and entropy sampling across diverse architectures and batch sizes.
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
        Work through these before moving on. The answer key is below each exercise — resist the urge to read ahead.
      </Prose>

      <H3>Exercise 1 (recall)</H3>
      <Prose>
        Write the three uncertainty sampling criteria — least confident, margin, and entropy — as argmax/argmin expressions over the unlabeled pool. For which of these three is the selected example identical for binary classification? Explain why.
      </Prose>
      <Callout type="answer" title="Answer 1">
        Least confident: argmin over x in U of max_y P(y|x,θ). Margin: argmin over x in U of [P(y1|x,θ) - P(y2|x,θ)] where y1, y2 are the top two classes. Entropy: argmax over x in U of -Σ_k P(y=k|x,θ) log P(y=k|x,θ). For binary classification, all three select the same point. Proof sketch: for K=2, let p = P(y=1|x,θ) and 1-p = P(y=0|x,θ). Least confident minimizes max(p, 1-p) — maximized when p=0.5. Margin minimizes |p - (1-p)| = |2p-1| — also maximized when p=0.5. Entropy maximizes -p log p - (1-p) log(1-p) — also maximized when p=0.5. All three agree at p=0.5 for binary classification. They diverge for K {">"} 2: entropy captures global spread across all classes while margin only looks at the top two.
      </Callout>

      <H3>Exercise 2 (derivation)</H3>
      <Prose>
        Derive the BALD acquisition function from first principles. Start from the mutual information between the model's prediction <Code>{"y"}</Code> and its parameters <Code>{"θ"}</Code>. What is the difference between predictive entropy and expected entropy, and why does BALD prefer points with high predictive entropy but low expected entropy?
      </Prose>
      <Callout type="answer" title="Answer 2">
        Mutual information I(y; θ | x, D) = H[y | x, D] - E_{p(θ|D)}[H[y | x, θ]]. The first term H[y | x, D] is the predictive entropy — entropy of the Bayesian model average (marginalizing over all weight samples). It is high for both aleatoric uncertainty (the example is inherently ambiguous) and epistemic uncertainty (the model is uncertain about its weights). The second term E[H[y | x, θ]] is the expected entropy under individual weight samples — average entropy of each committee member's prediction. It captures aleatoric uncertainty only: if individual models all assign 50/50 probability, the example is genuinely ambiguous regardless of which weights are used. The difference (BALD) = predictive entropy - expected entropy = epistemic uncertainty: the uncertainty due to not knowing the true model weights, which labeling can reduce. BALD prefers points where the average prediction is uncertain (high H[y|x,D]) but individual models are confident in different classes (low E[H[y|x,θ]]). These are points where the committee disagrees, not points that are inherently noisy — exactly what labeling should help resolve.
      </Callout>

      <H3>Exercise 3 (conceptual)</H3>
      <Prose>
        You are running pool-based active learning with entropy sampling and a logistic regression classifier on a 10-class problem. After 200 queries, you notice that the model's accuracy on 8 classes is near-perfect but it still fails on 2 classes. The entropy scores for examples from those 2 classes are consistently low. What is the likely cause, and how do you fix it?
      </Prose>
      <Callout type="answer" title="Answer 3">
        The likely cause is that the 2 failing classes are rare in the pool, so the model has seen very few examples of them and learned to predict them as one of the 8 well-represented classes with high confidence — incorrectly, but confidently. Logistic regression can become overconfident when examples from an underrepresented class are consistently classified as a majority class. Entropy is low (the model is confidently wrong) so the active learning strategy never queries those examples. This is a class imbalance failure mode. Fixes: (1) Stratify the pool — maintain a separate per-class uncertainty score and enforce a minimum number of queries from each class per round. (2) Use class-imbalance-aware query selection: weight entropy scores by the inverse predicted class frequency. (3) Apply class_weight='balanced' to the logistic regression to prevent majority-class overconfidence in the first place. (4) Monitor per-class accuracy during active learning, not just overall accuracy — a per-class breakdown would have caught this failure at query 50 rather than query 200.
      </Callout>

      <H3>Exercise 4 (debugging)</H3>
      <Prose>
        You implement entropy-based active learning for a ResNet-50 image classifier. After 500 queries, the validation accuracy is lower than a random-selection baseline trained on the same 500 labels. List three possible causes and one concrete fix for each.
      </Prose>
      <Callout type="answer" title="Answer 4">
        Cause 1: Miscalibrated model — raw ResNet softmax is overconfident. Entropy scores are uniformly low, queries are nearly random, and the active strategy provides no benefit. Fix: apply temperature scaling on a held-out validation set before using the model for pool scoring. Temperature T {">"} 1 spreads the softmax distribution to better reflect true uncertainty. Cause 2: Redundant batch selection. If you query one at a time and retrain after each, the 500 queries may cluster around the same high-uncertainty region (e.g., examples from a single visually confusing class pair). Fix: switch to batch-mode selection (B=50 per round) with coreset or BADGE to enforce diversity. Cause 3: Cold start failure. With ResNet-50 on ImageNet-scale images, the model needs substantial labeled data before its features are semantically meaningful. If the seed set is too small ({"<"} 50 per class), early uncertainty scores are garbage and the first 200 queries are effectively random or worse. Fix: start with a larger balanced seed (at least 10 examples per class) and potentially pre-train the backbone on unlabeled data (self-supervised pre-training with SimCLR or DINO) before starting the active learning loop — this ensures the embedding space is semantically structured from the beginning.
      </Callout>

      <H3>Exercise 5 (applied)</H3>
      <Prose>
        You have a pool of 100,000 unlabeled documents and a budget of 500 labels for a 5-class text classifier. You plan to select 50 documents per round (10 rounds). Compare and contrast coreset selection in TF-IDF space vs. coreset selection in sentence embedding space (e.g., BERT-large) for this task. Which do you recommend, and why?
      </Prose>
      <Callout type="answer" title="Answer 5">
        TF-IDF coreset: fast to compute (TF-IDF is a sparse matrix operation), no neural network needed. But TF-IDF distance is a bag-of-words measure — two documents can be semantically similar but lexically different (synonyms, paraphrases) and appear far apart in TF-IDF space. Coreset selection in TF-IDF space will select diverse vocabulary but may not select semantically diverse or informationally diverse examples. Dimensionality is high (vocabulary size), making exact distance computation expensive; use approximate nearest-neighbor search. BERT embedding coreset: much more expensive — requires a forward pass through BERT for all 100,000 documents (~hours on CPU, ~minutes on GPU). But BERT embeddings capture semantic similarity, so coreset selection in BERT space selects documents that are semantically representative and diverse. Near-duplicate paraphrases are close in embedding space and won't both be selected. Recommendation: BERT embeddings if you have GPU access. The computational cost (one-time, ~10 min on GPU) is worth it because the 500 labels are expensive, and maximizing their semantic diversity is more valuable than maximizing lexical diversity. Practical pipeline: (1) Compute BERT [CLS] embeddings for all 100,000 documents once. (2) Run k-center greedy on the embedding matrix. (3) Serve selected documents to annotators. Recompute embeddings after each round of training and rerun coreset — embeddings shift as the model fine-tunes.
      </Callout>

      <H3>Exercise 6 (synthesis)</H3>
      <Prose>
        A colleague argues: "Active learning is only useful when labeling is expensive. Since we now have large language models that can label data cheaply via zero-shot or few-shot prompting, active learning is becoming obsolete." Evaluate this argument. Under what conditions is it correct, and under what conditions is active learning still valuable even with LLM-based annotation?
      </Prose>
      <Callout type="answer" title="Answer 6">
        The argument has merit in a narrow regime but is overstated. When it holds: for standard NLP tasks (sentiment, topic classification, intent detection) where GPT-4-class models achieve near-human accuracy via prompting, using an LLM to label 10,000 examples costs {"<"} $50 and removes the bottleneck that motivates active learning. In this regime, random labeling with LLMs outperforms active selection with human annotators — cheaper, faster, and comparably accurate. Active learning is still valuable in four scenarios: (1) Domain-specific or specialized tasks where LLMs have poor zero-shot accuracy — radiology report classification, legal clause extraction, proprietary jargon. Here human experts are still the oracle, and label budget matters. (2) Tasks where LLM annotations are systematically biased and human correction is needed for a fraction of examples — active learning identifies which examples most need human review. (3) Latency-critical settings where you need a small, fast specialized model (not an API call to GPT-4). Actively selecting 500 labels to fine-tune a DistilBERT is superior to randomly selecting 500 LLM-generated labels because your model's uncertainty guides selection. (4) Distribution shift and rare events — LLMs fail on out-of-distribution inputs, which are exactly the high-uncertainty examples active learning would prioritize querying. In short: active learning is obsolete for commodity NLP tasks with well-calibrated LLMs as oracles. It remains essential wherever the oracle is imperfect, expensive, or the model must be specialized and efficient.
      </Callout>

    </div>
  ),
};

export default activeLearningContent;
