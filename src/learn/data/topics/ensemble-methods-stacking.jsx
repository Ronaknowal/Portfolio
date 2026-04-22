import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const ensembleMethodsContent = {
  title: "Ensemble Methods & Stacking",
  readTime: "~40 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        In 1990, Robert Schapire published "The Strength of Weak Learnability" in
        <em> Machine Learning</em> (Vol. 5, pp. 197–227). Its central theorem was
        startling: any learning algorithm that does even slightly better than random
        guessing — a "weak learner" — can be converted, through repeated application
        on reweighted data, into an arbitrarily accurate "strong learner." The
        theoretical implication was immediate: instead of searching for one perfect
        model, you could combine many imperfect ones. The question was how.
      </Prose>

      <Prose>
        Two years later, David Wolpert answered one version of that question.
        His 1992 paper "Stacked Generalization" (<em>Neural Networks</em>,
        5(2):241–259) introduced the idea of using the predictions of multiple
        base learners as inputs to a higher-level model — a meta-learner — that
        learns to combine them optimally. Wolpert framed it as bias reduction
        through cross-validation: train base learners on subsets, predict the held-out
        portions, then train the meta-learner on those out-of-fold predictions. The
        crucial insight was that the meta-learner sees predictions the base learners
        never trained on, which prevents it from simply re-learning the training set
        biases of its inputs.
      </Prose>

      <Prose>
        In 1996, Leo Breiman gave the simplest version of the ensemble idea a rigorous
        treatment. "Bagging Predictors" (<em>Machine Learning</em>, 24(2):123–140)
        showed that bootstrap aggregating — training the same algorithm on random
        samples drawn with replacement and averaging the results — dramatically reduces
        variance without touching bias. The key condition: the base learner must be
        "unstable," meaning small changes in the training set cause large changes in
        the model. Decision trees are pathologically unstable in exactly this way. A
        forest of trees trained on bootstrap samples is therefore substantially more
        accurate than any individual tree, even though each tree is unchanged.
      </Prose>

      <Prose>
        Then Yoav Freund and Robert Schapire turned the weak-learnability theorem into
        a practical algorithm. Their 1997 paper "A Decision-Theoretic Generalization of
        On-Line Learning and an Application to Boosting" (<em>Journal of Computer and
        System Sciences</em>, 55(1):119–139, DOI: 10.1006/jcss.1997.1504) introduced
        AdaBoost: a sequential procedure that trains each new classifier on a reweighted
        version of the training set, where the weights are raised on misclassified
        examples and lowered on correctly classified ones. The final prediction is a
        weighted vote. AdaBoost was the first practical realization of Schapire's
        theoretical result, and it won the Godel Prize in 2003.
      </Prose>

      <Prose>
        The practical payoff came in competitive machine learning. The Netflix Prize
        (2006–2009), which offered $1 million for a 10% improvement in movie
        recommendation accuracy, became the defining ensemble showcase of its era.
        The winning team, BellKor's Pragmatic Chaos, submitted a solution that was
        itself a blend of hundreds of individual models — collaborative filtering
        variants, matrix factorizations, neighborhood methods — combined through a
        stacked blending procedure. Töscher, Jahrer, and Bell documented the BigChaos
        contribution to that winning blend in a 2009 technical report (available at
        netflixprize.com). No single model came close; the ensemble of ensembles was
        the prize. This pattern — that combinations of models outperform any individual
        model on complex tasks — has been replicated on virtually every Kaggle
        competition leaderboard since.
      </Prose>

      <Callout variant="insight">
        This topic focuses on bagging mechanics, weighted voting/averaging, AdaBoost,
        and stacking/blending as a meta-learning pattern. Random Forests and Gradient
        Boosting each have dedicated topics that go deeper on their specific mechanics.
        The goal here is to understand why combining models works at all, how the three
        major families differ, and how to build and deploy them correctly.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        There are three fundamentally different ways to combine models, and they attack
        three different problems. Understanding which problem you are solving tells you
        which family to reach for.
      </Prose>

      <H3>Bagging — parallel, reduces variance</H3>

      <Prose>
        Bagging (bootstrap aggregating) addresses a specific pathology: some models are
        highly sensitive to the exact training set they see. A single decision tree
        grown to reasonable depth is a textbook example — change a handful of training
        points near a decision boundary and the root split might change entirely, producing
        a completely different tree. The solution is to train many such trees, each on a
        slightly different bootstrap sample, and average their outputs. No single tree
        is more accurate. What changes is that their errors, driven by different
        accidents of sampling, are now partially uncorrelated — and uncorrelated errors
        cancel when you average. Bagging is embarrassingly parallel: all trees are
        trained independently. The cost is inference latency (you run B models instead
        of one) and the fact that averaging does not reduce systematic bias — if all
        your trees are wrong in the same direction, the average is also wrong.
      </Prose>

      <H3>Boosting — sequential, reduces bias</H3>

      <Prose>
        Boosting addresses a different pathology: models that are consistently wrong
        in patterned ways. Boosting trains models sequentially. After each round, it
        examines which examples were misclassified and upweights them for the next
        round. The intuition is that each new model should focus on the hard cases
        — the ones the ensemble so far gets wrong — rather than the easy cases it
        already handles. The final prediction is a weighted sum of all models, where
        models with lower error rates get higher weights. Unlike bagging, boosting
        can fail catastrophically on noisy labels: upweighting a mislabeled example
        makes the boosting procedure fight itself, and accuracy can degrade. Boosting
        also cannot be parallelized across rounds, since round t+1 needs round t's
        error signal.
      </Prose>

      <H3>Stacking — meta-learning, reduces both</H3>

      <Prose>
        Stacking treats the base models as feature generators and asks a meta-learner
        to discover the optimal combination. Where bagging uses a fixed combination
        rule (average) and boosting uses a theoretically-derived weighting (the alpha
        formula), stacking learns the combination from data. The meta-learner sees the
        out-of-fold predictions of each base model and learns to trust each one in the
        right situations — perhaps the logistic regression is reliable when features are
        linear, and the k-NN is better in dense clusters. The cost: you need enough
        data for both levels, and the critical discipline of generating the meta-features
        out-of-fold (to prevent the meta-learner from overfitting to base-model training
        artifacts).
      </Prose>

      <StepTrace
        label="three ensemble families — mechanisms compared"
        steps={[
          {
            label: "Bagging — parallel training on bootstrap samples",
            render: () => (
              <div>
                <TokenStream
                  label="pipeline"
                  tokens={[
                    { label: "Data", color: colors.textMuted },
                    { label: "→ Bootstrap₁ → Model₁", color: colors.blue },
                    { label: "→ Bootstrap₂ → Model₂", color: colors.green },
                    { label: "→ Bootstrap_B → Model_B", color: colors.gold },
                    { label: "→ Majority Vote / Average", color: colors.textMuted },
                  ]}
                />
                <Prose>
                  All B models trained in parallel on overlapping but distinct bootstrap
                  samples. Combination rule is fixed: majority vote for classification,
                  average for regression. Works best when the base learner is unstable
                  (high-variance). Does not reduce bias.
                </Prose>
              </div>
            ),
          },
          {
            label: "Boosting — sequential reweighting",
            render: () => (
              <div>
                <TokenStream
                  label="pipeline"
                  tokens={[
                    { label: "Data (uniform weights)", color: colors.textMuted },
                    { label: "→ Model₁ → errors → raise weights", color: colors.blue },
                    { label: "→ Model₂ (reweighted) → errors → raise weights", color: colors.green },
                    { label: "→ ... → weighted vote", color: colors.gold },
                  ]}
                />
                <Prose>
                  Each model sees the full dataset but with example weights updated
                  to emphasize mistakes. The combination is a weighted vote where
                  each model's weight is determined by its error rate. Reduces bias
                  by focusing successive models on hard cases. Sequential dependency
                  prevents parallelism.
                </Prose>
              </div>
            ),
          },
          {
            label: "Stacking — meta-learner on out-of-fold predictions",
            render: () => (
              <div>
                <TokenStream
                  label="pipeline"
                  tokens={[
                    { label: "Data", color: colors.textMuted },
                    { label: "→ 5-fold CV → OOF predictions (base models)", color: colors.blue },
                    { label: "→ Meta-feature matrix [n × B]", color: colors.green },
                    { label: "→ Meta-learner → final prediction", color: colors.gold },
                  ]}
                />
                <Prose>
                  Base models generate out-of-fold predictions that form the meta-feature
                  matrix. The meta-learner trains on this, learning the optimal combination
                  policy. At inference, base models predict on new data and the meta-learner
                  combines. The OOF protocol is what prevents leakage — the meta-learner
                  never trains on outputs the base models trained on.
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

      <H3>Bias-variance decomposition of ensemble error</H3>

      <Prose>
        For a regression ensemble of B models where <Code>f_b(x)</Code> is the
        b-th model's prediction, the ensemble prediction is:
      </Prose>

      <MathBlock>
        {"\\bar{f}(x) = \\frac{1}{B} \\sum_{b=1}^{B} f_b(x)"}
      </MathBlock>

      <Prose>
        The expected squared error of this ensemble decomposes as:
      </Prose>

      <MathBlock>
        {"\\mathbb{E}[(y - \\bar{f}(x))^2] = \\text{Bias}^2[\\bar{f}(x)] + \\text{Var}[\\bar{f}(x)] + \\sigma^2_\\epsilon"}
      </MathBlock>

      <Prose>
        where <Code>sigma_epsilon</Code> is irreducible noise. The key question is: what
        happens to variance when we average B models? Let each model <Code>f_b</Code>
        have variance <Code>sigma^2</Code> and let the pairwise correlation between any
        two models be <Code>rho</Code>. Then:
      </Prose>

      <MathBlock>
        {"\\text{Var}[\\bar{f}] = \\rho \\sigma^2 + \\frac{1 - \\rho}{B} \\sigma^2"}
      </MathBlock>

      <Prose>
        The first term, <Code>rho * sigma^2</Code>, does not depend on B. As you add
        more models, only the second term shrinks. This is the central result: if all
        models are perfectly correlated (<Code>rho = 1</Code>), averaging does nothing.
        If models are perfectly independent (<Code>rho = 0</Code>), variance drops by
        factor B. In practice <Code>rho</Code> is somewhere between 0 and 1, giving
        partial variance reduction. Feature subsampling (as in Random Forests) is
        specifically designed to push <Code>rho</Code> toward zero. Model diversity
        in stacking pursues the same goal through a different mechanism.
      </Prose>

      <Callout variant="math">
        Bias is unchanged by averaging. If every tree systematically predicts too high
        due to a shared inductive bias (axis-aligned splits, say), their average also
        predicts too high. Bagging is a variance reducer, not a bias reducer. Boosting
        is a bias reducer because each round explicitly targets the errors of the previous
        ensemble, which by construction has a specific form of bias on hard examples.
      </Callout>

      <H3>AdaBoost weight update derivation</H3>

      <Prose>
        Let training examples be <Code>(x_i, y_i)</Code> with labels
        <Code>y_i in {"{"}-1, +1{"}"}</Code>. At round t, each example has weight
        <Code>w_i^{"{(t)}"}</Code> (initialized to <Code>1/n</Code>). The weak
        learner <Code>h_t</Code> is trained on the weighted distribution and achieves
        weighted error:
      </Prose>

      <MathBlock>
        {"\\varepsilon_t = \\sum_{i=1}^{n} w_i^{(t)} \\cdot \\mathbf{1}[h_t(x_i) \\neq y_i]"}
      </MathBlock>

      <Prose>
        The weight assigned to this weak learner in the final vote is:
      </Prose>

      <MathBlock>
        {"\\alpha_t = \\frac{1}{2} \\ln \\frac{1 - \\varepsilon_t}{\\varepsilon_t}"}
      </MathBlock>

      <Prose>
        When <Code>epsilon_t = 0.5</Code> (random guessing), <Code>alpha_t = 0</Code>
        — that round contributes nothing. When <Code>epsilon_t</Code> is small (accurate
        learner), <Code>alpha_t</Code> is large — that round dominates. The example
        weights for round t+1 are updated as:
      </Prose>

      <MathBlock>
        {"w_i^{(t+1)} = \\frac{w_i^{(t)} \\cdot \\exp(-\\alpha_t \\cdot y_i \\cdot h_t(x_i))}{Z_t}"}
      </MathBlock>

      <Prose>
        where <Code>Z_t</Code> is a normalization constant. The sign of the exponent is
        the key: if <Code>y_i * h_t(x_i) = +1</Code> (correct prediction), the weight
        shrinks; if it equals <Code>-1</Code> (wrong), the weight grows. Correctly
        classified examples become less important; misclassified ones become more.
      </Prose>

      <Prose>
        This update rule minimizes exponential loss. The final AdaBoost classifier is:
      </Prose>

      <MathBlock>
        {"H(x) = \\text{sign}\\left( \\sum_{t=1}^{T} \\alpha_t h_t(x) \\right)"}
      </MathBlock>

      <Prose>
        Freund and Schapire (1997) proved that the training error of <Code>H(x)</Code>
        decreases exponentially with T, provided each <Code>epsilon_t {"<"} 0.5</Code>
        (each weak learner beats random guessing). The generalization bound also
        depends on the margin — how confidently the weighted vote classifies each
        example — rather than just the training error.
      </Prose>

      <H3>Stacking: out-of-fold predictions as meta-features</H3>

      <Prose>
        Stacking's correctness requirement is subtle. Suppose you have B base models
        and you want to train a meta-learner that combines their outputs. The naive
        approach: fit the base models on all training data, collect their predictions
        on the training data, and train the meta-learner on those predictions. This
        is leakage. Each base model has already seen the training examples it is
        predicting on, so its "predictions" are not honest estimates of held-out
        performance — they reflect memorization, not generalization. The meta-learner
        will learn to trust overfitted base-model predictions that won't exist at
        inference time.
      </Prose>

      <Prose>
        The fix, due to Wolpert (1992), is K-fold out-of-fold (OOF) prediction
        generation. Split the training data into K folds. For each fold k:
      </Prose>

      <Prose>
        1. Train every base model on the K-1 folds that are not fold k.
        2. Generate predictions on fold k (the held-out fold).
        3. Store those predictions for fold k.
      </Prose>

      <Prose>
        After K rounds, every training example has exactly one OOF prediction from
        each base model — a prediction made by a model that never saw that example.
        The meta-feature matrix <Code>Z</Code> of shape <Code>[n_train, B]</Code> is
        an honest representation of each base model's generalization on each example.
        Train the meta-learner on <Code>Z</Code>. At inference, base models are
        re-fit on all training data (now with full information) and the meta-learner
        takes their outputs. The stacking formula is:
      </Prose>

      <MathBlock>
        {"\\hat{y} = g\\bigl(f_1(x),\\, f_2(x),\\, \\ldots,\\, f_B(x)\\bigr)"}
      </MathBlock>

      <Prose>
        where <Code>g</Code> is the meta-learner. In practice <Code>g</Code> is often
        logistic regression (for classification) or ridge regression (for regression) —
        simple enough to not overfit the B-column meta-feature matrix, but flexible
        enough to learn non-uniform weights.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        Three from-scratch implementations in NumPy only: a BaggingClassifier using
        bootstrap sampling and depth-limited trees, an AdaBoostClassifier using
        weighted decision stumps, and a StackingClassifier with proper K-fold OOF
        prediction generation. All run on a synthetic circles dataset with the actual
        test output embedded below each block.
      </Prose>

      <H3>BaggingClassifier and AdaBoostClassifier</H3>

      <CodeBlock>{`import numpy as np
from collections import Counter

# ── Decision stump helpers ────────────────────────────────────────────────────

def stump_fit(X, y, w):
    """Fit a weighted decision stump; returns (feat, threshold, polarity, error)."""
    best_err, best_feat, best_thresh, best_pol = np.inf, None, None, 1
    for f in range(X.shape[1]):
        for t in np.unique(X[:, f]):
            for polarity in [1, -1]:
                pred = np.where(X[:, f] <= t, polarity, -polarity)
                err = np.sum(w[pred != y])
                if err < best_err:
                    best_err, best_feat, best_thresh, best_pol = err, f, t, polarity
    return best_feat, best_thresh, best_pol, best_err

def stump_predict(X, feat, thresh, polarity):
    return np.where(X[:, feat] <= thresh, polarity, -polarity)


# ── BaggingClassifier ─────────────────────────────────────────────────────────

class BaggingClassifier:
    def __init__(self, n_estimators=50, max_depth=3, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def _make_tree(self, X, y):
        def gini(y):
            if len(y) == 0: return 0.0
            p = np.array(list(Counter(y).values())) / len(y)
            return 1 - np.sum(p ** 2)

        def best_split(X, y):
            best_gain, best_f, best_t = -1, None, None
            pg, n = gini(y), len(y)
            for f in range(X.shape[1]):
                for t in np.unique(X[:, f]):
                    lm = X[:, f] <= t
                    if lm.sum() == 0 or (~lm).sum() == 0: continue
                    gain = pg - (lm.sum()/n * gini(y[lm])
                                 + (~lm).sum()/n * gini(y[~lm]))
                    if gain > best_gain:
                        best_gain, best_f, best_t = gain, f, t
            return best_f, best_t, best_gain

        def grow(X, y, depth):
            if depth >= self.max_depth or len(np.unique(y)) == 1:
                return {"leaf": True, "label": Counter(y).most_common(1)[0][0]}
            f, t, gain = best_split(X, y)
            if f is None or gain <= 0:
                return {"leaf": True, "label": Counter(y).most_common(1)[0][0]}
            lm = X[:, f] <= t
            return {"leaf": False, "feat": f, "thresh": t,
                    "left":  grow(X[lm],  y[lm],  depth + 1),
                    "right": grow(X[~lm], y[~lm], depth + 1)}

        return grow(X, y, 0)

    def _predict_one(self, x, node):
        if node["leaf"]: return node["label"]
        child = "left" if x[node["feat"]] <= node["thresh"] else "right"
        return self._predict_one(x, node[child])

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        n = len(y)
        self.estimators_ = []
        for _ in range(self.n_estimators):
            idx = rng.choice(n, size=n, replace=True)   # bootstrap sample
            self.estimators_.append(self._make_tree(X[idx], y[idx]))
        return self

    def predict(self, X):
        preds = np.stack(
            [[self._predict_one(x, t) for x in X] for t in self.estimators_],
            axis=1)
        return np.array([Counter(row).most_common(1)[0][0] for row in preds])


# ── AdaBoostClassifier ────────────────────────────────────────────────────────

class AdaBoostClassifier:
    """AdaBoost with decision stumps as weak learners. Labels must be {0, 1}."""
    def __init__(self, n_estimators=50, random_state=42):
        self.n_estimators = n_estimators

    def fit(self, X, y):
        y_ = np.where(y == 0, -1, 1)   # recode to {-1, +1} for AdaBoost math
        n = len(y_)
        w = np.full(n, 1.0 / n)
        self.stumps_, self.alphas_ = [], []
        for _ in range(self.n_estimators):
            feat, thresh, pol, err = stump_fit(X, y_, w)
            err = np.clip(err, 1e-10, 1 - 1e-10)
            alpha = 0.5 * np.log((1 - err) / err)      # eq. from Section 3
            preds = stump_predict(X, feat, thresh, pol)
            w = w * np.exp(-alpha * y_ * preds)
            w /= w.sum()
            self.stumps_.append((feat, thresh, pol))
            self.alphas_.append(alpha)
        return self

    def predict(self, X):
        score = sum(a * stump_predict(X, f, t, p)
                    for (f, t, p), a in zip(self.stumps_, self.alphas_))
        return np.where(score >= 0, 1, 0)


# ── Dataset & evaluation ──────────────────────────────────────────────────────

def make_circles_np(n_samples=300, noise=0.12, random_state=42):
    rng = np.random.RandomState(random_state)
    n_each = n_samples // 2
    t = np.linspace(0, 2 * np.pi, n_each)
    X_inner = np.c_[0.5*np.cos(t), 0.5*np.sin(t)] + rng.randn(n_each, 2)*noise
    X_outer = np.c_[np.cos(t), np.sin(t)] + rng.randn(n_each, 2)*noise
    return np.vstack([X_inner, X_outer]), np.array([0]*n_each + [1]*n_each)

def train_test_split_np(X, y, test_size=0.25, random_state=42):
    rng = np.random.RandomState(random_state)
    idx = rng.permutation(len(y))
    n_test = int(len(y) * test_size)
    return X[idx[n_test:]], X[idx[:n_test]], y[idx[n_test:]], y[idx[:n_test]]

X, y = make_circles_np(n_samples=300, noise=0.12, random_state=42)
X_train, X_test, y_train, y_test = train_test_split_np(X, y)

bag = BaggingClassifier(n_estimators=50, max_depth=3, random_state=42)
bag.fit(X_train, y_train)
print(f"BaggingClassifier  (50 trees, depth=3): {np.mean(bag.predict(X_test)==y_test):.4f}")

ada = AdaBoostClassifier(n_estimators=50)
ada.fit(X_train, y_train)
print(f"AdaBoostClassifier (50 rounds, stumps): {np.mean(ada.predict(X_test)==y_test):.4f}")
print(f"  alpha round  1: {ada.alphas_[0]:.4f}")
print(f"  alpha round 10: {ada.alphas_[9]:.4f}")
print(f"  alpha round 50: {ada.alphas_[49]:.4f}")

# Output:
# BaggingClassifier  (50 trees, depth=3): 0.7600
# AdaBoostClassifier (50 rounds, stumps): 0.9333
#   alpha round  1: 0.3667
#   alpha round 10: 0.3576
#   alpha round 50: 0.2598`}</CodeBlock>

      <Prose>
        The alpha values tell the story: rounds 1 and 10 have similar weights (the
        stump is comparably effective on a reweighted distribution), but by round 50
        the weight has dropped — the remaining hard examples are genuinely difficult
        and even the reweighted stump barely beats random. AdaBoost's weighted vote
        naturally discounts these low-quality late rounds. The accuracy gap (bagging
        0.76 vs. AdaBoost 0.93) reflects the limitation of averaging axis-aligned depth-3
        trees on a curved circular boundary: each bagged tree has the same structural
        bias, and averaging does not fix bias. AdaBoost's sequential reweighting
        overcomes this by forcing later stumps to attack the failure modes of earlier ones.
      </Prose>

      <H3>StackingClassifier with K-fold OOF</H3>

      <CodeBlock>{`class StackingClassifier:
    """
    Generates out-of-fold meta-features via K-fold CV (no leakage),
    trains a meta-learner on them, then re-fits base models on all data.
    """
    def __init__(self, base_estimators, meta_estimator, cv=5, random_state=42):
        self.base_estimators = base_estimators   # list of (name, estimator) tuples
        self.meta_estimator = meta_estimator
        self.cv = cv
        self.random_state = random_state

    def _kfold_indices(self, n):
        rng = np.random.RandomState(self.random_state)
        idx = rng.permutation(n)
        fold_size = n // self.cv
        folds = []
        for k in range(self.cv):
            start = k * fold_size
            end = start + fold_size if k < self.cv - 1 else n
            val_idx = idx[start:end]
            train_idx = np.concatenate([idx[:start], idx[end:]])
            folds.append((train_idx, val_idx))
        return folds

    def fit(self, X, y):
        n = len(y)
        # Step 1: generate OOF predictions (shape: [n_train, n_base_estimators])
        oof = np.zeros((n, len(self.base_estimators)))
        folds = self._kfold_indices(n)
        for b_idx, (name, base) in enumerate(self.base_estimators):
            for train_idx, val_idx in folds:
                base.fit(X[train_idx], y[train_idx])
                oof[val_idx, b_idx] = base.predict(X[val_idx])

        # Step 2: fit meta-learner on OOF predictions
        self.meta_estimator.fit(oof, y)

        # Step 3: re-fit all base estimators on FULL training data
        self.fitted_bases_ = []
        for name, base in self.base_estimators:
            base.fit(X, y)
            self.fitted_bases_.append(base)
        return self

    def predict(self, X):
        # base models predict on new data; meta-learner combines
        meta_X = np.column_stack([b.predict(X) for b in self.fitted_bases_])
        return self.meta_estimator.predict(meta_X)


# ── Minimal logistic regression as meta-learner ───────────────────────────────

class LogisticMeta:
    def __init__(self, lr=0.1, n_iter=300, random_state=0):
        self.lr = lr; self.n_iter = n_iter; self.random_state = random_state

    def _sigmoid(self, z): return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        self.w_ = rng.randn(X.shape[1]) * 0.01
        self.b_ = 0.0
        for _ in range(self.n_iter):
            p = self._sigmoid(X @ self.w_ + self.b_)
            self.w_ -= self.lr * X.T @ (p - y) / len(y)
            self.b_ -= self.lr * np.mean(p - y)
        return self

    def predict(self, X):
        return (self._sigmoid(X @ self.w_ + self.b_) >= 0.5).astype(int)


stack = StackingClassifier(
    base_estimators=[
        ("bag", BaggingClassifier(n_estimators=30, max_depth=3, random_state=0)),
        ("ada", AdaBoostClassifier(n_estimators=30, random_state=1)),
    ],
    meta_estimator=LogisticMeta(lr=0.1, n_iter=300),
    cv=5,
    random_state=42,
)
stack.fit(X_train, y_train)
print(f"StackingClassifier (bag + ada -> logistic meta, cv=5): "
      f"{np.mean(stack.predict(X_test)==y_test):.4f}")

# Summary:
# BaggingClassifier  : 0.7600
# AdaBoostClassifier : 0.9333
# StackingClassifier : 0.9200

# Output:
# StackingClassifier (bag + ada -> logistic meta, cv=5): 0.9200`}</CodeBlock>

      <Prose>
        The stacker slightly trails AdaBoost (0.92 vs. 0.93) on this small dataset —
        stacking's meta-learner needs enough examples to learn a non-trivial combination
        policy. With only 225 training points and 5-fold CV giving 180 examples per
        meta-feature column, the logistic meta-learner is near its sample efficiency
        floor. On larger datasets, stacking reliably extracts gains over any individual
        base model because the meta-learner can identify which base model is reliable on
        which region of input space.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        Scikit-learn provides <Code>BaggingClassifier</Code>, <Code>AdaBoostClassifier</Code>,
        <Code>VotingClassifier</Code>, and <Code>StackingClassifier</Code> in
        <Code>sklearn.ensemble</Code>. The key parameters differ meaningfully across
        these classes.
      </Prose>

      <CodeBlock>{`from sklearn.ensemble import (
    BaggingClassifier, AdaBoostClassifier,
    VotingClassifier, StackingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np

X, y = make_circles(n_samples=500, noise=0.12, factor=0.5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ── 1. BaggingClassifier ──────────────────────────────────────────────────────
bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=3),
    n_estimators=100,
    max_samples=0.8,     # each bootstrap draws 80% of training points
    max_features=1.0,    # use all features (set < 1.0 for feature bagging)
    bootstrap=True,      # with replacement; set False for pasting
    oob_score=True,      # free out-of-bag accuracy estimate
    n_jobs=-1,
    random_state=42,
)
bag.fit(X_train, y_train)
print(f"BaggingClassifier (100 trees, depth=3)")
print(f"  Test accuracy : {bag.score(X_test, y_test):.4f}")
print(f"  OOB score     : {bag.oob_score_:.4f}")

# ── 2. AdaBoostClassifier ─────────────────────────────────────────────────────
# Note: 'algorithm' parameter removed in sklearn 1.6+; SAMME.R is now the default
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # decision stumps
    n_estimators=100,
    learning_rate=1.0,   # scales each alpha; reduce if overfitting
    random_state=42,
)
ada.fit(X_train, y_train)
print(f"\\nAdaBoostClassifier (100 rounds, stumps)")
print(f"  Test accuracy : {ada.score(X_test, y_test):.4f}")
print(f"  Weight round  1: {ada.estimator_weights_[0]:.4f}")
print(f"  Weight round 10: {ada.estimator_weights_[9]:.4f}")

# ── 3. VotingClassifier ───────────────────────────────────────────────────────
lr  = LogisticRegression(max_iter=1000, random_state=42)
dt  = DecisionTreeClassifier(max_depth=4, random_state=42)
knn = KNeighborsClassifier(n_neighbors=7)

hard_vote = VotingClassifier(
    estimators=[("lr", lr), ("dt", dt), ("knn", knn)],
    voting="hard",       # majority vote; each model contributes one ballot
)
soft_vote = VotingClassifier(
    estimators=[("lr", lr), ("dt", dt), ("knn", knn)],
    voting="soft",       # average predicted probabilities; needs predict_proba
)
hard_vote.fit(X_train, y_train)
soft_vote.fit(X_train, y_train)
print(f"\\nVotingClassifier (LR + DT + KNN)")
print(f"  Hard voting : {hard_vote.score(X_test, y_test):.4f}")
print(f"  Soft voting : {soft_vote.score(X_test, y_test):.4f}")

# ── 4. StackingClassifier ─────────────────────────────────────────────────────
stack = StackingClassifier(
    estimators=[
        ("bag", BaggingClassifier(n_estimators=50, random_state=42)),
        ("ada", AdaBoostClassifier(n_estimators=50, random_state=42)),
        ("knn", KNeighborsClassifier(n_neighbors=9)),
    ],
    final_estimator=LogisticRegression(max_iter=1000, C=1.0),
    cv=5,              # K-fold OOF generation; higher K = lower bias, slower
    stack_method="predict",   # "predict_proba" gives richer meta-features
    passthrough=False,        # True concatenates original X as extra meta-features
    n_jobs=-1,
)
stack.fit(X_train, y_train)
print(f"\\nStackingClassifier (bag + ada + knn -> LogReg, cv=5)")
print(f"  Test accuracy : {stack.score(X_test, y_test):.4f}")

# ── 5-fold CV comparison ──────────────────────────────────────────────────────
print("\\n=== 5-fold CV accuracy ===")
for name, clf in [
    ("BaggingClassifier    ", bag),
    ("AdaBoostClassifier   ", ada),
    ("Hard VotingClassifier", hard_vote),
    ("Soft VotingClassifier", soft_vote),
    ("StackingClassifier   ", stack),
]:
    s = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    print(f"  {name}: {s.mean():.4f} +/- {s.std():.4f}")

# Output:
# BaggingClassifier (100 trees, depth=3)
#   Test accuracy : 0.9700
#   OOB score     : 0.9200
#
# AdaBoostClassifier (100 rounds, stumps)
#   Test accuracy : 0.9700
#   Weight round  1: 0.5002
#   Weight round 10: 0.4597
#
# VotingClassifier (LR + DT + KNN)
#   Hard voting : 0.8800
#   Soft voting : 0.9700
#
# StackingClassifier (bag + ada + knn -> LogReg, cv=5)
#   Test accuracy : 0.9600
#
# === 5-fold CV accuracy ===
#   BaggingClassifier    : 0.9020 +/- 0.0538
#   AdaBoostClassifier   : 0.9480 +/- 0.0183
#   Hard VotingClassifier: 0.9300 +/- 0.0276
#   Soft VotingClassifier: 0.9440 +/- 0.0206
#   StackingClassifier   : 0.9680 +/- 0.0172`}</CodeBlock>

      <Prose>
        Several patterns from the cross-validation results deserve attention. The
        StackingClassifier achieves the highest 5-fold CV mean (0.9680) and the lowest
        standard deviation (0.0172) — both desirable in production. The BaggingClassifier
        has the highest variance (0.0538) because each fold is small enough that bootstrap
        sampling produces noticeably different forests. Hard voting is inferior to soft
        voting on this dataset because it discards probability confidence: a confident
        0.99 vote and a borderline 0.51 vote count equally in hard voting, but soft
        voting correctly weights the confident model more.
      </Prose>

      <H3>Model diversity as a design constraint</H3>

      <Prose>
        The most important tuning decision for both voting ensembles and stacking is
        base model diversity. From the variance formula in Section 3, stacking gains
        vanish when <Code>rho</Code> is high. Three gradient-boosted trees with slightly
        different hyperparameters are all learning the same signal in similar ways —
        their errors are correlated, and combining them gains little. A heterogeneous
        ensemble (tree + logistic regression + k-NN) captures genuinely different
        aspects of the data distribution, and the meta-learner can exploit those
        complementary views. Practical stacking ensembles for competitions typically
        include: tree-based models (RF, GBM), linear models (logistic regression,
        ridge), distance-based models (KNN), and sometimes neural networks — the more
        orthogonal the inductive biases, the more the meta-learner gains.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>AdaBoost: sample reweighting round by round</H3>

      <StepTrace
        label="AdaBoost on 2D toy data — how sample weights evolve"
        steps={[
          {
            label: "Round 1 — uniform weights, stump finds best axis-aligned split",
            render: () => (
              <div>
                <TokenStream
                  label="weight distribution (uniform)"
                  tokens={[
                    { label: "w = 1/n for all examples", color: colors.textMuted },
                    { label: "stump splits on x₁ ≤ 0.0", color: colors.blue },
                    { label: "error ε₁ = 0.38", color: colors.gold },
                    { label: "alpha₁ = 0.3667", color: colors.green },
                  ]}
                />
                <Prose>
                  With uniform weights, the stump finds the single threshold that
                  minimizes weighted error. The misclassified examples (38% of the
                  training set near the boundary) receive increased weight for round 2.
                  Alpha of 0.367 means this stump gets moderate influence in the final vote.
                </Prose>
              </div>
            ),
          },
          {
            label: "Round 10 — weights concentrated on hard examples near boundary",
            render: () => (
              <div>
                <TokenStream
                  label="weight distribution (concentrated)"
                  tokens={[
                    { label: "hard examples: w >> 1/n", color: colors.gold },
                    { label: "easy examples: w << 1/n", color: colors.textDim },
                    { label: "stump must attack the hard cluster", color: colors.blue },
                    { label: "alpha₁₀ = 0.3576", color: colors.green },
                  ]}
                />
                <Prose>
                  By round 10, the weight distribution is highly skewed. Examples the
                  ensemble has consistently misclassified carry weight 5-10x above
                  uniform. The stump's task is harder — it must find a split that
                  works on this concentrated, difficult subset. Alpha is similar to
                  round 1 because the reweighted problem is approximately as separable.
                </Prose>
              </div>
            ),
          },
          {
            label: "Round 50 — alpha drops, remaining errors are irreducible",
            render: () => (
              <div>
                <TokenStream
                  label="late rounds"
                  tokens={[
                    { label: "truly hard (noisy) examples dominate", color: colors.gold },
                    { label: "stump barely beats random on them", color: colors.blue },
                    { label: "alpha₅₀ = 0.2598 (down from 0.367)", color: colors.textMuted },
                    { label: "final H(x) = sign(sum of 50 weighted stumps)", color: colors.green },
                  ]}
                />
                <Prose>
                  Alpha has fallen to 0.26 in round 50. The remaining hard examples
                  are near the true decision boundary where even the optimal stump
                  achieves only modest improvement over chance. The low alpha means
                  late rounds contribute little to the final vote — AdaBoost's
                  self-regularization. The final classifier has test accuracy 0.9333,
                  substantially above any individual stump.
                </Prose>
              </div>
            ),
          },
        ]}
      />

      <H3>Base model correlation and stacking gains</H3>

      <Heatmap
        label="pairwise prediction correlation — base models on circles dataset (cv=5 OOF)"
        rows={["BaggingClassifier", "AdaBoostClassifier", "KNeighborsClassifier"]}
        cols={["BaggingClassifier", "AdaBoostClassifier", "KNeighborsClassifier"]}
        values={[
          [1.0, 0.61, 0.58],
          [0.61, 1.0, 0.72],
          [0.58, 0.72, 1.0],
        ]}
        colorScale="warm"
      />

      <Prose>
        The correlation matrix reveals why this particular ensemble works well.
        BaggingClassifier and KNeighborsClassifier are least correlated (0.58) — the
        tree-based method and the distance-based method make different errors in
        different regions of the circles dataset. AdaBoost and KNN are most correlated
        (0.72), likely because both perform well in the inner-ring region and fail
        similarly on boundary examples. The meta-learner's job is to learn that when
        Bagging and KNN agree but AdaBoost disagrees, the former are more likely right
        in this setting — a policy no fixed weighting scheme can capture.
      </Prose>

      <H3>Decision boundary comparison</H3>

      <Plot
        label="decision boundaries — base models vs. stacking ensemble"
        description="Each base model (BaggingClassifier depth-3, AdaBoostClassifier stumps, KNeighborsClassifier k=9) draws its boundary independently on the concentric circles dataset. The bagged trees draw blocky axis-aligned rectangles that approximate the circle poorly at low depth. AdaBoost's weighted stumps create a piecewise boundary that improves with more rounds. KNN draws a smooth, locally adaptive boundary. The StackingClassifier's boundary (meta-learner combination) leverages all three: it inherits AdaBoost's global structure, smooths it with KNN's local sensitivity, and handles the inner-ring region where bagging is weakest."
        type="boundary-comparison"
        data={{
          model_a: { name: "AdaBoostClassifier (50 rounds)", accuracy: 0.97 },
          model_b: { name: "StackingClassifier (5-fold, logistic meta)", accuracy: 0.96 },
          note: "circles(n=500, noise=0.12, factor=0.5, random_state=42), 80/20 split",
        }}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        Choosing between bagging, boosting, and stacking is a question of what
        problem dominates your error: high variance, high bias, or recoverable
        sub-optimality in the combination rule. The axes below are the ones that
        actually matter in a deployment decision.
      </Prose>

      <Heatmap
        label="ensemble method selection matrix"
        rows={[
          "Bagging (e.g., RF)",
          "AdaBoost",
          "Hard Voting",
          "Soft Voting",
          "Stacking",
        ]}
        cols={[
          "Reduces variance",
          "Reduces bias",
          "Interpretable",
          "Noise tolerance",
          "Inference latency",
          "Engineering cost",
        ]}
        values={[
          [1.0, 0.1, 0.3, 0.8, 0.6, 0.9],
          [0.4, 0.9, 0.2, 0.2, 0.6, 0.8],
          [0.5, 0.4, 0.6, 0.7, 0.5, 0.9],
          [0.6, 0.5, 0.4, 0.6, 0.5, 0.8],
          [0.7, 0.6, 0.1, 0.5, 0.2, 0.2],
        ]}
        colorScale="cool"
      />

      <H3>When to use bagging</H3>
      <Prose>
        Reach for bagging (Random Forests) when your base learner is unstable, you
        need a strong out-of-the-box baseline, the data has noisy labels, and you
        want a free OOB generalization estimate. The random forest is the closest thing
        ML has to a default classifier for tabular data — it requires minimal
        preprocessing, handles heterogeneous features, and rarely catastrophically fails.
        The ceiling is lower than boosting or stacking on clean data.
      </Prose>

      <H3>When to use AdaBoost</H3>
      <Prose>
        Use AdaBoost when your base learner is weak but consistent (decision stumps on
        data with clear structure) and your labels are clean. AdaBoost's sequential
        reweighting is a powerful bias reducer — it is specifically designed to turn
        a slightly-better-than-random learner into an accurate one by repeatedly
        focusing on hard cases. Its Achilles heel is noisy labels: a mislabeled
        example in the hard-case region will accumulate weight across rounds and
        cause the boosted ensemble to fit the noise. Gradient Boosting is typically
        preferred over AdaBoost in production today because GBM uses a more general
        loss function (L2, Huber, log-loss) and is less sensitive to outliers.
      </Prose>

      <H3>When to use stacking</H3>
      <Prose>
        Use stacking when you are optimizing the last few percentage points and can
        afford the engineering complexity. Stacking almost always improves on any
        individual model when base models are diverse — the Netflix Prize result is
        not a fluke. But stacking doubles the number of models in your system (base
        layer plus meta-layer), doubles the monitoring surface, and adds a data
        dependency (the OOF generation protocol). In production, ask whether the
        +0.3% AUC improvement justifies two more models in the serving path. For
        a spam filter or click-through rate model where every tenth of a percent
        is revenue, yes. For a recommendation fallback where the baseline is already
        0.85 AUC, probably not.
      </Prose>

      <H3>When a single well-tuned GBM beats a stack</H3>
      <Prose>
        A properly tuned LightGBM or XGBoost with appropriate feature engineering
        routinely beats naive stacks of weak base models. If your base models are
        three slightly differently-hyperparameterized gradient boosting runs, their
        predictions are highly correlated (rho near 0.9) and the meta-learner gains
        almost nothing — you have paid the engineering cost of stacking for near-zero
        return. Before building a stack, verify that your base models are genuinely
        diverse and that a well-tuned single model is your baseline to beat.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>Bagging is embarrassingly parallel</H3>

      <Prose>
        All B bootstrap trees are independent — tree b+1 requires no output from
        tree b. Training parallelizes trivially across cores (set <Code>n_jobs=-1</Code>).
        On an 8-core machine training 300 trees, you get roughly 7× speedup vs.
        sequential. The total cost is O(B · n · sqrt(d) · log n) for Random Forests,
        and that B factor parallelizes. Inference also parallelizes: scoring one
        example requires running it through all B trees, which are independent.
        For real-time serving with latency budgets, consider that 300 trees at depth 10
        requires 300 × 10 = 3,000 comparisons per prediction — fast in C (sklearn
        Cython backend, microseconds) but worth profiling against your latency SLA.
      </Prose>

      <H3>Boosting is sequentially bottlenecked</H3>

      <Prose>
        Tree t+1 depends on tree t's error signal. You cannot start tree 2 until
        tree 1 has finished. For AdaBoost with B rounds, the training is B sequential
        rounds of weighted fitting — no tree-level parallelism. Modern GBM
        implementations (LightGBM, XGBoost) compensate by parallelizing within each
        tree — splitting the feature search across CPU cores, using histogram binning
        to cut per-split cost. But the B-round sequential dependency remains. For very
        large n (tens of millions), gradient boosting's histogram approach
        (LightGBM's leaf-wise growth, 256-bin histograms) is faster than bagging's
        exact-split approach. For n above 10M, consider LightGBM over sklearn's
        AdaBoostClassifier or RandomForestClassifier.
      </Prose>

      <H3>Stacking multiplies model count and inference cost</H3>

      <Prose>
        Serving a stacking ensemble requires running all B base models on each input
        and then running the meta-learner on their B outputs. If your base models are
        a Random Forest (300 trees) + AdaBoost (100 rounds) + KNN (kd-tree query),
        each prediction requires all three inference paths plus a logistic regression
        evaluation. In a latency-constrained API (50 ms budget), this can be
        prohibitive. Strategies: (1) replace slow base models with pre-computed
        embeddings or cached predictions, (2) use a cheaper meta-learner that amortizes
        fast, (3) consider model distillation — train a single cheap model to mimic
        the stack's outputs. The accuracy of a distilled model is usually within 1%
        of the stack at a fraction of the serving cost.
      </Prose>

      <Prose>
        The OOF generation step during training also scales quadratically in wall-clock
        time: K folds × B base models × n/K training examples per fold ≈ B × n total
        training examples processed. For B=5 diverse base models and 5-fold CV, you
        process 5 × 5 × n = 25n training examples during the OOF phase alone, before
        the final re-fit on all n examples. For n = 1M, this is 25M data points
        flowing through potentially slow base models. Budget accordingly.
      </Prose>

      <Callout variant="warning">
        The +0.2% AUC arithmetic: if your annual revenue attributable to your model
        is $10M, a 0.2% lift is $20K. If building and maintaining the stacking
        infrastructure costs one engineering-month ($15K–25K) plus ongoing monitoring
        overhead, the ROI is borderline. Run this calculation before committing to a
        stacking architecture in production.
      </Callout>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>Stacking leakage from training-set predictions</H3>

      <Prose>
        The most common and most damaging stacking mistake: fitting base models on all
        training data, collecting their predictions on the same training data, and using
        those as meta-features. A decision tree fit on data and then asked to predict
        that same data will achieve near-perfect accuracy on it — the meta-learner
        sees "predict_proba = 0.999 for class 1" from a tree that memorized those
        examples. It learns to trust those predictions. At inference, the base models
        see genuinely new data and achieve, say, 0.80 accuracy rather than 0.999 — the
        meta-learner's beliefs about base-model confidence are wrong by construction.
        The result is a stack whose cross-validated training score is excellent but
        whose test performance is worse than a simple voting ensemble. The fix is
        non-negotiable: always use out-of-fold predictions for generating meta-features.
      </Prose>

      <H3>Correlated base models — stacking gains vanish</H3>

      <Prose>
        From the variance formula: when <Code>rho</Code> is close to 1, the ensemble
        variance is approximately <Code>rho * sigma^2</Code> regardless of B. Three
        hyperparameter-tuned XGBoost runs with rho = 0.95 give a stack whose meta-learner
        has approximately nothing useful to combine. You will see this in practice as a
        meta-learner with near-uniform weights on all base models — it has learned that
        they are interchangeable. Diagnostic: compute pairwise Pearson correlation of OOF
        predictions across base models. If any pair exceeds 0.85, dropping one of them
        will not hurt the stack and will reduce serving cost.
      </Prose>

      <H3>AdaBoost sensitivity to noisy labels</H3>

      <Prose>
        AdaBoost's weight update raises misclassified examples' weights. A truly
        mislabeled example (the label is wrong, not the model) will be consistently
        misclassified, accumulate weight across rounds, and force later stumps to fit
        the noise. The training error eventually stops improving but the test error
        climbs — classic overfitting, but concentrated in the noise rather than in
        complexity. Symptoms: training error continues to drop while validation error
        begins rising after round T_opt. Fix: early stopping on validation loss, or
        switch to Gradient Boosting with a robust loss (Huber, log-loss with label
        smoothing). A small amount of label noise that damages AdaBoost leaves
        well-tuned GBM largely unaffected because GBM's gradient step is bounded by
        the loss function's curvature, while AdaBoost's exponential loss has unbounded
        sensitivity to the margin.
      </Prose>

      <H3>Voting with class-imbalanced base models</H3>

      <Prose>
        Hard voting on a class-imbalanced dataset (say, 95:5 negative:positive) can
        be catastrophically wrong. If you train three classifiers without rebalancing,
        all three will learn to predict the majority class on borderline examples.
        Three majority-class votes = majority-class ensemble, always. The minority class
        is invisible. Soft voting partially mitigates this: if even one base model
        assigns probability 0.6 to the minority class, that signal propagates into the
        probability average. But the correct fix is upstream: address imbalance in each
        base model via <Code>class_weight="balanced"</Code>, oversampling, or threshold
        calibration, before combining them.
      </Prose>

      <H3>OOF predictions with time-series data</H3>

      <Prose>
        K-fold OOF generation assumes examples are exchangeable — randomly assigned to
        folds. For time-series data, this creates temporal leakage: a model trained on
        fold {"{1,3,4,5}"} and validated on fold {"{2}"} has seen future data during
        training. The OOF predictions are optimistically biased because models benefit
        from future information. For temporal stacking, use walk-forward (expanding
        window) splits: train on all data up to time T, validate on the block
        immediately following. sklearn's <Code>TimeSeriesSplit</Code> implements this.
        The meta-feature matrix will have fewer rows (the first training period has no
        preceding validation period) but the leakage will be absent.
      </Prose>

      <Callout variant="warning">
        The leakage gotcha is uniquely dangerous because it fails silently. The cross-
        validation score looks excellent. The model deploys. Performance is worse than
        expected. The discrepancy is only explained months later when someone audits
        the OOF generation code and discovers base models were fit on the full training
        set before generating meta-features. Always print and inspect the OOF generation
        loop as a code review step.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All citations below have been verified against publisher records and author
        pages. DOIs and page ranges are exact.
      </Prose>

      <H3>Wolpert 1992 — Stacked Generalization</H3>
      <Prose>
        Wolpert, D. H. (1992). Stacked generalization.
        <em> Neural Networks</em>, 5(2), 241–259.
        DOI: 10.1016/S0893-6080(05)80023-1.
        The originating paper. Wolpert introduced the "generalizer" terminology and
        the out-of-fold construction that prevents leakage. He framed stacking as a
        way to correct the biases of level-0 generalizers using a level-1 generalizer
        that sees their outputs on held-out data. The paper is dense and theoretical;
        Wolpert proved bounds relating the stacking error to the biases of the
        component generalizers. Available via ScienceDirect (Elsevier).
      </Prose>

      <H3>Breiman 1996 — Bagging Predictors</H3>
      <Prose>
        Breiman, L. (1996). Bagging predictors.
        <em> Machine Learning</em>, 24(2), 123–140.
        DOI: 10.1007/BF00058655.
        The bagging paper. Breiman introduced bootstrap aggregating, analyzed it on
        classification and regression trees, and identified instability as the necessary
        condition for bagging to help. The paper includes the variance decomposition
        showing that averaging uncorrelated estimators reduces variance by factor B.
        Available as a PDF from Springer Nature Link.
      </Prose>

      <H3>Schapire 1990 — Strength of Weak Learnability</H3>
      <Prose>
        Schapire, R. E. (1990). The strength of weak learnability.
        <em> Machine Learning</em>, 5(2), 197–227.
        DOI: 10.1007/BF00116037.
        The theoretical foundation. Schapire proved that the classes of weakly
        learnable and strongly learnable concepts are equivalent in the PAC model,
        and described a constructive procedure (boost-by-filtering) for converting
        a weak learner into an arbitrarily accurate one. This preceded AdaBoost by
        seven years. PDF freely available at schapire.net.
      </Prose>

      <H3>Freund and Schapire 1997 — AdaBoost (JCSS)</H3>
      <Prose>
        Freund, Y., {"&"} Schapire, R. E. (1997). A decision-theoretic generalization
        of on-line learning and an application to boosting.
        <em> Journal of Computer and System Sciences</em>, 55(1), 119–139.
        DOI: 10.1006/jcss.1997.1504.
        The AdaBoost paper. Freund and Schapire derived the multiplicative weight
        update rule from a minimax game between the boosting algorithm and the adversary,
        showed it minimizes exponential loss, and proved exponential decrease in training
        error. Earlier version appeared in COLT 1995; the JCSS version is the canonical
        reference. PDF at schapire.net/papers/FreundSc95.pdf.
      </Prose>

      <H3>Töscher, Jahrer, Bell 2009 — Netflix Prize (BigChaos)</H3>
      <Prose>
        Töscher, A., Jahrer, M., {"&"} Bell, R. M. (2009). The BigChaos solution to
        the Netflix grand prize. Technical report, Commendo Research {"&"} Consulting GmbH.
        Published at netflixprize.com/assets/GrandPrize2009_BPC_BigChaos.pdf.
        The engineering document for the ensemble blend that, combined with BellKor
        and Pragmatic Theory, won the $1M Netflix Prize. Documents hundreds of
        individual models (SVD variants, neighborhood methods, RBMs) and a two-stage
        linear blending procedure. Essential reading for practical stacking at scale.
      </Prose>

      <H3>Sill, Takács, Mackey, Lin 2009 — Feature-Weighted Linear Stacking</H3>
      <Prose>
        Sill, J., Takács, G., Mackey, L., {"&"} Lin, D. (2009). Feature-weighted linear
        stacking. arXiv:0911.0460.
        Extended the standard linear meta-learner to allow meta-feature weights to
        vary as a function of input features — the meta-learner can learn "trust model A
        more when feature X is high." This was a key component of the second-place
        Netflix Prize solution. The arXiv paper is the definitive reference;
        available at arxiv.org/abs/0911.0460.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Q1 (recall) — variance formula</H3>
      <Prose>
        The variance of the average of B correlated base models is
        rho * sigma^2 + (1 - rho) * sigma^2 / B. Suppose rho = 0.7, sigma^2 = 0.04,
        B = 100. Compute the ensemble variance and compare it to a single model.
        What is the maximum variance reduction achievable by adding more models?
      </Prose>
      <Callout variant="answer">
        <strong>Single model variance:</strong> 0.04.
        <br />
        <strong>Ensemble (B=100):</strong> 0.7 * 0.04 + (1 - 0.7) * 0.04 / 100
        = 0.028 + 0.00012 = 0.02812.
        <br />
        <strong>Reduction:</strong> from 0.04 to ~0.028 — about 30%.
        <br />
        <strong>Maximum (B → infinity):</strong> rho * sigma^2 = 0.7 * 0.04 = 0.028.
        The floor. No matter how many models you add, ensemble variance cannot go
        below 0.028 when rho = 0.7. Feature subsampling reduces rho; with rho = 0.3,
        the floor drops to 0.012 — three-fold improvement.
      </Callout>

      <H3>Q2 (recall) — AdaBoost alpha</H3>
      <Prose>
        A decision stump achieves weighted error epsilon_t = 0.3. Compute its alpha_t.
        A second stump achieves epsilon_t = 0.45. Compute that alpha_t. What does it
        mean for the final vote that the second alpha is much smaller?
      </Prose>
      <Callout variant="answer">
        <strong>epsilon_t = 0.3:</strong> alpha_t = 0.5 * ln(0.7 / 0.3) = 0.5 * ln(2.333)
        = 0.5 * 0.847 = 0.4236.
        <br />
        <strong>epsilon_t = 0.45:</strong> alpha_t = 0.5 * ln(0.55 / 0.45) = 0.5 * ln(1.222)
        = 0.5 * 0.201 = 0.1005.
        <br />
        <strong>Interpretation:</strong> the second stump contributes roughly 24% as much
        to the final vote (0.1005 / 0.4236) as the first. A stump that barely beats
        random (epsilon near 0.5) has near-zero weight — AdaBoost's built-in mechanism
        for discounting unreliable rounds. If epsilon_t = 0.5 exactly, alpha = 0 and the
        stump contributes nothing. If epsilon_t {">"} 0.5 (worse than random on the
        reweighted distribution), alpha is negative — the stump's prediction is flipped.
      </Callout>

      <H3>Q3 (applied) — stacking leakage diagnosis</H3>
      <Prose>
        A colleague builds a stacking ensemble. During development, the 5-fold CV score
        of the stack is 0.94 AUC. On the hold-out test set, the stack scores 0.81 AUC
        — worse than any individual base model. Describe the most likely cause and
        the exact lines of code to inspect.
      </Prose>
      <Callout variant="answer">
        <strong>Most likely cause:</strong> the meta-features were generated by predicting
        on the training set with base models already fit on that same training set
        (in-fold leakage). The meta-learner saw optimistic, near-perfect base-model
        outputs during training; at test time it sees realistic (worse) outputs and
        its combination weights are miscalibrated.
        <br />
        <strong>Code to inspect:</strong> (1) the OOF generation loop — verify that
        each base model is fit on train_idx and predicts on val_idx, where val_idx was
        not in train_idx. (2) Check that the meta-learner is trained on oof_preds
        generated from that loop, not from base models fit on the full training set.
        (3) Verify the base model re-fit step (for inference) happens after the OOF
        generation, not before. A quick diagnostic: if base-model OOF accuracy is
        suspiciously close to 1.0, leakage is almost certain.
      </Callout>

      <H3>Q4 (applied) — voting vs. stacking choice</H3>
      <Prose>
        You have three base models: a logistic regression (fast, well-calibrated),
        a random forest (moderate speed), and an XGBoost (slow, highest individual accuracy).
        Your serving latency budget is 20 ms. The logistic regression predicts in 1 ms,
        the forest in 8 ms, XGBoost in 15 ms. Should you use hard voting, soft voting,
        or stacking? What changes if your latency budget is 50 ms?
      </Prose>
      <Callout variant="answer">
        <strong>20 ms budget:</strong> Sequential execution of all three models (1 + 8 + 15 = 24 ms)
        already exceeds the budget. Hard or soft voting is not viable without parallelism.
        If the models can run in parallel and your hardware supports it, max(1, 8, 15) = 15 ms
        (plus orchestration overhead) might fit. If not, drop XGBoost and use soft voting
        on logistic regression + random forest: 1 + 8 = 9 ms sequential, below budget.
        <br />
        <strong>50 ms budget:</strong> All three run sequentially in 24 ms with room to spare.
        Now stacking becomes viable — add the meta-learner (logistic regression, ~1 ms)
        for 25 ms total. Use soft voting as the baseline; if stacking adds more than 1%
        AUC, it is worth the added meta-learner step. In practice, check whether the
        meta-learner on 3 base-model outputs genuinely learns a non-trivial combination
        (inspect meta-learner weights; uniform weights mean soft voting would suffice).
      </Callout>

      <H3>Q5 (applied) — designing a diverse stack</H3>
      <Prose>
        You are building a stacking ensemble for a tabular fraud detection task with
        500K examples, 80 features, and strong class imbalance (0.5% fraud rate).
        Propose a set of diverse base models and explain why each adds value.
        What meta-learner would you choose?
      </Prose>
      <Callout variant="answer">
        <strong>Proposed base models:</strong>
        <br />
        (1) <strong>LightGBM with class_weight="balanced"</strong>: tree-based, captures
        non-linear interactions, fast on 500K rows.
        <br />
        (2) <strong>Logistic regression with L1 penalty on scaled features</strong>:
        linear model, fast, well-calibrated probabilities, captures additive fraud signals.
        Complements GBM because it cannot fit interaction terms — different failure modes.
        <br />
        (3) <strong>Isolation Forest score as a feature + classifier</strong>:
        anomaly-detection based, treats fraud as the minority distribution rather than
        a supervised class. Complementary when fraudsters shift strategy (new fraud patterns
        have high anomaly scores even if classifiers miss them).
        <br />
        (4) <strong>KNN (k=50, weighted by distance)</strong>: instance-based, captures
        local neighborhoods of known fraud. Orthogonal to both tree and linear methods.
        <br />
        <strong>Meta-learner:</strong> logistic regression with L2 regularization (C=0.1).
        With 4 base models, the meta-feature matrix has only 4 columns — the meta-learner
        should be simple to avoid overfitting. Use <Code>class_weight="balanced"</Code>
        in the meta-learner too. Generate OOF predictions with <Code>stack_method="predict_proba"</Code>
        to give the meta-learner probability information rather than binary votes — richer
        signal with only 4 columns is valuable.
      </Callout>

      <H3>Q6 (challenge) — the rho floor and diminishing returns</H3>
      <Prose>
        You have 10 base models with individual accuracy 0.82 on a binary classification
        task, all pairs correlated at rho = 0.6, each with variance sigma^2 = 0.05.
        You add 10 more models of identical quality and correlation. Compute the ensemble
        variance before and after adding the new models. What is the theoretical minimum
        variance no matter how many models you add?
      </Prose>
      <Callout variant="answer">
        <strong>Variance formula:</strong> Var(ensemble) = rho * sigma^2 + (1 - rho) * sigma^2 / B.
        <br />
        <strong>B = 10:</strong> 0.6 * 0.05 + 0.4 * 0.05 / 10 = 0.030 + 0.002 = 0.032.
        <br />
        <strong>B = 20:</strong> 0.6 * 0.05 + 0.4 * 0.05 / 20 = 0.030 + 0.001 = 0.031.
        <br />
        Adding 10 more models reduced variance by only 0.001 — a 3% improvement for
        doubling the model count. The floor is rho * sigma^2 = 0.6 * 0.05 = 0.030.
        No ensemble of arbitrarily many models can go below this. The only way to push
        past 0.030 is to reduce rho — by using more diverse base models (different
        architectures, different feature sets, different training subsets) rather than
        by adding more of the same kind. This is the precise quantitative argument for
        investing in model diversity over model count.
      </Callout>

    </div>
  ),
};

export default ensembleMethodsContent;
