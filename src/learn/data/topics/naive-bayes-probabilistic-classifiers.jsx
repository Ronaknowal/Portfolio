import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const naiveBayesContent = {
  title: "Naive Bayes & Probabilistic Classifiers",
  readTime: "~35 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        In 1763, two years after Thomas Bayes died, his friend Richard Price sent an unpublished manuscript to the Royal Society in London. The paper — "An Essay towards solving a Problem in the Doctrine of Chances" — was read aloud on 23 December 1763 and published in <em>Philosophical Transactions</em>, volume 53, pages 370–418. Bayes had been working on a question about inverse probability: given that an event has been observed to happen a certain number of times, what can we say about the probability that governs it? His answer was the theorem that now carries his name. The problem he solved was narrow — a billiards-table thought experiment about uniform priors — but the machinery he built, Bayes' rule, is general.
      </Prose>

      <Prose>
        Fifty years later, working entirely independently, Pierre-Simon Laplace formulated the same rule in full generality in his 1814 <em>Essai philosophique sur les probabilités</em>. Laplace used it to estimate the probability that the sun would rise tomorrow (given it had risen every day for thousands of years), to study population statistics, and to correct measurement errors in astronomy. What we call Bayesian reasoning is, historically, Laplace's mature framework built on Bayes' foundational insight.
      </Prose>

      <Prose>
        The jump from probability theory to machine learning came in 1961. M. E. Maron, working at RAND Corporation, published "Automatic Indexing: An Experimental Inquiry" in the <em>Journal of the ACM</em> (volume 8, issue 3, pages 404–417). Maron's problem was document classification: given a collection of technical documents, assign each to a subject category automatically based on word occurrences. His method — computing the probability of each category given the words present, using Bayes' rule and the assumption that words occur independently — is exactly what we now call Multinomial Naive Bayes for text. The paper predates scikit-learn by half a century and precedes the term "Naive Bayes" itself.
      </Prose>

      <Prose>
        The practical importance of that independence assumption — naive because it is almost certainly wrong — became clear in the spam-filtering era. Sahami, Dumais, Heckerman, and Horvitz published "A Bayesian Approach to Filtering Junk E-Mail" at the AAAI Workshop on Learning for Text Categorization in 1998. They showed that a Naive Bayes classifier trained on a few hundred labeled emails substantially outperformed hand-crafted rule-based filters. The independence assumption was wrong — words in emails are correlated — but the argmax class prediction was consistently right. This empirical regularity became one of the most cited facts in applied ML: Naive Bayes works far better than its assumptions deserve.
      </Prose>

      <Prose>
        The reasons are concrete: Naive Bayes trains in a single pass through the data at <Code>O(n·d)</Code> cost, where <Code>n</Code> is the number of samples and <Code>d</Code> is the number of features. It updates incrementally — a new email arrives, you update word counts. It handles extremely high-dimensional sparse data (a vocabulary of 100,000 words is routine) without the numerical issues that plague discriminative classifiers. And it has no iterative optimization, so it never fails to converge. These properties made it the workhorse of early NLP and it remains competitive today for text, spam filtering, document routing, and any setting where labeled data is scarce and fast iteration matters.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        The core question Naive Bayes answers: given this input <Code>x</Code>, which class <Code>c</Code> was most likely to have generated it? This is a generative framing. Instead of learning a direct mapping from inputs to labels (as logistic regression does), Naive Bayes builds a model of how each class produces data, and then inverts that model using Bayes' rule to classify new inputs.
      </Prose>

      <Prose>
        Bayes' rule is the inversion formula. The probability of class <Code>c</Code> given observation <Code>x</Code> — the posterior — equals the likelihood of seeing <Code>x</Code> under class <Code>c</Code>, times the prior probability of that class, divided by the probability of seeing <Code>x</Code> at all:
      </Prose>

      <MathBlock>
        {"P(c \\mid x) = \\frac{P(x \\mid c) \\cdot P(c)}{P(x)}"}
      </MathBlock>

      <Prose>
        The denominator <Code>P(x)</Code> is the same for all classes, so for classification (comparing posteriors across classes) we can drop it and just maximize the numerator:
      </Prose>

      <MathBlock>
        {"\\hat{c} = \\arg\\max_c \\; P(c) \\cdot P(x \\mid c)"}
      </MathBlock>

      <Prose>
        This is the MAP (Maximum A Posteriori) classifier. The prior <Code>P(c)</Code> encodes our baseline belief about class frequency before seeing any data. The likelihood <Code>P(x | c)</Code> is the hard part: <Code>x</Code> is a vector of <Code>d</Code> features, so <Code>P(x | c)</Code> is a joint distribution over <Code>d</Code> variables. Estimating an arbitrary joint distribution requires exponentially many parameters in <Code>d</Code>.
      </Prose>

      <Prose>
        The naive assumption breaks the joint down: assume all features are conditionally independent given the class. Then the joint factorizes into a product of per-feature terms:
      </Prose>

      <MathBlock>
        {"P(x \\mid c) = \\prod_{j=1}^{d} P(x_j \\mid c)"}
      </MathBlock>

      <Prose>
        Each <Code>P(x_j | c)</Code> is now a simple univariate distribution — a Gaussian, a multinomial over counts, or a Bernoulli over presence/absence. The number of parameters to estimate is linear in <Code>d</Code>, not exponential. This is why Naive Bayes can handle vocabulary sizes in the hundreds of thousands without breaking.
      </Prose>

      <Prose>
        The intuition for why this wrong assumption still works: for classification we only need to get the argmax right, not the posterior probabilities themselves. Even if the factorized <Code>P(x | c)</Code> assigns wildly incorrect absolute probabilities (because features are correlated and we double-count evidence), the class with the higher true posterior usually still has the higher factorized posterior. The ranking is preserved even when the values are wrong. This fails when posteriors are very close together, when class imbalance is severe, or when you need calibrated probabilities rather than just a decision — all covered in Section 9.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 Bayes' rule and the MAP classifier</H3>

      <Prose>
        Let <Code>y ∈ {"{1, …, K}"}</Code> be the class label and <Code>x ∈ ℝᵈ</Code> be the feature vector. The MAP classifier selects the class with the highest posterior:
      </Prose>

      <MathBlock>
        {"\\hat{y} = \\arg\\max_{k} \\; P(y=k) \\cdot P(x \\mid y=k)"}
      </MathBlock>

      <Prose>
        Applying the naive independence assumption, <Code>P(x | y=k)</Code> factorizes as <Code>∏ⱼ P(xⱼ | y=k)</Code>. Working in log-space (to avoid numerical underflow from multiplying many small probabilities):
      </Prose>

      <MathBlock>
        {"\\hat{y} = \\arg\\max_{k} \\left[ \\log P(y=k) + \\sum_{j=1}^{d} \\log P(x_j \\mid y=k) \\right]"}
      </MathBlock>

      <Prose>
        This is the complete classification rule. The three standard Naive Bayes variants differ only in the distribution assumed for <Code>P(xⱼ | y=k)</Code>.
      </Prose>

      <H3>3.2 Gaussian Naive Bayes</H3>

      <Prose>
        For continuous features, assume each feature is normally distributed within each class. For class <Code>k</Code> and feature <Code>j</Code>, estimate the class-conditional mean <Code>μₖⱼ</Code> and variance <Code>σ²ₖⱼ</Code> from training data:
      </Prose>

      <MathBlock>
        {"P(x_j \\mid y=k) = \\frac{1}{\\sqrt{2\\pi\\sigma_{kj}^2}} \\exp\\!\\left(-\\frac{(x_j - \\mu_{kj})^2}{2\\sigma_{kj}^2}\\right)"}
      </MathBlock>

      <Prose>
        Training: one pass through the data to compute per-class per-feature sample means and variances. No iterative optimization. The decision boundary between two classes is not always a hyperplane — because each class has its own variance estimate, the boundary can be quadratic (an ellipse or parabola in 2D). This is visible in the Section 6 plot.
      </Prose>

      <H3>3.3 Multinomial Naive Bayes</H3>

      <Prose>
        For discrete count features — word counts in a document — assume each feature follows a multinomial distribution within each class. Let <Code>θₖⱼ = P(word j | class k)</Code> be the probability of word <Code>j</Code> given class <Code>k</Code>. The log-likelihood of document <Code>x</Code> (a word-count vector) under class <Code>k</Code> is:
      </Prose>

      <MathBlock>
        {"\\log P(x \\mid y=k) = \\sum_{j=1}^{d} x_j \\cdot \\log \\theta_{kj}"}
      </MathBlock>

      <Prose>
        Training: estimate <Code>θₖⱼ</Code> as the relative frequency of word <Code>j</Code> among all words in class-<Code>k</Code> documents. Raw counts collapse to an MLE:
      </Prose>

      <MathBlock>
        {"\\hat{\\theta}_{kj} = \\frac{\\sum_{i: y_i=k} x_{ij}}{\\sum_{j'} \\sum_{i: y_i=k} x_{ij'}}"}
      </MathBlock>

      <H3>3.4 Bernoulli Naive Bayes</H3>

      <Prose>
        For binary presence/absence features (did word <Code>j</Code> appear in the document at all?), assume each feature is Bernoulli. Let <Code>pₖⱼ = P(xⱼ=1 | y=k)</Code>. The log-likelihood is:
      </Prose>

      <MathBlock>
        {"\\log P(x \\mid y=k) = \\sum_{j=1}^{d} \\left[ x_j \\log p_{kj} + (1-x_j) \\log (1 - p_{kj}) \\right]"}
      </MathBlock>

      <Prose>
        Unlike Multinomial NB, Bernoulli NB explicitly penalizes the absence of words: if a word that typically appears in spam is absent, that is evidence against spam. Multinomial NB simply ignores absent words (zero counts contribute zero to the sum).
      </Prose>

      <H3>3.5 Laplace / Lidstone smoothing</H3>

      <Prose>
        If a word never appears in any training spam document, the MLE gives <Code>θ₁ⱼ = 0</Code>, and a single occurrence of that word in a test document drives <Code>log P(x | spam)</Code> to <Code>-∞</Code> — the model becomes completely certain the document is not spam, regardless of all other words. This is the zero-frequency problem. Laplace smoothing (add-1 smoothing) is the standard fix: add a pseudocount <Code>α</Code> to every feature count before normalizing:
      </Prose>

      <MathBlock>
        {"\\hat{\\theta}_{kj} = \\frac{\\left(\\sum_{i: y_i=k} x_{ij}\\right) + \\alpha}{\\left(\\sum_{j'} \\sum_{i: y_i=k} x_{ij'}\\right) + \\alpha d}"}
      </MathBlock>

      <Prose>
        With <Code>α = 1</Code> (Laplace smoothing) every word has at least one pseudocount. With <Code>α {"<"} 1</Code> (Lidstone smoothing) the prior is lighter. As <Code>α → 0</Code> the smoothed estimate approaches the MLE; as <Code>α → ∞</Code> it approaches the uniform distribution over words. In sklearn, this is the <Code>alpha</Code> hyperparameter.
      </Prose>

      <H3>3.6 Log-sum-exp for numerical stability</H3>

      <Prose>
        Multiplying many small probabilities underflows to zero in floating point. Working in log-space turns products into sums (Section 3.1 above). When you need to convert log-posteriors back to probabilities — for example, to compute <Code>predict_proba</Code> — use the log-sum-exp trick:
      </Prose>

      <MathBlock>
        {"\\log \\sum_k e^{a_k} = a^* + \\log \\sum_k e^{a_k - a^*}, \\quad a^* = \\max_k a_k"}
      </MathBlock>

      <Prose>
        Subtracting <Code>a*</Code> before exponentiating ensures at least one term equals 1 and none overflow. NumPy exposes this as <Code>scipy.special.logsumexp</Code>; sklearn uses it internally in all its Naive Bayes implementations.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        All code below was run with NumPy only. Outputs are verbatim terminal results.
      </Prose>

      <H3>4a. Gaussian Naive Bayes from scratch</H3>

      <CodeBlock language="python">
{`import numpy as np

class GaussianNB:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.priors_, self.means_, self.vars_ = {}, {}, {}
        for c in self.classes_:
            Xc = X[y == c]
            self.priors_[c] = len(Xc) / len(y)
            self.means_[c]  = Xc.mean(axis=0)
            self.vars_[c]   = Xc.var(axis=0) + 1e-9   # epsilon: numerical stability
        return self

    def _log_likelihood(self, x, c):
        mu, var = self.means_[c], self.vars_[c]
        # log of Gaussian PDF per feature, summed (independence assumption)
        return -0.5 * np.sum(np.log(2 * np.pi * var) + (x - mu)**2 / var)

    def predict_log_proba(self, X):
        log_posts = []
        for c in self.classes_:
            log_prior = np.log(self.priors_[c])
            ll = np.array([self._log_likelihood(x, c) for x in X])
            log_posts.append(log_prior + ll)
        return np.column_stack(log_posts)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]

np.random.seed(42)
n = 200
X0 = np.random.randn(n // 2, 2) + np.array([-2, -2])
X1 = np.random.randn(n // 2, 2) + np.array([ 2,  2])
X  = np.vstack([X0, X1])
y  = np.hstack([np.zeros(n // 2, dtype=int), np.ones(n // 2, dtype=int)])

gnb = GaussianNB()
gnb.fit(X, y)
preds = gnb.predict(X)
acc   = np.mean(preds == y)

print(f"class 0 prior: {gnb.priors_[0]:.4f}")
# Output: class 0 prior: 0.5000

print(f"class 1 prior: {gnb.priors_[1]:.4f}")
# Output: class 1 prior: 0.5000

print(f"class 0 mean:  {gnb.means_[0].round(4)}")
# Output: class 0 mean:  [-2.1156 -1.966 ]

print(f"class 1 mean:  {gnb.means_[1].round(4)}")
# Output: class 1 mean:  [2.1282 2.0435]

print(f"class 0 var:   {gnb.vars_[0].round(4)}")
# Output: class 0 var:   [0.7259 0.9877]

print(f"class 1 var:   {gnb.vars_[1].round(4)}")
# Output: class 1 var:   [1.0699 0.8651]

print(f"train accuracy: {acc:.4f}")
# Output: train accuracy: 0.9950`}
      </CodeBlock>

      <Prose>
        The model estimates separate means and variances for each class-feature pair. Class 0 clusters around <Code>(-2.1, -2.0)</Code>, class 1 around <Code>(2.1, 2.0)</Code>, consistent with the data generation process. The variances differ between classes (the Gaussian assumption allows distinct covariance per class, unlike LDA which forces a shared covariance). Training accuracy of 99.5% reflects that the two Gaussians barely overlap.
      </Prose>

      <H3>4b. Multinomial Naive Bayes with Laplace smoothing — spam toy example</H3>

      <CodeBlock language="python">
{`import numpy as np

class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha     # Laplace / Lidstone smoothing

    def fit(self, X, y):
        """
        X : (n_samples, n_features) integer word counts
        y : (n_samples,) integer class labels
        """
        self.classes_            = np.unique(y)
        self.log_priors_         = {}
        self.log_likelihoods_    = {}   # log P(feature j | class c)

        for c in self.classes_:
            Xc                        = X[y == c]
            self.log_priors_[c]       = np.log(len(Xc) / len(y))
            counts                    = Xc.sum(axis=0) + self.alpha
            self.log_likelihoods_[c]  = np.log(counts / counts.sum())
        return self

    def predict_log_proba(self, X):
        log_posts = []
        for c in self.classes_:
            lp = self.log_priors_[c] + X @ self.log_likelihoods_[c]
            log_posts.append(lp)
        return np.column_stack(log_posts)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]

# ---- Tiny spam corpus ----
# vocabulary: ['free', 'money', 'win', 'prize', 'meeting', 'agenda', 'report', 'quarter']
vocab = ['free', 'money', 'win', 'prize', 'meeting', 'agenda', 'report', 'quarter']

spam_docs = np.array([
    [3, 2, 1, 1, 0, 0, 0, 0],   # "free free free money money win prize"
    [2, 3, 2, 0, 0, 0, 0, 0],   # "free free money money money win win"
    [1, 1, 3, 2, 0, 0, 0, 0],   # "free money win win win prize prize"
    [4, 2, 0, 1, 0, 0, 0, 0],   # "free free free free money money prize"
])
ham_docs = np.array([
    [0, 0, 0, 0, 2, 3, 1, 1],   # "meeting meeting agenda agenda agenda report quarter"
    [0, 0, 0, 0, 1, 1, 3, 2],   # "meeting agenda report report report quarter quarter"
    [0, 1, 0, 0, 3, 2, 1, 0],   # "money meeting meeting meeting agenda agenda report"
    [0, 0, 0, 0, 2, 1, 2, 3],   # "meeting meeting agenda report report quarter quarter quarter"
])

X_corpus = np.vstack([spam_docs, ham_docs])
y_corpus  = np.array([1, 1, 1, 1, 0, 0, 0, 0])   # 1=spam, 0=ham

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_corpus, y_corpus)

print("log P(class=0 ham):  ", round(mnb.log_priors_[0], 4))
# Output: log P(class=0 ham):   -0.6931

print("log P(class=1 spam): ", round(mnb.log_priors_[1], 4))
# Output: log P(class=1 spam):  -0.6931

print("\nlog P(word | spam):")
for w, lp in zip(vocab, mnb.log_likelihoods_[1]):
    print(f"  {w:10s}: {lp:.4f}")
# Output:
#   free      : -1.1856
#   money     : -1.3863
#   win       : -1.6376
#   prize     : -1.9741
#   meeting   : -3.5835
#   agenda    : -3.5835
#   report    : -3.5835
#   quarter   : -3.5835

# Classify new document: "free free money" -> likely spam
test_doc = np.array([[2, 1, 0, 0, 0, 0, 0, 0]])
log_posts = mnb.predict_log_proba(test_doc)
pred      = mnb.predict(test_doc)

print(f"\nTest: 'free free money'")
print(f"log-posterior [ham, spam]: {log_posts[0].round(4)}")
# Output: log-posterior [ham, spam]: [-10.8328  -4.4507]

print(f"Predicted: {'spam' if pred[0] == 1 else 'ham'}")
# Output: Predicted: spam

preds_all = mnb.predict(X_corpus)
print(f"\nTraining accuracy: {np.mean(preds_all == y_corpus):.4f}")
# Output: Training accuracy: 1.0000`}
      </CodeBlock>

      <Prose>
        The log-posterior for "free free money" is <Code>-4.45</Code> for spam vs. <Code>-10.83</Code> for ham — a difference of 6.38 log-units, corresponding to a posterior spam probability of about 99.8%. The model has learned that "free" has log-probability <Code>-1.19</Code> under spam (appears frequently) vs. <Code>-3.61</Code> under ham (appears rarely). Each occurrence of "free" contributes a log-likelihood ratio of <Code>(-1.19) - (-3.61) = +2.42</Code> in favor of spam — a multiplicative factor of about 11.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        Scikit-learn provides four Naive Bayes variants. The API follows the standard sklearn estimator interface: <Code>fit</Code>, <Code>predict</Code>, <Code>predict_proba</Code>, <Code>partial_fit</Code> (for out-of-core / streaming). All outputs below are verbatim terminal results.
      </Prose>

      <H3>5a. GaussianNB on continuous data</H3>

      <CodeBlock language="python">
{`import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(
    n_samples=500, n_features=10, n_informative=5,
    n_redundant=2, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

gnb = GaussianNB()    # no hyperparameters — estimates mean/var from data
gnb.fit(X_train, y_train)

print(f"test accuracy:  {accuracy_score(y_test, gnb.predict(X_test)):.4f}")
# Output: test accuracy:  0.8500

print(f"class means (class 0, first 5 features): {gnb.theta_[0, :5].round(4)}")
# Output: class means (class 0, first 5 features): [ 1.4285 -0.9464 -0.0593  0.999  -0.0476]

print(f"class means (class 1, first 5 features): {gnb.theta_[1, :5].round(4)}")
# Output: class means (class 1, first 5 features): [-0.5509  0.116  -0.5267 -0.1276 -0.0307]

# var_smoothing: adds epsilon * max(var) to all variances — prevents zero-variance features
gnb_smoothed = GaussianNB(var_smoothing=1e-8)
gnb_smoothed.fit(X_train, y_train)
print(f"with var_smoothing=1e-8: {accuracy_score(y_test, gnb_smoothed.predict(X_test)):.4f}")
# Output: with var_smoothing=1e-8: 0.8500`}
      </CodeBlock>

      <H3>5b. MultinomialNB and ComplementNB on text data</H3>

      <CodeBlock language="python">
{`import numpy as np
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Hand-built spam/ham corpus (20 docs)
spam_msgs = [
    "free money win prize cash", "win cash prize free money",
    "claim your free prize now", "money back guarantee free offer",
    "free credit score win now", "earn money fast free win",
    "click here free offer cash", "big cash prize free today",
    "free gift claim now win",   "lottery winner claim free prize",
]
ham_msgs = [
    "team meeting agenda tomorrow", "quarterly report please review",
    "project update attached report", "schedule review meeting please",
    "budget review report quarterly", "agenda for team standup",
    "report due next quarter",       "project deadline meeting review",
    "please review attached agenda", "team offsite agenda tomorrow",
]

corpus = spam_msgs + ham_msgs
labels = np.array([1] * 10 + [0] * 10)

cv = CountVectorizer()
X_counts = cv.fit_transform(corpus).toarray()
print(f"vocab size: {len(cv.vocabulary_)}")
# Output: vocab size: 42

X_tr, X_te, y_tr, y_te = train_test_split(
    X_counts, labels, test_size=0.3, random_state=7
)

# --- MultinomialNB ---
mnb = MultinomialNB(alpha=1.0)   # alpha: Laplace smoothing strength
mnb.fit(X_tr, y_tr)
print(f"\nMultinomialNB (alpha=1.0) test accuracy: {accuracy_score(y_te, mnb.predict(X_te)):.4f}")
# Output: MultinomialNB (alpha=1.0) test accuracy: 1.0000

feature_names = cv.get_feature_names_out()
print("Top 5 words for ham  (class 0):",
      [feature_names[i] for i in np.argsort(mnb.feature_log_prob_[0])[-5:][::-1]])
# Output: Top 5 words for ham  (class 0): ['agenda', 'team', 'review', 'report', 'attached']

print("Top 5 words for spam (class 1):",
      [feature_names[i] for i in np.argsort(mnb.feature_log_prob_[1])[-5:][::-1]])
# Output: Top 5 words for spam (class 1): ['free', 'win', 'now', 'offer', 'cash']

# predict_proba on unseen message
test_msg = cv.transform(["free money win prize"]).toarray()
print(f"\npredict_proba('free money win prize'): {mnb.predict_proba(test_msg).round(4)}")
# Output: predict_proba('free money win prize'): [[0.0094 0.9906]]

# --- ComplementNB: better for imbalanced text ---
# Trains on the complement of each class — corrects for NB's tendency to
# overfit the majority class in imbalanced corpora.
cnb = ComplementNB(alpha=1.0)
cnb.fit(X_tr, y_tr)
print(f"\nComplementNB (alpha=1.0) test accuracy: {accuracy_score(y_te, cnb.predict(X_te)):.4f}")
# Output: ComplementNB (alpha=1.0) test accuracy: 1.0000

# --- BernoulliNB: word presence/absence ---
X_tr_bin = (X_tr > 0).astype(float)
X_te_bin = (X_te > 0).astype(float)
bnb = BernoulliNB(alpha=0.5)
bnb.fit(X_tr_bin, y_tr)
print(f"\nBernoulliNB (alpha=0.5, binarized) test accuracy: {accuracy_score(y_te, bnb.predict(X_te_bin)):.4f}")
# Output: BernoulliNB (alpha=0.5, binarized) test accuracy: 1.0000`}
      </CodeBlock>

      <Callout type="info" title="Which NB variant to use">
        MultinomialNB: word counts or TF-IDF features — the standard for text classification. ComplementNB (Rennie et al. 2003): use instead of Multinomial when classes are imbalanced; it trains each class's model on the complement set, correcting for NB's optimistic class-conditional assumptions. BernoulliNB: binary feature vectors (word present or absent) — better than Multinomial when document length varies enormously or when you care about the signal from absent words. GaussianNB: continuous features — sensor readings, embeddings, tabular data where features are plausibly Gaussian within each class.
      </Callout>

      <H3>5c. Online / streaming learning with partial_fit</H3>

      <CodeBlock language="python">
{`from sklearn.naive_bayes import MultinomialNB
import numpy as np

# MultinomialNB supports partial_fit for out-of-core / streaming updates.
# You must pass all possible classes in the first call.
mnb_stream = MultinomialNB(alpha=1.0)

# Simulate a stream: feed 5 mini-batches of 4 docs each
X_batches = [X_tr[i*4:(i+1)*4] for i in range(len(X_tr) // 4)]
y_batches  = [y_tr[i*4:(i+1)*4] for i in range(len(y_tr)  // 4)]

for batch_idx, (Xb, yb) in enumerate(zip(X_batches, y_batches)):
    if batch_idx == 0:
        mnb_stream.partial_fit(Xb, yb, classes=np.array([0, 1]))
    else:
        mnb_stream.partial_fit(Xb, yb)

acc_stream = accuracy_score(y_te, mnb_stream.predict(X_te))
print(f"streaming partial_fit accuracy: {acc_stream:.4f}")
# Output: streaming partial_fit accuracy: 1.0000`}
      </CodeBlock>

      <Prose>
        <Code>partial_fit</Code> is unique to Naive Bayes (and a handful of other sklearn classifiers). Because training reduces to accumulating sufficient statistics — word counts, class-conditional means and variances — each mini-batch's contribution is just an additive update to those statistics. There is no loss surface, no gradient, no learning rate. The model after 100 mini-batches is identical to the model trained on all 100 mini-batches at once.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6a. Decision boundary: Gaussian NB vs. logistic regression</H3>

      <Prose>
        Gaussian Naive Bayes allows each class to have its own covariance (encoded via per-class per-feature variances). When the two classes have different variances, the decision boundary becomes quadratic — an ellipse or parabola in 2D. Logistic regression is always linear. The plot below shows two clusters with deliberately unequal spreads: class 0 is tight (variance ≈ 0.5), class 1 is spread out (variance ≈ 2.0). GNB traces the quadratic boundary; logistic regression forces a straight line.
      </Prose>

      <Plot
        title="Gaussian NB (quadratic boundary) vs. logistic regression (linear boundary)"
        description="Class 0 centered at (-2,-2) with σ²≈0.5; class 1 centered at (2,2) with σ²≈2.0. GNB traces the quadratic boundary between unequal-variance Gaussians. Logistic regression is constrained to a hyperplane."
        xLabel="feature 1"
        yLabel="feature 2"
        series={[
          {
            label: "class 0 (tight spread)",
            type: "scatter",
            color: colors.gold,
            points: (() => {
              const pts = [];
              let s = 42;
              const rand = () => { s = (s * 1664525 + 1013904223) & 0xffffffff; return (s >>> 0) / 0xffffffff; };
              const randn = () => { const u = 1 - rand(), v = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
              for (let i = 0; i < 60; i++) pts.push([-2 + randn() * 0.7, -2 + randn() * 0.7]);
              return pts;
            })(),
          },
          {
            label: "class 1 (wide spread)",
            type: "scatter",
            color: colors.green,
            points: (() => {
              const pts = [];
              let s = 99;
              const rand = () => { s = (s * 1664525 + 1013904223) & 0xffffffff; return (s >>> 0) / 0xffffffff; };
              const randn = () => { const u = 1 - rand(), v = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
              for (let i = 0; i < 60; i++) pts.push([2 + randn() * 1.4, 2 + randn() * 1.4]);
              return pts;
            })(),
          },
          {
            label: "logistic regression boundary (linear)",
            type: "line",
            color: colors.textMuted,
            points: [[-4, -4], [5, 5]],
          },
          {
            label: "Gaussian NB boundary (quadratic — approximated)",
            type: "line",
            color: "#e06c75",
            points: [[-4, -1.8], [-1.8, 0], [0, 1.0], [1.0, 2.2], [2.2, 5]],
          },
        ]}
      />

      <H3>6b. MultinomialNB step trace — 3-document toy corpus</H3>

      <Prose>
        The following trace walks through MultinomialNB fitting and prediction on a minimal 5-word vocabulary. It shows exactly what numbers are being computed at each step.
      </Prose>

      <StepTrace
        label="MultinomialNB step-by-step — vocab: ['free', 'money', 'win', 'meeting', 'agenda']"
        steps={[
          {
            label: "Step 1 — Training corpus",
            render: () => (
              <Prose>
                3 documents, 2 classes. spam1: "free free money" → [2,1,0,0,0]. spam2: "free money win" → [1,1,1,0,0]. ham1: "meeting agenda" → [0,0,0,1,1]. Class priors: P(spam)=2/3=0.6667, P(ham)=1/3=0.3333.
              </Prose>
            ),
          },
          {
            label: "Step 2 — Estimate log P(word | spam) with α=1 smoothing",
            render: () => (
              <Prose>
                Spam word counts: free=3, money=2, win=1, meeting=0, agenda=0. Total raw = 6. With Laplace (α=1): counts = [4,3,2,1,1], total = 11. log θ: free=-1.0116, money=-1.2993, win=-1.7047, meeting=-2.3979, agenda=-2.3979.
              </Prose>
            ),
          },
          {
            label: "Step 3 — Estimate log P(word | ham) with α=1 smoothing",
            render: () => (
              <Prose>
                Ham word counts: free=0, money=0, win=0, meeting=1, agenda=1. Total raw = 2. With Laplace (α=1): counts = [1,1,1,2,2], total = 7. log θ: free=-1.9459, money=-1.9459, win=-1.9459, meeting=-1.2528, agenda=-1.2528. Note that unseen words (free, money, win) still get smoothed counts — zero-frequency problem solved.
              </Prose>
            ),
          },
          {
            label: "Step 4 — Classify test doc: 'free' → [1,0,0,0,0]",
            render: () => (
              <Prose>
                log P(spam | 'free') = log(0.6667) + 1×(-1.0116) = -0.4055 + (-1.0116) = -1.4171. log P(ham | 'free') = log(0.3333) + 1×(-1.9459) = -1.0986 + (-1.9459) = -3.0445. log-posterior ratio: -1.4171 - (-3.0445) = +1.6274. Predicted class: spam (higher log-posterior).
              </Prose>
            ),
          },
          {
            label: "Step 5 — Intuition: why 'free' strongly implies spam",
            render: () => (
              <Prose>
                The log-likelihood ratio for 'free' is (-1.0116) - (-1.9459) = +0.9343 per occurrence. Each time 'free' appears in a document, the model gets 0.9343 log-units of evidence for spam — a multiplicative factor of e^0.9343 ≈ 2.5. Two occurrences of 'free' gives a factor of 6.3 in favor of spam. This is exactly the Bayesian update: the posterior shifts proportionally to how much more likely the observation is under spam vs. ham.
              </Prose>
            ),
          },
        ]}
      />

      <H3>6c. Per-class word log-probability heatmap</H3>

      <Prose>
        The heatmap shows log P(word | class) for the 8-word toy corpus from Section 4b. Darker cells indicate higher log-probability (more characteristic of that class). "Free," "money," "win," and "prize" score high for spam; "meeting," "agenda," "report," and "quarter" score high for ham. The contrast is clean because the toy corpus has no word overlap — real corpora are noisier.
      </Prose>

      <Heatmap
        label="log P(word | class) — Multinomial NB, 8-word toy corpus"
        colLabels={["free", "money", "win", "prize", "meeting", "agenda", "report", "quarter"]}
        rowLabels={["ham", "spam"]}
        matrix={[
          [-3.61, -2.92, -3.61, -3.61, -1.41, -1.53, -1.53, -1.67],
          [-1.19, -1.39, -1.64, -1.97, -3.58, -3.58, -3.58, -3.58],
        ]}
        colorScale="gold"
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <StepTrace
        label="when to use Naive Bayes — and when not to"
        steps={[
          {
            label: "NB wins: text classification",
            render: () => (
              <Prose>
                Multinomial and Complement NB are competitive with SVMs and logistic regression on bag-of-words text at a fraction of the training cost. Vocabulary sizes of 100k+ are handled trivially — the model is just a matrix of word log-probabilities. For document routing, spam filtering, language detection, and short-text categorization, NB is often the correct first choice. Start here before reaching for fine-tuned transformers.
              </Prose>
            ),
          },
          {
            label: "NB wins: small labeled datasets",
            render: () => (
              <Prose>
                Naive Bayes has few parameters to estimate: one mean and one variance per class-feature pair for Gaussian NB, or one probability per class-word pair for Multinomial NB. With 50 training samples, logistic regression and neural networks overfit; GaussianNB fits stably because its generative structure imposes strong inductive bias. If your labeled set has fewer than ~500 samples, NB is worth serious consideration.
              </Prose>
            ),
          },
          {
            label: "NB wins: speed and streaming",
            render: () => (
              <Prose>
                Training is a single pass, O(n·d). No gradient descent, no convergence monitoring, no hyperparameter search for the optimizer. For streaming data where the model must update continuously (email arriving in real time, log-line classification), partial_fit is uniquely suited — each batch update is an additive count increment, not a full retrain. Latency-sensitive production systems benefit from NB's microsecond prediction time.
              </Prose>
            ),
          },
          {
            label: "NB wins: high-dimensional sparse features",
            render: () => (
              <Prose>
                Logistic regression with L2 regularization and gradient descent slows dramatically as feature dimensionality grows. NB parameter count scales linearly with d, estimation is closed-form, and the sparse structure of bag-of-words features is exploited naturally — only non-zero word counts contribute to the log-posterior sum.
              </Prose>
            ),
          },
          {
            label: "NB loses: strongly correlated features",
            render: () => (
              <Prose>
                When features are highly correlated — e.g., "excellent" and "great" both appear in positive reviews — Multinomial NB double-counts their evidence. The model becomes overconfident: the posterior is pushed further toward one class than the true posterior warrants. This shows up as extreme probabilities near 0 or 1 even when the true probability is 0.6. Use logistic regression or gradient boosting instead; they learn inter-feature relationships via shared weights or splits.
              </Prose>
            ),
          },
          {
            label: "NB loses: calibrated probability outputs required",
            render: () => (
              <Prose>
                If downstream decisions depend on accurate probability values (e.g., a medical triage system where P(disease)=0.3 vs. P(disease)=0.7 has different clinical implications), NB's probabilities are unreliable. The independence assumption systematically pushes posteriors toward 0 and 1. Use CalibratedClassifierCV from sklearn with method='isotonic' or method='sigmoid' to post-process NB outputs, or switch to logistic regression which is better calibrated by construction.
              </Prose>
            ),
          },
          {
            label: "NB loses: continuous features with non-Gaussian distributions",
            render: () => (
              <Prose>
                GaussianNB assumes unimodal Gaussian distributions per class per feature. Bimodal distributions, heavy tails, or skewed features will fool it. A feature with a bimodal within-class distribution will have its variance overestimated, reducing discriminative power. If your continuous features look non-Gaussian, consider transforming them (log transform for right-skewed features), or use a discriminative classifier that makes no distributional assumption.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>8.1 Training complexity: O(n·d) — genuinely fast</H3>

      <Prose>
        Training Multinomial NB on a corpus of <Code>n</Code> documents each with <Code>d</Code> distinct word types requires a single pass to accumulate word counts per class. This is <Code>O(n·d)</Code> time and <Code>O(K·d)</Code> memory, where <Code>K</Code> is the number of classes. Compare with logistic regression (gradient descent: <Code>O(n·d)</Code> per epoch, many epochs needed) or an SVM (kernel methods: <Code>O(n²)</Code> to <Code>O(n³)</Code> for training). Gaussian NB is similarly a single pass to compute per-class means and variances. There is no iterative loop to converge, no learning rate to tune, no loss function to monitor.
      </Prose>

      <Prose>
        For <Code>n = 1,000,000</Code> documents and <Code>d = 100,000</Code> word types, NB training completes in seconds. Logistic regression on the same corpus with a sparse solver (saga) takes minutes to hours depending on convergence. This is not a marginal difference — in production systems where models need to retrain frequently (daily or hourly), NB's training speed is a decisive operational advantage.
      </Prose>

      <H3>8.2 Streaming / online updates: O(1) per sample</H3>

      <Prose>
        NB's sufficient statistics are counts and counts of counts. When a new labeled document arrives, updating the model requires only incrementing the appropriate word counts and class count — constant time per document, regardless of the existing training set size. No other standard classifier supports this. Logistic regression requires re-running gradient descent; SVMs require re-solving the dual quadratic program; random forests require rebuilding trees.
      </Prose>

      <Prose>
        Sklearn's <Code>partial_fit</Code> implements this exactly. The internal state is just two arrays: <Code>feature_count_</Code> (K × d, class-conditional word counts) and <Code>class_count_</Code> (K, total documents per class). Each <Code>partial_fit</Code> call adds the new batch's counts to these arrays and recomputes log-probabilities. The model after seeing 1,000,000 emails via <Code>partial_fit</Code> in 1,000-email batches is identical to the model trained on all 1,000,000 emails at once.
      </Prose>

      <H3>8.3 What doesn't scale: the independence assumption</H3>

      <Prose>
        The computational bottleneck is not compute — it is model quality. As datasets grow larger, patterns of feature correlation become statistically detectable, and classifiers that model those correlations (logistic regression, gradient boosting) exploit them while NB ignores them. Empirically, on text classification benchmarks, NB's accuracy advantage over logistic regression diminishes as training set size increases beyond ~10,000 documents. On very large datasets, discriminative classifiers learn to use the full feature correlation structure and typically win.
      </Prose>

      <Prose>
        The practical implication: use NB for fast prototyping and as a baseline regardless of dataset size. If NB already achieves acceptable performance, ship it. If performance is insufficient and the dataset is large, switch to logistic regression or gradient boosting. NB's quality ceiling is set by the independence assumption; its compute ceiling is essentially unlimited.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>9.1 Zero-frequency problem</H3>

      <Prose>
        Any word that appears in test documents but never in training documents gets probability zero under MLE. One zero probability drives the entire log-posterior to <Code>-∞</Code>, making the model completely certain about the wrong class. This is not an edge case — it happens routinely with any test vocabulary that exceeds the training vocabulary. <strong>Always use Laplace or Lidstone smoothing</strong> (<Code>alpha {">"} 0</Code> in sklearn). The <Code>alpha=0</Code> default in some textbook implementations is dangerous in production.
      </Prose>

      <H3>9.2 Correlated features inflate confidence</H3>

      <Prose>
        When features are correlated, the independence assumption double-counts evidence. In a spam filter, "free offer" and "free gift" might appear together in 90% of spam emails. A document containing both "offer" and "gift" gets treated as if both provided independent evidence — the posterior gets pushed further toward spam than the true posterior. The predicted class is usually still correct, but the associated probability is miscalibrated toward extremes. This is the primary reason NB probabilities cannot be used for anything requiring accurate confidence scores without post-processing.
      </Prose>

      <H3>9.3 Poor probability calibration</H3>

      <Prose>
        Naive Bayes systematically produces overconfident posterior probabilities — values near 0 and 1 appear far more often than they should. A well-calibrated model that predicts P=0.8 for 100 events should see that event occur about 80 times. NB calibration curves typically bow toward the corners. The fix in sklearn is <Code>CalibratedClassifierCV</Code>:
      </Prose>

      <CodeBlock language="python">
{`from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB

# 'isotonic' calibration: nonparametric, good for many samples
# 'sigmoid' calibration: Platt scaling, good for small calibration sets
cal_mnb = CalibratedClassifierCV(MultinomialNB(alpha=1.0), method='isotonic', cv=5)
cal_mnb.fit(X_tr, y_tr)
# cal_mnb.predict_proba() now returns calibrated probabilities`}
      </CodeBlock>

      <Prose>
        Calibration is a post-processing step that maps the raw predicted probabilities to more accurate ones. It does not change the predicted class — just the associated confidence. Use it whenever downstream decisions depend on probability values rather than just the argmax.
      </Prose>

      <H3>9.4 Log-sum-exp and numerical stability</H3>

      <Prose>
        If you implement <Code>predict_proba</Code> by exponentiating log-posteriors and then normalizing, the exponentiation step can overflow (if log-posteriors are large and positive) or underflow to zero (if they are large and negative). Always use the log-sum-exp trick: subtract the maximum log-posterior before exponentiating. Sklearn handles this internally. In custom implementations, use <Code>scipy.special.logsumexp</Code> or implement it manually as shown in Section 3.6.
      </Prose>

      <H3>9.5 Class imbalance — use ComplementNB for text</H3>

      <Prose>
        Standard Multinomial NB is sensitive to class imbalance. The majority class has more training documents, so its word counts are larger, and the model tends to assign higher log-likelihoods to test documents regardless of their actual content. ComplementNB (Rennie et al., ICML 2003) directly addresses this: instead of estimating P(word | class), it estimates P(word | complement of class) and uses the complement parameters for classification. This corrects for the majority-class bias and consistently outperforms standard Multinomial NB on imbalanced corpora. Use <Code>sklearn.naive_bayes.ComplementNB</Code> as a drop-in replacement whenever class frequencies are unequal.
      </Prose>

      <H3>9.6 Gaussian NB on non-Gaussian features</H3>

      <Prose>
        GaussianNB's Gaussian assumption is violated by count data (always non-negative, often zero-inflated), binary data (Bernoulli, not Gaussian), and any heavy-tailed or multimodal distribution. Applying GaussianNB to raw word counts instead of continuous embeddings is a common mistake. Check feature distributions first: <Code>plt.hist(X[:, j])</Code> for a few features. If they look nothing like Gaussians, either transform them (log1p for counts) or switch to MultinomialNB / BernoulliNB as appropriate.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All citations below were WebSearch-verified for author, year, venue, and main contribution. Read them in this order to follow the intellectual lineage from probability theory to production NLP.
      </Prose>

      <StepTrace
        label="primary literature"
        steps={[
          {
            label: "Bayes 1763 — The foundational theorem",
            render: () => (
              <Prose>
                Bayes, T. (1763). "An Essay towards solving a Problem in the Doctrine of Chances." Communicated by Richard Price. <em>Philosophical Transactions of the Royal Society of London</em>, 53, 370–418. Published posthumously two years after Bayes' death. Price edited the manuscript and added his own appendix. The paper introduced the concept of inverse probability and the formula for updating beliefs given evidence — the theorem that now bears Bayes' name. The original is available via the Royal Society Publishing archive (DOI: 10.1098/rstl.1763.0053). Laplace's 1812 <em>Théorie analytique des probabilités</em> and his 1814 philosophical essay independently derived the same rule and extended it to continuous priors, giving the theorem its modern generality.
              </Prose>
            ),
          },
          {
            label: "Maron 1961 — Naive Bayes applied to text classification",
            render: () => (
              <Prose>
                Maron, M.E. (1961). "Automatic Indexing: An Experimental Inquiry." <em>Journal of the ACM</em>, 8(3), 404–417. DOI: 10.1145/321075.321084. Available via ACM Digital Library. Maron, working at RAND Corporation, applied Bayes' rule with a conditional independence assumption to automatically classify technical documents into subject categories based on word occurrences. This is the earliest published application of what we now call Naive Bayes to text. The paper is empirical: Maron built a working system on 968 scientific abstracts and reported classification accuracy, demonstrating that the naive independence assumption worked surprisingly well in practice. This practical finding — that the naive model performs well despite its wrong assumption — remained largely untheorized for decades.
              </Prose>
            ),
          },
          {
            label: "Sahami, Dumais, Heckerman, Horvitz 1998 — Bayesian spam filtering",
            render: () => (
              <Prose>
                Sahami, M., Dumais, S., Heckerman, D., and Horvitz, E. (1998). "A Bayesian Approach to Filtering Junk E-Mail." <em>AAAI Technical Report WS-98-05</em>, AAAI Workshop on Learning for Text Categorization, Madison, Wisconsin, July 27, 1998. Available at cdn.aaai.org/Workshops/1998/WS-98-05/WS98-05-009.pdf. This paper brought Naive Bayes to widespread attention in the ML community by applying it to the practical, high-stakes problem of email spam. The key contributions: framing spam detection as a decision-theoretic problem with asymmetric misclassification costs (false negatives — missing spam — are worse than false positives — blocking legitimate mail); showing that domain-specific features (message headers, specific phrases) combined with raw text produced substantially better filters than rule-based systems; and reporting that a Naive Bayes classifier trained on a few hundred labeled emails generalized well. The paper became one of the most cited works in applied ML from the 1990s.
              </Prose>
            ),
          },
          {
            label: "Rennie, Shih, Teevan, Karger 2003 — Complement NB and fixing NB's assumptions",
            render: () => (
              <Prose>
                Rennie, J.D.M., Shih, L., Teevan, J., and Karger, D.R. (2003). "Tackling the Poor Assumptions of Naive Bayes Text Classifiers." <em>Proceedings of the 20th International Conference on Machine Learning (ICML)</em>, Washington DC, August 21–24, 2003, 616–623. Available via ACM DL (10.5555/3041838.3041916) and Microsoft Research. This paper is the definitive analysis of <em>why</em> standard Multinomial NB underperforms on text and what to do about it. The authors identify three systematic problems: (1) the multinomial model treats document length as informative, biasing long-document classes; (2) training imbalance causes majority-class overconfidence; (3) correlated features cause probability overestimation. Their fix — Complement NB, which estimates each class's model from the complement set — addresses problems (2) and (3) and produces a fast algorithm competitive with SVMs on 20 Newsgroups. ComplementNB is now in sklearn and is the recommended NB variant for imbalanced text classification.
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
        Write the MAP classification rule for Naive Bayes. What is the "naive" assumption, and why is it called naive? What does working in log-space accomplish numerically?
      </Prose>
      <Callout type="answer" title="Answer 1">
        MAP rule: ŷ = argmax_k [log P(y=k) + Σⱼ log P(xⱼ | y=k)]. The naive assumption is conditional independence of features given the class: P(x | y=k) = ∏ⱼ P(xⱼ | y=k). It is called naive because features in real data are almost never independent — "free" and "money" in emails are correlated — but the assumption is made anyway for tractability. Log-space serves two purposes: (1) it turns a product of many small probabilities into a sum, preventing numerical underflow to 0.0 in floating point; (2) it is computationally equivalent (argmax of log = argmax of original) so classification is unchanged.
      </Callout>

      <H3>Exercise 2 (derivation)</H3>
      <Prose>
        Derive the Laplace-smoothed estimate of <Code>θₖⱼ</Code> (word probability) for Multinomial NB. What happens as <Code>α → 0</Code>? What happens as <Code>α → ∞</Code>? Why is <Code>α = 0</Code> dangerous in production?
      </Prose>
      <Callout type="answer" title="Answer 2">
        The Laplace-smoothed estimate is θ̂ₖⱼ = (Nₖⱼ + α) / (Nₖ + α·d), where Nₖⱼ = total count of word j in class k documents, Nₖ = total word count in class k documents, d = vocabulary size. As α → 0: the estimate approaches the MLE Nₖⱼ/Nₖ — no smoothing. As α → ∞: the estimate approaches 1/d — uniform distribution over words, ignoring data entirely. α = 0 is dangerous because any word absent from class k training documents gets θ̂ₖⱼ = 0, so log θ̂ₖⱼ = -∞. A single occurrence of that word in a test document makes log P(x | y=k) = -∞, and the model assigns posterior 0 to class k — garbage output due to one unseen word.
      </Callout>

      <H3>Exercise 3 (conceptual)</H3>
      <Prose>
        Explain why Naive Bayes decision boundaries can be quadratic while logistic regression boundaries are always linear. Give the specific condition under which Gaussian NB and logistic regression produce the same linear boundary.
      </Prose>
      <Callout type="answer" title="Answer 3">
        Logistic regression models P(y=1 | x) as σ(wᵀx + b) — a linear function of x passed through a sigmoid. The decision boundary (where P=0.5) is always a hyperplane: wᵀx + b = 0. Gaussian NB computes log P(y=k) + Σⱼ log N(xⱼ; μₖⱼ, σ²ₖⱼ) for each class. When the log-Gaussian terms are expanded, they include quadratic terms -(xⱼ - μₖⱼ)²/(2σ²ₖⱼ). When two classes have different variances, the quadratic terms from the two classes don't cancel when taking the difference, leaving a quadratic boundary. The boundary becomes linear exactly when all classes share the same per-feature variance (σ²₀ⱼ = σ²₁ⱼ for all j) — identical to the assumption made by Linear Discriminant Analysis (LDA). In that case, GaussianNB and LDA produce the same boundary, and with the further assumption of Gaussian features, the boundary is the same hyperplane as logistic regression (though the weight estimates differ).
      </Callout>

      <H3>Exercise 4 (debugging)</H3>
      <Prose>
        You train MultinomialNB on an email spam dataset and evaluate it. On the training set, accuracy is 99%. On a held-out test set from a different time period (6 months later), accuracy drops to 61% — barely above a majority-class baseline. List three possible causes and one fix for each.
      </Prose>
      <Callout type="answer" title="Answer 4">
        Cause 1: Vocabulary drift. New spam uses words absent from training data. NB assigns those words near-zero probability (or exactly zero without smoothing), making predictions unreliable. Fix: retrain the model periodically on recent data, and ensure alpha {">"} 0 for smoothing. Cause 2: Concept drift. The statistical relationship between words and spam has changed (spammers adapted). Fix: use a sliding window of recent labeled data rather than the full historical corpus, or use partial_fit to incrementally update the model as new labeled data arrives. Cause 3: Different preprocessing between train and test. If the tokenizer or vocabulary cutoff differs between the two time periods (e.g., max_features in CountVectorizer is applied separately), the feature spaces are misaligned. Fix: fit the vectorizer on training data only (cv.fit on train, cv.transform on both), never refit on test.
      </Callout>

      <H3>Exercise 5 (applied)</H3>
      <Prose>
        You have a 10-class document classification problem with severely imbalanced classes (class 0: 60% of data, class 9: 0.5% of data). You try MultinomialNB and find it never predicts class 9. What sklearn class addresses this directly, and what is the conceptual reason it helps?
      </Prose>
      <Callout type="answer" title="Answer 5">
        Use ComplementNB (sklearn.naive_bayes.ComplementNB). Conceptually: standard Multinomial NB estimates P(word | class k) from the word counts within class k. For the rare class (class 9), there are few training documents, so the word probability estimates are high-variance and the prior log P(class 9) is very negative (log(0.005) = -5.3). The model almost never overcomes this prior penalty. ComplementNB instead estimates P(word | NOT class k) for each class k — training the model on the complement set. For the rare class, the complement set is large (99.5% of data), giving stable estimates. Classification is then "which class's complement model least explains this document," which naturally gives the rare class more consideration. In practice, ComplementNB consistently outperforms standard MultinomialNB on imbalanced text corpora by 2–10 percentage points of accuracy.
      </Callout>

      <H3>Exercise 6 (synthesis)</H3>
      <Prose>
        A colleague says: "I need calibrated probabilities for my medical classification system, so Naive Bayes is ruled out." Is this correct? If not, describe a complete pipeline that uses Naive Bayes but produces well-calibrated outputs.
      </Prose>
      <Callout type="answer" title="Answer 6">
        Incorrect — or rather, premature. Naive Bayes <em>raw</em> probabilities are poorly calibrated, but this is fixable. A complete pipeline: (1) Train MultinomialNB (or GaussianNB) as normal. (2) Wrap it in CalibratedClassifierCV(MultinomialNB(alpha=1.0), method='isotonic', cv=5). The isotonic regression calibrator learns a monotone mapping from NB's raw scores to calibrated probabilities using cross-validated held-out predictions. (3) Evaluate calibration with a reliability diagram (sklearn.calibration.CalibrationDisplay) and Expected Calibration Error (ECE). After calibration, predicted probabilities of 0.8 should be correct about 80% of the time. The tradeoff: calibration requires a validation set (auto-handled by cv=5), and isotonic calibration needs at least a few hundred samples per class to fit stably. If the dataset is very small (under 200 samples total), Platt scaling (method='sigmoid') is more appropriate as it has fewer parameters.
      </Callout>

    </div>
  ),
};

export default naiveBayesContent;
