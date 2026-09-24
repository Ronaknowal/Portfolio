"""Generate complete executed examples. Source strings are formatted without AST changes."""
from pathlib import Path
import ast
import contextlib
import io
import json
import platform
import warnings

import black
import numpy as np
import scipy
import sklearn

examples = []


def example(identifier, title, question, code):
    original_ast = ast.dump(ast.parse(code), include_attributes=False)
    formatted = black.format_str(code, mode=black.FileMode())
    assert original_ast == ast.dump(ast.parse(formatted), include_attributes=False)
    output = io.StringIO()
    with warnings.catch_warnings(record=True) as captured, contextlib.redirect_stdout(output):
        warnings.simplefilter("always")
        exec(compile(formatted, identifier + ".py", "exec"), {"__name__": "__main__"})
    if captured:
        raise AssertionError([(str(item.message), identifier) for item in captured])
    examples.append({
        "id": identifier, "title": title, "question": question,
        "code": formatted, "expected": output.getvalue().rstrip(),
    })


example("normalized-scores", "Normalize scores without losing their scale",
        "If both exponentiated scores round to zero, does the evidence say that the classes are equally likely?",
r'''
import math
from fractions import Fraction as F


def log_probabilities(scores):
    if not scores or any(math.isnan(s) or s == math.inf for s in scores):
        raise ValueError("Use finite scores or negative infinity")
    maximum = max(scores)
    if maximum == -math.inf:
        raise ValueError("Every class assigns zero likelihood: posterior undefined")
    shifted = [s - maximum for s in scores]
    if any(s != -math.inf and not math.isfinite(s - maximum) for s in scores):
        raise ValueError("Score difference exceeds arithmetic range")
    correction = math.log(math.fsum(math.exp(s) for s in shifted))
    return [s - correction for s in shifted]


joint = [F(1, 3) * F(1, 7), F(2, 3) * F(4, 11)]
print("free-token posterior [ham, spam]:", [str(p / sum(joint)) for p in joint])
scores = [-1001.0, -1000.0]
print("direct exponentials:", [math.exp(s) for s in scores])
print("normalized:", [round(math.exp(p), 6) for p in log_probabilities(scores)])
print("one possible class:", [math.exp(p) for p in log_probabilities([-math.inf, -3.0])])
try:
    log_probabilities([-math.inf, -math.inf])
except ValueError as error:
    print("zero-evidence response:", error)
''')

count_class = r'''
import numpy as np
from scipy.special import logsumexp


class CountNaiveBayes:
    """Integer-count plug-in MNB; bounded educational input, not a full sklearn estimator."""

    def __init__(self, alpha=1.0):
        if not np.isfinite(alpha) or not 0 <= alpha <= 100:
            raise ValueError("Use finite alpha from 0 to 100")
        self.alpha = float(alpha)

    @staticmethod
    def _counts(X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or not X.shape[0] or not X.shape[1]:
            raise ValueError("Use a nonempty sample-by-feature matrix")
        if not np.isfinite(X).all() or (X < 0).any() or (X > 1e6).any():
            raise ValueError("Counts must be finite and between 0 and 1e6")
        if not np.equal(X, np.floor(X)).all():
            raise ValueError("This event-model implementation uses integer counts")
        return X

    def fit(self, X, y):
        X = self._counts(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("One label is required per row")
        self.classes_, inverse = np.unique(y, return_inverse=True)
        self.class_count_ = np.bincount(inverse).astype(float)
        self.feature_count_ = np.zeros((len(self.classes_), X.shape[1]))
        np.add.at(self.feature_count_, inverse, X)
        smoothed = self.feature_count_ + self.alpha
        total = smoothed.sum(axis=1, keepdims=True)
        if (total == 0).any() or not np.isfinite(total).all():
            raise ValueError("A class has no token mass; use positive smoothing")
        self.theta_ = smoothed / total
        with np.errstate(divide="ignore"):
            self.feature_log_prob_ = np.log(self.theta_)
        self.class_log_prior_ = np.log(self.class_count_ / len(X))
        return self

    def log_joint(self, X):
        X = self._counts(X)
        if X.shape[1] != self.feature_count_.shape[1]:
            raise ValueError("Use the fitted vocabulary in the same column order")
        scores = []
        for row in X:
            active = row > 0  # Prevent the undefined floating-point operation 0 * -inf.
            scores.append(self.class_log_prior_ +
                          (self.feature_log_prob_[:, active] * row[active]).sum(axis=1))
        return np.asarray(scores)

    def predict_log_proba(self, X):
        scores = self.log_joint(X)
        normalizer = logsumexp(scores, axis=1, keepdims=True)
        if not np.isfinite(normalizer).all():
            raise ValueError("Every class has zero likelihood for at least one row")
        return scores - normalizer

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        return self.classes_[self.predict_log_proba(X).argmax(axis=1)]
'''
eight_word_data = r'''
vocabulary = ["free", "money", "win", "prize", "meeting", "agenda", "report", "quarter"]
X = np.array([
    [3, 2, 1, 1, 0, 0, 0, 0], [2, 3, 2, 0, 0, 0, 0, 0],
    [1, 1, 3, 2, 0, 0, 0, 0], [4, 2, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 3, 1, 1], [0, 0, 0, 0, 1, 1, 3, 2],
    [0, 1, 0, 0, 3, 2, 1, 0], [0, 0, 0, 0, 2, 1, 2, 3],
])
y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
'''
example("count-classifier", "Fit a complete count classifier on the retained eight-word corpus",
        "Why does the money column receive a nonzero ham count before any smoothing?",
        count_class + eight_word_data + r'''
model = CountNaiveBayes(alpha=1).fit(X, y)
query = np.array([[2, 1, 0, 0, 0, 0, 0, 0], [0] * 8])
print("class order:", model.classes_.tolist())
print("token totals:", model.feature_count_.sum(axis=1).astype(int).tolist())
print("class word counts:", model.feature_count_.astype(int).tolist())
print("log word probabilities:", model.feature_log_prob_.round(4).tolist())
print("query joint scores:", model.log_joint(query).round(6).tolist())
print("query probabilities:", model.predict_proba(query).round(6).tolist())
print("training accuracy:", float(np.mean(model.predict(X) == y)))
from sklearn.naive_bayes import MultinomialNB
reference = MultinomialNB(alpha=1).fit(X, y)
print("matches sklearn:", np.allclose(model.predict_proba(query), reference.predict_proba(query)))
''')

example("event-models", "Count repetitions, observe absences, and declare category support",
        "Does repeating free change the Bernoulli observation? What does a reserved category buy you?",
r'''
import numpy as np
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, MultinomialNB

X = np.array([[2, 1, 0, 0, 0], [1, 1, 1, 0, 0], [0, 0, 0, 1, 1]])
y = np.array([1, 1, 0])
queries = np.array([[1, 0, 0, 0, 0], [2, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
for model in [MultinomialNB(alpha=1), BernoulliNB(alpha=1, binarize=0.0)]:
    model.fit(X, y)
    print(type(model).__name__, "P(spam):", model.predict_proba(queries)[:, 1].round(6).tolist())

# Status: idle=0, busy=1, offline=2, other=3 (reserved by a fixed input schema).
# Link: wired=0, radio=1. Each column has its own category universe.
status_link = np.array([[0, 0], [0, 1], [1, 0], [2, 1], [2, 0], [1, 1]])
fault = np.array([0, 0, 0, 1, 1, 1])
categorical = CategoricalNB(alpha=1, min_categories=[4, 2]).fit(status_link, fault)
print("category counts:", [row.tolist() for row in categorical.category_count_])
print("offline/radio and other/wired:", categorical.predict_proba([[2, 1], [3, 0]]).round(6).tolist())
try:
    categorical.predict_proba([[4, 0]])
except IndexError:
    print("category 4: outside the declared/fitted support; reject or map by the fixed schema")
''')

example("count-law", "Separate ordered tokens, count vectors and integrated prediction",
        "Why can the multinomial coefficient be dropped for classification but not from an actual count probability?",
r'''
from fractions import Fraction as F
from math import factorial
from itertools import product


def multinomial_mass(counts, probabilities):
    if any(type(k) is not int or k < 0 for k in counts):
        raise ValueError("Use nonnegative integer counts")
    if len(counts) != len(probabilities) or sum(probabilities) != 1:
        raise ValueError("Use a probability for every category")
    coefficient = factorial(sum(counts))
    for k in counts:
        coefficient //= factorial(k)
    result = F(coefficient)
    for k, probability in zip(counts, probabilities):
        result *= probability ** k
    return result


def integrated_mass(counts, shapes):
    if len(counts) != len(shapes) or any(type(a) is not int or a <= 0 for a in shapes):
        raise ValueError("Use matching positive integer Dirichlet shapes")
    if any(type(k) is not int or k < 0 for k in counts):
        raise ValueError("Use nonnegative integer counts")
    numerator = factorial(sum(counts))
    denominator = 1
    for k, a in zip(counts, shapes):
        denominator *= factorial(k)
        for j in range(k):
            numerator *= a + j
    for j in range(sum(counts)):
        denominator *= sum(shapes) + j
    return F(numerator, denominator)


theta = [F(4, 5), F(1, 5)]
print("ordered AB:", theta[0] * theta[1])
print("one A and one B:", multinomial_mass([1, 1], theta))
print("three-token masses:", [str(multinomial_mass([k, 3-k], theta)) for k in range(4)])
print("sum:", sum(multinomial_mass([k, 3-k], theta) for k in range(4)))
for counts in ([2, 0], [1, 1]):
    print("counts", counts, "plug-in:", multinomial_mass(counts, theta),
          "integrated:", integrated_mass(counts, [4, 1]))
plugin = [multinomial_mass([2, 0], p) for p in (theta, theta[::-1])]
integrated = [integrated_mass([2, 0], a) for a in ([4, 1], [1, 4])]
print("equal-prior class0 posterior, plug-in:", plugin[0] / sum(plugin))
print("equal-prior class0 posterior, integrated:", integrated[0] / sum(integrated))
''')

example("gaussian-fit", "Estimate Gaussian evidence and normalize the resulting scores",
        "Does the 99.5% training result measure performance on unseen data?",
r'''
import numpy as np
from scipy.special import logsumexp


class DiagonalGaussianNB:
    """Absolute variance floor in the declared squared feature units."""
    def __init__(self, variance_floor=1e-9):
        if not np.isfinite(variance_floor) or variance_floor <= 0:
            raise ValueError("Use a positive finite variance floor")
        self.variance_floor = variance_floor

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or y.ndim != 1 or len(y) != len(X) or not len(X):
            raise ValueError("Use a nonempty matrix and one label per row")
        if not np.isfinite(X).all() or np.max(np.abs(X)) > 1e100:
            raise ValueError("Input exceeds the supported arithmetic range")
        self.classes_, counts = np.unique(y, return_counts=True)
        self.prior_ = counts / len(y)
        self.mean_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])
        self.variance_ = np.array([X[y == c].var(axis=0, ddof=0) for c in self.classes_]) + self.variance_floor
        return self

    def log_joint(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.mean_.shape[1] or not np.isfinite(X).all():
            raise ValueError("Use finite rows in the fitted feature schema")
        with np.errstate(over="ignore", invalid="ignore"):
            scores = np.log(self.prior_) - 0.5 * (
                np.log(2 * np.pi) + np.log(self.variance_)
                + (X[:, None, :] - self.mean_) ** 2 / self.variance_
            ).sum(axis=2)
        if not np.isfinite(scores).all():
            raise ValueError("Gaussian scores exceed the supported arithmetic range")
        return scores

    def predict_log_proba(self, X):
        scores = self.log_joint(X)
        return scores - logsumexp(scores, axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_log_proba(X).argmax(axis=1)]


# Preserve the original lesson's RandomState draw sequence.
rng = np.random.RandomState(42)
X = np.vstack([rng.randn(100, 2) + [-2, -2], rng.randn(100, 2) + [2, 2]])
y = np.repeat([0, 1], 100)
model = DiagonalGaussianNB().fit(X, y)
print("priors:", model.prior_.tolist())
print("means:", model.mean_.round(4).tolist())
print("variances:", model.variance_.round(4).tolist())
print("training accuracy:", format(np.mean(model.predict(X) == y), ".4f"))
print("P(class | origin):", np.exp(model.predict_log_proba([[0, 0]])).round(6).tolist())
print("normalization:", np.exp(model.predict_log_proba([[0, 0]])).sum(axis=1).round(12).tolist())
''')

example("gaussian-boundaries", "Check an actual boundary and fit a separate comparison",
        "Will fitting a logistic model force its coefficients to equal the known Gaussian model's boundary?",
r'''
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

center = np.array([-10/3, -10/3])
radius = np.sqrt(200/9 - (6-np.log(4)) / .75)
angles = np.linspace(0, 2*np.pi, 101)
boundary = center + radius * np.column_stack([np.cos(angles), np.sin(angles)])
difference = (multivariate_normal.logpdf(boundary, [2, 2], 2*np.eye(2))
              - multivariate_normal.logpdf(boundary, [-2, -2], .5*np.eye(2)))
print("analytic circle center/radius:", center.round(6).tolist(), round(radius, 6))
print("equal-score residual below 1e-12:", bool(np.max(np.abs(difference)) < 1e-12))

# Retained original 500-row fixture; one held-out split, not a benchmark.
X, y = make_classification(n_samples=500, n_features=10, n_informative=5,
                          n_redundant=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
print("test rows:", len(y_test), "majority baseline:", round(np.mean(y_test == np.bincount(y_train).argmax()), 4))
for model in [GaussianNB(), GaussianNB(var_smoothing=1e-8),
              LogisticRegression(C=1.0, max_iter=2000)]:
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    print(type(model).__name__, "accuracy", format(accuracy_score(y_test, prediction), ".4f"),
          "log loss", format(log_loss(y_test, model.predict_proba(X_test)), ".6f"))
''')

example("dependence", "Calculate the actual risk of counting copies as fresh evidence",
        "At two copies the model ties; at three it changes class. What actually changed in the observation?",
r'''
from fractions import Fraction as F
from itertools import product

prior = F(1, 5)
positive_rates = [F(2, 5), F(4, 5)]  # normal, fault
for copies in [1, 2, 3, 5]:
    correct = F(0)
    brier = F(0)
    probabilities = []
    for positive in [True, False]:
        likelihoods = positive_rates if positive else [1-p for p in positive_rates]
        joint = [(1-prior) * likelihoods[0], prior * likelihoods[1]]
        fake_joint = [(1-prior) * likelihoods[0] ** copies, prior * likelihoods[1] ** copies]
        p = fake_joint[1] / sum(fake_joint)
        predicted = int(p > F(1, 2))  # Ties choose normal.
        correct += joint[predicted]
        brier += joint[0] * p**2 + joint[1] * (1-p)**2
        probabilities.append(str(p))
    print("copies", copies, "P(fault | +/-)", probabilities,
          "actual accuracy", str(correct), "Brier", format(float(brier), ".6f"))
print("true P(fault | +):", prior * positive_rates[1] /
      ((1-prior)*positive_rates[0] + prior*positive_rates[1]))

rows = [(a, b, int(a != b)) for a, b in product([0, 1], repeat=2)]
for label in [0, 1]:
    selected = [(a, b) for a, b, y in rows if y == label]
    print("XOR class", label, "feature-positive rates:",
          [str(F(sum(row[j] for row in selected), len(selected))) for j in [0, 1]])
print("XOR NB has identical marginal models; the pair still determines the class.")
''')

example("complement", "Pool the other classes and verify the sign of the decision",
        "Does a larger negative log-complement score mean the document matches or contradicts that complement?",
r'''
import numpy as np
from scipy.special import logsumexp
from sklearn.naive_bayes import ComplementNB

# Three routing classes, one aggregate training document per class.
X = np.array([[8, 2, 0], [1, 7, 2], [0, 2, 8]])
y = np.array([0, 1, 2])
query = np.array([[2, 0, 1]])
complement_counts = X.sum(axis=0) - X
theta = (complement_counts + 1) / (complement_counts.sum(axis=1, keepdims=True) + 3)
print("complement counts:", complement_counts.tolist())
print("complement probabilities:", theta.round(6).tolist())
for normalize in [False, True]:
    weights = -np.log(theta)
    if normalize:
        weights /= weights.sum(axis=1, keepdims=True)
    scores = query @ weights.T
    model = ComplementNB(alpha=1, norm=normalize).fit(X, y)
    print("norm", normalize, "scores:", scores.round(6).tolist(),
          "class:", model.predict(query).tolist())
    print("weights match:", bool(np.allclose(weights, model.feature_log_prob_)))
    print("softmax matches:", bool(np.allclose(
        np.exp(scores-logsumexp(scores, axis=1, keepdims=True)), model.predict_proba(query))))
print("These normalized scores are not a fitted generative P(class | document).")
''')

text_data = r'''
spam = [
    "free money win prize cash", "win cash prize free money",
    "claim your free prize now", "money back guarantee free offer",
    "free credit score win now", "earn money fast free win",
    "click here free offer cash", "big cash prize free today",
    "free gift claim now win", "lottery winner claim free prize",
]
ham = [
    "team meeting agenda tomorrow", "quarterly report please review",
    "project update attached report", "schedule review meeting please",
    "budget review report quarterly", "agenda for team standup",
    "report due next quarter", "project deadline meeting review",
    "please review attached agenda", "team offsite agenda tomorrow",
]
texts = spam + ham
labels = np.array([1] * 10 + [0] * 10)
'''
example("text-pipeline", "Fit vocabulary only where training is allowed",
        "An unseen test word is ignored by this fitted vectorizer. How does that differ from a zero word likelihood inside its vocabulary?",
r'''
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
''' + text_data + r'''
train_text, test_text, y_train, y_test = train_test_split(
    texts, labels, test_size=.3, random_state=7, stratify=labels)
inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=19)
search = GridSearchCV(
    Pipeline([("words", CountVectorizer()), ("nb", MultinomialNB())]),
    {"nb__alpha": [0.5, 1.0, 2.0]}, scoring="neg_log_loss", cv=inner, refit=True)
search.fit(train_text, y_train)
best = search.best_estimator_
print("training/test rows:", len(y_train), len(y_test))
print("selected alpha using training folds:", search.best_params_["nb__alpha"])
print("final training-only vocabulary:", len(best.named_steps["words"].vocabulary_))
baseline = DummyClassifier(strategy="prior").fit(np.zeros((len(y_train), 1)), y_train)
for name, model in [
    ("prior baseline", baseline),
    ("selected MNB", best),
    ("fixed Bernoulli", Pipeline([("words", CountVectorizer()), ("nb", BernoulliNB())]).fit(train_text, y_train)),
    ("fixed Complement", Pipeline([("words", CountVectorizer()), ("nb", ComplementNB())]).fit(train_text, y_train)),
]:
    inputs = np.zeros((len(y_test), 1)) if name == "prior baseline" else test_text
    print(name, "accuracy", format(accuracy_score(y_test, model.predict(inputs)), ".4f"),
          "log loss", format(log_loss(y_test, model.predict_proba(inputs)), ".6f"))
print("selected MNB confusion [true rows/predicted columns, ham/spam]:",
      confusion_matrix(y_test, best.predict(test_text), labels=[0, 1]).tolist())
probe = ["free money win prize", "zephyronic qxz", ""]
print("probe nonzero feature counts:", best.named_steps["words"].transform(probe).getnnz(axis=1).tolist())
print("probe probabilities [ham, spam]:", best.predict_proba(probe).round(6).tolist())
print("A six-message test is a workflow check, not evidence for deployment.")
''')

example("streaming", "Consume the remainder and compare sufficient statistics",
        "Does updating in batches need to revisit earlier rows, and does accumulating old counts forget them?",
r'''
import numpy as np
from sklearn.naive_bayes import MultinomialNB
''' + eight_word_data + r'''
# Fourteen labeled rows; every batch uses the same fixed eight-column vocabulary.
X = np.concatenate([X, X[:6]], axis=0)
y = np.concatenate([y, y[:6]])
online = MultinomialNB(alpha=1)
batch_sizes = []
for start in range(0, len(y), 4):
    stop = min(start + 4, len(y))
    batch_sizes.append(stop-start)
    online.partial_fit(X[start:stop], y[start:stop], classes=np.array([0, 1]))
batch = MultinomialNB(alpha=1).fit(X, y)
print("batch sizes:", batch_sizes)
print("class counts:", online.class_count_.astype(int).tolist())
print("same sufficient counts:", bool(np.array_equal(online.feature_count_, batch.feature_count_)))
print("same probabilities:", bool(np.allclose(online.predict_proba(X), batch.predict_proba(X))))
print("floor-based loop would consume:", (len(y)//4)*4, "of", len(y), "rows")
print("This model accumulates all fourteen rows; it does not apply a moving window.")
''')

example("calibrated-report", "Separate base fitting, probability calibration and the final report",
        "Can calibration change an argmax, and does one lower Brier score establish calibration for every subgroup?",
r'''
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score


def draw_cases(rng, count):
    y = rng.binomial(1, .3, size=count)
    reading = 1.5*y + rng.normal(size=count)
    return np.column_stack([reading, reading]), y  # Exact copies, not two measurements.


def report(name, y, p):
    print(name, "accuracy", format(accuracy_score(y, p > .5), ".4f"),
          "Brier", format(brier_score_loss(y, p), ".6f"),
          "log loss", format(log_loss(y, p), ".6f"))


rng = np.random.default_rng(2026)
X_fit, y_fit = draw_cases(rng, 800)
X_cal, y_cal = draw_cases(rng, 800)
X_test, y_test = draw_cases(rng, 1200)
base = GaussianNB().fit(X_fit, y_fit)
calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
calibrated.fit(X_cal, y_cal)  # No refit of frozen base; these rows fit the calibrator.
raw = base.predict_proba(X_test)[:, 1]
adjusted = calibrated.predict_proba(X_test)[:, 1]
report("prior-only", y_test, np.full(len(y_test), y_fit.mean()))
report("raw duplicate GNB", y_test, raw)
report("sigmoid calibrated", y_test, adjusted)
print("changed .5 decisions:", int(np.sum((raw > .5) != (adjusted > .5))))
observed, predicted = calibration_curve(y_test, adjusted, n_bins=5, strategy="uniform")
print("mean prediction by nonempty bin:", predicted.round(4).tolist())
print("observed positive fractions:", observed.round(4).tolist())
counts, _ = np.histogram(adjusted, bins=np.linspace(0, 1, 6))
print("bin sizes:", counts.tolist())
false_positive_cost, false_negative_cost = 4, 1
threshold = false_positive_cost / (false_positive_cost + false_negative_cost)
actions = adjusted > threshold  # Ties choose class0.
realized = np.where(actions & (y_test == 0), false_positive_cost,
                    np.where(~actions & (y_test == 1), false_negative_cost, 0))
print("declared-cost threshold:", threshold, "realized test cost per case:", format(realized.mean(), ".6f"))
print("Fixed synthetic law and seed; finite test metrics do not guarantee future calibration.")
''')

target = Path("src/learn/data/naive-bayes-examples.js")
target.write_text("export const naiveBayesExamples = " + json.dumps(examples, indent=2, ensure_ascii=False) + ";\n", encoding="utf-8")
record = {"generatedAt": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
          "versions": {"python": platform.python_version(), "numpy": np.__version__,
                       "scipy": scipy.__version__, "sklearn": sklearn.__version__},
          "examples": [{"id": item["id"], "stdout": item["expected"]} for item in examples]}
Path("scratch/naive-bayes-generation").mkdir(exist_ok=True)
Path("scratch/naive-bayes-generation/results.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(json.dumps(record, indent=2))
