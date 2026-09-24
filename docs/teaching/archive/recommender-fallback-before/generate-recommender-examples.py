"""Write complete, independently runnable teaching programs and actual stdout."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import sys
import textwrap
import ast
import black

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "scratch/recommender-native"
DEST.mkdir(parents=True, exist_ok=True)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
examples = {}


def add(key, title, question, code):
    code = textwrap.dedent(code).strip() + "\n"
    formatted = black.format_str(code, mode=black.Mode(line_length=88))
    assert ast.dump(ast.parse(code), include_attributes=False) == ast.dump(ast.parse(formatted), include_attributes=False)
    code = formatted
    path = DEST / f"{key}.py"
    path.write_text(code, encoding="utf-8")
    run = subprocess.run([sys.executable, "-X", "utf8", "-I", str(path)],
                         capture_output=True, text=True, encoding="utf-8", timeout=120)
    if run.returncode:
        raise RuntimeError(f"{key}: {run.stdout}\n{run.stderr}")
    examples[key] = dict(title=title, question=question, code=code.rstrip(),
                         expected=run.stdout.rstrip(), language="python")


RATINGS = '''
import numpy as np

ratings = np.array([
    [5, 3, np.nan, 1, np.nan, 4, np.nan],
    [4, np.nan, 4, 1, 2, np.nan, 3],
    [np.nan, 3, np.nan, np.nan, 4, 3, np.nan],
    [1, np.nan, np.nan, 5, 4, np.nan, 2],
    [np.nan, 1, 5, 4, np.nan, np.nan, 3],
], dtype=float)
observed = np.isfinite(ratings)
'''
COUNTS = '''
import numpy as np

counts = np.array([
    [10, 3, 0, 1, 0, 15, 0],
    [8, 0, 5, 1, 2, 0, 4],
    [0, 6, 0, 0, 9, 12, 0],
    [1, 0, 0, 20, 7, 0, 3],
    [0, 1, 14, 8, 0, 0, 6],
], dtype=float)
'''

add("evidence", "Audit the evidence before fitting", "How many cells are actually observed? Would an observed rating of zero disappear?", RATINGS + '''
print("users, items:", ratings.shape)
print("observed, missing:", int(observed.sum()), int((~observed).sum()))
print("observed mean:", round(ratings[observed].mean(), 3))
print("user means:", np.round(np.nanmean(ratings, axis=1), 3).tolist())
print("user 0 unseen:", np.flatnonzero(~observed[0]).tolist())
changed = np.array([0., np.nan, 4.])
print("zero is observed:", np.isfinite(changed).tolist())
print("zero-inclusive mean:", changed[np.isfinite(changed)].mean())
# Missing values never enter an explicit squared-error objective.
predictions = np.full(ratings.shape, 3.)
print("constant-3 observed SSE:", float(((ratings[observed] - predictions[observed])**2).sum()))
''')

add("neighbors", "Expose the evidence behind an item neighbor", "For user 0's unseen item 2, what changes when a one-person overlap is rejected?", RATINGS + '''
def predict(user, item, centered=True, minimum_overlap=2, shrinkage=2., k=3, signed=False):
    if observed[user, item]:
        raise ValueError("This example predicts an unobserved pair.")
    user_means = np.nanmean(ratings, axis=1)
    fallback = float(user_means[user])
    evidence = []
    for other in np.flatnonzero(observed[user]):
        shared = observed[:, item] & observed[:, other]
        count = int(shared.sum())
        center = user_means[shared] if centered else 0.
        left, right = ratings[shared, item] - center, ratings[shared, other] - center
        norm_product = np.linalg.norm(left) * np.linalg.norm(right)
        cosine = float(left @ right / norm_product) if norm_product else 0.
        weight = cosine * count / (count + shrinkage) if count >= minimum_overlap and norm_product else 0.
        signal = ratings[user, other] - (fallback if centered else 0.)
        if weight > 0 or (signed and weight != 0):
            evidence.append((int(other), count, weight, float(signal)))
    evidence.sort(key=lambda row: (-abs(row[2]), row[0]))
    selected = evidence[:k]
    denominator = sum(abs(row[2]) for row in selected)
    value = (fallback if centered else 0.) + sum(row[2] * row[3] for row in selected) / denominator if denominator else fallback
    return value, selected

for centered, support, shrinkage in [(False, 1, 0.), (True, 2, 2.), (True, 4, 2.)]:
    value, selected = predict(0, 2, centered, support, shrinkage)
    print(f"centered={centered}, minimum={support}, score={value:.6f}")
    print("  selected (item, overlap, weight, signal):", [tuple(round(x, 6) if isinstance(x, float) else x for x in row) for row in selected])
''')

add("gradient", "Check the old-value gradient before training", "Which factor values must the item update use? Can a finite-difference calculation detect a sequential-update mistake?", '''
import numpy as np

mean, rating, penalty, rate = 3., 5., .1, .05
state = np.array([.4, -.2, .5, .3, .1, -.1])

def objective(values):
    p, q, bu, bi = values[:2], values[2:4], values[4], values[5]
    error = rating - mean - bu - bi - p @ q
    return .5 * (error**2 + penalty * (values @ values))

def gradient(values):
    p, q, bu, bi = values[:2], values[2:4], values[4], values[5]
    error = rating - mean - bu - bi - p @ q
    return np.r_[-error*q + penalty*p, -error*p + penalty*q,
                 -error + penalty*bu, -error + penalty*bi]

analytic = gradient(state)
step = 1e-6
numeric = np.array([(objective(state + step*row) - objective(state - step*row))/(2*step)
                    for row in np.eye(6)])
assert np.allclose(analytic, numeric, atol=1e-8)
updated = state - rate*analytic
print("gradient:", np.round(analytic, 6).tolist())
print("new state:", np.round(updated, 6).tolist())
print(f"local objective: {objective(state):.6f} -> {objective(updated):.6f}")
for changed_rate in (.05, .5, 1.):
    print(f"rate={changed_rate}: objective={objective(state-changed_rate*analytic):.6f}")
''')

SGD_FUNCTIONS = '''
def fit_mf(records, users, items, rank=2, penalty=.1, rate=.02, epochs=160, seed=7):
    # This objective sums a regularized local term for every observed record.
    # A user/item with d records therefore has its penalty counted d times.
    rng = np.random.default_rng(seed)
    mean = float(np.mean([row[2] for row in records]))
    p, q = rng.normal(0, .1, (users, rank)), rng.normal(0, .1, (items, rank))
    bu, bi = np.zeros(users), np.zeros(items)
    seen_users, seen_items = {row[0] for row in records}, {row[1] for row in records}
    for user in set(range(users)) - seen_users:
        p[user] = 0.
    for item in set(range(items)) - seen_items:
        q[item] = 0.
    history = []
    for epoch in range(epochs):
        rolling_sse = 0.
        for position in rng.permutation(len(records)):
            user, item, rating = records[position][:3]
            old_p, old_q = p[user].copy(), q[item].copy()
            old_bu, old_bi = bu[user], bi[item]
            error = rating - mean - old_bu - old_bi - old_p @ old_q
            rolling_sse += error**2
            p[user] += rate * (error*old_q - penalty*old_p)
            q[item] += rate * (error*old_p - penalty*old_q)
            bu[user] += rate * (error - penalty*old_bu)
            bi[item] += rate * (error - penalty*old_bi)
        fixed_errors, objective = [], 0.
        for user, item, rating, *rest in records:
            error = rating - mean - bu[user] - bi[item] - p[user] @ q[item]
            fixed_errors.append(error**2)
            objective += .5*(error**2 + penalty*(p[user]@p[user] + q[item]@q[item] + bu[user]**2 + bi[item]**2))
        if not np.isfinite(objective):
            raise ArithmeticError("Training left the finite numerical range.")
        history.append((np.sqrt(rolling_sse/len(records)), np.sqrt(np.mean(fixed_errors)), objective))
    return dict(mean=mean, p=p, q=q, bu=bu, bi=bi, seen_users=seen_users, seen_items=seen_items, history=history)

def score(model, user, item):
    known_user = user in model["seen_users"]
    known_item = item in model["seen_items"]
    value = model["mean"]
    if known_user:
        value += model["bu"][user]
    if known_item:
        value += model["bi"][item]
    if known_user and known_item:
        value += model["p"][user] @ model["q"][item]
    return float(value)
'''

add("explicitFit", "Fit only observed ratings and recompute the objective", "Are the last rolling error, end-of-epoch training RMSE and held-out error interchangeable?", RATINGS + SGD_FUNCTIONS + '''
records = [(int(user), int(item), float(ratings[user, item])) for user, item in zip(*np.where(observed))]
model = fit_mf(records, 5, 7)
for epoch in (0, 39, 79, 159):
    rolling, fixed, objective = model["history"][epoch]
    print(f"epoch={epoch+1}: rolling={rolling:.6f}, fixed training={fixed:.6f}, objective={objective:.6f}")
ranking = sorted((score(model, 0, item), item) for item in range(7) if not observed[0, item])
print("user 0 unseen, descending:", [(item, round(value, 6)) for value, item in sorted(ranking, key=lambda row: (-row[0], row[1]))])
print("No held-out score was computed in this mechanism demonstration.")
''')

add("observedAls", "Solve a masked factor block, then alternate", "Why does an exact block solve lower this objective without proving a globally best factorization?", RATINGS + '''
def als(ratings, rank=2, penalty=.3, sweeps=12, seed=4):
    observed = np.isfinite(ratings)
    users, items = ratings.shape
    rng = np.random.default_rng(seed)
    p, q = rng.normal(0, .1, (users, rank)), rng.normal(0, .1, (items, rank))
    # Once-per-entity regularization; this example has no bias terms.
    def objective():
        return float(((ratings[observed] - (p @ q.T)[observed])**2).sum()
                     + penalty*((p*p).sum()+(q*q).sum()))
    values = [objective()]
    for _ in range(sweeps):
        for user in range(users):
            selected = observed[user]
            design = q[selected]
            p[user] = np.linalg.solve(design.T@design + penalty*np.eye(rank), design.T@ratings[user, selected])
        values.append(objective())
        for item in range(items):
            selected = observed[:, item]
            design = p[selected]
            q[item] = np.linalg.solve(design.T@design + penalty*np.eye(rank), design.T@ratings[selected, item])
        values.append(objective())
    assert np.all(np.diff(values) <= 1e-8)
    return p, q, values

p, q, objectives = als(ratings)
print("first four objective values:", [round(x, 6) for x in objectives[:4]])
print("final objective:", round(objectives[-1], 6))
print("user 0 unseen scores:", np.round((p@q.T)[0, [2, 4, 6]], 6).tolist())
# Both rank-one matrices agree on the two observed diagonal entries.
first = np.array([[1., .5], [2., 1.]])
second = np.array([[1., 2.], [.5, 1.]])
print("same observations, different completions:", first.tolist(), second.tolist())
assert np.linalg.matrix_rank(first) == np.linalg.matrix_rank(second) == 1
''')

add("implicitBlock", "Compare a dense objective with its sparse Gram solve", "Which term preserves the cost of an unobserved item? What changes if that term is omitted?", '''
import numpy as np

items = np.array([[1., 0.], [0., 1.], [1., 1.]])
counts = np.array([2., 0., 1.])
alpha, penalty = 2., 1.
target = (counts > 0).astype(float)
confidence = 1. + alpha*counts
normal = items.T @ (confidence[:, None]*items) + penalty*np.eye(2)
right = items.T @ (confidence*target)
dense = np.linalg.solve(normal, right)
selected = counts > 0
q = items[selected]
sparse_normal = items.T@items + q.T@((confidence[selected]-1)[:, None]*q) + penalty*np.eye(2)
sparse_right = q.T@confidence[selected]
cached = np.linalg.solve(sparse_normal, sparse_right)
augmented = np.vstack((np.sqrt(confidence)[:, None]*items, np.sqrt(penalty)*np.eye(2)))
augmented_target = np.r_[np.sqrt(confidence)*target, [0., 0.]]
independent = np.linalg.lstsq(augmented, augmented_target, rcond=None)[0]
assert np.allclose(dense, cached) and np.allclose(dense, independent)
print("normal:", normal.tolist(), "right:", right.tolist())
print("user factors:", np.round(dense, 6).tolist())
print("all item scores:", np.round(items@dense, 6).tolist())
observed_only = np.linalg.solve(q.T@(confidence[selected, None]*q)+penalty*np.eye(2), q.T@confidence[selected])
print("if missing terms are removed:", np.round(observed_only, 6).tolist())
empty = np.linalg.solve(items.T@items+penalty*np.eye(2), np.zeros(2))
print("empty-user factors:", empty.tolist())
''')

add("implicitFit", "Train the preserved count matrix with exact sparse corrections", "Do unobserved scores become probabilities? Does each exact user/item sweep lower the declared objective?", COUNTS + '''
def fit_implicit(counts, rank=3, alpha=10., penalty=.1, sweeps=10, seed=5):
    users, items = counts.shape
    target, confidence = (counts > 0).astype(float), 1.+alpha*counts
    rng = np.random.default_rng(seed)
    p, q = rng.normal(0, .1, (users, rank)), rng.normal(0, .1, (items, rank))
    def objective():
        return float((confidence*(target-p@q.T)**2).sum()+penalty*((p*p).sum()+(q*q).sum()))
    history = [objective()]
    for _ in range(sweeps):
        for factors, other, data in ((p, q, counts), (q, p, counts.T)):
            gram = other.T@other
            for row in range(len(factors)):
                selected = data[row] > 0
                design = other[selected]
                conf = 1.+alpha*data[row, selected]
                normal = gram + design.T@((conf-1)[:, None]*design) + penalty*np.eye(rank)
                factors[row] = np.linalg.solve(normal, design.T@conf)
            history.append(objective())
    assert np.all(np.diff(history) <= 1e-7)
    return p, q, history

p, q, history = fit_implicit(counts)
print("objective initially / after first sweep / finally:", [round(history[j], 6) for j in (0, 2, -1)])
unseen = np.flatnonzero(counts[0] == 0)
print("unseen scores:", [(int(item), round(float(p[0]@q[item]), 6)) for item in unseen])
print("Scores are unconstrained factor products, not calibrated probabilities.")
''')

add("bpr", "Differentiate and train a sampled positive-negative pair", "At a tied score, why does a pairwise update have a nonzero signal?", '''
import numpy as np

def pair_loss(state, penalty):
    user, positive, negative = state.reshape(3, 2)
    gap = user@(positive-negative)
    return float(np.logaddexp(0., -gap) + .5*penalty*(state@state))

def pair_gradient(state, penalty):
    user, positive, negative = state.reshape(3, 2)
    gap = user@(positive-negative)
    signal = np.exp(-np.logaddexp(0., gap))
    return np.r_[-signal*(positive-negative)+penalty*user,
                 -signal*user+penalty*positive,
                 signal*user+penalty*negative]

state, penalty, rate = np.array([1., .5, .5, 1., 1., 0.]), .1, .1
step = 1e-6
numeric = np.array([(pair_loss(state+step*row, penalty)-pair_loss(state-step*row, penalty))/(2*step) for row in np.eye(6)])
assert np.allclose(numeric, pair_gradient(state, penalty), atol=1e-8)
for iteration in range(6):
    user, positive, negative = state.reshape(3, 2)
    print(f"step={iteration}: gap={user@(positive-negative):.6f}, local loss={pair_loss(state, penalty):.6f}")
    state -= rate*pair_gradient(state, penalty)
# This repeated pair illustrates its objective; a recommender samples many valid triples.
''')

METRICS = '''
import math

def metrics(order, grades, cutoff, eligible=None):
    eligible = set(range(len(grades))) if eligible is None else set(eligible)
    if cutoff < 1 or len(set(order)) != len(order) or not set(order) <= eligible:
        raise ValueError("Use a positive cutoff and unique eligible items.")
    relevant = {item for item in eligible if grades[item] > 0}
    selected = order[:cutoff]
    hits, precision_sum, reciprocal = 0, 0., 0.
    dcg = 0.
    for rank, item in enumerate(selected, 1):
        dcg += (2**grades[item]-1)/math.log2(rank+1)
        if item in relevant:
            hits += 1
            precision_sum += hits/rank
            reciprocal = reciprocal or 1/rank
    ideal = sum((2**grade-1)/math.log2(rank+1) for rank, grade in enumerate(sorted((grades[item] for item in eligible), reverse=True)[:cutoff], 1))
    return dict(precision=hits/cutoff, recall=hits/len(relevant) if relevant else None,
                ndcg=dcg/ideal if ideal else None, reciprocal_rank=reciprocal if relevant else None,
                average_precision=precision_sum/min(cutoff, len(relevant)) if relevant else None,
                fill_rate=len(selected)/cutoff,
                candidate_recall_ceiling=min(cutoff, len(set(order)&relevant))/len(relevant) if relevant else None)
'''
add("ranking", "Recompute the slate, denominator and retrieval ceiling", "Can reranking recover a relevant item that candidate generation never returned?", METRICS + '''
for order, grades, cutoff in [([2, 0, 4, 1, 3], [1, 1, 0, 0, 0], 3),
                               ([2, 0, 4], [1, 1, 0, 0, 0], 3),
                               ([0, 1], [0, 0, 0], 3),
                               ([2, 0, 1], [2, 1, 3], 2)]:
    result = metrics(order, grades, cutoff)
    print("order", order, "K", cutoff, {key: round(value, 6) if value is not None else None for key, value in result.items()})
''')

add("coldItem", "Map metadata into an existing factor space", "How can a new item get a score before it has interactions, and what assumption is still required?", '''
import numpy as np

# Synthetic training items: columns are two metadata features.
features = np.array([[1., 0.], [0., 1.], [1., 1.], [2., 1.]])
item_factors = np.array([[.8, .1], [.1, .9], [.9, .8], [1.7, .8]])
penalty = .2
mapping = np.linalg.solve(features.T@features+penalty*np.eye(2), features.T@item_factors)
augmented = np.vstack((features, np.sqrt(penalty)*np.eye(2)))
targets = np.vstack((item_factors, np.zeros((2, 2))))
assert np.allclose(mapping, np.linalg.lstsq(augmented, targets, rcond=None)[0])
user, new_features = np.array([.4, .7]), np.array([1., .5])
new_factor = new_features@mapping
print("mapping:", np.round(mapping, 6).tolist())
print("new item factor:", np.round(new_factor, 6).tolist())
print("personalized dot score:", round(float(user@new_factor), 6))
rotation = np.array([[0., -1.], [1., 0.]])
assert np.allclose(user@new_factor, (user@rotation)@(new_factor@rotation))
print("joint factor rotation preserves the score; factor axes are not unique.")
''')

add("policy", "Enumerate a known randomized logging policy", "Does a logged click average of .48 estimate a half-A/half-B policy's value?", '''
from fractions import Fraction as F

logging, target, reward_chance = [F(4, 5), F(1, 5)], [F(1, 2), F(1, 2)], [F(2, 5), F(4, 5)]
logged = sum(p*q for p, q in zip(logging, reward_chance))
value = sum(p*q for p, q in zip(target, reward_chance))
mean, second = F(0), F(0)
for action in range(2):
    for reward in (0, 1):
        mass = logging[action]*(reward_chance[action] if reward else 1-reward_chance[action])
        weighted = target[action]/logging[action]*reward
        mean += mass*weighted
        second += mass*weighted**2
        print("action", action, "reward", reward, "mass", mass, "weighted reward", weighted)
assert mean == value
print("logged mean:", logged, "target mean:", value, "IPS expectation:", mean)
print("one-request IPS variance:", second-mean**2)
# A finite realized sample is not the expectation: 16 A, 4 B; 6 A and 3 B clicks.
estimate = (target[0]/logging[0]*6 + target[1]/logging[1]*3)/20
print("one declared 20-request sample: raw", F(9, 20), "IPS", estimate)
print("If B has zero logging probability and positive target mass, its mean is not identified from those logs.")
''')

add("implicitLibrary", "Check the current implicit API against a direct solve", "Are the CSR entries raw counts or confidence weights, and which axis contains users?", COUNTS + '''
import implicit
from scipy.sparse import csr_matrix
from implicit.cpu.als import AlternatingLeastSquares

alpha, penalty = 2., .2
confidence = csr_matrix(np.where(counts > 0, 1.+alpha*counts, 0.))
model = AlternatingLeastSquares(factors=3, regularization=penalty, alpha=1.,
                               dtype=np.float64, use_cg=False, iterations=12,
                               num_threads=1, random_state=11)
model.fit(confidence, show_progress=False)
print("implicit version:", implicit.__version__)
print("user factors, item factors:", model.user_factors.shape, model.item_factors.shape)
# The last item sweep changes Q. Recalculate this user against that final Q.
user = model.recalculate_user(0, confidence[0])
q = model.item_factors
c = 1.+alpha*counts[0]
target = (counts[0] > 0).astype(float)
direct = np.linalg.solve(q.T@(c[:, None]*q)+penalty*np.eye(3), q.T@(c*target))
assert np.allclose(user, direct, atol=1e-8)
ids, scores = model.recommend(0, confidence[0], N=3, recalculate_user=True, filter_already_liked_items=True)
print("unseen recommendations:", [(int(item), round(float(value), 6)) for item, value in zip(ids, scores)])
print("recalculated user matches the declared dense confidence objective:", bool(np.allclose(user, direct, atol=1e-8)))
''')

add("surpriseLibrary", "Execute Surprise's own regularized MF predictions", "Does a method named SVD fill missing cells and then call a matrix decomposition?", RATINGS + '''
import pandas as pd
import surprise
from surprise import Dataset, Reader, SVD

records = [(str(user), str(item), float(ratings[user, item])) for user, item in zip(*np.where(observed))]
frame = pd.DataFrame(records, columns=["user", "item", "rating"])
train = Dataset.load_from_df(frame, Reader(rating_scale=(1, 5))).build_full_trainset()
model = SVD(n_factors=2, n_epochs=160, lr_all=.02, reg_all=.1, random_state=7)
model.fit(train)
print("Surprise version:", surprise.__version__)
for item in (2, 4, 6):
    prediction = model.predict("0", str(item), clip=False)
    print(f"user 0, item {item}: {prediction.est:.6f}")
print(f"unknown user, known item 0: {model.predict('new', '0', clip=False).est:.6f}")
print(f"both unknown: {model.predict('new', 'new', clip=False).est:.6f}")
assert np.isclose(model.predict("new", "new", clip=False).est, train.global_mean)
print("This fits the full toy matrix for API inspection; it is not a test evaluation.")
''')

add("evaluation", "Run an honest small timestamped comparison", "What is selected on validation data, what is refit, and which test cohort is reported?", RATINGS + SGD_FUNCTIONS + '''
# Integer time is an invented event order, not a benchmark timestamp.
base = [(int(user), int(item), float(ratings[user, item])) for user, item in zip(*np.where(observed))]
validation = [(0, 2, 4.), (1, 1, 2.), (2, 0, 4.), (3, 2, 2.), (4, 4, 2.)]
test = [(0, 6, 4.), (1, 5, 3.), (2, 3, 1.), (3, 1, 2.), (4, 0, 2.), (5, 0, 4.)]
events = [(*row, time) for time, row in enumerate(base+validation+test, 1)]

def evaluate(cutoff=20, test_start=26):
    train = [row for row in events if row[3] <= cutoff]
    valid = [row for row in events if cutoff < row[3] < test_start]
    held = [row for row in events if row[3] >= test_start]
    if not train or not valid or not held:
        raise ValueError("All three time partitions must be nonempty.")
    configurations = [(rank, penalty) for rank in (1, 2, 3) for penalty in (.05, .2)]
    validation_scores = []
    for rank, penalty in configurations:
        fitted = fit_mf(train, 6, 7, rank=rank, penalty=penalty)
        rmse = np.sqrt(np.mean([(score(fitted, u, i)-r)**2 for u, i, r, t in valid]))
        validation_scores.append((float(rmse), rank, penalty))
    # Ties are resolved by rank, then penalty. Test ratings have not been inspected.
    best_rmse, rank, penalty = min(validation_scores)
    final_train = train+valid
    fitted = fit_mf(final_train, 6, 7, rank=rank, penalty=penalty)
    mean = np.mean([row[2] for row in final_train])
    def item_baseline(item):
        values = [r for u, i, r, t in final_train if i == item]
        return (sum(values)+3*mean)/(len(values)+3)
    print(f"split sizes={len(train)}/{len(valid)}/{len(held)}; selected rank={rank}, penalty={penalty}, validation RMSE={best_rmse:.6f}")
    for name, subset in (("all", held), ("warm users", [row for row in held if row[0] in fitted["seen_users"]]), ("cold users", [row for row in held if row[0] not in fitted["seen_users"]])):
        if not subset:
            print(name, "n=0; RMSE undefined")
            continue
        baseline = np.sqrt(np.mean([(item_baseline(i)-r)**2 for u, i, r, t in subset]))
        mf = np.sqrt(np.mean([(score(fitted, u, i)-r)**2 for u, i, r, t in subset]))
        print(f"{name}: n={len(subset)}, shrunk item RMSE={baseline:.6f}, MF RMSE={mf:.6f}")
    # Popularity ranking uses only pre-test events; no test feedback is folded back in.
    history = {(u, i) for u, i, r, t in final_train}
    popular = {i: sum(1 for u, j, r, t in final_train if j == i) for i in range(7)}
    eligible = [i for i in range(7) if (0, i) not in history]
    print("user 0 eligible unseen:", eligible)
    print("popularity order:", sorted(eligible, key=lambda i: (-popular[i], i)))
    print("MF order:", sorted(eligible, key=lambda i: (-score(fitted, 0, i), i)))
    print("Release times are 0 for these seven items; changing that assumption requires request-time eligibility.")

evaluate()
print("Changed cutoff, same untouched test period:")
evaluate(cutoff=18)
''')

destination = ROOT / "src/learn/data/recommender-examples.js"
destination.write_text("// Complete scripts and actual stdout from generate-recommender-examples.py.\nexport const recommenderExamples = " + json.dumps(examples, ensure_ascii=False, indent=2) + ";\n", encoding="utf-8")
record = dict(timestamp=datetime.now(timezone.utc).isoformat(), programs={key: dict(codeSha256=hashlib.sha256(value["code"].encode()).hexdigest(), stdoutSha256=hashlib.sha256(value["expected"].encode()).hexdigest(), stdout=value["expected"]) for key, value in examples.items()})
(DEST / "execution.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
print(f"Executed and wrote {len(examples)} complete programs.")
