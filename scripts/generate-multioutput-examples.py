"""Generate complete lesson programs with actual, deterministic stdout."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import os
import subprocess
import sys
import textwrap
import black

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "scratch/multioutput/native"
DEST.mkdir(parents=True, exist_ok=True)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
examples = {}


def add(key, title, question, code):
    code = textwrap.dedent(code).strip() + "\n"
    formatted = black.format_str(code, mode=black.Mode(line_length=88))
    assert ast.dump(ast.parse(code)) == ast.dump(ast.parse(formatted))
    path = DEST / f"{key}.py"
    path.write_text(formatted, encoding="utf-8")
    result = subprocess.run([sys.executable, "-X", "utf8", "-I", str(path)], capture_output=True, text=True, encoding="utf-8", timeout=120)
    if result.returncode:
        raise RuntimeError(f"{key}\n{result.stdout}\n{result.stderr}")
    examples[key] = dict(title=title, question=question, code=formatted.rstrip(), expected=result.stdout.rstrip(), language="python")


add("schema", "Keep the label vocabulary attached to the columns", "Will repeating a tag create another positive label? What happens to a tag absent from the declared vocabulary?", '''
from sklearn.preprocessing import MultiLabelBinarizer

vocabulary = ["code", "hardware", "energy"]
messages = [["code", "energy"], ["hardware", "hardware"], []]
encoder = MultiLabelBinarizer(classes=vocabulary)
encoded = encoder.fit_transform(messages)
print("columns:", encoder.classes_.tolist())
print("matrix:", encoded.tolist())
print("decoded:", encoder.inverse_transform(encoded))

def checked_transform(rows):
    unknown = set().union(*map(set, rows)) - set(vocabulary)
    if unknown:
        raise ValueError(f"Unrecognized labels: {sorted(unknown)}")
    return encoder.transform(rows)

try:
    checked_transform([["network"]])
except ValueError as error:
    print(error)
print("Unreviewed energy annotation stays None, not 0:", [1, 0, None])
''')

add("metrics", "Calculate errors before choosing an average", "The four rows contain three mistakes. Why are Hamming loss and exact-set accuracy both 0.25 here, despite measuring different things?", '''
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, jaccard_score

truth = np.array([[1,0,1], [0,1,1], [1,1,0], [0,0,1]])
pred = np.array([[1,0,1], [0,1,0], [1,0,0], [0,1,1]])
tp = int(((truth == 1) & (pred == 1)).sum())
fp = int(((truth == 0) & (pred == 1)).sum())
fn = int(((truth == 1) & (pred == 0)).sum())
print("TP, FP, FN:", tp, fp, fn)
print("Hamming loss / exact-set accuracy:", hamming_loss(truth, pred), accuracy_score(truth, pred))
for average in ["micro", "macro", "samples"]:
    print(average, "F1/Jaccard:", round(f1_score(truth, pred, average=average, zero_division=0), 6), round(jaccard_score(truth, pred, average=average, zero_division=0), 6))
empty = np.zeros((1, 3), dtype=int)
print("Both-empty sample F1:", f1_score(empty, empty, average="samples", zero_division=0))
all_positive = np.ones((2, 5), dtype=int)
for name, candidate in [("A", [[1]*5, [0]*5]), ("B", [[1,1,0,0,0]]*2)]:
    print(name, "mean J/F1:", round(jaccard_score(all_positive, candidate, average="samples", zero_division=0), 6), round(f1_score(all_positive, candidate, average="samples", zero_division=0), 6))
''')

add("numpyHeads", "Fit independent heads and a greedy chain from scratch", "On the original controlled data, will a chain necessarily beat the independent model? Keep the learning rule and split fixed before comparing.", '''
import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss, accuracy_score, f1_score

X, Y = make_multilabel_classification(n_samples=1000, n_features=20, n_classes=5, n_labels=2, allow_unlabeled=False, random_state=42)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
mean, scale = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
train = np.c_[np.ones(len(X_train)), (X_train-mean)/scale]
test = np.c_[np.ones(len(X_test)), (X_test-mean)/scale]

def sigmoid(scores):
    return 1 / (1 + np.exp(-np.clip(scores, -500, 500)))

def fit(features, target):
    weights = np.zeros(features.shape[1])
    for _ in range(300):
        probability = sigmoid(features @ weights)
        weights -= 0.05 * features.T @ (probability-target) / len(target)
    return weights

independent = np.column_stack([sigmoid(test @ fit(train, Y_train[:, label])) >= 0.5 for label in range(5)]).astype(int)
chain = np.zeros_like(Y_test)
for label in range(5):
    # Training uses actual earlier labels; prediction has only earlier predictions.
    weights = fit(np.c_[train, Y_train[:, :label]], Y_train[:, label])
    chain[:, label] = sigmoid(np.c_[test, chain[:, :label]] @ weights) >= 0.5
constant = np.broadcast_to(Y_train.mean(axis=0) >= 0.5, Y_test.shape)
for name, prediction in [("constant", constant), ("independent", independent), ("chain", chain)]:
    print(f"{name}: Hamming={hamming_loss(Y_test,prediction):.4f}; subset={accuracy_score(Y_test,prediction):.4f}; micro-F1={f1_score(Y_test,prediction,average='micro',zero_division=0):.4f}")
print("Stable mean summed cell loss at zero scores:", round(float(np.mean(np.sum(np.logaddexp(0, np.zeros_like(Y_train))-Y_train*0, axis=1))), 6))
''')

add("joint", "Choose a decision under an exact joint distribution", "Which label vector minimizes expected Hamming loss, and which minimizes the chance of any mistake?", '''
from fractions import Fraction as F
from itertools import product

states = list(product([0, 1], repeat=2))
mass = dict(zip(states, map(lambda count: F(count, 20), [6, 5, 1, 8])))
marginals = [sum(p for state, p in mass.items() if state[j]) for j in range(2)]
print("marginals:", list(map(str, marginals)))
for action in states:
    hamming = sum(p * F(sum(a != y for a, y in zip(action, state)), 2) for state, p in mass.items())
    print(action, "Hamming risk", hamming, "subset risk", 1-mass[action])
for order in [(0, 1), (1, 0)]:
    action = [0, 0]
    first, second = order
    action[first] = int(marginals[first] >= F(1, 2))
    support = sum(p for state, p in mass.items() if state[first] == action[first])
    conditional = sum(p for state, p in mass.items() if state[first] == action[first] and state[second]) / support
    action[second] = int(conditional >= F(1, 2))
    print("greedy order", order, "conditional", conditional, "action", action)
''')

add("chainApi", "Inspect what a fitted chain's probability method returns", "Does the second number returned by predict_proba sum over both possible values of the first label?", '''
from itertools import product
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain

rng = np.random.default_rng(9)
X = rng.normal(size=(240, 2))
first = rng.binomial(1, 1/(1+np.exp(-X[:, 0])))
second = rng.binomial(1, 1/(1+np.exp(-(2*first-0.4+0.5*X[:, 1]))))
Y = np.c_[first, second]
chain = ClassifierChain(LogisticRegression(C=1, max_iter=500), order=[0,1], cv=None, chain_method="predict").fit(X, Y)
row = np.array([[0.2, -0.3]])
api = chain.predict_proba(row)[0]
first_hard = chain.estimators_[0].predict(row)[0]
path_second = chain.estimators_[1].predict_proba(np.c_[row, [first_hard]])[0,1]
leaf_mass = {}
for a, b in product([0,1], repeat=2):
    p_a = chain.estimators_[0].predict_proba(row)[0,a]
    p_b_given_a = chain.estimators_[1].predict_proba(np.c_[row, [a]])[0,b]
    leaf_mass[a,b] = p_a*p_b_given_a
marginal_second = sum(p for (_, b), p in leaf_mass.items() if b)
assert np.isclose(api[1], path_second)
assert np.isclose(sum(leaf_mass.values()), 1)
print("API path probabilities:", np.round(api, 6).tolist())
print("First predicted feature:", int(first_hard))
print("Second marginal from all leaves:", round(marginal_second, 6))
print("Fitted joint mass:", [(key, round(float(value), 6)) for key,value in leaf_mass.items()])
''')

add("association", "Separate pooled association from conditional dependence", "Can two labels be independent after seeing X even though their pooled co-occurrence table is strongly associated?", '''
from fractions import Fraction as F

def independent(p):
    return [(1-p)**2, (1-p)*p, p*(1-p), p*p]

low, high = independent(F(1,10)), independent(F(9,10))
pooled = [(a+b)/2 for a,b in zip(low,high)]
marginal = pooled[2]+pooled[3]
print("Pooled 00,01,10,11:", list(map(str,pooled)))
print("Pooled P(B=1):", pooled[1]+pooled[3])
print("Pooled P(B=1|A=1):", pooled[3]/marginal)
print("Within low/high P(B=1|A=1):", low[3]/(low[2]+low[3]), high[3]/(high[2]+high[3]))
print("Weighted score for p=.1, positive weight 9:", F(9,10)/(F(9,10)+F(9,10)))
''')

add("thresholds", "Search validation decisions, with ties and a named objective", "Does optimizing each label's F1 necessarily maximize pooled micro-F1? Enumerate this tiny validation set rather than guessing.", '''
from itertools import product
import numpy as np
from sklearn.metrics import f1_score

scores = np.array([[.9,.9],[.8,.8],[.8,.7],[.55,.6],[.4,.4],[.2,.1]])
truth = np.array([[0,0],[0,1],[0,1],[0,0],[0,0],[1,1]])
cuts = [sorted(set([0., 1.01, *scores[:,j]])) for j in range(2)]
def score(threshold, average):
    return f1_score(truth, scores >= threshold, average=average, zero_division=0)

per_label = np.array([max(cuts[j], key=lambda t: f1_score(truth[:,j], scores[:,j]>=t, zero_division=0)) for j in range(2)])
micro = max(product(*cuts), key=lambda t: score(np.array(t), "micro"))
shared_candidates = sorted(set([0.,1.01,*scores.ravel()]))
shared = max(shared_candidates, key=lambda t: score(t,"micro"))
for name, threshold in [("per-label F1",per_label),("joint micro",np.array(micro)),("shared micro",shared)]:
    print(name, "threshold", np.asarray(threshold).tolist(), "macro", round(score(threshold,"macro"),6), "micro", round(score(threshold,"micro"),6))
print("At t=.8 both equal .8 scores stay selected:", (scores[:,0]>=.8).astype(int).tolist())
print("No-positive choice:", (scores[:,0]>=1.01).astype(int).tolist())
''')

add("powerset", "Encode combinations and combine overlapping labelsets", "Can a plain observed-class label-powerset model emit 11 if it only learned classes 00, 01 and 10?", '''
import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.array([[-2],[-1.8],[-1],[-.8],[1],[1.2]])
Y = np.array([[0,0],[0,0],[0,1],[0,1],[1,0],[1,0]])
vocabulary, classes = np.unique(Y, axis=0, return_inverse=True)
model = LogisticRegression(C=1).fit(X, classes)
pred = vocabulary[model.predict([[0.5],[2]])]
print("Observed vector classes:", vocabulary.tolist())
print("Decoded predictions:", pred.tolist())
print("11 is an available class:", any(np.array_equal(row,[1,1]) for row in vocabulary))

# A transparent vote example, not an implementation of the full RAkEL trainer.
subsets = [(0,1),(1,2),(0,2)]
local_predictions = [(1,0),(1,1),(0,1)]
positive, coverage = np.zeros(3), np.zeros(3)
for subset, prediction in zip(subsets, local_predictions):
    for label, value in zip(subset,prediction):
        positive[label] += value
        coverage[label] += 1
votes = positive/coverage
print("Overlap vote fractions:", votes.tolist())
print("Tie-positive decoded labels:", (votes>=.5).astype(int).tolist())
''')

add("categorical", "Give each categorical output its own class vocabulary", "Why does predict_proba return two arrays of different widths here?", '''
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

X = np.array([[a,b] for a in [-2,-1,0,1,2] for b in [-1,1]])
Y = np.column_stack([np.where(X[:,0]<0,"low",np.where(X[:,0]>0,"high","middle")), np.where(X[:,1]<0,"cool","warm")])
model = MultiOutputClassifier(LogisticRegression(C=2,max_iter=500)).fit(X,Y)
query = np.array([[-1,1],[0,-1]])
print("Per-output classes:", [estimator.classes_.tolist() for estimator in model.estimators_])
print("Predictions:", model.predict(query).tolist())
print("Probability shapes:", [list(array.shape) for array in model.predict_proba(query)])
''')

add("regression", "Compare vector and column-wise least squares on held-out numbers", "If both approaches use the same unconstrained linear features and squared loss, should fitting the columns together change the solution?", '''
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.multioutput import MultiOutputRegressor, RegressorChain

rng = np.random.default_rng(23)
X = rng.normal(size=(120,3))
noise = rng.normal(size=(120,2)) @ np.array([[.2,3.0],[0,4.0]])
Y = np.c_[1+2*X[:,0]-X[:,1], 30+15*X[:,0]+5*X[:,2]] + noise
train, test = slice(0,90), slice(90,None)
design = np.c_[np.ones(90),X[train]]
coefficients = np.linalg.lstsq(design,Y[train],rcond=None)[0]
columns = np.column_stack([np.linalg.lstsq(design,Y[train,j],rcond=None)[0] for j in range(2)])
assert np.allclose(coefficients,columns)
vector = LinearRegression().fit(X[train],Y[train])
wrapper = MultiOutputRegressor(LinearRegression()).fit(X[train],Y[train])
assert np.allclose(vector.predict(X[test]),wrapper.predict(X[test]))
scales = Y[train].std(axis=0)  # Frozen from training only.
for name, prediction in [("mean",np.broadcast_to(Y[train].mean(axis=0),Y[test].shape)),("OLS",vector.predict(X[test])),("ridge",Ridge(alpha=1).fit(X[train],Y[train]).predict(X[test])),("linear chain",RegressorChain(LinearRegression(),order=[0,1]).fit(X[train],Y[train]).predict(X[test]))]:
    errors = prediction-Y[test]
    rmse = np.sqrt((errors**2).mean(axis=0))
    print(name,"RMSE [C,Wh]",np.round(rmse,6).tolist(),"scaled MSE",round(float(np.mean((errors/scales)**2)),6))
print("OLS column solutions agree:", bool(np.allclose(coefficients,columns)))
''')

add("sharedSplit", "Watch units change a shared tree's compromise", "If energy is measured in hundreds of Wh, does the best common split change even though each physical observation is the same?", '''
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor

X = np.arange(4)[:,None]
Y = np.array([[0,0],[0,100],[2,100],[2,100]],dtype=float)
for energy_scale in [1,100]:
    scales = np.array([1,energy_scale])
    shared = DecisionTreeRegressor(max_depth=1,random_state=0).fit(X,Y/scales)
    prediction = shared.predict(X)*scales
    print("Energy scale",energy_scale,"shared cut",shared.tree_.threshold[0],"raw SSE",np.round(((Y-prediction)**2).sum(axis=0),6).tolist())
separate = MultiOutputRegressor(DecisionTreeRegressor(max_depth=1,random_state=0)).fit(X,Y)
print("Separate cuts:",[float(model.tree_.threshold[0]) for model in separate.estimators_])
print("Separate training SSE:",((Y-separate.predict(X))**2).sum(axis=0).tolist())
''')

add("groupShrinkage", "Check a shared feature penalty against its closed form", "Why can a grouped penalty keep both coefficients where a separate absolute-value penalty would remove one?", '''
import numpy as np
from sklearn.linear_model import MultiTaskLasso, Lasso

# X.T X / n = I, so each feature-row objective separates exactly.
X = np.sqrt(3)*np.eye(3)
Z = np.array([[3.,4.],[.5,2.],[0.,0.]])
Y = X@Z
penalty = 2.
norms = np.linalg.norm(Z,axis=1)
factors = np.maximum(0,1-penalty/np.maximum(norms,1e-30))
expected = Z*factors[:,None]
group = MultiTaskLasso(alpha=penalty,fit_intercept=False,tol=1e-12,max_iter=10000).fit(X,Y).coef_.T
separate = np.column_stack([Lasso(alpha=penalty,fit_intercept=False,tol=1e-12).fit(X,Y[:,j]).coef_ for j in range(2)])
assert np.allclose(group,expected)
print("Grouped coefficients:",np.round(group,6).tolist())
print("Separate coefficients:",np.round(separate,6).tolist())
print("Grouped formula matches actual estimator:",bool(np.allclose(group,expected)))
''')

add("compression", "Retained label variance can miss every rare positive", "Does retaining more than 95 percent of centered label energy guarantee useful recall for every label?", '''
import numpy as np

# First two tags repeat together; a third rare tag is orthogonal after centering.
common = np.tile([0.,1.],100)
rare = np.zeros(200)
rare[[0,1]] = 1
Y = np.c_[common,common,rare]
mean = Y.mean(axis=0)
centered = Y-mean
_, singular, right = np.linalg.svd(centered,full_matrices=False)
basis = right[:1].T
reconstructed = centered@basis@basis.T+mean
prediction = reconstructed>=.5
print("Retained centered energy:",round(float(singular[0]**2/(singular**2).sum()),6))
print("Rare reconstructed range:",np.round([reconstructed[:,2].min(),reconstructed[:,2].max()],6).tolist())
print("Rare recall:",float(prediction[rare==1,2].mean()))
print("Uniform single-positive conditional mean:",(np.ones(5)/5).tolist())
''')

TEXT = '''
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.metrics import f1_score, hamming_loss, accuracy_score

# Deliberately small, hand-authored examples: not a real benchmark.
# Each group is one source case; its paraphrases never cross a split.
cases = [
    ("train", (1,0,0), ["Python loop raises an exception", "Fix the Python loop error"]),
    ("train", (0,1,0), ["Replace the loose sensor cable", "A connector on the sensor is broken"]),
    ("train", (0,0,1), ["Measure electricity consumed overnight", "The electricity bill reports high energy use"]),
    ("train", (1,1,0), ["Python cannot read the sensor cable", "Debug software that reads a sensor connector"]),
    ("train", (1,0,1), ["Python estimates electricity use", "Fix software computing energy consumption"]),
    ("train", (0,1,1), ["A battery connector wastes energy", "Replace a battery cable to reduce power loss"]),
    ("train", (1,1,1), ["Debug Python for the battery sensor energy meter", "Software reads battery hardware electricity consumption"]),
    ("train", (0,0,0), ["Where is the meeting room", "Please update the workshop opening hours"]),
    ("validation", (1,0,0), ["A Python function has a software bug", "Debug the broken Python function"]),
    ("validation", (0,1,1), ["Check the battery cable power loss", "Energy is wasted by a battery connector"]),
    ("validation", (1,1,0), ["Software cannot read a hardware sensor", "Debug Python sensor readings"]),
    ("validation", (0,0,1), ["Track electricity consumption", "Our electricity usage is rising"]),
    ("test", (1,0,1), ["Repair Python energy estimates", "Software predicts electricity consumption"]),
    ("test", (0,1,0), ["The hardware sensor cable snapped", "Replace the broken connector"]),
    ("test", (1,1,1), ["Debug software on a battery electricity sensor", "Python reads the battery energy hardware"]),
    ("test", (0,0,0), ["What time does the meeting begin", "Where can workshop visitors park"]),
]
texts, labels, groups, splits = [], [], [], []
for group, (split, target, variants) in enumerate(cases):
    for text in variants:
        texts.append(text); labels.append(target); groups.append(group); splits.append(split)
Y, groups, splits = np.array(labels), np.array(groups), np.array(splits)
train, validation, test = [np.flatnonzero(splits==name) for name in ["train","validation","test"]]
assert not (set(groups[train]) & set(groups[test]))
assert not (set(groups[train]) & set(groups[validation]))
assert not (set(groups[validation]) & set(groups[test]))
vectorizer = TfidfVectorizer(ngram_range=(1,2))
X_train = vectorizer.fit_transform([texts[i] for i in train])
X_validation = vectorizer.transform([texts[i] for i in validation])
X_test = vectorizer.transform([texts[i] for i in test])
models = {
    "independent": OneVsRestClassifier(LogisticRegression(C=2,max_iter=1000)),
    "chain 012": ClassifierChain(LogisticRegression(C=2,max_iter=1000),order=[0,1,2]),
    "chain 210": ClassifierChain(LogisticRegression(C=2,max_iter=1000),order=[2,1,0]),
}
for model in models.values():
    model.fit(X_train,Y[train])
validation_scores = {name:model.predict_proba(X_validation) for name,model in models.items()}
'''
add("textReport", "Run a complete grouped multi-tag text comparison", "Select the model and threshold using only validation. Is the selected model also better than the prevalence baseline on this small test cohort?", TEXT + '''
thresholds = [.25,.4,.5,.6,.75]
settings = [(f1_score(Y[validation],score>=threshold,average="micro",zero_division=0),name,threshold) for name,score in validation_scores.items() for threshold in thresholds]
# Ties: use the first listed candidate, avoiding an unreported test-set tie-break.
best = max(range(len(settings)),key=lambda index:settings[index][0])
validation_f1,name,threshold = settings[best]
print("Rows train/validation/test:",len(train),len(validation),len(test))
print("Selected:",name,"threshold",threshold,"validation micro-F1",round(validation_f1,6))
predictions = models[name].predict_proba(X_test)>=threshold
baseline = np.broadcast_to(Y[train].mean(axis=0)>=.5,Y[test].shape)
print("Test positives [code,hardware,energy]:",Y[test].sum(axis=0).tolist())
for label,pred in [("baseline",baseline),("selected",predictions)]:
    print(label,"Hamming",round(hamming_loss(Y[test],pred),6),"subset",round(accuracy_score(Y[test],pred),6),"micro-F1",round(f1_score(Y[test],pred,average="micro",zero_division=0),6),"label-F1",np.round(f1_score(Y[test],pred,average=None,zero_division=0),6).tolist())
print("Predicted test rows:",predictions.astype(int).tolist())
''')

add("changedReport", "Change the validation objective, then freeze again", "If missing a tag costs three times as much as an extra tag, which validation setting is selected? Do not use the test labels to choose it.", TEXT + '''
thresholds = [.25,.4,.5,.6,.75]
def cost(truth,prediction):
    false_positive = ((truth==0)&prediction).sum()
    false_negative = ((truth==1)&~prediction).sum()
    return int(false_positive+3*false_negative)
settings = [(cost(Y[validation],score>=threshold),name,threshold) for name,score in validation_scores.items() for threshold in thresholds]
best = min(range(len(settings)),key=lambda index:settings[index][0])
validation_cost,name,threshold = settings[best]
prediction = models[name].predict_proba(X_test)>=threshold
print("Selected by validation FP+3FN:",name,threshold,"cost",validation_cost)
print("Frozen test cost:",cost(Y[test],prediction),"over",Y[test].size,"label decisions")
print("Frozen test micro-F1:",round(f1_score(Y[test],prediction,average="micro",zero_division=0),6))
print("Predicted test rows:",prediction.astype(int).tolist())
''')

target = ROOT / "src/learn/data/multioutput-examples.js"
target.write_text("// Generated from complete, actually executed programs.\nexport const multioutputExamples = " + json.dumps(examples,ensure_ascii=False,indent=2) + ";\n",encoding="utf-8")
record = {"timestamp":datetime.now(timezone.utc).isoformat(),"python":sys.version,"examples":{key:{"codeSha256":hashlib.sha256(value["code"].encode()).hexdigest(),"stdoutSha256":hashlib.sha256(value["expected"].encode()).hexdigest(),"stdout":value["expected"]} for key,value in examples.items()}}
(DEST / "execution.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
print(json.dumps({"programs":len(examples),"record":str(DEST / "execution.json")},indent=2))
