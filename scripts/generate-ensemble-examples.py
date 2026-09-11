"""Generate complete runnable examples and a native-fitted prediction map."""
import ast
import contextlib
import io
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import black

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "scratch/ensemble-methods"
WORK.mkdir(exist_ok=True)
examples = []


def add(name, title, question, code, note=""):
    raw = textwrap.dedent(code).strip() + "\n"
    formatted = black.format_str(raw, mode=black.Mode(line_length=88))
    assert ast.dump(ast.parse(raw)) == ast.dump(ast.parse(formatted))
    examples.append({"id": name, "title": title, "question": question, "code": formatted, "note": note})


add("combination-arithmetic", "Account for every signed error", "Why can the loss of the mean be smaller than the mean loss, yet still worse than the best member?", r'''
    from fractions import Fraction as F

    a = [F(-3), F(-1), F(2), F(2)]
    b = [F(1), F(1), F(-1), F(-3)]
    mean = lambda values: sum(values) / len(values)
    denominator = sum((x - z) ** 2 for x, z in zip(a, b))
    weight = sum(z * (z - x) for x, z in zip(a, b)) / denominator
    combined = [weight * x + (1 - weight) * z for x, z in zip(a, b)]
    average_loss = weight * mean([x*x for x in a]) + (1-weight)*mean([z*z for z in b])
    disagreement = mean([weight*(x-r)**2 + (1-weight)*(z-r)**2 for x,z,r in zip(a,b,combined)])
    print("weight A:", weight)
    print("combined errors:", [str(value) for value in combined])
    print("average member loss:", average_loss)
    print("disagreement:", disagreement)
    print("combined loss:", mean([value*value for value in combined]))
    assert average_loss-disagreement == mean([value*value for value in combined])
    # A new case: A is exact, B is wrong by 2. The mean is worse than A.
    print("exact A, error-2 B, equal mean error:", F(0, 2) + F(2, 2))
    error = F(3, 10)
    independent_majority = 3*error**2*(1-error) + error**3
    print("three independent 0.3-error ballots:", independent_majority)
    print("three copies of one 0.3-error ballot:", error)
''')

add("bootstrap-oob", "Fit each bootstrap bag and inspect who is out of bag", "Which fitted models may supply an OOB prediction for row A, and what happens if all models saw A?", r'''
    from fractions import Fraction
    import numpy as np
    from sklearn.tree import DecisionTreeRegressor

    x = np.arange(6., dtype=float).reshape(-1, 1)
    y = np.array([1., 2., 2., 6., 7., 8.])
    bags = [[0,0,2,3,3,5], [1,2,2,4,4,5], [0,1,1,3,4,4]]
    predictions = []
    for index, draws in enumerate(bags):
        model = DecisionTreeRegressor(max_depth=1, random_state=7).fit(x[draws], y[draws])
        predictions.append(model.predict(x))
        count = np.bincount(draws, minlength=6)
        threshold = model.tree_.threshold[0]
        print(f"bag {index+1}: counts={count.tolist()}, threshold={threshold:.1f}")
        print("  predictions:", np.round(predictions[-1], 6).tolist())
    eligible = [index for index, draws in enumerate(bags) if 0 not in draws]
    print("A eligible bags:", [index+1 for index in eligible])
    oob = np.mean([predictions[index][0] for index in eligible]) if eligible else None
    print("A OOB prediction:", round(float(oob), 6) if oob is not None else None)
    print("A all-model prediction:", round(float(np.mean(predictions, axis=0)[0]), 6))
    missing = Fraction(5, 6)**6
    print("specified row absent in a random six-draw bag:", missing)
    print("expected distinct rows:", 6*(1-missing))
    print("no eligible model among three independent random bags:", (1-missing)**3)
''')

add("signed-adaboost", "A complete binary AdaBoost loop with explicit boundaries", "Distinguish the row weights, the stump error and its signed vote weight; when does this implementation stop?", r'''
    import numpy as np

    def train_adaboost(x, labels, rounds=8):
        x = np.asarray(x, dtype=float)
        y = np.asarray(labels, dtype=float)
        if x.ndim != 1 or y.shape != x.shape or not len(x):
            raise ValueError("Use equally sized, nonempty one-dimensional inputs.")
        if not np.isfinite(x).all() or not np.isin(y, [-1,1]).all():
            raise ValueError("Finite inputs and signed labels -1/+1 are required.")
        if type(rounds) is not int or not 1 <= rounds <= 50:
            raise ValueError("Use 1-50 rounds in this teaching implementation.")
        if np.max(np.abs(x)) > 1e6:
            raise ValueError("Input exceeds this example's supported numeric scale.")
        unique = np.unique(x)
        thresholds = np.r_[unique[0]-1, unique[:-1]/2+unique[1:]/2, unique[-1]+1]
        weights = np.full(len(y), 1/len(y))
        scores = np.zeros(len(y))
        history = []
        fitted = []
        for round_index in range(rounds):
            candidates = []
            for threshold in thresholds:
                for polarity in [-1,1]:
                    prediction = np.where(x <= threshold, polarity, -polarity)
                    error = float(weights[prediction != y].sum())
                    candidates.append((error, float(threshold), polarity, prediction))
            # Sort traversal is fixed; tiny arithmetic ties keep the first candidate.
            best = candidates[0]
            for candidate in candidates[1:]:
                if candidate[0] < best[0]-1e-14:
                    best = candidate
            error, threshold, polarity, prediction = best
            if error == 0:
                return {"status":"perfect", "perfect":(threshold,polarity), "fitted":fitted, "history":history}
            if error >= .5-1e-14:
                return {"status":"no-edge", "perfect":None, "fitted":fitted, "history":history}
            alpha = .5*(np.log1p(-error)-np.log(error))
            before = weights.copy()
            updated = weights*np.exp(-alpha*y*prediction)
            normalizer = float(updated.sum())
            weights = updated/normalizer
            scores += alpha*prediction
            fitted.append((threshold,polarity,float(alpha)))
            history.append({"error":error,"alpha":float(alpha),"before":before,"after":weights.copy(),"normalizer":normalizer,"loss":float(np.mean(np.exp(-y*scores)))})
        return {"status":"round-limit", "perfect":None, "fitted":fitted, "history":history}

    def predict(model, x):
        x = np.asarray(x, dtype=float)
        if model["perfect"] is not None:
            threshold, polarity = model["perfect"]
            return np.where(x <= threshold, polarity, -polarity)
        scores = np.zeros(x.shape)
        for threshold, polarity, alpha in model["fitted"]:
            scores += alpha*np.where(x <= threshold, polarity, -polarity)
        return np.where(scores >= 0, 1, -1)

    model = train_adaboost([0,1,2,3,4,5], [-1,-1,1,1,-1,1], rounds=3)
    for index, row in enumerate(model["history"], 1):
        print(f"round {index}: error={row['error']:.6f}, alpha={row['alpha']:.6f}, loss={row['loss']:.6f}")
        print("  next weights:", np.round(row["after"], 6).tolist())
    print("new-query labels:", predict(model, [.5,2.5,4.5]).tolist())
    print("perfect fixture:", train_adaboost([0,1,2,3], [-1,-1,1,1])["status"])
    print("contradictory fixture:", train_adaboost([0,0,1,1], [-1,1,-1,1])["status"])
''')

# Conserve the useful original complete NumPy implementation byte-for-byte.
original = json.loads((ROOT / "docs/teaching/evidence/ensemble-original-review.json").read_text(encoding="utf-8"))
examples.append({"id":"full-numpy-comparison", "title":"Full NumPy trees and stumps on a fixed circles example", "question":"Does this one observed difference establish that one ensemble family always wins? Locate the finite-error clipping convention in the AdaBoost fit.", "code":original["programs"][0]["code"], "note":"This complete fixed-data implementation is preserved. Its clipping convention is explained in the lesson; use the separate boundary-explicit loop when studying zero error or no edge."})

add("manual-oof", "Build and use the exact six-row stack", "Why are the combiner's training predictions different from predictions made after refitting the bases on all six rows?", r'''
    import numpy as np
    from sklearn.base import clone
    from sklearn.linear_model import LinearRegression
    from sklearn.neighbors import KNeighborsRegressor

    def convex_weight(predictions, target):
        errors = predictions-target[:,None]
        difference = errors[:,0]-errors[:,1]
        denominator = float(difference@difference)
        return .5 if denominator == 0 else float(np.clip(errors[:,1]@(-difference)/denominator, 0, 1))

    x = np.arange(6., dtype=float).reshape(-1,1)
    y = np.array([1.,2.,2.,6.,7.,8.])
    bases = [KNeighborsRegressor(1), LinearRegression()]
    held_out = [np.array([0,3]), np.array([1,4]), np.array([2,5])]
    oof = np.full((len(y),len(bases)), np.nan)
    coverage = np.zeros(len(y), dtype=int)
    for held in held_out:
        train = np.setdiff1d(np.arange(len(y)), held)
        assert not set(train)&set(held)
        coverage[held] += 1
        for column, base in enumerate(bases):
            oof[held,column] = clone(base).fit(x[train],y[train]).predict(x[held])
    assert np.all(coverage==1) and np.isfinite(oof).all()
    weight = convex_weight(oof,y)
    full_bases = [clone(base).fit(x,y) for base in bases]
    queries = np.array([.5,2.5,4.5]).reshape(-1,1)
    new_predictions = np.column_stack([model.predict(queries) for model in full_bases])
    prediction = new_predictions@np.array([weight,1-weight])
    print("OOF matrix:", np.round(oof,6).tolist())
    print(f"honest nearest weight: {weight:.6f}")
    leaked = np.column_stack([model.predict(x) for model in full_bases])
    print(f"in-sample nearest weight: {convex_weight(leaked,y):.6f}")
    print("new predictions:", np.round(prediction,6).tolist())
    # These are explicitly NEW observed outcomes for a small worked check.
    new_y = np.array([1.5,4.,7.5])
    print(f"worked new-row stack MSE: {np.mean((prediction-new_y)**2):.6f}")
    print(f"worked new-row nearest MSE: {np.mean((new_predictions[:,0]-new_y)**2):.6f}")
''')

add("api-conventions", "Check the installed ensemble contracts", "Which meta-feature is a probability and which is a margin? Why do stored SAMME weights differ from the signed-vote alpha?", r'''
    import numpy as np
    import sklearn
    from sklearn.ensemble import AdaBoostClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    from sklearn.tree import DecisionTreeClassifier

    x = np.arange(6., dtype=float).reshape(-1,1)
    y = np.array([0,0,1,1,0,1])
    ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=3, random_state=7).fit(x,y)
    print("sklearn:", sklearn.__version__)
    print("SAMME errors:", np.round(ada.estimator_errors_,6).tolist())
    print("SAMME weights:", np.round(ada.estimator_weights_,6).tolist())
    print("signed binary alpha:", np.round(ada.estimator_weights_/2,6).tolist())

    x = np.array([[-3.,0.],[-2.,1.],[-1.,0.],[0.,1.],[1.,0.],[2.,1.],[3.,0.],[4.,1.]])
    y = np.array([0,0,1,0,1,1,0,1])
    stack = StackingClassifier(
        estimators=[("bayes",GaussianNB()),("margin",make_pipeline(StandardScaler(),LinearSVC(random_state=3)))],
        final_estimator=LogisticRegression(), cv=2, stack_method="auto")
    stack.fit(x,y)
    transformed = stack.transform(x)
    print("selected methods:", stack.stack_method_)
    print("full-refit transform shape:", transformed.shape)
    assert np.allclose(transformed[:,0],stack.estimators_[0].predict_proba(x)[:,1])
    assert np.allclose(transformed[:,1],stack.estimators_[1].decision_function(x))
    print("column 0 = P(class 1); column 1 = margin: checked")
    # transform(x) is the full-refit representation, not the OOF training matrix.
''')

add("forward-group-oof", "Keep time and entity boundaries inside the complete fit", "Which early rows lack a forward prediction, and why must they stay out of meta-training?", r'''
    import numpy as np
    from sklearn.base import clone
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.model_selection import GroupKFold, TimeSeriesSplit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = np.arange(12., dtype=float).reshape(-1,1)
    y = 2 + .4*x[:,0] + np.array([0,.2,-.1,.3,0,-.2,.1,0,.2,-.1,.1,-.2])
    bases = [make_pipeline(StandardScaler(),Ridge(alpha=1)), LinearRegression()]
    oof = np.full((len(y),2),np.nan)
    for train,held in TimeSeriesSplit(n_splits=3).split(x):
        assert max(train)<min(held)
        for column,base in enumerate(bases):
            oof[held,column] = clone(base).fit(x[train],y[train]).predict(x[held])
        print("time fit:", train.tolist(), "predict:", held.tolist())
    covered = np.isfinite(oof).all(axis=1)
    meta = Ridge(alpha=1).fit(oof[covered],y[covered])
    refits = [clone(base).fit(x,y) for base in bases]
    next_x = np.array([[12.]])
    next_meta = np.column_stack([model.predict(next_x) for model in refits])
    print("uncovered prefix:", np.flatnonzero(~covered).tolist())
    print(f"next-time prediction: {meta.predict(next_meta)[0]:.6f}")

    groups = np.repeat(np.arange(6),2)
    group_oof = np.full((len(y),2),np.nan)
    for train,held in GroupKFold(3).split(x,y,groups):
        assert not set(groups[train])&set(groups[held])
        for column,base in enumerate(bases):
            group_oof[held,column] = clone(base).fit(x[train],y[train]).predict(x[held])
    assert np.isfinite(group_oof).all()
    print("group-safe covered rows:", int(np.isfinite(group_oof).all(axis=1).sum()))
    print("These are DIFFERENT evaluation targets: future rows versus new groups.")
''')

add("calibrated-average", "An exact probability law exposes the calibration gap", "Verify calibration by grouping cases with the same forecast. Does averaging those forecasts preserve the group event rates?", r'''
    from fractions import Fraction as F
    from itertools import product

    rows=[]
    for a,b in product([0,1],repeat=2):
        truth=F(a+b,2)
        pa,pb=F(1+2*a,4),F(1+2*b,4)
        rows.append((F(1,4),truth,pa,pb,(pa+pb)/2))
    for column,name in [(2,"A"),(3,"B"),(4,"mean")]:
        print(name)
        for forecast in sorted({row[column] for row in rows}):
            group=[row for row in rows if row[column]==forecast]
            rate=sum(row[0]*row[1] for row in group)/sum(row[0] for row in group)
            print(f"  forecast {forecast}: event chance {rate}")
        # E[(p-Y)^2 | signals] = p^2 - 2p E[Y|signals] + E[Y|signals].
        risk=sum(mass*(row[column]**2-2*row[column]*truth+truth) for row in rows for mass,truth in [row[:2]])
        print("  exact Brier risk:",risk)
''')

add("context-interactions", "A learned linear fit can use nonlinear meta-features", "What changes when products are added, and why is this not the same as merely adding the context column?", r'''
    import numpy as np

    # Synthetic honest model outputs; no base fit uses these outcome rows.
    rows=np.array([[a,b,context] for context in [0.,1.] for a,b in [(0.,0.),(0.,1.),(1.,0.),(1.,1.)]])
    a,b,context=rows.T
    target=(1-context)*a+context*b
    ordinary=np.column_stack([np.ones(len(rows)),a,b,context])
    interactions=np.column_stack([(1-context)*a,context*b])
    first=np.linalg.lstsq(ordinary,target,rcond=None)[0]
    second=np.linalg.lstsq(interactions,target,rcond=None)[0]
    print("constant-slope coefficients:",np.round(first,6).tolist())
    print(f"constant-slope training MSE: {np.mean((ordinary@first-target)**2):.6f}")
    print("interaction coefficients:",np.round(second,6).tolist())
    print(f"interaction training MSE: {np.mean((interactions@second-target)**2):.6f}")
    new=np.array([[.2,.8,0.],[.2,.8,1.],[.2,.8,.25]])
    na,nb,nc=new.T
    prediction=np.column_stack([(1-nc)*na,nc*nb])@second
    print("changed new-row predictions:",np.round(prediction,6).tolist())
    print("This exact constructed relation is not evidence of deployment improvement.")
''')

PROTOCOL = r'''
import numpy as np
from sklearn.base import clone
from sklearn.datasets import make_circles
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

def experiment(seed=71, noise=.14, omit_tree=False):
    x,y=make_circles(n_samples=360,noise=noise,factor=.5,random_state=seed)
    indices=np.arange(len(y))
    train,rest=train_test_split(indices,test_size=.4,stratify=y,random_state=23)
    valid,test=train_test_split(rest,test_size=.5,stratify=y[rest],random_state=29)
    bases=[("line",make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000))),
           ("tree",DecisionTreeClassifier(max_depth=4,random_state=31)),
           ("neighbor",make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=9)))]
    if omit_tree:
        bases=[item for item in bases if item[0]!="tree"]
    candidates={name:clone(model) for name,model in bases}
    candidates["soft"]=VotingClassifier(estimators=bases,voting="soft",n_jobs=1)
    candidates["stack"]=StackingClassifier(estimators=bases,final_estimator=LogisticRegression(C=1,max_iter=2000),
        cv=StratifiedKFold(3,shuffle=True,random_state=37),n_jobs=1)
    candidates["prior"]=DummyClassifier(strategy="prior")
    fitted={name:model.fit(x[train],y[train]) for name,model in candidates.items()}
    validation={name:log_loss(y[valid],model.predict_proba(x[valid])) for name,model in fitted.items()}
    selected=min(validation,key=lambda name:(validation[name],name))
    # Select action threshold using only validation; false negative costs four.
    pv=fitted[selected].predict_proba(x[valid])[:,1]
    thresholds=np.unique(np.r_[0.,pv,1.])
    def cost(threshold):
        action=pv>=threshold
        return int(np.sum(action&(y[valid]==0))+4*np.sum((~action)&(y[valid]==1)))
    threshold=min(thresholds,key=lambda value:(cost(value),value))
    report={}
    # Candidate set and selection are frozen before this test report is opened.
    for name,model in fitted.items():
        p=model.predict_proba(x[test])[:,1]
        report[name]={"validationLogLoss":validation[name],"testLogLoss":log_loss(y[test],np.c_[1-p,p]),
            "testBrier":brier_score_loss(y[test],p),"testAccuracy":accuracy_score(y[test],model.predict(x[test]))}
    p=fitted[selected].predict_proba(x[test])[:,1]
    action=p>=threshold
    matrix=confusion_matrix(y[test],action,labels=[0,1])
    print("split sizes:",len(train),len(valid),len(test))
    for name,values in report.items():
        print(f"{name}: valLL={values['validationLogLoss']:.6f}, testLL={values['testLogLoss']:.6f}, Brier={values['testBrier']:.6f}, accuracy={values['testAccuracy']:.6f}")
    print("selected on validation:",selected)
    print(f"selected validation cost threshold: {threshold:.6f}")
    print("test confusion [TN,FP;FN,TP]:",matrix.tolist())
    print("test action cost:",int(matrix[0,1]+4*matrix[1,0]))
    print("Keep this finite report separate from a guarantee of calibration or superiority.")
    return {"x":x,"y":y,"train":train,"valid":valid,"test":test,"fitted":fitted,"report":report,"selected":selected,"threshold":threshold}

run=experiment()
'''
add("held-out-comparison", "A complete baseline, voting and stacking comparison", "Which rule does validation select, and does its final test result support the same ranking? Keep the decision fixed after viewing test results.", PROTOCOL)
add("changed-protocol", "A checkable changed capstone", "Change the seed to 83, noise to .20 and omit the tree base. What changes in the selected rule, loss and decision cost?", PROTOCOL.replace("run=experiment()", "run=experiment(seed=83,noise=.20,omit_tree=True)"))

for example in examples:
    file=WORK/(example["id"]+".py")
    file.write_text(example["code"],encoding="utf-8")
    execution=subprocess.run([sys.executable,str(file)],capture_output=True,text=True,timeout=180,cwd=ROOT)
    if execution.returncode or execution.stderr.strip():
        raise RuntimeError(example["id"]+"\n"+execution.stderr)
    example["output"]=execution.stdout.rstrip()

# Generate all diagram values from the very same complete displayed experiment.
scope={}
with contextlib.redirect_stdout(io.StringIO()):
    exec(examples[-2]["code"],scope)
run=scope["run"]
np=scope["np"]
coordinates=np.linspace(-1.5,1.5,25)
points=np.array([[a,b] for b in coordinates for a in coordinates])
map_data={
    "provenance":{"kind":"native fitted synthetic experiment","programId":"held-out-comparison","seed":71,"noise":.14,"trainRows":216,"validationRows":72,"testRows":72,"grid":"25 by25 points; displayed cells are local approximations, not exact mathematical boundaries"},
    "coordinates":coordinates.tolist(),
    "models":{name:{"probabilities":model.predict_proba(points)[:,1].tolist(),"report":run["report"][name]} for name,model in run["fitted"].items() if name!="prior"},
    "testPoints":[{"x":float(run["x"][index,0]),"y":float(run["x"][index,1]),"label":int(run["y"][index])} for index in run["test"]],
    "selected":run["selected"]
}
(ROOT/"src/learn/data/ensemble-prediction-map.json").write_text(json.dumps(map_data,separators=(',',':'))+'\n',encoding="utf-8")
(ROOT/"src/learn/data/ensemble-methods-examples.js").write_text('// Generated by scripts/generate-ensemble-examples.py; complete executed programs.\nexport const ensembleExamples = '+json.dumps(examples,indent=2,ensure_ascii=False)+';\n',encoding="utf-8")
print(json.dumps({"examples":[{"id":example["id"],"output":example["output"]} for example in examples],"mapModels":list(map_data["models"]),"gridPoints":len(points)},indent=2))
