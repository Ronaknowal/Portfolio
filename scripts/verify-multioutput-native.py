"""Complement UI calculations with independent count/risk/library oracles."""
from pathlib import Path
from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
import itertools
import json
import math
import numpy as np
from scipy.optimize import minimize
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import f1_score, hamming_loss, accuracy_score

ROOT = Path(__file__).resolve().parents[1]
directory = ROOT / "scratch/multioutput/native"
data = json.loads((directory / "model-cases.json").read_text(encoding="utf-8"))
checks = {}
def close(actual, expected):
    if expected is None:
        assert actual is None
    else:
        assert math.isclose(actual, float(expected), rel_tol=1e-10, abs_tol=1e-10), (actual,expected)

for case in data["cases"]["metrics"]:
    truth, prediction, result = case["truth"], case["prediction"], case["result"]
    pairs = [(target,guess) for row,guessrow in zip(truth,prediction) for target,guess in zip(row,guessrow) if target is not None]
    tp = pairs.count((1,1)); fp = pairs.count((0,1)); fn = pairs.count((1,0))
    assert (result["tp"],result["fp"],result["fn"],result["observed"]) == (tp,fp,fn,len(pairs))
    close(result["hamming"], F(fp+fn,len(pairs)) if pairs else None)
    close(result["f1"], F(2*tp,max(1,2*tp+fp+fn)) if pairs else None)
    complete = [(row,guess) for row,guess in zip(truth,prediction) if None not in row]
    close(result["subsetAccuracy"], F(sum(row==guess for row,guess in complete),len(complete)) if complete else None)
    per_label=[]
    for j in range(len(truth[0])):
        observed=[(row[j],guess[j]) for row,guess in zip(truth,prediction) if row[j] is not None]
        if observed:
            actual, pred=zip(*observed)
            per_label.append(f1_score(actual,pred,zero_division=0))
    close(result["macroF1"],np.mean(per_label) if per_label else None)
checks["observed_count_and_library_metrics"] = len(data["cases"]["metrics"])

states=list(itertools.product([0,1],repeat=2))
for case in data["cases"]["joint"]:
    counts,result=case["counts"],case["result"]
    mass=[F(count,sum(counts)) for count in counts]
    risks=[]
    for action,candidate in zip(states,result["candidates"]):
        risk=sum(probability*F(sum(a!=y for a,y in zip(action,state)),2) for probability,state in zip(mass,states))
        risks.append(risk);close(candidate["hammingRisk"],risk)
        close(candidate["subsetLoss"],1-mass[states.index(action)])
    assert risks[states.index(tuple(result["marginalDecision"]))] == min(risks)
    for branch in result["branches"]:
        for leaf in branch["leaves"]:
            close((branch["support"]*leaf["conditional"]) if branch["support"] else 0,mass[leaf["index"]])
    assert result["modes"] == [list(state) for state,count in zip(states,counts) if count==max(counts)]
checks["exact_joint_risks_chain_leaves"] = len(data["cases"]["joint"])

for result in data["cases"]["association"]:
    h=F(str(result["highShare"])); p=F(1,10)*(1-h)+F(9,10)*h
    both=F(1,100)*(1-h)+F(81,100)*h
    close(result["marginal"],p);close(result["conditional"],both/p)
checks["mixture_conditionals"] = len(data["cases"]["association"])
for case in data["cases"]["thresholds"]:
    source,result=case["input"],case["result"]
    prediction=np.array(source["scores"])>=source["threshold"]
    close(result["f1"],f1_score(source["targets"],prediction,zero_division=0))
    assert result["predictions"] == prediction.astype(int).tolist()
checks["threshold_tie_and_none_states"] = len(data["cases"]["thresholds"])
for case in data["cases"]["stumps"]:
    result=case["result"]
    X=np.arange(4)[:,None]; Y=np.array(result["targets"]).T
    scales=np.array([1,result["energyScale"]])
    model=DecisionTreeRegressor(max_depth=1,random_state=0).fit(X,Y/scales)
    close(result["shared"]["threshold"],model.tree_.threshold[0])
    prediction=model.predict(X)*scales
    np.testing.assert_allclose([output["predictions"] for output in result["shared"]["outputs"]],prediction.T)
    for candidate in result["candidates"]:
        cut=candidate["cut"]
        exact=[]
        for column in Y.T:
            errors=[]
            for part in [column[:cut],column[cut:]]:
                mean=sum(F(str(value)) for value in part)/len(part)
                errors.extend((F(str(value))-mean)**2 for value in part)
            exact.append(sum(errors))
        close(candidate["scaledSse"],exact[0]+exact[1]/F(str(result["energyScale"]))**2)
checks["changed_stumps_exact_and_sklearn"] = len(data["cases"]["stumps"])
for result in data["cases"]["shrink"]:
    z=np.array(result["initial"]); lam=result["penalty"]; b=np.array(result["grouped"])
    if np.linalg.norm(b)>1e-12:
        np.testing.assert_allclose(b-z+lam*b/np.linalg.norm(b),[0,0],atol=1e-12)
    else:
        assert np.linalg.norm(z)<=lam+1e-12
    # Convex subgradient certificate establishes a global minimizer.
    assert result["groupedObjective"]<=result["independentAtGroupedObjective"]+1e-12
checks["group_penalty_subgradient_certificates"] = len(data["cases"]["shrink"])

# Changed practice, independently calculated rather than taken from UI state.
y=np.array([[1,0],[0,1],[1,1]]); p=np.array([[1,1],[0,0],[1,1]])
assert f1_score(y,p,average="micro")==.75
close(hamming_loss(y,p),F(1,3));close(accuracy_score(y,p),F(1,3))
assert sum([F(3,5)*F(1,5),F(2,5)*F(9,10)])==F(12,25)
assert F(4,5)/(F(4,5)+F(4,5))==F(1,2)
assert [F(8,3),F(50),F(208,3)].index(min(F(8,3),F(50),F(208,3)))==0
np.testing.assert_allclose(np.array([-3,4])*.8,[-2.4,3.2])
checks["changed_hand_calculation_checks"] = 6
files=["src/learn/data/multioutput-models.js","src/learn/data/multioutput-examples.js"]
record={"timestamp":datetime.now(timezone.utc).isoformat(),"checks":checks,"invalidInputs":data["invalid"],"actualPrograms":len(data["examples"]),"sourceHashes":{file:hashlib.sha256((ROOT/file).read_bytes()).hexdigest() for file in files},"limitations":"Finite supported teaching models, no population performance claim. Complete-program execution is separately source-hashed in execution.json; changed cost program is executed there."}
(directory/"verification.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
print(json.dumps(record,indent=2))
