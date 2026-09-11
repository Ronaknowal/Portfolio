from pathlib import Path
from datetime import datetime, timezone
from fractions import Fraction as F
import contextlib
import hashlib
import io
import json
import warnings
import numpy as np
import mpmath as mp
from scipy.optimize import minimize_scalar

directory = Path("scratch/linear-logistic-independent-review")
examples = json.loads((directory / "examples.json").read_text(encoding="utf-8"))
namespaces = {}
for e in examples:
    namespace = {}
    output = io.StringIO()
    with contextlib.redirect_stdout(output), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        exec(compile(e["code"], e["id"], "exec"), namespace)
    assert not caught
    assert output.getvalue().rstrip() == e["expected"]
    namespaces[e["id"]] = namespace

# Verify the actual program's objective/gradient under changed rows and both labels,
# using independently differentiated 80-digit likelihoods.
mp.mp.dps = 80
ns = namespaces["logistic-fit"]
rng = np.random.default_rng(613)
gradient_cases = 0
for n in [4,7,13]:
    for _ in range(5):
        xx = rng.integers(-4,5,size=n)
        yy = rng.integers(0,2,size=n)
        ns["design"] = np.column_stack([np.ones(n),xx])
        ns["y"] = yy
        ns["penalty"] = .3
        for theta in [[-.7,.2],[.5,-1.2],[1.1,1.4]]:
            def exact(a,b):
                return sum(mp.log(1+mp.exp((1-2*int(y))*(a+b*int(x)))) for x,y in zip(xx,yy))/n + mp.mpf(".3")*b*b/2
            actual_loss,actual_gradient = ns["objective"](np.array(theta))
            reference = exact(*map(mp.mpf,theta))
            derivative = [mp.diff(lambda a: exact(a,mp.mpf(theta[1])),mp.mpf(theta[0])),
                          mp.diff(lambda b: exact(mp.mpf(theta[0]),b),mp.mpf(theta[1]))]
            assert np.allclose(actual_gradient,np.array(derivative,dtype=float),rtol=2e-12,atol=2e-12)
            assert abs(actual_loss-float(reference))<2e-12
            gradient_cases += 1

# The weighted-probability statement: an independently solved expected-loss problem.
weighted_cases = 0
for q in [F(1,10),F(1,3),F(4,5)]:
    for a,b in [(1,4),(3,1),(2,7),(5,5)]:
        optimum = a*q/(a*q+b*(1-q))
        result=minimize_scalar(lambda z: -float(a*q)*np.log(z)-float(b*(1-q))*np.log1p(-z),
                               bounds=(1e-8,1-1e-8),method="bounded",options={"xatol":1e-12})
        assert abs(result.x-float(optimum))<2e-8
        weighted_cases += 1

# Confirm actual held-out and nonlinear pipelines learned transforms on fit rows.
held = namespaces["held-out-pipeline"]
for _,_,pipeline,_ in held["candidates"]:
    assert np.allclose(pipeline.named_steps["standardscaler"].mean_,held["X_train"].mean(axis=0))
assert len(held["X_test"]) == len(held["y_test"]) == 60
changed=namespaces["changed-report"]
assert len(set(changed["train"]) & set(changed["valid"])) == 0
assert len(set(changed["valid"]) & set(changed["test"])) == 0
assert len(set(changed["train"]) & set(changed["test"])) == 0
for _,_,_,pipeline in changed["candidates"]:
    expanded=pipeline.named_steps["polynomialfeatures"].transform(changed["X"][changed["train"]])
    assert np.allclose(pipeline.named_steps["standardscaler"].mean_,expanded.mean(axis=0))

# Independent leverage/inference calculation: centered exact design at changed locations.
uncertainty=namespaces["uncertainty"]
for location in [F(-1),F(1,2),F(7,2),F(9)]:
    h=F(1,6)+(location-F(5,2))**2/F(35,2)
    query=np.array([1,float(location)])
    mean_variance=query@uncertainty["covariance"]@query
    assert abs(mean_variance-uncertainty["variance"]*float(h))<1e-13
    assert uncertainty["variance"]+mean_variance>mean_variance

result={"checkedAt":datetime.now(timezone.utc).isoformat(),
        "programs":[{"id":e["id"],"sha256":hashlib.sha256(e["code"].encode()).hexdigest(),"stdoutMatch":True} for e in examples],
        "changedObjectiveGradientStates":gradient_cases,"weightedRiskMinima":weighted_cases,
        "fitOnlyScalerChecks":7,"exactCenteredLeverageLocations":4,
        "scope":"Complementary checks of actual displayed source, not a duplicate of the author suite"}
(directory/"native.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
print(json.dumps(result,indent=2))
