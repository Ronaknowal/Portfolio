"""Bounded complementary review of actual frozen Differential Geometry helpers.

Run with scratch/lesson-tools/Scripts/python.exe from the repository root.
The author suites are deliberately not imported: rotations, solid angles,
congruence identities and restricted eigenspaces supply different checks.
"""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone

import numpy as np
from scipy.linalg import expm, logm
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "scratch/differential-geometry-independent-review"
OUT.mkdir(parents=True, exist_ok=True)

export = r"""
import * as m from './src/learn/data/differential-geometry-models.js';
import {differentialGeometryExamples} from './src/learn/data/differential-geometry-examples.js';
const states=[];
for(const wedge of [25,55,85,115,145])
  for(const reverse of [false,true])
    for(const initial of [-135,30,120])
      states.push({wedge,reverse,initial,state:m.transportTriangle(wedge,initial,3,reverse,2.5)});
const metrics=[];
for(const shear of [-1.3,-.4,.7,1.4]) for(const cost of [.6,1.3,2.7])
  metrics.push({shear,cost,state:m.metricDifferential(shear,cost,137)});
console.log(JSON.stringify({examples:differentialGeometryExamples,states,metrics}));
"""
payload = json.loads(subprocess.check_output(["node", "--input-type=module", "-e", export], cwd=ROOT, text=True))
archive = json.loads((ROOT / "docs/teaching/evidence/differential-geometry-original-content.json").read_text())
assert payload["examples"]["original"]["code"] == archive["blocks"][0]["text"]
assert payload["examples"]["original"]["expected"] == archive["blocks"][1]["text"]
packet = json.loads((ROOT / "docs/teaching/evidence/differential-geometry-author-review.json").read_text())
for source in packet["sources"]:
    assert hashlib.sha256((ROOT/source["path"]).read_bytes()).hexdigest() == source["sha256"]

namespaces = {}
for key, example in payload["examples"].items():
    namespace = {}
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(example["code"], f"displayed-{key}.py", "exec"), namespace)
    assert stdout.getvalue().strip() == example["expected"].strip(), (key, stdout.getvalue())
    namespaces[key] = namespace

max_error = 0.0
def close(actual, expected, atol=3e-10):
    global max_error
    difference = float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
    max_error = max(max_error, difference)
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=3e-10)

def skew(axis):
    x, y, z = axis
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])

def rotate_between(x, y):
    axis = np.cross(x, y)
    sine = np.linalg.norm(axis)
    if sine == 0:
        return np.eye(3)
    angle = np.arctan2(sine, x@y)
    return expm(skew(axis/sine)*angle)

rng = np.random.default_rng(92371)
map_cases = 0
for _ in range(48):
    x = rng.normal(size=3)
    x /= np.linalg.norm(x)
    direction = rng.normal(size=3)
    direction -= (direction@x)*x
    direction /= np.linalg.norm(direction)
    length = rng.uniform(.025, 2.6)
    tangent = direction*length
    expected = expm(skew(np.cross(x, direction))*length)@x
    actual = namespaces["maps"]["sphere_exp"](x, tangent)
    close(actual, expected)
    close(namespaces["maps"]["sphere_log"](x, actual), tangent)
    arrow = rng.normal(size=3)
    arrow -= (arrow@x)*x
    close(namespaces["transport"]["transport"](x, actual, arrow), rotate_between(x, actual)@arrow)
    map_cases += 1

triangle_cases = 0
for entry in payload["states"]:
    state = entry["state"]
    vertices = np.asarray(state["vertices"])/2.5
    arrow = np.array(state["initial"])
    initial = arrow.copy()
    for x,y in zip(vertices, vertices[1:]):
        arrow = rotate_between(x,y)@arrow
    close(arrow, state["finalVector"])
    a,b,c = vertices[:3]
    # Oriented solid angle of a short spherical triangle, independent of
    # this fixture's meridian/wedge area expression.
    solid_angle = 2*np.arctan2(np.linalg.det(np.stack([a,b,c])), 1+a@b+b@c+c@a)
    turn = np.arctan2(a@np.cross(initial,arrow), initial@arrow)
    close(turn, solid_angle)
    close(turn, state["finalTurn"])
    close(abs(solid_angle)*2.5**2, state["enclosedArea"])
    triangle_cases += 1

for entry in payload["metrics"]:
    shear,cost,state = entry["shear"],entry["cost"],entry["state"]
    S = np.array([[1.,shear],[0.,1.]])
    physical_metric = np.diag([1.,cost**2])
    expected = np.linalg.solve(physical_metric,[2.,-1.])
    close(state["worldGradient"], expected)
    close(S@state["gradient"], expected)
    for coordinate in state["coordinateEllipse"][::16]:
        world = S@coordinate
        close(world@physical_metric@world,1)

spd_cases = 0
for _ in range(24):
    factor_a, factor_b = rng.normal(size=(2,2)),rng.normal(size=(2,2))
    A,B = factor_a@factor_a.T+np.eye(2),factor_b@factor_b.T+2*np.eye(2)
    P = np.array([[1.2,.35],[-.2,.85]])
    midpoint = namespaces["spd"]["affine_path"](A,B,.5)
    # The positive midpoint uniquely solves X A^{-1} X=B.
    close(midpoint@np.linalg.solve(A,midpoint),B)
    mapped = namespaces["spd"]["affine_path"](P@A@P.T,P@B@P.T,.5)
    close(mapped,P@midpoint@P.T)
    distance = namespaces["spd"]["affine_distance"](A,B)
    close(namespaces["spd"]["affine_distance"](P@A@P.T,P@B@P.T),distance)
    for fraction in [.2,.7]:
        X = namespaces["spd"]["affine_path"](A,B,fraction)
        close(namespaces["spd"]["affine_distance"](A,X),fraction*distance)
    # The log-Euclidean midpoint generally differs for noncommuting inputs.
    log_midpoint = expm((logm(A)+logm(B))/2)
    assert np.linalg.norm(log_midpoint-midpoint)>1e-9
    spd_cases += 1

optimization = []
for initial in ([0.,1.,1.],[1.,0.,1.],[0.,0.,1.],[1.,2.,3.]):
    A = np.diag([2.,5.,11.])
    answer,status,history = namespaces["optimize"]["minimize_rayleigh"](A,initial,tolerance=1e-6)
    nonzero = np.flatnonzero(initial)
    restricted_minimum = np.diag(A)[nonzero].min()
    close(answer@A@answer,restricted_minimum,2e-10)
    assert status == "stationary"
    assert all(b[0] <= a[0]+1e-12 for a,b in zip(history,history[1:]))
    optimization.append({"initial":initial,"status":status,"value":float(answer@A@answer),"restrictedReference":float(restricted_minimum)})

fisher_cases = 0
for lower,upper in [(.001,.12),(.03,.94),(.21,.37),(.81,.998)]:
    measured = quad(lambda p:1/np.sqrt(p*(1-p)),lower,upper,epsabs=1e-11)[0]
    expected = namespaces["fisher"]["fisher_coordinate"](upper)-namespaces["fisher"]["fisher_coordinate"](lower)
    close(measured,expected)
    fisher_cases += 1

results = {
    "checkedAt":datetime.now(timezone.utc).isoformat(),"passed":True,
    "authorFreeze":packet["frozenAt"],"sources":packet["sources"],
    "actualPrograms":len(namespaces),"originalCodeOutputConserved":True,"rotatedMapTransportCases":map_cases,
    "orientedSolidAngleLoops":triangle_cases,"metricCases":len(payload["metrics"]),
    "noncommutingCongruenceCases":spd_cases,"restrictedOptimizerCases":optimization,
    "FisherArcIntegrals":fisher_cases,"maximumAbsoluteDiscrepancy":max_error,
    "limits":"Bounded complementary cases, not arbitrary-range floating-point certification or a rerun of the author suite."
}
(OUT/"results.json").write_text(json.dumps(results,indent=2)+"\n")
print(json.dumps({k:v for k,v in results.items() if k != "sources"},indent=2))
