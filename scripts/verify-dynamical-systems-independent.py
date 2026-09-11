"""Complementary reviewer checks; no production edits or author-test imports."""
import contextlib
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import platform
import subprocess

import numpy as np
import scipy
from scipy.integrate import solve_ivp, quad

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "scratch/dynamical-systems-independent-review"
OUT.mkdir(parents=True, exist_ok=True)
export = r"""
import * as m from './src/learn/data/dynamical-systems-models.js';
import { dynamicalSystemsExamples as examples } from './src/learn/data/dynamical-systems-examples.js';
const hopf=[];
for(const a of [-.7,-1e-14,0,1e-14,.15,.6]) for(const start of [[.35,-.8],[-.6,.25]]) {
  const input={mode:'hopf',parameter:a,initial:start,duration:2*Math.PI,steps:80};
  hopf.push({input,state:m.planarTrace(input)});
}
const oscillators=[];
for(const step of [.125,.375,1.25,2,2.125]) for(const initial of [[-.25,.75],[1.5,-.5]]) {
  const input={step,initial,steps:16};
  oscillators.push({input,state:m.oscillatorTrace(input)});
}
const words=['LLRLRRLRRL','RRLLRLRLLR','LRLRLRLRLR','RRRRRRRRRR','LLLLLLLLLL'];
const cylinders=words.map(word=>m.tentCylinder(word));
const coordinate=[.03,.11,.27,.63,.89,.98].map(x=>({x,cdf:m.logisticInvariantCdf(x)}));
console.log(JSON.stringify({examples,hopf,oscillators,cylinders,coordinate}));
"""
result = subprocess.run(["node", "--input-type=module", "-e", export], cwd=ROOT,
                        capture_output=True, encoding="utf-8", check=True)
data = json.loads(result.stdout)
spaces = {}
counts = {"actual_programs": 0, "hopf_radial_and_angle": 0,
          "exact_modified_forms": 0, "native_modified_forms": 0,
          "ten_branch_cylinders": 0, "invariant_probability_integrals": 0,
          "lorenz_variational_volume": 0, "native_changed_cubic": 0}
errors = []


def close(actual, expected, atol=1e-10, rtol=1e-10):
    a, b = np.asarray(actual), np.asarray(expected)
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)
    errors.append(float(np.max(np.abs(a-b))))


for key, example in data["examples"].items():
    program = OUT / (key + ".py")
    program.write_text(example["code"], encoding="utf-8")
    completed = subprocess.run([__import__("sys").executable, str(program)],
                               capture_output=True, encoding="utf-8", check=True)
    assert completed.stdout.rstrip("\n") == example["expected"], key
    assert completed.stderr == ""
    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(example["code"], str(program), "exec"), namespace)
    spaces[key] = namespace
    counts["actual_programs"] += 1
original = json.loads((ROOT / "docs/teaching/evidence/dynamical-systems-original-content.json").read_text(encoding="utf-8"))
assert data["examples"]["original"]["code"] == original["blocks"][0]["code"]
assert data["examples"]["original"]["expected"] == original["blocks"][1]["code"]

for case in data["hopf"]:
    a = case["input"]["parameter"]
    initial = case["input"]["initial"]
    radius0 = math.hypot(*initial)
    for row in case["state"]["rows"]:
        time = row["time"]
        # Solve the linear equation for inverse squared radius with 90 digits.
        # This tests the author's expm1/z expression by a separate state variable.
        with localcontext() as ctx:
            ctx.prec = 90
            A, T = Decimal(str(a)), Decimal(str(time))
            inverse0 = 1 / sum(Decimal(str(v))**2 for v in initial)
            inverse = inverse0 + 2*T if A == 0 else (
                1/A + (inverse0 - 1/A)*(-2*A*T).exp())
            radius = float(1 / inverse.sqrt())
        close(row["radius"], radius, atol=2e-14, rtol=2e-13)
        rotation = np.array([[math.cos(time), -math.sin(time)],
                             [math.sin(time), math.cos(time)]])
        close(row["position"], rotation @ initial * radius/radius0,
              atol=2e-14, rtol=2e-13)
        close(spaces["hopf"]["radius_at"](a, radius0, time), radius,
              atol=2e-14, rtol=2e-13)
        counts["hopf_radial_and_angle"] += 1


def exact_oscillator(step, initial, count):
    h = Fraction(step)
    q, p = map(Fraction, initial)
    modified0 = (q*q+p*p-h*q*p)/2
    for _ in range(count):
        # Matrix multiplication independently expresses kick-then-drift.
        q, p = (1-h*h)*q+h*p, -h*q+p
        assert (q*q+p*p-h*q*p)/2 == modified0
    return q, p, modified0


for case in data["oscillators"]:
    args = case["input"]
    for n, row in enumerate(case["state"]["rows"]):
        q, p, invariant = exact_oscillator(args["step"], args["initial"], n)
        close(row["symplectic"], [float(q), float(p)], atol=1e-9)
        # At h>2, large unstable coordinates cancel in this indefinite form.
        # Use a magnitude-based binary64 evaluation allowance, not a fixed
        # absolute tolerance that pretends cancellation cannot lose digits.
        form_scale = float(q*q+p*p+abs(Fraction(args["step"])*q*p))/2
        close(row["modifiedEnergy"], float(invariant),
              atol=max(1e-10, 16*np.finfo(float).eps*form_scale))
        counts["exact_modified_forms"] += 1
for h, start, n in itertools.product([.125,.375], [(-.75,.25),(.5,1.25)], [0,7,31]):
    q, p, invariant = exact_oscillator(h, start, n)
    _, energy, modified = spaces["energy"]["integrate_oscillator"](h, n, start)
    close(energy, float((q*q+p*p)/2))
    close(modified, float(invariant))
    counts["native_modified_forms"] += 1

for case in data["cylinders"]:
    word = case["word"]
    # Pull each endpoint backward, composing from the LAST itinerary branch.
    def pullback(value):
        for branch in reversed(word):
            value = value/2 if branch == "L" else 1-value/2
        return value
    lo, hi = sorted([pullback(Fraction(0)), pullback(Fraction(1))])
    close(case["interval"], [float(lo), float(hi)], atol=0, rtol=0)
    assert hi-lo == Fraction(1, 2**len(word))
    zero, one = pullback(Fraction(0)), pullback(Fraction(1))
    periodic = zero / (1-(one-zero))
    close(case["periodicPoint"], float(periodic), atol=1e-15)
    value = periodic
    for branch in word:
        assert value <= Fraction(1,2) if branch == "L" else value >= Fraction(1,2)
        value = 2*value if value <= Fraction(1,2) else 2*(1-value)
    assert value == periodic
    counts["ten_branch_cylinders"] += 1
for case in data["coordinate"]:
    mass = quad(lambda x: 1/(math.pi*math.sqrt(x*(1-x))), 0, case["x"],
                epsabs=1e-12, epsrel=1e-12)[0]
    close(case["cdf"], mass, atol=3e-12)
    counts["invariant_probability_integrals"] += 1

for rho, initial, horizon in itertools.product([.8,7.5,24], [[.2,-.3,1.2]], [.025,.1,.4]):
    def augmented(time, combined):
        x,y,z = combined[:3]
        jacobian = np.array([[-10,10,0],[rho-z,-1,-x],[y,x,-8/3]])
        velocity = spaces["lorenz"]["lorenz"](time, combined[:3], rho)
        return np.r_[velocity, (jacobian @ combined[3:].reshape(3,3)).ravel()]
    solved = solve_ivp(augmented, (0,horizon), np.r_[initial,np.eye(3).ravel()],
                       method="DOP853", rtol=2e-12, atol=2e-14, max_step=.002)
    assert solved.success
    matrix = solved.y[3:,-1].reshape(3,3)
    sign, logdet = np.linalg.slogdet(matrix)
    assert sign == 1
    close(logdet, -41*horizon/3, atol=4e-10)
    assert np.linalg.svd(matrix, compute_uv=False)[0] > 0
    independent = spaces["lorenz"]["solve_lorenz"](initial,rho,horizon,1e-11)
    close(independent.y[:,-1], solved.y[:3,-1], atol=4e-10)
    counts["lorenz_variational_volume"] += 1
for initial, time, attracting in itertools.product([-.6,.15,.9], [.02,.25], [True,False]):
    field = lambda t,y: -y**3 if attracting else y**3
    solved = solve_ivp(field,(0,time),[initial],method="DOP853",rtol=1e-12,atol=1e-14)
    close(spaces["scalar"]["cubic_solution"](initial,time,attracting),solved.y[0,-1],atol=1e-11)
    counts["native_changed_cubic"] += 1

freeze = json.loads((ROOT / "docs/teaching/evidence/dynamical-systems-author-review.json").read_text(encoding="utf-8"))
for item in freeze["production"]:
    assert hashlib.sha256((ROOT/item["path"]).read_bytes()).hexdigest() == item["sha256"], item["path"]
record = {"verifiedAt":datetime.now(timezone.utc).isoformat(), "passed":True,
          "environment":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__},
          "authorFrozenAt":freeze["authorFrozenAt"], "production":freeze["production"],
          "cases":counts,"numericAssertions":len(errors),"maxAbsoluteDiscrepancy":max(errors),
          "originalProgramPreserved":True,
          "limits":"Complementary finite oracles and mathematical source review, not a new long-time chaos proof, exhaustive numerical API certification, or duplicate full browser run."}
(OUT/"results.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
print(json.dumps(record,indent=2))
