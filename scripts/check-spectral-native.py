"""Bounded native replay and derivative checks for the spectral lesson; no refits."""
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import re
import sys
from pathlib import Path
import numpy as np
import scipy
import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm
from torch.nn.utils.parametrize import remove_parametrizations

ROOT = Path(__file__).resolve().parents[1]
ID = "spectral-normalization-gradient-penalty"
DRAFT = ROOT / "docs/teaching/drafts" / ID
OUT = ROOT / "docs/teaching/deep-learning-completion" / ID
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)
checks = []
def check(name, actual, expected, tolerance=1e-6):
    error = float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
    assert error <= tolerance, (name, error, tolerance)
    checks.append(dict(name=name, passed=True, maxAbsoluteError=error, tolerance=tolerance))

def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, DRAFT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
study = load("spectral_study", "critic-regularization-study.py")
sensitivity = load("spectral_sensitivity", "sensitivity-calculations.py")
torch.set_num_threads(2)
source = (DRAFT/"lesson.md").read_text(encoding="utf-8")
fence = chr(96)*3
embedded = re.search(fence+r"python\n([\s\S]*?)"+fence, source).group(1)
assert embedded.strip() == (DRAFT/"critic-regularization-study.py").read_text(encoding="utf-8").strip()
checks.append(dict(name="Complete inline training program equals canonical executable", passed=True))
recorded = json.loads((DRAFT/"calculated-inputs.json").read_text())
assert hashlib.sha256((DRAFT/"digits-400.csv").read_bytes()).hexdigest() == recorded["protocol"]["csv_sha256"]
rows = list(csv.DictReader((DRAFT/"digits-400.csv").open()))
pixels = np.array([[int(row[f"pixel_{j}"]) for j in range(64)] for row in rows]).reshape(-1,8,8)
ids = np.array([int(row["source_id"]) for row in rows])
profiles = np.c_[pixels[:,:,:4].sum((1,2)),pixels[:,:,4:].sum((1,2))]/512
check("Every original image produces its recorded ink profile", profiles, recorded["measurements"], 0)
groups = {}
for i,pair in enumerate(map(tuple,profiles)): groups.setdefault(pair,[]).append(i)
order = np.random.default_rng(91).permutation(len(groups))
keys = list(groups)
roles = {name:np.array([i for g in selected for i in groups[keys[g]]]) for name, selected in
         zip(["fit","development","assessment"],[order[:236],order[236:315],order[315:]])}
for name,index in roles.items(): check("Whole-profile group split "+name, ids[index],recorded["protocol"]["roles"][name],0)
bootstrap = profiles[np.random.default_rng(2026).choice(roles["fit"],256,replace=True)]
check("Every empirical bootstrap point", bootstrap,recorded["bootstrap"]["generated"],0)
for role,index in roles.items():
    check("Bootstrap directional discrepancy "+role,study.projected_distance(bootstrap,profiles[index]),recorded["bootstrap"]["metrics"][role],1e-12)

# Run the complete companion program, routing only its output away from historical
# evidence. Input reads still use the prepared study.
class CalculationPaths:
    def __truediv__(self, name):
        return DRAFT/name if name=="calculated-inputs.json" else OUT/("native-"+name)
sensitivity.ROOT = CalculationPaths()
with contextlib.redirect_stdout(io.StringIO()):
    sensitivity.main()
checks.append(dict(name="Complete sensitivity program executed, including SVD gradient and parametrization export",passed=True))

fixtures = dict(matrix=[],convolution=[],linear=[],probes=[],frozen=[],library=[])
for w in [[[2.,1.],[0.,1.]],[[0.,0.],[0.,0.]],[[1.,2.],[2.,4.]],[[-3.,2.],[.4,-1.]],[[.2,0],[0,.2]]]:
    values = np.linalg.svd(w,compute_uv=False)
    fixtures["matrix"].append(dict(weight=w,singular=values.tolist()))
for kernel in [[1.,1.],[1.,2.],[-2.,.7],[0.,0.]]:
    a,b=kernel
    for mode,matrix in [("valid",[[a,b,0],[0,a,b]]),("disjoint",[[a,b,0,0],[0,0,a,b]]),("circular",[[a,b,0,0],[0,a,b,0],[0,0,a,b],[b,0,0,a]])]:
        fixtures["convolution"].append(dict(kernel=kernel,mode=mode,matrix=matrix,norm=float(np.linalg.svd(matrix,compute_uv=False)[0])))
for values in [[3.,4.],[2.,-1.],[0.,0.],[.2,.1]]:
    for kind in ["target-one","one-sided","zero"]:
        w=torch.tensor(values,dtype=torch.float64,requires_grad=True)
        r=w.norm()
        penalty=2*(r*r if kind=="zero" else torch.relu(r-1)**2 if kind=="one-sided" else (r-1)**2)
        derivative=torch.autograd.grad(penalty,w)[0]
        fixtures["linear"].append(dict(weight=values,kind=kind,penalty=float(penalty.detach()),gradient=derivative.tolist(),zeroConvention=values==[0.,0.]))
for values in [[-.5,.5,1.5],[-.5,.5,2.5],[-1.,2.,4.]]:
    fixtures["probes"].append(dict(points=values,slopes=[None if x==2 else 1+4*(x>2) for x in values]))

for fit in recorded["fits"]:
    key=fit["method"]+"-"+str(fit["seed"])
    g=sensitivity.forward(recorded["evaluation_latents"],fit["generator_layers"],"generator")
    d=sensitivity.forward(profiles,fit["critic_layers"],"critic").ravel()
    check(key+" all256 generator outputs",g,fit["generated"])
    check(key+" all400 critic scores",d,fit["critic_values"])
    for role,index in roles.items():
        check(key+" "+role+" discrepancy from current frozen outputs",study.projected_distance(g,profiles[index]),fit["metrics"][role])
    for history in fit["history"]:
        check(key+" history "+str(history["step"])+" discrepancy",study.projected_distance(history["generated"],profiles[roles["development"]]),history["development_projected_w1"],1e-12)
    points=torch.tensor(fit["grid_coordinates"],dtype=torch.float64,requires_grad=True)
    value=points
    for j,layer in enumerate(fit["critic_layers"]):
        value=nn.functional.linear(value,torch.tensor(layer["weight"],dtype=torch.float64),torch.tensor(layer["bias"],dtype=torch.float64))
        if j<2: value=nn.functional.leaky_relu(value,.2)
    gradient=torch.autograd.grad(value.sum(),points)[0]
    check(key+" all1681 grid scores",value.detach().numpy().ravel(),fit["grid_score"])
    check(key+" all1681 grid input derivatives",gradient.detach().numpy(),fit["grid_gradient"])
    spectra=[np.linalg.svd(layer["weight"],compute_uv=False) for layer in fit["critic_layers"]]
    check(key+" matrix product bound",np.prod([s[0] for s in spectra]),fit["matrix_product_bound"],2e-6)
    fresh=[[.13,.77],[-.4,1.2],[0.,0.],[.9,.08]]
    p=torch.tensor(fresh,dtype=torch.float64,requires_grad=True);v=p
    for j,layer in enumerate(fit["critic_layers"]):
        v=nn.functional.linear(v,torch.tensor(layer["weight"],dtype=torch.float64),torch.tensor(layer["bias"],dtype=torch.float64))
        if j<2:v=nn.functional.leaky_relu(v,.2)
    grad=torch.autograd.grad(v.sum(),p)[0]
    fixtures["frozen"].append(dict(key=key,points=fresh,critic=v.detach().tolist(),gradients=grad.tolist(),generated=sensitivity.forward(fresh,fit["generator_layers"],"generator").tolist()))

# Actual library state trace with deliberately controlled buffers.
layer=spectral_norm(nn.Linear(2,2,bias=False,dtype=torch.float64),n_power_iterations=1)
param=layer.parametrizations.weight[0]
with torch.no_grad():
    layer.parametrizations.weight.original.copy_(torch.tensor([[2.,1.],[0.,1.]],dtype=torch.float64))
    param._u.copy_(torch.tensor([1.,1.],dtype=torch.float64)/np.sqrt(2))
    param._v.copy_(torch.tensor([1.,1.],dtype=torch.float64)/np.sqrt(2))
for training in [True,True,False,False,True]:
    layer.train(training);effective=layer.weight
    fixtures["library"].append(dict(training=training,u=param._u.tolist(),v=param._v.tolist(),effective=effective.detach().tolist()))
check("Evaluation access preserves singular buffers",fixtures["library"][2]["u"],fixtures["library"][3]["u"],0)
layer.eval();before=layer(torch.tensor([[.2,.7]],dtype=torch.float64)).detach()
remove_parametrizations(layer,"weight",leave_parametrized=True)
check("Actual library export preserves function",before,layer(torch.tensor([[.2,.7]],dtype=torch.float64)).detach(),0)

# Exercise canonical models, GP, and optimizer code for one bounded update.
# This is a graph check, not a rerun of the nine saved fitting campaigns.
for method in ["clipping","gradient-penalty","spectral-normalization"]:
    torch.manual_seed(17);g,d=study.Generator(),study.Critic()
    if method=="spectral-normalization":
        for layer in d.layers:spectral_norm(layer,n_power_iterations=1)
    og=torch.optim.Adam(g.parameters(),lr=.001,betas=(0.,.9));od=torch.optim.Adam(d.parameters(),lr=.001,betas=(0.,.9))
    real=torch.tensor(profiles[:8],dtype=torch.float32)
    fake=g(torch.randn(8,2)).detach();scores=d(torch.cat([real,fake]))
    penalty=study.gradient_penalty(d,real,fake,torch.Generator().manual_seed(71)) if method=="gradient-penalty" else scores.new_zeros(())
    loss=scores[8:].mean()-scores[:8].mean()+10*penalty
    old=[p.detach().clone() for p in d.parameters()]
    od.zero_grad();loss.backward();od.step()
    assert any(not torch.equal(a,b) for a,b in zip(old,d.parameters()))
    assert all(p.grad is None for p in g.parameters())
    if method=="clipping":
        with torch.no_grad():
            for p in d.parameters():p.clamp_(-.1,.1)
    d.eval();d.requires_grad_(False);og.zero_grad()
    old=[p.detach().clone() for p in g.parameters()]
    loss=-d(g(torch.randn(8,2))).mean();loss.backward();og.step()
    assert any(not torch.equal(a,b) for a,b in zip(old,g.parameters()))
    assert all(torch.isfinite(p).all() for p in list(g.parameters())+list(d.parameters()))
    checks.append(dict(name=method+" canonical D/G optimizer steps, detached fake, retained generator derivative and finite parameters",passed=True))

(OUT/"native-fixtures.json").write_text(json.dumps(fixtures,indent=2,allow_nan=False)+"\n")
(OUT/"native-checks.json").write_text(json.dumps(dict(topicId=ID,passed=True,checks=checks,environment=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,maximumThreads=2),limits=["Original nine600-step fits reused; all final outputs, metrics and recorded histories replayed. One bounded actual D/G update per method checks derivative/optimizer contracts.","No new GAN benchmark, GPU/mixed-precision claim or browser acceptance."]),indent=2)+"\n")
print(len(checks),"native spectral checks passed")
