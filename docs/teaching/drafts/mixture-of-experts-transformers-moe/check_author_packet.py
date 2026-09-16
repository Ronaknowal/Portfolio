"""Bounded completed-manuscript checks; no model fitting."""
from pathlib import Path
import csv, hashlib, json, re
import numpy as np
import torch
from moe_calculations import route_fixture,capacity
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
lesson=(HERE/"lesson.md").read_text(encoding="utf-8-sig")
spec=(HERE/"visual-specifications.md").read_text(encoding="utf-8-sig")
source=(HERE/"moe_study.py").read_text()
embedded=re.findall(r"~~~python\n(.*?)~~~",lesson,re.S)
assert len(embedded)==1 and embedded[0].strip()==source.strip()
namespace={"__name__":"author_code_check","__file__":str(HERE/"moe_study.py")}
exec(compile(embedded[0],str(HERE/"lesson.md"),"exec"),namespace)
fits=json.loads((HERE/"fitted-models.json").read_text())
model=namespace["DigitTransformer"](True)
model.load_state_dict({k:torch.tensor(v) for k,v in fits["moe_001_17"].items()})
model.eval()
calculated=json.loads((HERE/"calculated-inputs.json").read_text())
fresh=calculated["trained_fixtures"]["7"]["variants"]["clean"]
with torch.no_grad():
    logits=model(torch.tensor([fresh["pixels"]]))[0]
assert np.max(abs(logits.numpy()-np.array(fresh["logits"])))==0

rows=list(csv.DictReader((HERE/"optical-digits.csv").open()))
assert len(rows)==950
assert {role:sum(r["role"]==role for r in rows) for role in ("fit","validation","assessment")}==dict(fit=500,validation=150,assessment=300)
assert len({tuple(r[f"pixel_{j}"] for j in range(64)) for r in rows})==950
assert hashlib.sha256((HERE/"optical-digits.csv").read_bytes()).hexdigest()=="9ad440c38a738b66ba8ca85ed9097ccafe9e99846517f02cdd6512a79dde9c24"
original=REPO/"src/learn/data/topics/mixture-of-experts-transformers-moe.jsx"
assert hashlib.sha256(original.read_bytes()).hexdigest()=="fad5a8e9f48f6d51a7a94825aa09ded73e7ef84443253762e4659d6e64d0f230"
plain=re.sub(r"~~~.*?~~~","",lesson,flags=re.S)
assert plain.count(r"\(")==plain.count(r"\)") and plain.count(r"\[")==plain.count(r"\]")
assert lesson.count("<details>")==lesson.count("</details>")==14
assert not re.search(r"<details[^>]*\bopen\b",lesson)
assert len(re.findall(r"\*\*Figure \d+",lesson))==11
assert len(re.findall(r"\*\*Investigation \d+",lesson))==5
assert len(re.findall(r"^## I\d",spec,re.M))==5
practice=re.split(r"^## 12\.",lesson.split("## 11. Practice")[1],flags=re.M)[0]
assert len(re.findall(r"^### \d+\.",practice,re.M))==12
study=json.loads((HERE/"study-results.json").read_text())
assert len(study["runs"])==12
for run in study["runs"]:
    selected=min(run["curve"],key=lambda c:c["validation_ce_after"])["step"]
    assert selected==run["selected_step"]
    for role,measurement in run["measurements"].items():
        confusion=np.array(measurement["confusion"])
        assert confusion.sum()==measurement["count"] and confusion.trace()==measurement["correct"]
        if "route_counts" in measurement:assert sum(measurement["route_counts"])==measurement["count"]*16*2

extras={}
probabilities=np.array([[.6,.3,.1],[.5,.4,.1],[.2,.7,.1],[.1,.2,.7]])
extras["fresh_balance"]=[]
for changed in (False,True):
    p=probabilities.copy()
    if changed:p[0]=[.2,.7,.1]
    q=np.bincount(p.argmax(1),minlength=3)/4
    value=3*np.dot(q,p.mean(0))
    extras["fresh_balance"].append(dict(q=q.tolist(),P=p.mean(0).tolist(),loss=float(value)))
assert np.allclose([r["loss"] for r in extras["fresh_balance"]],[1.0125,1.125])
extras["fresh_cost"]=[]
for n,m,k in [(10,80,2),(20,40,4)]:
    extras["fresh_cost"].append(dict(total_expert_parameters=3*48*m*n,active_expert_macs=3*48*m*k,
        router_parameters=48*n,payload_bytes=2*24*k*48*4))
assert [r["payload_bytes"] for r in extras["fresh_cost"]]==[18432,36864]
extras["practice_routes"]=route_fixture([2,5,3],[[1],[5],[-2]],2)
assert np.allclose(extras["practice_routes"]["output"],[19/8])
assert np.allclose(extras["practice_routes"]["sum_output_gradient"],[0,105/64,-105/64])
extras["practice_capacity"]=capacity([[0,1]]*6,4)
assert len(extras["practice_capacity"]["dropped"])==4
assert extras["practice_capacity"]["fully_dropped"]==[4,5]
extras["scatter"]= (.25*np.array([2,1])+.75*np.array([-1,3])).tolist()
assert extras["scatter"]==[-.25,2.5]
extras["practice_bias"]=float(np.dot(np.array([.4,.8])/1.2,[3,-1]))
assert abs(extras["practice_bias"]-1/3)<1e-12
expert_choice=np.array([[.7,.2,.1],[.6,.3,.1],[.2,.2,.6],[.1,.8,.1]])
extras["expert_choice_figure"]=dict(row_choices=expert_choice.argmax(1).tolist(),column_choices=expert_choice.argmax(0).tolist())
assert extras["expert_choice_figure"]==dict(row_choices=[0,0,2,1],column_choices=[0,3,2])
assert np.allclose(calculated["trained_fixtures"]["0"]["variants"]["clean"]["logits"],
                   calculated["trained_fixtures"]["0"]["variants"]["disable_0"]["logits"],rtol=0,atol=0)
assert fresh["pixels"]==calculated["trained_fixtures"]["7"]["variants"]["upper_left_patch_zero"]["pixels"]
assert all(f["batch_companion_null_max"]<2e-5 for f in calculated["trained_fixtures"].values())
output=dict(status="author checks passed; implementation and independent review deferred",
    displayed_source_sha256=hashlib.sha256(source.encode()).hexdigest(),displayed_source_matches=True,
    figures=11,investigations=5,practice_questions=12,closed_disclosures=14,declared_fits=12,
    saved_model_checks=calculated["checks"],additional_calculations=extras)
(HERE/"author-checks.json").write_text(json.dumps(output,indent=2)+"\n")
print(json.dumps({k:v for k,v in output.items() if k!="additional_calculations"},indent=2))
