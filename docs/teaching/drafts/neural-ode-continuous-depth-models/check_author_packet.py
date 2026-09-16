"""Content author closure: no fit, runtime publication or browser campaign."""
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import numpy as np
import torch
from neural_ode_study import DepthClassifier, VectorField, integrate, load_data, evaluate
from ode_calculations import adaptive_heun, fixed_solve, observations, scalar_gradient

DIRECTORY=Path(__file__).resolve().parent
ROOT=DIRECTORY.parents[3]


def main():
    manuscript=(DIRECTORY/"lesson.md").read_text(encoding="utf-8")
    specification=(DIRECTORY/"visual-specifications.md").read_text(encoding="utf-8")
    result=json.loads((DIRECTORY/"study-results.json").read_text())
    fixtures=json.loads((DIRECTORY/"calculated-inputs.json").read_text())
    snapshots=json.loads((DIRECTORY/"fitted-models.json").read_text())
    original=ROOT/"src/learn/data/topics/neural-ode-continuous-depth-models.jsx"
    original_hash=hashlib.sha256(original.read_bytes()).hexdigest()
    assert original_hash=="f022724f84307be8059de01876003b5e8b4ce91329aa21410b5f6d9071e73cee"
    assert hashlib.sha256((DIRECTORY/"iris.csv").read_bytes()).hexdigest()==result["dataset_sha256"]
    features,labels,roles,metadata=load_data()
    assert metadata==result["data"]
    assert [len(roles[key]) for key in roles]==[90,30,29]
    assert len(set(sum(roles.values(),[])))==149
    assert metadata["duplicate_groups"]==[[102,143]]
    assert len(snapshots)==len(result["runs"])==12
    maximum_error=0.
    for snapshot in snapshots:
        model=DepthClassifier(snapshot["kind"])
        model.load_state_dict({key:torch.tensor(value) for key,value in snapshot["state"].items()})
        report=next(row for row in result["runs"] if row["kind"]==snapshot["kind"] and row["seed"]==snapshot["seed"])
        assert sum(p.numel() for p in model.parameters())==report["parameters"]
        assert report["selected_step"]==min(report["curves"],key=lambda row:row["validation"]["cross_entropy"])["step"]
        for name,indices in roles.items():
            actual=evaluate(model,features[indices],labels[indices])
            maximum_error=max(maximum_error,abs(actual["cross_entropy"]-report[name]["cross_entropy"]))
            assert actual["correct"]==report[name]["correct"] and actual["count"]==report[name]["count"]
    assert maximum_error<1e-12
    blocks=re.findall(r"~~~python\n(.*?)\n~~~",manuscript,re.S)
    assert len(blocks)==2
    namespace={}
    exec(blocks[0],namespace)
    torch.manual_seed(77)
    field=VectorField(4)
    initial=torch.randn(3,4)
    inline=namespace["rk4_step"](field,0.,initial,.25)
    expected=integrate(field,initial,1,"rk4",.25)[-1]
    assert torch.equal(inline,expected)
    training=(DIRECTORY/"neural_ode_study.py").read_text(encoding="utf-8").strip()
    assert blocks[1].strip()==training
    compile(blocks[1],"displayed full study","exec")
    assert manuscript.count("<details>")==manuscript.count("</details>")==20
    assert not re.search(r"<details[^>]+\bopen\b",manuscript)
    assert len(set(re.findall(r"\[Figure (O\d{2})",manuscript)))==12
    assert len(set(re.findall(r"\*\*Investigation (O-I\d)",manuscript)))==6
    for number in range(1,7):
        assert "## O-I"+str(number) in specification
    # Independent practice arithmetic, plus meaningful invariants.
    practice=dict(decay_euler=2*.7**2,decay_exact=2*math.exp(-.6),
        parameters=5*16+16+16*4+4+4*3+3,
        normalized_local_error=.01/.011,one_step_gradient=(1.5*(1-.2)-1)*.2*1.5,
        observed_zero=.42*math.exp(-.4),missing=.6*math.exp(-.4),
        log_density_change=-1.5*(.4-.1),volume=math.exp(.45),density=math.exp(-.45),
        flow_mean_velocity=0.,flow_mean_loss=16.)
    assert abs(practice["observed_zero"]-.2815344193349685)<1e-12
    assert abs(practice["one_step_gradient"]-.06)<1e-12
    zero=adaptive_heun(lambda time,state:np.array([-2.,-50.])*state,[0.,0.],.4,.01)
    assert zero["endpoint"]==[0.,0.] and zero["rejected"]==0
    for method in ["euler","rk4"]:
        constant=fixed_solve(lambda time,state:np.zeros_like(state),[.6,.8],1.2,4,method)
        assert np.array_equal(constant,np.tile([.6,.8],(5,1)))
        for rate in [0.,-.5]:
            grad=scalar_gradient(rate,0.,1.,.7,4,method)
            assert grad["gradient"]==0.
    causal=observations([.1,.7,1.4],[1.,-1.,.5],1.)
    future_edit=observations([.1,.7,1.4],[1.,-1.,-2.],1.)
    assert causal==future_edit
    assert observations([],[],1.)["state"]==0.
    assert fixtures["independent_network_max_error"]<2e-12
    report=dict(scope="content author checks only; no refitting or implementation",
        source_sha256=original_hash,dataset_sha256=result["dataset_sha256"],
        fits_reconciled=12,maximum_saved_metric_error=maximum_error,roles={key:len(value) for key,value in roles.items()},
        inline_rk4_exact_parity=True,full_displayed_program_bound=True,
        closed_practice_details=18,other_closed_details=2,figure_anchors=12,investigations=6,
        practice_calculations=practice,zero_future_and_constant_nulls=True,
        independent_network_max_error=fixtures["independent_network_max_error"])
    (DIRECTORY/"author-checks.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
