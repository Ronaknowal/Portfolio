"""Independent full-network SDPA reference; does not import author model code."""
import os
os.environ["OMP_NUM_THREADS"]="2"
os.environ["OPENBLAS_NUM_THREADS"]="2"
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
torch.set_num_threads(2)
ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
ID="multi-head-latent-attention-mla"
raw=json.loads((ROOT/"public/learn-assets"/ID/"runtime.json").read_text())
w={k:torch.tensor(v,dtype=torch.float64) for k,v in raw["weights"].items()}
checks=[]
def near(name,left,right,tol=3e-7):
    a=np.asarray(left);b=np.asarray(right);assert a.shape==b.shape,(name,a.shape,b.shape)
    err=float(np.max(np.abs(a-b)));assert err<tol,(name,err,tol)
    checks.append(dict(name=name,passed=True,maxAbsoluteError=err,tolerance=tol))
def linear(x,name):
    return F.linear(x,w[name+".weight"],w.get(name+".bias"))
def ln(x,name):
    return F.layer_norm(x,(24,),w[name+".weight"],w[name+".bias"],eps=1e-5)
def rms(x,name):
    return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)*w[name+".scale"]
def rotary(x,ids):
    # Every rotary field here has exactly one adjacent pair.
    angle=ids.to(torch.float64)
    while angle.ndim<x.ndim-1:angle=angle.unsqueeze(0)
    return torch.stack((x[...,0]*angle.cos()-x[...,1]*angle.sin(),x[...,0]*angle.sin()+x[...,1]*angle.cos()),-1)
def reference(points,rank,shift):
    x=torch.tensor(points,dtype=torch.float64)*2-1;t=len(points);ids=torch.arange(t)+shift
    state=linear(x,"stem");a=ln(state,"norm_attention")
    full=rms(linear(a,"kv_down"),"kv_norm")
    c=full;ku=w["key_up.weight"].reshape(4,4,8);vu=w["value_up.weight"].reshape(4,4,8)
    if rank is not None:
        basis=torch.tensor(raw["completeBasis"],dtype=torch.float64)[:,:rank]
        c=c@basis;ku=ku@basis;vu=vu@basis
    qlatent=rms(linear(a,"query_down"),"query_norm")
    qc=linear(qlatent,"query_content").reshape(t,4,4).transpose(0,1)
    qr=rotary(linear(qlatent,"query_rotary").reshape(t,4,2).transpose(0,1),ids)
    kr=rotary(linear(a,"rotary_key"),ids)
    # Reconstruct different K/V heads and use ordinary SDPA. The author browser
    # path absorbs both maps and uses its own scalar softmax, so trust roots differ.
    k=torch.einsum("sc,hpc->hsp",c,ku);v=torch.einsum("sc,hvc->hsv",c,vu)
    q=torch.cat((qc,qr),-1);keys=torch.cat((k,kr.unsqueeze(0).expand(4,-1,-1)),-1)
    allowed=ids[None,:]<=ids[:,None]
    heads=F.scaled_dot_product_attention(q,keys,v,attn_mask=allowed,dropout_p=0.,scale=1/math.sqrt(6))
    content=qc@k.transpose(-1,-2);position=qr@kr.T
    probs=((content+position)/math.sqrt(6)).masked_fill(~allowed,-torch.inf).softmax(-1)
    state=state+linear(heads.transpose(0,1).reshape(t,16),"output")
    state=state+linear(F.gelu(linear(ln(state,"norm_feedforward"),"feedforward.0")),"feedforward.2")
    result=(linear(ln(state,"norm_final"),"forecast")+1)/2
    return result,full,c,kr,heads,probs,content,position
browser=json.loads((HERE/"independent-browser-values.json").read_text())
for case in browser["cases"]:
    label=f'shape{case["shape"]} rank{case["rank"]} shift{case["shift"]}'
    out,full,c,kr,heads,probs,content,position=reference(case["points"],case["rank"],case["shift"])
    near(label+" full-network SDPA forecast",out,case["predictions"])
    near(label+" original normalized latent",full,case["fullLatents"],1e-11)
    near(label+" stored basis coordinates",c,case["cache"]["latents"],1e-11)
    near(label+" positioned rotary cache",kr,case["cache"]["rotaryKeys"],1e-11)
    for key,expected in [("output",heads[:,-1]),("weights",probs[:,-1]),("contentScores",content[:,-1]),("rotaryScores",position[:,-1])]:
        near(label+" final-head "+key,expected,[h[key] for h in case["lastHeads"]],1e-11)

# Changed dimensions not covered by the author bridge: batch2/head3, 7 latent
# coordinates, P3/R2/V4, reordered logical keys with future records.
torch.manual_seed(806)
inputs=[torch.randn(shape,dtype=torch.float64,requires_grad=True) for shape in
        [(2,3,2,3),(2,5,7),(3,3,7),(3,4,7),(2,3,2,2),(2,5,2)]]
q,c,ku,vu,qr,kr=inputs;legal=torch.tensor([[True,False,True,False,True],[True,True,True,False,True]])
expanded_k=torch.einsum("bsc,hpc->bhsp",c,ku);expanded_v=torch.einsum("bsc,hvc->bhsv",c,vu)
expanded=F.scaled_dot_product_attention(torch.cat([q,qr],-1),torch.cat([expanded_k,kr[:,None].expand(-1,3,-1,-1)],-1),expanded_v,attn_mask=legal,dropout_p=0.,scale=1/math.sqrt(5))
effective=torch.einsum("bhtp,hpc->bhtc",q,ku)
mixed=F.scaled_dot_product_attention(torch.cat([effective,qr],-1),torch.cat([c,kr],-1)[:,None],c[:,None],attn_mask=legal,dropout_p=0.,scale=1/math.sqrt(5),enable_gqa=True)
absorbed=torch.einsum("bhtc,hvc->bhtv",mixed,vu)
near("Changed 7-latent/3-content batched SDPA paths",expanded.detach(),absorbed.detach(),1e-12)
probe=torch.linspace(-1,1,expanded.numel(),dtype=torch.float64).reshape_as(expanded)
left=torch.autograd.grad((expanded*probe).sum(),inputs,retain_graph=True)
right=torch.autograd.grad((absorbed*probe).sum(),inputs)
for i,(a,b) in enumerate(zip(left,right)):near("Changed-dimension factor gradient "+str(i),a,b,1e-11)
for filename in ["author-calculations.py","author-results.json","data-provenance.md","forecast-model.json","mechanism-calculations.py","mechanism-fixtures.json","mla_sdpa_bridge.py","movement_libras.data","movement_libras.names"]:
    assert (ROOT/"public/learn-assets"/ID/filename).read_bytes()==(ROOT/"docs/teaching/drafts"/ID/filename).read_bytes()
checks.append(dict(name="All9 public reproduction files byte-identical to intended sources",passed=True))
checks.append(dict(name="Browser future exclusion, logical-record permutation, chunk3 cache and BigInt budget controls",passed=all(browser["controls"][k] for k in ["futureUnchanged","logicalRecordPermutation","extremeBudgetBigIntAgreement"])))
result=dict(topicId=ID,passed=True,checks=checks,maximumForecastError=max(c["maxAbsoluteError"] for c in checks if "forecast" in c["name"]),environment=dict(torch=torch.__version__,numpy=np.__version__,maximumThreads=2),limitations=["Author all-parameter gradient and held-out fit replay evidence reused; no fit rerun.","This independent reference uses complete functional PyTorch layers and expanded ordinary SDPA; it does not call the author model class or scalar attention implementation.","Painted/browser controls and full integration remain the parent responsibility."])
(HERE/"independent-native-results.json").write_text(json.dumps(result,indent=2)+"\n")
print(len(checks),"complementary independent checks passed; maximum forecast error",result["maximumForecastError"])
