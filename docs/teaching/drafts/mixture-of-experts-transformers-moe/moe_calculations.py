"""Constructed routing arithmetic and saved-model author checks; never fits."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
import torch.nn.functional as F
from moe_study import DigitTransformer,load_data,evaluate
HERE=Path(__file__).resolve().parent

def route_fixture(masses, outputs, k, renormalize=True):
    logits=torch.log(torch.tensor(masses,dtype=torch.float64)).requires_grad_()
    experts=torch.tensor(outputs,dtype=torch.float64)
    selected_logits,selected=logits.topk(k)
    weights=selected_logits.softmax(0) if renormalize else logits.softmax(0)[selected]
    result=(weights[:,None]*experts[selected]).sum(0)
    gradient=torch.autograd.grad(result.sum(),logits)[0]
    return dict(masses=masses,experts=outputs,k=k,renormalize=renormalize,
                selected=selected.tolist(),weights=weights.detach().tolist(),
                output=result.detach().tolist(),sum_output_gradient=gradient.tolist())

def capacity(selected,capacity,order=None):
    used=[0]*4; retained=[]; dropped=[]; outputs=np.zeros(len(selected))
    for token in (range(len(selected)) if order is None else order):
        for slot,expert in enumerate(selected[token]):
            if used[expert]<capacity:
                used[expert]+=1
                retained.append([token,slot,expert])
                outputs[token] += .5*(expert+1)
            else: dropped.append([token,slot,expert])
    kept=set(r[0] for r in retained)
    return dict(selected=selected,capacity=capacity,retained=retained,dropped=dropped,
                fully_dropped=[i for i in range(len(selected)) if i not in kept],output=outputs.tolist(),used=used)

def dense_reference(model,pixels):
    batch=len(pixels)
    patches=pixels.reshape(batch,4,2,4,2).permute(0,1,3,2,4).reshape(batch,16,4)
    hidden=model.project(patches)+model.position
    qkv=model.qkv(model.norm_attention(hidden)).reshape(batch,16,3,2,8)
    query,key,value=qkv.permute(2,0,3,1,4).unbind(0)
    context=((query@key.transpose(-1,-2)/np.sqrt(8)).softmax(-1)@value).transpose(1,2).reshape(batch,16,16)
    hidden=hidden+model.attention_output(context)
    inputs=model.norm_ffn(hidden)
    scores=model.router(inputs)
    values,selected=scores.topk(2,-1)
    gates=torch.zeros_like(scores).scatter(-1,selected,values.softmax(-1))
    all_outputs=torch.stack([expert(inputs) for expert in model.experts],dim=-2)
    combined=(gates[...,None]*all_outputs).sum(-2)
    logits=model.classifier(model.norm_final(hidden+combined).mean(1))
    fractions=torch.bincount(selected.flatten(),minlength=4).to(scores.dtype)/selected.numel()
    auxiliary=4*(fractions.detach()*scores.softmax(-1).mean((0,1))).sum()
    return logits,auxiliary

def main():
    result={}
    result["routing"]={name:route_fixture(masses,outputs,k,renorm) for name,masses,outputs,k,renorm in [
        ("worked_selected",[4,2,1],[[2,0],[0,3],[-1,1]],2,True),
        ("worked_full",[4,2,1],[[2,0],[0,3],[-1,1]],2,False),
        ("fresh_selected",[1,3,2,4],[[2,-1],[1,2],[-1,4],[3,0]],2,True),
        ("fresh_changed",[1,3,5,4],[[2,-1],[1,2],[-1,4],[3,0]],2,True),
        ("fresh_top1",[1,3,2,4],[[2,-1],[1,2],[-1,4],[3,0]],1,True),
        ("fresh_top1_full",[1,3,2,4],[[2,-1],[1,2],[-1,4],[3,0]],1,False),
        ("constant_null",[1,3,2,4],[[2,-1]]*4,2,True)]}
    selected=[[0,1],[0,2],[0,3],[1,2],[0,1]]
    result["capacity"]={str(c):capacity(selected,c) for c in (2,3,4)}
    result["capacity"]["reordered"]=capacity(selected,2,[4,0,1,2,3])
    logits=torch.log(torch.tensor([[.51,.49]]*3+[[.001,.999]],dtype=torch.float64)).requires_grad_()
    prob=logits.softmax(-1); fraction=torch.tensor([.75,.25],dtype=torch.float64)
    auxiliary=2*(fraction*prob.mean(0)).sum()
    result["auxiliary"]=dict(probabilities=prob.detach().tolist(),fractions=fraction.tolist(),
                            value=auxiliary.item(),gradient=torch.autograd.grad(auxiliary,logits)[0].tolist())
    result["z_loss"]=[]
    for p in ([.5,.5],[.99,.01]):
        for shift in (0.,3.):
            scores=torch.log(torch.tensor(p,dtype=torch.float64))+shift
            result["z_loss"].append(dict(p=p,shift=shift,probabilities=scores.softmax(0).tolist(),
                                        loss=scores.logsumexp(0).square().item()))
    affinity=np.array([.8,.7,.6,.1]);bias=np.array([0.,0.,.3,0.])
    chosen=np.argsort(-(affinity+bias),kind="stable")[:2]
    result["selection_bias"]=dict(affinity=affinity.tolist(),bias=bias.tolist(),selected=chosen.tolist(),
                                  weights=(affinity[chosen]/affinity[chosen].sum()).tolist())
    result["costs"]={str((e,m,k)):dict(expert_parameters=3*64*m,total_expert_parameters=e*3*64*m,
        router_parameters=64*e,active_expert_macs=k*3*64*m,
        forward_remote_payload_bytes=2*32*k*64*2) for e,m,k in [(8,128,2),(16,64,4)]}

    rows,pixels,labels,roles=load_data()
    fits=json.loads((HERE/"fitted-models.json").read_text())
    study=json.loads((HERE/"study-results.json").read_text())
    maximum_metric_error=0.; maximum_reference_error=0.;maximum_gradient_error=0.
    for run in study["runs"]:
        model=DigitTransformer(run["condition"]!="dense")
        model.load_state_dict({name:torch.tensor(value) for name,value in fits[run["key"]].items()});model.eval()
        for role,index in roles.items():
            measured=evaluate(model,pixels[index],labels[index]); saved=run["measurements"][role]
            maximum_metric_error=max(maximum_metric_error,abs(measured["cross_entropy"]-saved["cross_entropy"]))
            assert measured["correct"]==saved["correct"] and measured["confusion"]==saved["confusion"]
        if model.sparse:
            model.double()
            x=pixels[roles["validation"][:4]].double()
            logits,aux=model(x)
            loss=F.cross_entropy(logits,labels[roles["validation"][:4]])+.01*aux
            gradients=torch.autograd.grad(loss,tuple(model.parameters()),allow_unused=True)
            gradients=tuple(torch.zeros_like(p) if g is None else g for p,g in zip(model.parameters(),gradients))
            reference,reference_aux=dense_reference(model,x)
            ref_loss=F.cross_entropy(reference,labels[roles["validation"][:4]])+.01*reference_aux
            ref_gradients=torch.autograd.grad(ref_loss,tuple(model.parameters()))
            maximum_reference_error=max(maximum_reference_error,(logits-reference).abs().max().item())
            maximum_gradient_error=max(maximum_gradient_error,max((a-b).abs().max().item() for a,b in zip(gradients,ref_gradients)))
    fixtures={}
    for label in (0,7):
        index=next(int(i) for i in roles["validation"] if int(labels[i])==label)
        record=rows[index]; image=pixels[index:index+1]
        model=DigitTransformer(True)
        model.load_state_dict({name:torch.tensor(value) for name,value in fits["moe_001_17"].items()});model.eval()
        variants={}
        for variant in ("clean","lower_half_zero","upper_left_patch_zero","temperature_2","disable_0"):
            edited=image.clone().reshape(1,8,8)
            if variant=="lower_half_zero": edited[:,4:,:]=0
            if variant=="upper_left_patch_zero":edited[:,:2,:2]=0
            with torch.no_grad():
                logits,aux,trace=model(edited.reshape(1,64),temperature=2. if variant=="temperature_2" else 1.,
                                       disabled_expert=0 if variant=="disable_0" else None,trace=True)
            variants[variant]={"pixels":edited.flatten().tolist(),"class_probabilities":logits.softmax(-1)[0].tolist(),
                               **{name:value.detach().tolist() for name,value in trace.items()}}
        with torch.no_grad():
            single=model(image)[0]
            repeated=model(torch.cat((pixels[roles["validation"][:3]],image),0))[0][-1:]
        fixtures[str(label)]=dict(source_file=record["source_file"],source_id=int(record["source_id"]),label=label,
                                  variants=variants,batch_companion_null_max=(single-repeated).abs().max().item())
    result["trained_fixtures"]=fixtures
    result["checks"]=dict(saved_metric_max_error=maximum_metric_error,dense_dispatch_max_output_error=maximum_reference_error,
                          all_parameter_gradient_max_error=maximum_gradient_error,
                          data_sha256=hashlib.sha256((HERE/"optical-digits.csv").read_bytes()).hexdigest())
    assert maximum_metric_error==0 and maximum_reference_error<1e-12 and maximum_gradient_error<1e-12
    (HERE/"calculated-inputs.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result["checks"]))
    for label,fixture in fixtures.items():
        print(label,fixture["source_id"],{k:dict(prediction=int(np.argmax(v["class_probabilities"])),
            label_probability=v["class_probabilities"][int(label)],counts=v["counts"]) for k,v in fixture["variants"].items()})
if __name__=="__main__":main()
