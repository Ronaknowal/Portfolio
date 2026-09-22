"""Run beside digits-400.csv. CPU; Python3.12, torch2.14, numpy2.3.5, sklearn1.9.1."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split

torch.set_num_threads(1)
EPSILON = 1e-5


def centered_norm(values, axes, scale=1., shift=0., epsilon=EPSILON):
    mean = values.mean(dim=axes, keepdim=True)
    variance = (values - mean).square().mean(dim=axes, keepdim=True)
    return scale * (values - mean) / torch.sqrt(variance + epsilon) + shift


def rms_norm(values, scale=1., epsilon=EPSILON):
    return scale * values / torch.sqrt(values.square().mean(-1, keepdim=True) + epsilon)


def group_norm(values, groups, scale, shift, epsilon=EPSILON):
    batch, channels, height, width = values.shape
    if groups < 1 or channels % groups:
        raise ValueError("groups must be positive and divide channels")
    grouped = values.reshape(batch, groups, channels // groups, height, width)
    normalized = centered_norm(grouped, (2, 3, 4), epsilon=epsilon).reshape_as(values)
    return normalized * scale.reshape(1, channels, 1, 1) + shift.reshape(1, channels, 1, 1)


def batch_norm(values, scale, shift, running_mean, running_variance,
               training=True, momentum=.1, epsilon=EPSILON):
    channels = values.shape[1]
    shape = (1, channels, 1, 1)
    if training:
        count = values.numel() // channels
        if count <= 1:
            raise ValueError("training BatchNorm needs more than one value per channel")
        mean = values.mean((0, 2, 3), keepdim=True)
        variance = (values - mean).square().mean((0, 2, 3), keepdim=True)
        with torch.no_grad():
            running_mean.lerp_(mean.reshape(channels), momentum)
            running_variance.lerp_(variance.reshape(channels) * count / (count - 1), momentum)
    else:
        mean = running_mean.reshape(shape)
        variance = running_variance.reshape(shape)
    return scale.reshape(shape) * (values - mean) / (variance + epsilon).sqrt() + shift.reshape(shape)


def fixtures():
    dtype = torch.float64
    values = torch.arange(1., 17., dtype=dtype).reshape(2, 4, 1, 2)
    ones, zeros = torch.ones(4, dtype=dtype), torch.zeros(4, dtype=dtype)
    variants = dict(
        batch=centered_norm(values, (0,2,3)),
        layer=centered_norm(values, (1,2,3)),
        group=group_norm(values, 2, ones, zeros),
        instance=centered_norm(values, (2,3)))
    changed = values.clone()
    changed[1, 0, 0, 0] += 10
    changed_variants = dict(
        batch=centered_norm(changed,(0,2,3)),
        layer=centered_norm(changed,(1,2,3)),
        group=group_norm(changed,2,ones,zeros),
        instance=centered_norm(changed,(2,3)))
    influence = {key: float((changed_variants[key][0]-output[0]).abs().max())
                 for key, output in variants.items()}
    token_cases = []
    for row in ([1.,3.],[11.,13.],[-1.,1.],[5.,5.],[0.,0.]):
        token = torch.tensor(row, dtype=dtype)
        token_cases.append(dict(input=row, layer=centered_norm(token,(-1,)).tolist(),
                                rms=rms_norm(token).tolist()))
    parity = {}
    references = dict(batch=nn.BatchNorm2d(4,eps=EPSILON,dtype=dtype),
                      layer=nn.LayerNorm((4,1,2),eps=EPSILON,dtype=dtype),
                      group=nn.GroupNorm(2,4,eps=EPSILON,dtype=dtype),
                      instance=nn.InstanceNorm2d(4,eps=EPSILON,affine=True,dtype=dtype))
    for key, reference in references.items():
        parity[key]=float((reference(values)-variants[key]).abs().max().detach())
    parity["rms"]=float((nn.RMSNorm(2,eps=EPSILON,dtype=dtype)(values)
                         -rms_norm(values)).abs().max().detach())
    reference=nn.BatchNorm2d(4,eps=EPSILON,momentum=.1,dtype=dtype)
    running_mean=zeros.clone();running_variance=ones.clone()
    own=batch_norm(values,ones,zeros,running_mean,running_variance)
    expected=reference(values)
    batch_state_errors=dict(training=float((own-expected).abs().max().detach()),
        mean=float((running_mean-reference.running_mean).abs().max()),
        variance=float((running_variance-reference.running_var).abs().max()))
    reference.eval()
    own_eval=batch_norm(values,ones,zeros,running_mean,running_variance,training=False)
    batch_state_errors["evaluation"]=float((own_eval-reference(values)).abs().max().detach())
    # Independent input-gradient comparison against library LN and GN.
    coefficients=torch.linspace(-.8,1.2,16,dtype=dtype).reshape_as(values)
    gradient_errors={}
    for key in ("layer","group"):
        x=values.clone().requires_grad_()
        own=centered_norm(x,(1,2,3)) if key=="layer" else group_norm(x,2,ones,zeros)
        g1=torch.autograd.grad((own*coefficients).sum(),x)[0]
        g2=torch.autograd.grad((references[key](x)*coefficients).sum(),x)[0]
        gradient_errors[key]=float((g1-g2).abs().max())
    bn=nn.BatchNorm2d(1,eps=EPSILON,momentum=.1,dtype=dtype)
    x=torch.tensor([1.,3.,5.,7.],dtype=dtype).reshape(2,1,1,2)
    train_output=bn(x).detach()
    bn.eval()
    eval_output=bn(x).detach()
    bn_record=dict(input=x.flatten().tolist(),train=train_output.flatten().tolist(),
                   running_mean=bn.running_mean.tolist(),running_variance=bn.running_var.tolist(),
                   evaluation=eval_output.flatten().tolist())
    bn2=nn.BatchNorm2d(1,eps=EPSILON,dtype=dtype)
    single_spatial=bn2(torch.tensor([1.,3.],dtype=dtype).reshape(1,1,1,2)).detach()
    try:
        bn2(torch.ones((1,1,1,1),dtype=dtype))
        singleton_rejected=False
    except ValueError:
        singleton_rejected=True
    gamma=torch.tensor([1.,2.],dtype=dtype,requires_grad=True)
    beta=torch.tensor([0.,0.],dtype=dtype,requires_grad=True)
    x=torch.tensor([1.,3.],dtype=dtype,requires_grad=True)
    y=centered_norm(x,(-1,),gamma,beta)
    loss=(y-torch.tensor([0.,1.],dtype=dtype)).square().mean()
    gx,gg,gb=torch.autograd.grad(loss,(x,gamma,beta))
    after=centered_norm(x,(-1,),gamma-.1*gg,beta-.1*gb)
    update=dict(input=x.tolist(),output=y.detach().tolist(),loss=loss.item(),
                input_gradient=gx.tolist(),scale_gradient=gg.tolist(),shift_gradient=gb.tolist(),
                updated_output=after.detach().tolist(),
                updated_loss=float((after-torch.tensor([0.,1.],dtype=dtype)).square().mean().detach()))
    return dict(tensor=values.tolist(),outputs={k:v.tolist() for k,v in variants.items()},
                influence_first_sample=influence,token_cases=token_cases,forward_errors=parity,
                gradient_errors=gradient_errors,batch_state_errors=batch_state_errors,batch_train_eval=bn_record,
                single_image_spatial=single_spatial.flatten().tolist(),
                singleton_rejected=singleton_rejected,affine_update=update,
                group_one_vs_layer=float((group_norm(values,1,ones,zeros)
                                         -variants["layer"]).abs().max()),
                group_four_vs_instance=float((group_norm(values,4,ones,zeros)
                                             -variants["instance"]).abs().max()))


def digit_runs():
    data=np.genfromtxt(Path(__file__).with_name("digits-400.csv"),delimiter=",",names=True)
    x=torch.tensor(np.column_stack([data[f"pixel_{i}"] for i in range(64)])/16,
                   dtype=torch.float32)
    y=torch.tensor(data["digit"].astype(int),dtype=torch.long)
    train,valid=train_test_split(np.arange(len(data)),test_size=.3,random_state=22,
                                 stratify=data["digit"])
    records=[]
    for seed in (1,2,3):
        for name in ("none","batch","layer","rms"):
            torch.manual_seed(seed)
            normalization={"none":lambda:nn.Identity(),
                           "batch":lambda:nn.BatchNorm1d(32,eps=EPSILON),
                           "layer":lambda:nn.LayerNorm(32,eps=EPSILON),
                           "rms":lambda:nn.RMSNorm(32,eps=EPSILON)}[name]()
            model=nn.Sequential(nn.Linear(64,32),normalization,nn.Tanh(),nn.Linear(32,10))
            optimizer=torch.optim.SGD(model.parameters(),lr=.1)
            generator=torch.Generator().manual_seed(1000+seed)
            history=[]
            for epoch in range(51):
                if epoch>0:
                    model.train()
                    shuffled=torch.tensor(train)[torch.randperm(len(train),generator=generator)]
                    for batch in shuffled.split(28):
                        loss=F.cross_entropy(model(x[batch]),y[batch])
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                if epoch in (0,1,5,20,50):
                    model.eval()
                    with torch.no_grad():
                        train_logits=model(x[train]);validation_logits=model(x[valid])
                        history.append(dict(epoch=epoch,
                            train_evaluation_loss=F.cross_entropy(train_logits,y[train]).item(),
                            validation_loss=F.cross_entropy(validation_logits,y[valid]).item(),
                            validation_correct=int((validation_logits.argmax(-1)==y[valid]).sum())))
            records.append(dict(seed=seed,normalization=name,history=history))
    return dict(train_source_ids=data["source_id"][train].astype(int).tolist(),
                validation_source_ids=data["source_id"][valid].astype(int).tolist(),records=records)


if __name__=="__main__":
    result=dict(environment=dict(torch=torch.__version__,numpy=np.__version__),
                mechanisms=fixtures(),digits=digit_runs())
    Path(__file__).with_name("calculated-inputs.json").write_text(json.dumps(result,indent=2),
                                                               encoding="utf-8")
    print(json.dumps(result["mechanisms"],indent=2))
    for run in result["digits"]["records"]:
        print(run["seed"],run["normalization"],run["history"][-1])
