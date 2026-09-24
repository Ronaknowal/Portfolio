"""Complete small CPU residual experiments. Run beside digits-400.csv."""
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split

torch.set_num_threads(1)


class Refinement(nn.Module):
    def __init__(self, width, mode, depth, seed):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.lower = nn.Linear(width, width)
        self.upper = nn.Linear(width, width)
        generator = torch.Generator().manual_seed(seed)
        for layer in (self.lower, self.upper):
            nn.init.xavier_normal_(layer.weight, generator=generator)
            nn.init.zeros_(layer.bias)
        self.mode = mode
        if mode == "rezero":
            self.scale = nn.Parameter(torch.zeros(()))
        else:
            self.register_buffer("scale", torch.tensor(1/math.sqrt(depth) if mode == "scaled" else 1.))

    def forward(self, inputs):
        correction = self.upper(torch.tanh(self.lower(self.norm(inputs))))
        if self.mode == "plain":
            return correction
        if correction.shape != inputs.shape:
            raise ValueError("residual addition requires the same shape and coordinate meaning")
        return inputs + self.scale*correction


class DigitNetwork(nn.Module):
    def __init__(self, depth, mode, seed):
        super().__init__()
        self.stem = nn.Linear(64, 32)
        self.blocks = nn.ModuleList([Refinement(32, mode, depth, seed*1000+i)
                                    for i in range(depth)])
        self.head = nn.Linear(32, 10)
        for offset, layer in ((100, self.stem), (200, self.head)):
            nn.init.xavier_normal_(layer.weight, generator=torch.Generator().manual_seed(seed+offset))
            nn.init.zeros_(layer.bias)

    def states(self, pixels, omit=None):
        value = torch.tanh(self.stem(pixels))
        outputs = [value]
        for index, block in enumerate(self.blocks):
            if index != omit:
                value = block(value)
            outputs.append(value)
        return self.head(value), outputs

    def forward(self, pixels):
        return self.states(pixels)[0]


def score(model, pixels, targets, omit=None):
    model.eval()
    with torch.no_grad():
        logits, _ = model.states(pixels, omit=omit)
        return {"ce": F.cross_entropy(logits, targets).item(),
                "correct": int((logits.argmax(-1)==targets).sum()), "count": len(targets)}


def diagnostics(model, pixels, targets):
    model.eval()
    logits, states = model.states(pixels)
    loss = F.cross_entropy(logits, targets)
    derivatives = torch.autograd.grad(loss, [model.stem.weight, *states])
    return {"stem_weight_gradient_norm": derivatives[0].norm().item(),
            "layers": [{"layer": i, "mean_square": state.detach().square().mean().item(),
                        "activation_gradient_rms": gradient.square().mean().sqrt().item()}
                       for i, (state,gradient) in enumerate(zip(states,derivatives[1:]))]}


def fit_models(train_pixels, train_targets, validation_pixels, validation_targets):
    records = []
    configurations = [(0,"stem_only")] + [(depth,mode) for depth in (2,6,12)
                                for mode in ("plain","residual","scaled","rezero")]
    for seed in (1,2,3):
        for depth,mode in configurations:
            model = DigitNetwork(depth,mode,seed)
            before = {name: value.detach().clone() for name,value in model.named_parameters()}
            optimizer = torch.optim.Adam(model.parameters(),lr=.003)
            trace = []
            for step in range(251):
                if step in (0,1,25,100,250):
                    trace.append({"step": step, "training": score(model,train_pixels,train_targets),
                                  "validation": score(model,validation_pixels,validation_targets),
                                  "diagnostics": diagnostics(model,train_pixels[:32],train_targets[:32])})
                if step==250:
                    break
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(train_pixels),train_targets)
                if not torch.isfinite(loss):
                    raise ValueError("non-finite training loss; keep and investigate this configuration")
                loss.backward()
                optimizer.step()
            displacement = {name: (value.detach()-before[name]).norm().item()
                            for name,value in model.named_parameters()}
            ablations = [{"omitted_block": i, **score(model,validation_pixels,validation_targets,omit=i)}
                         for i in range(depth)] if mode!="plain" else []
            record = {"seed":seed,"depth":depth,"mode":mode,"trace":trace,
                      "trainable_count":sum(value.numel() for value in model.parameters()),
                      "parameter_displacements":displacement,
                      "scales":[block.scale.item() for block in model.blocks],
                      "validation_block_ablations":ablations}
            records.append(record)
            end=trace[-1]
            print(seed,depth,mode,round(end["training"]["ce"],6),
                  round(end["validation"]["ce"],6),end["validation"]["correct"],flush=True)
    return records


def mechanisms():
    dtype=torch.float64
    records={}
    x=torch.tensor([2.,-1.],dtype=dtype,requires_grad=True)
    weight=torch.tensor([[.1,.2],[0.,-.5]],dtype=dtype,requires_grad=True)
    target=torch.tensor([1.,0.],dtype=dtype)
    correction=weight@x
    output=x+correction
    loss=.5*(output-target).square().sum()
    gx,gw=torch.autograd.grad(loss,(x,weight))
    with torch.no_grad():
        updated=weight-.1*gw
        new_output=x+updated@x
    upstream=output.detach()-target
    records["one_update"]={"input":x.tolist(),"weight":weight.tolist(),"target":target.tolist(),
        "correction":correction.tolist(),"output":output.tolist(),"loss":loss.item(),
        "upstream":upstream.tolist(),"branch_input_gradient":(weight.detach().T@upstream).tolist(),
        "total_input_gradient":gx.tolist(),"weight_gradient":gw.tolist(),
        "new_weight":updated.tolist(),"new_output":new_output.tolist(),
        "new_loss":(.5*(new_output-target).square().sum()).item()}
    changed=weight.detach().clone()
    changed[0,0]=-.1
    changed_output=x.detach()+changed@x.detach()
    records["one_weight_edit"]={"weight":changed.tolist(),"output":changed_output.tolist(),
                               "loss":(.5*(changed_output-target).square().sum()).item()}
    assert torch.allclose(gx,upstream+weight.detach().T@upstream)
    scalar=[]
    for slope in (-1.,-.5,0.,.1,1.):
        scalar.append({"branch_slope":slope,"block_derivative":1+slope,
                       "ten_block_derivative":(1+slope)**10})
    records["scalar_derivatives"]=scalar
    x=torch.tensor([-2.,1.],dtype=dtype)
    zero=lambda value: torch.zeros_like(value)
    records["zero_branch_order"]={
        "input":x.tolist(),"pure_output":(x+zero(x)).tolist(),
        "post_relu_output":F.relu(x+zero(x)).tolist(),
        "pure_jacobian":torch.func.jacrev(lambda v:v+zero(v))(x).tolist(),
        "post_relu_jacobian":torch.func.jacrev(lambda v:F.relu(v+zero(v)))(x).tolist()}
    normalized=lambda value:F.layer_norm(value,(2,),eps=1e-5)
    records["zero_branch_norm"]={
        "pre_output":x.tolist(),"post_output":normalized(x).tolist(),
        "post_jacobian":torch.func.jacrev(normalized)(x).tolist(),
        "post_constant_direction":(torch.func.jacrev(normalized)(x)@torch.ones(2,dtype=dtype)).tolist()}
    gates=[]
    for initial_scale in (0.,.001,1.):
        value=torch.tensor([1.,2.],dtype=dtype)
        matrix=torch.tensor([[.2,-.1],[.1,.3]],dtype=dtype,requires_grad=True)
        alpha=torch.tensor(initial_scale,dtype=dtype,requires_grad=True)
        residual=matrix@value
        result=value+alpha*residual
        loss=.5*result.square().sum()
        ga,gm=torch.autograd.grad(loss,(alpha,matrix))
        gates.append({"scale":initial_scale,"branch":residual.tolist(),"output":result.tolist(),
                      "scale_gradient":ga.item(),"branch_weight_gradient":gm.tolist(),
                      "scale_after_sgd_0_1":initial_scale-.1*ga.item()})
    records["gates"]=gates
    # Zero final linear projection can learn; zero ReLU branch cannot here.
    openings=[]
    for kind in ("zero_final_linear","zero_relu"):
        value=torch.tensor([1.,2.],dtype=dtype)
        first=torch.eye(2,dtype=dtype,requires_grad=True)
        last=torch.zeros(2,2,dtype=dtype,requires_grad=True)
        feature=F.relu(first@value)
        branch=last@feature if kind=="zero_final_linear" else F.relu(last@feature)
        result=value+branch
        gradient_first,gradient_last=torch.autograd.grad(.5*result.square().sum(),(first,last))
        openings.append({"kind":kind,"output":result.tolist(),
                         "first_gradient":gradient_first.tolist(),"last_gradient":gradient_last.tolist()})
    records["zero_last"]=openings
    # Exact nonlinear counterexample to distributing F over a sum.
    f=lambda value:value**2
    value=1.
    records["path_expansion"]={"input":value,"actual":value+f(value)+f(value+f(value)),
                              "invalid_distributed":value+f(value)+f(value)+f(f(value))}
    projection=torch.tensor([[1.,0.],[0.,1.],[1.,1.]],dtype=dtype)
    value=torch.tensor([2.,-1.],dtype=dtype)
    records["projection"]={"matrix":projection.tolist(),"input":value.tolist(),
                           "skip_output":(projection@value).tolist(),
                           "upstream":[1.,2.,3.],
                           "skip_gradient":(projection.T@torch.tensor([1.,2.,3.],dtype=dtype)).tolist()}
    records["changed_projection"]={"input":[1.,3.],"upstream":[2.,-1.,4.],
        "output":(projection@torch.tensor([1.,3.],dtype=dtype)).tolist(),
        "skip_gradient":(projection.T@torch.tensor([2.,-1.,4.],dtype=dtype)).tolist()}
    saved=[]
    def pack(tensor):
        saved.append(list(tensor.shape))
        return tensor
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda tensor:tensor):
        left=torch.tensor([1.,2.],dtype=dtype,requires_grad=True)
        right=torch.tensor([3.,4.],dtype=dtype,requires_grad=True)
        (left+right).sum().backward()
    records["pure_addition_saved_tensors"]=saved
    records["euler"]=[{"step":step,"multiplier":1-step,"after_ten":(1-step)**10}
                      for step in (.1,1.,2.5)]
    return records


def run():
    directory=Path(__file__).resolve().parent
    data=np.genfromtxt(directory/"digits-400.csv",delimiter=",",names=True,dtype=np.int64)
    pixels=torch.tensor(np.column_stack([data[f"pixel_{i}"] for i in range(64)]),dtype=torch.float32)/16
    targets=torch.tensor(data["digit"],dtype=torch.long)
    training,validation=train_test_split(np.arange(400),test_size=.3,stratify=data["digit"],random_state=22)
    output={"versions":{"torch":torch.__version__,"numpy":np.__version__},
            "training_source_ids":data["source_id"][training].tolist(),
            "validation_source_ids":data["source_id"][validation].tolist(),
            "mechanisms":mechanisms(),
            "fits":fit_models(pixels[training],targets[training],pixels[validation],targets[validation])}
    (directory/"calculated-inputs.json").write_text(json.dumps(output,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    run()
