"""Complete CPU teaching experiment; reads the adjacent attributed digits CSV."""
from pathlib import Path
import itertools
import json
import platform
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent


def mask_values(x, probability, training, mode="element"):
    """Inverted Bernoulli mask; probability means drop, not keep."""
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between zero and one")
    if mode not in ("element", "channel", "row", "batch"):
        raise ValueError("unknown mask mode")
    if mode == "channel" and x.ndim != 4:
        raise ValueError("channel mode requires [batch, channel, height, width]")
    if not training or probability == 0:
        return x
    if probability == 1:
        return x * 0
    shape = {"element": x.shape,
             "channel": (x.shape[0], x.shape[1], 1, 1) if x.ndim == 4 else (),
             "row": (x.shape[0],) + (1,) * (x.ndim - 1),
             "batch": (1,) * x.ndim}[mode]
    mask = x.new_empty(shape).bernoulli_(1 - probability)
    return x * mask / (1 - probability)


def fixtures():
    dtype = torch.float64
    h = torch.tensor([1., 2.], dtype=dtype, requires_grad=True)
    w = torch.tensor([1., -.5], dtype=dtype, requires_grad=True)
    mask = torch.tensor([1., 0.], dtype=dtype)
    output = (w * h * mask / .5).sum()
    loss = .5 * (output - 1).square()
    loss.backward()
    updated = w.detach() - .1 * w.grad
    new_output = (updated * h.detach() * mask / .5).sum()
    one_update = dict(input=h.detach().tolist(), weight=w.detach().tolist(),
                      mask=mask.tolist(), output=output.item(), loss=loss.item(),
                      weight_gradient=w.grad.tolist(), input_gradient=h.grad.tolist(),
                      updated_weight=updated.tolist(), updated_output=new_output.item(),
                      updated_loss=(.5 * (new_output-1).square()).item())
    enumerations = []
    for p in (.25, .5):
        records = []
        for bits in itertools.product((0, 1), repeat=2):
            probability = np.prod([(1-p) if bit else p for bit in bits])
            values = np.array([1., 2.]) * bits / (1-p)
            signed_sum = (bits[0] - bits[1]) / (1-p)
            records.append(dict(mask=list(bits), probability=float(probability),
                                values=values.tolist(), relu_signed_sum=max(0., signed_sum)))
        mean = sum(r["probability"] * np.array(r["values"]) for r in records)
        variance = sum(r["probability"] * (np.array(r["values"])-mean)**2 for r in records)
        enumerations.append(dict(drop_probability=p, outcomes=records, mean=mean.tolist(),
                                 variance=variance.tolist(),
                                 expected_relu=sum(r["probability"]*r["relu_signed_sum"] for r in records)))
    grid = torch.arange(1,17,dtype=dtype).reshape(2,2,2,2)
    patterns = {
        "element": torch.tensor([1,0,1,0,0,1,0,1,1,1,0,0,0,0,1,1],dtype=dtype).reshape(2,2,2,2),
        "channel": torch.tensor([1,0,0,1],dtype=dtype).reshape(2,2,1,1),
        "row": torch.tensor([1,0],dtype=dtype).reshape(2,1,1,1),
        "batch": torch.zeros(1,1,1,1,dtype=dtype)}
    granularity = {name: dict(mask_shape=list(m.shape),mask=m.tolist(),
                              output=(grid*m/.5).tolist()) for name,m in patterns.items()}
    x = torch.tensor([2.,-1.],dtype=dtype)
    correction = torch.tensor([.5,1.],dtype=dtype)
    branch = [dict(keep=m, output=(x+m*correction/.5).tolist(),
                   wrong_whole_output=(m*(x+correction)/.5).tolist()) for m in (0,1)]
    schedules = []
    for length in (1,4,12):
        for endpoint in (.2,.5):
            original = [endpoint*(i+1)/length for i in range(length)]
            zero_first = [0.] if length == 1 else [endpoint*i/(length-1) for i in range(length)]
            schedules.append(dict(blocks=length,endpoint=endpoint,
                                  original_rates=original, original_active=sum(1-r for r in original),
                                  zero_first_rates=zero_first, zero_first_active=sum(1-r for r in zero_first)))
    bn = nn.BatchNorm1d(1,momentum=1.,affine=False).double()
    bn.train()
    train_values = torch.tensor([[0.],[2.],[0.],[6.]],dtype=dtype)
    train_output = bn(train_values)
    bn.eval()
    clean_values = torch.tensor([[1.],[3.]],dtype=dtype)
    eval_output = bn(clean_values)
    normalization = dict(noisy_values=train_values.flatten().tolist(),
                         population_mean=2.,population_variance=6.,
                         running_mean=bn.running_mean.tolist(), running_variance=bn.running_var.tolist(),
                         train_output=train_output.flatten().tolist(),eval_output=eval_output.flatten().tolist(),
                         clean_ln=F.layer_norm(torch.tensor([1.,3.],dtype=dtype),(2,)).tolist(),
                         masked_ln=F.layer_norm(torch.tensor([2.,0.],dtype=dtype),(2,)).tolist())
    probe = nn.Sequential(nn.BatchNorm1d(2), nn.Dropout(.5))
    probe.train()
    with torch.no_grad():
        first = probe(torch.tensor([[1.,2.],[3.,4.],[5.,6.],[7.,8.]]))
    after_no_grad = int(probe[0].num_batches_tracked)
    probe.eval()
    probe[1].train()
    before = probe[0].running_mean.clone()
    with torch.no_grad():
        second = probe(torch.tensor([[1.,2.],[3.,4.]]))
    state = dict(after_no_grad_batches=after_no_grad,
                 after_mc_batches=int(probe[0].num_batches_tracked),
                 mc_running_mean_changed=not torch.equal(before,probe[0].running_mean),
                 no_grad_output_requires_grad=first.requires_grad,
                 mc_output_requires_grad=second.requires_grad)
    attention = torch.tensor([.25,.75],dtype=dtype)
    attention_masked = attention*torch.tensor([1.,0.])/.5
    boundary_input = torch.tensor([1.,-2.],dtype=dtype,requires_grad=True)
    zero = mask_values(boundary_input,1.,True)
    zero.sum().backward()
    boundaries = dict(p1_train=zero.detach().tolist(),
                      p1_gradient=boundary_input.grad.tolist(),
                      p1_eval=mask_values(boundary_input,1.,False).detach().tolist(),
                      p0_train=mask_values(boundary_input,0.,True).detach().tolist())
    calls = {"count": 0}
    def counted_branch(values):
        calls["count"] += 1
        return values * 2
    eager = mask_values(counted_branch(x),1.,True,"batch")
    eager_calls = calls["count"]
    calls["count"] = 0
    keep_branch = False
    lazy = counted_branch(x) if keep_branch else torch.zeros_like(x)
    execution = dict(eager_calls=eager_calls,lazy_calls=calls["count"],
                     eager_output=eager.tolist(),lazy_output=lazy.tolist())
    hypothetical = np.array([[.9,.1],[.1,.9]])
    entropy_mean = float(-(hypothetical.mean(0)*np.log(hypothetical.mean(0))).sum())
    mean_entropy = float(-(hypothetical*np.log(hypothetical)).sum(1).mean())
    return dict(one_update=one_update,enumerations=enumerations,granularity=granularity,
                branch=branch,schedules=schedules,normalization=normalization,state=state,
                attention=dict(original=attention.tolist(),masked=attention_masked.tolist(),
                               row_sum=attention_masked.sum().item()),
                boundaries=boundaries,execution=execution,
                hypothetical_disagreement=dict(entropy_mean=entropy_mean,
                    mean_entropy=mean_entropy,disagreement=entropy_mean-mean_entropy),
                shared_mask_covariance=2.,independent_mask_covariance=0.)


class DigitModel(nn.Module):
    def __init__(self, seed, family, probability=0., mode="row"):
        super().__init__()
        torch.manual_seed(seed)
        self.family, self.probability, self.mode = family, probability, mode
        self.stem = nn.Linear(64,64)
        self.dropout = nn.Dropout(probability if family == "mlp" else 0.)
        count = 1 if family == "mlp" else 4
        self.layers = nn.ModuleList([nn.Linear(64,64) for _ in range(count)])
        self.head = nn.Linear(64,10)

    def forward(self,x):
        h = torch.tanh(self.stem(x))
        if self.family == "mlp":
            h = self.dropout(h)
            h = self.dropout(torch.tanh(self.layers[0](h)))
        else:
            for index,layer in enumerate(self.layers):
                correction = .5 * torch.tanh(layer(h))
                probability = self.probability * index / 3
                h = h + mask_values(correction,probability,self.training,self.mode)
        return self.head(h)


def metrics_from_probabilities(probabilities,targets):
    probabilities = probabilities.double()
    one_hot = F.one_hot(targets,10)
    return dict(cross_entropy=float(-probabilities[torch.arange(len(targets)),targets].log().mean()),
                correct=int((probabilities.argmax(1)==targets).sum()),count=len(targets),
                brier=float((probabilities-one_hot).square().sum(1).mean()))


def measure(model,x,y):
    model.eval()
    with torch.no_grad():
        return metrics_from_probabilities(model(x).softmax(1),y)


def mc_measure(model,x,y):
    model.eval()
    for module in model.modules():
        if isinstance(module,nn.Dropout):
            module.train()
    torch.manual_seed(7001)
    with torch.no_grad():
        samples = torch.stack([model(x).softmax(1) for _ in range(100)])
    model.eval()
    probabilities = samples.mean(0)
    tiny = torch.finfo(samples.dtype).tiny
    entropy = -(probabilities*probabilities.clamp_min(tiny).log()).sum(1)
    mean_entropy = -(samples*samples.clamp_min(tiny).log()).sum(2).mean(0)
    return dict(metrics=metrics_from_probabilities(probabilities,y),
                sample_probabilities=samples[:,:3].tolist(),mean_probabilities=probabilities.tolist(),
                predictive_entropy=entropy.tolist(),
                disagreement=(entropy-mean_entropy).tolist(),
                probability_std=samples.std(0,correction=1).tolist(),
                targets=y.tolist(),draws=100)


def run():
    rows=np.genfromtxt(HERE/"digits-400.csv",delimiter=",",names=True)
    pixels=np.column_stack([rows[f"pixel_{i}"] for i in range(64)])
    x=torch.tensor(pixels/16,dtype=torch.float32)
    y=torch.tensor(rows["digit"].astype(int),dtype=torch.long)
    train,validation=train_test_split(np.arange(len(rows)),test_size=.3,stratify=y.numpy(),random_state=22)
    configurations=[("mlp",p,"row") for p in (0.,.2,.5,.8)]
    configurations += [("residual",0.,"row")]
    configurations += [("residual",p,mode) for p in (.2,.5) for mode in ("row","batch")]
    results=[];mc=None
    for seed in (1,2,3):
        for family,probability,mode in configurations:
            model=DigitModel(seed,family,probability,mode)
            optimizer=torch.optim.Adam(model.parameters(),lr=.003)
            torch.manual_seed(1000+seed)
            trace=[]
            for step in range(401):
                if step in (0,1,25,100,200,400):
                    trace.append(dict(step=step,train=measure(model,x[train],y[train]),
                                      validation=measure(model,x[validation],y[validation])))
                if step==400:break
                model.train();optimizer.zero_grad(set_to_none=True)
                loss=F.cross_entropy(model(x[train]),y[train])
                loss.backward();optimizer.step()
            record=dict(seed=seed,family=family,drop_probability=probability,mode=mode,
                        parameters=sum(p.numel() for p in model.parameters()),trace=trace)
            results.append(record)
            print(seed,family,probability,mode,trace[-1]["validation"],flush=True)
            if seed==1 and family=="mlp" and probability==.5:
                mc=mc_measure(model,x[validation],y[validation])
    output=dict(versions=dict(python=platform.python_version(),torch=torch.__version__,
                              numpy=np.__version__,sklearn=sklearn.__version__),
                split=dict(train_source_ids=rows["source_id"][train].astype(int).tolist(),
                           validation_source_ids=rows["source_id"][validation].astype(int).tolist()),
                mechanisms=fixtures(),fits=results,monte_carlo=mc)
    (HERE/"calculated-inputs.json").write_text(json.dumps(output,indent=2,allow_nan=False),encoding="utf-8")


if __name__=="__main__":
    run()
