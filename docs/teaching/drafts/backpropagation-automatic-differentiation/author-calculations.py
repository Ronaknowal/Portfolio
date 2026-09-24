"""Bounded author evidence for the written examples; no app/runtime checks."""
from pathlib import Path
import importlib.util
import json
import math
import sys
import numpy as np
import torch
from torch.nn import functional as F
import sklearn

directory = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("teaching_autodiff", directory/"teaching-autodiff.py")
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)
Tensor = engine.Tensor
torch.set_num_threads(1)
record = {"versions":{"python":"3.12.14","numpy":np.__version__,
                      "torch":torch.__version__,"sklearn":sklearn.__version__}}
record["graph"]=[]
for coefficient in [2., -1., 0.]:
    x=Tensor(3.,True)
    square=x*x
    loss=square+coefficient*square
    loss.backward()
    record["graph"].append({"coefficient":coefficient,"x":3.,"square":9.,
                            "loss":float(loss.data),"gradient":float(x.grad)})
    first=x.grad.copy()
    loss.backward()
    assert np.array_equal(first,x.grad)
cases=[
    ("repeat",np.array([.3,-.7]),lambda x:(x*x+x).sum(),lambda x:(x*x+x).sum()),
    ("broadcast",np.array([[.3],[-.7]]),
     lambda x:(x+np.array([1.,2.,3.])).sum(),
     lambda x:(x+torch.tensor([1.,2.,3.],dtype=torch.float64)).sum()),
    ("tanh",np.array([.3,-.7]),lambda x:x.tanh().sum(),lambda x:x.tanh().sum()),
    ("relu",np.array([.3,-.7,0]),lambda x:x.relu().sum(),lambda x:x.relu().sum()),
    ("exp_log",np.array([.3,.7]),lambda x:x.log().exp().sum(),lambda x:x.log().exp().sum()),
    ("matrix",np.array([[.3,-.7],[1.2,.4]]),
     lambda x:(x@Tensor(np.array([[1.,2.,3.],[4.,5.,6.]]))).sum(),
     lambda x:(x@torch.tensor([[1.,2.,3.],[4.,5.,6.]],dtype=torch.float64)).sum()),
]
record["primitiveChecks"]=[]
for name,values,ours,theirs in cases:
    x=Tensor(values,True); loss=ours(x);loss.backward()
    tx=torch.tensor(values,requires_grad=True,dtype=torch.float64)
    target=theirs(tx);target.backward()
    error=np.max(np.abs(x.grad-tx.grad.numpy()))
    assert error<1e-12
    record["primitiveChecks"].append({"case":name,"gradient":x.grad.tolist(),"maxAbsError":float(error)})
logits=Tensor([[1000.,-1000.],[1.,2.]],True)
loss=logits.cross_entropy(np.array([1,0]));loss.backward()
tx=torch.tensor(logits.data.copy(),requires_grad=True,dtype=torch.float64)
tloss=F.cross_entropy(tx,torch.tensor([1,0]));tloss.backward()
assert np.allclose(logits.grad,tx.grad.numpy(),atol=1e-12)
record["stableLoss"]={"loss":float(loss.data),"torchLoss":float(tloss.detach()),"gradient":logits.grad.tolist()}
parameters=engine.make_parameters(5,(2,3,2))
features=np.array([[.2,-.4],[.8,1.1]])
labels=np.array([1,0])
loss=engine.predict(features,parameters).cross_entropy(labels)
loss.backward()
torch_parameters=[torch.tensor(p.data.copy(),requires_grad=True,dtype=torch.float64) for p in parameters]
tw1,tb1,tw2,tb2=torch_parameters
tloss=F.cross_entropy(torch.tanh(torch.tensor(features)@tw1+tb1)@tw2+tb2,torch.tensor(labels))
tloss.backward()
record["networkGradientChecks"]=[]
for index,(parameter,tparameter) in enumerate(zip(parameters,torch_parameters)):
    error=float(np.max(np.abs(parameter.grad-tparameter.grad.numpy())))
    assert error<1e-12
    fd=np.zeros_like(parameter.data)
    for position in np.ndindex(parameter.data.shape):
        plus=[p.data.copy() for p in parameters]
        minus=[p.data.copy() for p in parameters]
        plus[index][position]+=1e-5
        minus[index][position]-=1e-5
        lp=engine.predict(features,[Tensor(p) for p in plus]).cross_entropy(labels)
        lm=engine.predict(features,[Tensor(p) for p in minus]).cross_entropy(labels)
        fd[position]=(lp.data-lm.data)/2e-5
    record["networkGradientChecks"].append({"parameter":index,"shape":list(parameter.data.shape),
        "maxTorchAbsError":error,"maxFiniteDifferenceAbsError":float(np.max(np.abs(fd-parameter.grad)))})
record["finiteDifferences"]=[]
for step in [1.,.1,.01,1e-3,1e-5,1e-7,1e-9,1e-11,1e-13,1e-15]:
    difference=(math.sin(1+step)-math.sin(1-step))/(2*step)
    offset=((1e12+(1+step))-(1e12+(1-step)))/(2*step)
    record["finiteDifferences"].append({"step":step,"sinDerivative":difference,
        "sinAbsoluteError":abs(difference-math.cos(1)), "offsetDerivative":offset})
record["nonsmooth"]={"reluAtZero":0.,"centralAtZero":.5,"squareAtZero":0.}
point=np.array([.3,.7]);direction=np.array([1.,2.]);seed=np.array([1.,-1.,2.])
def vector_function(x):
    return torch.stack([x[0]*x[1],torch.sin(x[0]),x[1]**2])
tx=torch.tensor(point,dtype=torch.float64)
value,jvp=torch.func.jvp(vector_function,(tx,),(torch.tensor(direction),))
value,pullback=torch.func.vjp(vector_function,tx)
vjp=pullback(torch.tensor(seed))[0]
record["products"]={"value":value.tolist(),"jvp":jvp.tolist(),"vjp":vjp.tolist(),
                    "dualityLeft":float(torch.tensor(seed)@jvp),
                    "dualityRight":float(vjp@torch.tensor(direction))}
tx=torch.tensor([.3,.7],requires_grad=True,dtype=torch.float64)
loss=tx[0]**2+tx[0]*tx[1]+3*tx[1]**2
gradient=torch.autograd.grad(loss,tx,create_graph=True)[0]
hvp=torch.autograd.grad(gradient@torch.tensor(direction),tx)[0]
record["higherOrder"]={"gradient":gradient.tolist(),"hvp":hvp.tolist()}
record["updates"]=[]
for rate in [0.,.1,1.]:
    w,b=1.-rate*(-2.),0.-rate*(-1.)
    predictions=w*np.array([1.,2.])+b
    loss=float(np.mean((predictions-np.array([1.,3.]))**2))
    record["updates"].append({"rate":rate,"weight":w,"bias":b,"predictions":predictions.tolist(),"mse":loss})
features=torch.arange(1.,6.,dtype=torch.float64);targets=2*features
record["accumulation"]={}
for mode in ["full","weighted","unweighted"]:
    weight=torch.tensor(0.,requires_grad=True,dtype=torch.float64)
    if mode=="full":
        ((weight*features-targets)**2).mean().backward()
    else:
        for indices in [slice(0,2),slice(2,5)]:
            errors=weight*features[indices]-targets[indices]
            loss=(errors**2).mean()
            if mode=="weighted":loss=loss*len(errors)/5
            loss.backward()
    record["accumulation"][mode]=weight.grad.item()
record["xorTraining"]=engine.train_xor()
record["digitTraining"]=engine.train_digits()
(directory/"calculated-inputs.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
print(json.dumps({key:record[key] for key in ["graph","stableLoss","products","higherOrder","updates","accumulation","digitTraining"]}))
