import importlib.util
import json
import hashlib
from pathlib import Path
import sys
import numpy as np
sys.dont_write_bytecode = True
path = Path('public/learn-assets/backpropagation/teaching-autodiff.py')
receipt = Path('docs/teaching/evidence/backprop-independent-semantics.json')
receipt.write_text(json.dumps({'status':'running','scope':'Complementary semantic invariants; not yet passed all checks.'},indent=2)+'\n',encoding='utf-8')
spec = importlib.util.spec_from_file_location('backprop_review_engine', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Tensor = module.Tensor
checks = []
def checked(name, actual, expected, tol=1e-12):
    np.testing.assert_allclose(actual, expected, atol=tol, rtol=tol)
    checks.append({'name': name, 'actual': np.asarray(actual).tolist(), 'expected': np.asarray(expected).tolist()})
# Nonuniform output weighting and repeated operands exercise the declared seed API.
x = Tensor(np.array([[.2,-.7,1.1],[1.4,.3,-.5]]), True)
result = x*x + .25*x
seed = np.array([[1.,-2.,3.],[-.5,.7,1.2]])
result.backward(seed)
checked('Repeated operand with nonuniform vector-output seed',x.grad,seed*(2*x.data+.25))
# Three forward uses of one vector, followed by square then mean: reduces exactly once.
bias = Tensor(np.array([.4,-.6]), True)
inputs = np.array([[1.,2.],[-1.,3.],[2.,-.5]])
outputs = Tensor(inputs)+bias
loss = (outputs*outputs).mean()
loss.backward()
checked('Broadcast bias plus mean reduces axes and includes one mean factor',bias.grad,2*(inputs+bias.data).sum(axis=0)/6)
# Constant frozen weight still transmits input gradients, independent of its own grad flag.
x = Tensor([[.2,-.7]],True)
weight = Tensor([[1.,2.],[-.3,.8]],False)
result = x@weight
result.backward(np.array([[.4,-1.3]]))
checked('Frozen matrix transmits weighted input gradient',x.grad,np.array([[.4,-1.3]])@weight.data.T)
checked('Frozen matrix does not accumulate parameter gradient',weight.grad,np.zeros((2,2)))
# CE is invariant under arbitrary per-row additive logit shifts.
base = np.array([[-.4,1.2],[.7,-.2]])
labels = np.array([1,0])
a = Tensor(base,True); b = Tensor(base+np.array([[1000.],[-1000.]]),True)
la=a.cross_entropy(labels); lb=b.cross_entropy(labels)
la.backward();lb.backward()
checked('Row-specific logit offsets preserve stable CE',la.data,lb.data)
checked('Row-specific logit offsets preserve logit gradients',a.grad,b.grad)
checked('Each row CE gradient sums to zero',a.grad.sum(axis=1),[0,0])
# One parameter matrix occurs twice in one matmul, then feeds a scalar weighted loss.
w = Tensor([[.2,-.7],[1.1,.3]], True)
weighting = np.array([[1.,2.],[-.5,.8]])
(w@w).backward(weighting)
checked('Same matrix occupies both matmul operands',w.grad,weighting@w.data.T+w.data.T@weighting)
original=np.array([1.,2.]); immutable=Tensor(original,True);original[0]=99
checked('Constructor copies input before external mutation',immutable.data,[1,2])
try:
    immutable.data[0]=17
    raise AssertionError('Forward values unexpectedly writable')
except ValueError:
    checks.append({'name':'Forward values reject ordinary item mutation','passed':True})
try:
    (immutable*immutable).backward()
    raise AssertionError('Non-scalar backward accepted an implicit seed')
except ValueError:
    checks.append({'name':'Non-scalar backward requires explicit seed','passed':True})
try:
    immutable.backward(np.array([[1.,1.]]))
    raise AssertionError('Mismatched seed shape accepted')
except ValueError:
    checks.append({'name':'Mismatched explicit seed shape rejected','passed':True})
for label in [np.array([1.,0.]),np.array([[1],[0]]),np.array([2,0])]:
    try:
        Tensor(base,True).cross_entropy(label)
        raise AssertionError('Invalid labels accepted')
    except ValueError:
        pass
checks.append({'name':'Floating, broadcast-shaped and out-of-range labels rejected','passed':True})
record={'status':'passed','reviewer':'independent-backprop-review','scope':'Complementary semantic invariants on teaching engine only; no rerun of the full training/native campaign.','sourceHash':hashlib.sha256(path.read_bytes()).hexdigest(),'verifierHash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'checks':checks}
receipt.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':'passed','namedChecks':len(checks),'engineHash':record['sourceHash']}))
