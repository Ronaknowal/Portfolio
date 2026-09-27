"""Complementary review: native full frozen reader and fresh input derivatives.

Does not import the author model or repeat fitting / distributed transport tests.
"""
from pathlib import Path
import json
import inspect
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
DRAFT = ROOT / 'docs/teaching/drafts/ring-attention-sequence-parallelism'
torch.set_num_threads(1)
packet = json.loads((DRAFT / 'movement-attention-model.json').read_text())
weights = {k: torch.tensor(v, dtype=torch.float64) for k, v in packet['state_dict'].items()}
data = np.loadtxt(DRAFT / 'movement_libras.data', delimiter=',')

def forward(points):
    stem = F.linear(2*points-1, weights['stem.weight'], weights['stem.bias']).tanh()
    q, k, v = [F.linear(stem, weights['mixer.'+name+'.weight']).view(45, 2, 12).transpose(0, 1)
               for name in ['query', 'key', 'value']]
    output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.)
    joined = output.transpose(0, 1).reshape(45, 24)
    pooled = (stem + F.linear(joined, weights['mixer.output.weight'])).mean(0)
    logits = F.linear(pooled, weights['classifier.weight'], weights['classifier.bias'])
    return dict(features=stem, query=q, key=k, value=v, output=output,
                logits=logits, probabilities=logits.softmax(0))

cases = []
source = data[76, :90].reshape(45, 2).copy()
source[[0, 21, 44]] = [[.13, .82], [.97, .02], [.45, .68]]
fresh = data[19, :90].reshape(45, 2)[np.roll(np.arange(45)[::-1], 7)].copy()
fresh[13] = [.01, .99]
for label, points in [('source77-three-point-edit', source),
                      ('source20-permuted-edited', fresh),
                      ('uninformative-constant', np.full((45, 2), .5))]:
    x = torch.tensor(points, dtype=torch.float64, requires_grad=True)
    result = forward(x)
    loss = .5*result['logits'].square().sum()
    gradient, = torch.autograd.grad(loss, x)
    cases.append(dict(name=label, points=points.tolist(),
                      expected={k:v.detach().tolist() for k,v in result.items()},
                      loss=loss.item(), gradient=gradient.tolist()))

rng = np.random.default_rng(86741)
attention = []
for label, scale in [('ordinary', .8), ('large-magnitude', 25.)]:
    q, k = [torch.tensor(rng.normal(size=(3, 9, 4))*scale, dtype=torch.float64, requires_grad=True) for _ in range(2)]
    v = torch.tensor(rng.normal(size=(3, 9, 2)), dtype=torch.float64, requires_grad=True)
    upstream = torch.tensor(rng.normal(size=(3, 9, 2)), dtype=torch.float64)
    positions=np.array([3, 0, 2, 1, 1, 0, 3, 2, 4]);documents=np.array([0,0,0,0,1,1,1,1,1])
    mask=(positions[None,:]<=positions[:,None])&(documents[None,:]==documents[:,None])
    mask[[1,5]]=False
    mask[:,7]=False  # a padding record is not a legal key
    actual=F.scaled_dot_product_attention(q,k,v,attn_mask=torch.tensor(mask),dropout_p=0.)
    grads=torch.autograd.grad((actual*upstream).sum(),[q,k,v])
    attention.append(dict(name=label,query=q.detach().tolist(),key=k.detach().tolist(),value=v.detach().tolist(),
                          upstream=upstream.tolist(),mask=mask.tolist(),positions=positions.tolist(),documents=documents.tolist(),
                          output=actual.detach().tolist(),gradients=[g.tolist() for g in grads]))

def arrays(value):
    if isinstance(value, dict): return sum(arrays(v) for v in value.values())
    if isinstance(value, list): return sum(arrays(v) for v in value)
    if isinstance(value, (int,float)) and not isinstance(value,bool):
        assert np.isfinite(value)
        return 1
    return 0

structured={}
for name in ['movement-attention-model.json','partition-results.json','systems-results.json']:
    value=json.loads((DRAFT/name).read_text(),parse_constant=lambda text: (_ for _ in ()).throw(ValueError(text)))
    structured[name]=dict(topLevelKeys=list(value),numericFields=arrays(value))
assert data.shape==(360,91) and np.isfinite(data).all()
assert sum(t.numel() for t in weights.values())==2751
assert 'async_op' not in inspect.signature(torch.distributed.batch_isend_irecv).parameters
result=dict(passed=True,torch=torch.__version__,threads=1,fullModelCases=cases,attentionCases=attention,
            packetInspection=structured,rawShape=list(data.shape),rawNumericValues=int(data.size),
            apiSignature=str(inspect.signature(torch.distributed.batch_isend_irecv)),
            limits=['Native SDPA and input-autograd reference only; no new training, GPU or transport benchmark.',
                    'Existing actual Gloo P1/P2/P3/L8 evidence is reused for unchanged canonical transport routines.'])
(HERE/'independent-native-fixtures.json').write_text(json.dumps(result,allow_nan=False,separators=(',',':'))+'\n')
print('Three full native readers,270 input gradients and two asymmetric/empty/saturated attention cases saved.')
