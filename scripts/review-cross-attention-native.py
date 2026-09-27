"""Independent functional references; no lesson implementation imports."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)
rng = np.random.default_rng(927)
torch.manual_seed(927)
cases = []
for index in range(24):
    readers, slots, keys, values = 2+index%3, 2+index%5, 2+index%3, 1+index%4
    q = rng.normal(size=(readers, keys))
    k = rng.normal(size=(slots, keys))
    v = rng.normal(size=(slots, values))
    allowed = rng.random((readers, slots)) > .35
    allowed[:, 0] = True
    scores = torch.tensor(q) @ torch.tensor(k).T / keys**.5
    weights = scores.masked_fill(~torch.tensor(allowed), -torch.inf).softmax(-1)
    cases.append(dict(q=q.tolist(), k=k.tolist(), v=v.tolist(), allowed=allowed.tolist(),
                      weights=weights.tolist(), output=(weights@torch.tensor(v)).tolist()))
models = []
for index in range(12):
    width, heads = [(4, 2), (8, 2), (12, 3)][index%3]
    layer = nn.MultiheadAttention(width, heads, dropout=0, batch_first=True).double().eval()
    q = torch.randn(1, 2+index%3, width, dtype=torch.float64)
    m = torch.randn(1, 3+index%4, width, dtype=torch.float64)
    allowed = torch.rand(q.shape[1], m.shape[1]) > .35
    allowed[:, 0] = True
    with torch.no_grad():
        out, weights = layer(q, m, m, attn_mask=~allowed, average_attn_weights=False)
    models.append(dict(heads=heads, query=q[0].tolist(), memory=m[0].tolist(),
                       allowed=allowed.tolist(), state={k:v.tolist() for k,v in layer.state_dict().items()},
                       output=out[0].tolist(), weights=weights[0].tolist()))
gates = []
for index in range(21):
    x, value, target, alpha = rng.uniform(-3, 3, 4)
    if index == 0: alpha = 0.
    if index == 1: value = 0.
    a, v = [torch.tensor(t, dtype=torch.float64, requires_grad=True) for t in (alpha, value)]
    y = x + a.tanh()*v
    loss = .5*(y-target)**2
    da, dv = torch.autograd.grad(loss, (a,v))
    gates.append(dict(inputs=dict(x=x,value=value,target=target,alpha=alpha,rate=.07),
                      output=float(y.detach()), loss=float(loss.detach()),
                      da=float(da), dv=float(dv)))
destination=Path('docs/teaching/deep-learning-completion/interleaved-cross-attention-architectures')
(destination/'independent-fixtures.json').write_text(json.dumps(dict(reads=cases,models=models,gates=gates),indent=2)+'\n')
print('Created 24 native reads, 12 full multihead references and 21 autograd gates.')
