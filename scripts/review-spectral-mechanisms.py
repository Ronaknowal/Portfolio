"""Independent SVD/autograd and frozen-function oracles; no training or author-model import."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(3819)
root = Path(__file__).resolve().parents[1]
output = root / 'docs/teaching/deep-learning-completion/spectral-normalization-gradient-penalty/independent-fixtures.json'
fixtures = dict(matrices=[], convolutions=[], penalties=[], forwards=[], library=[])
for _ in range(24):
    w = rng.uniform(-3, 3, (2, 2)); h = rng.normal(size=(2, 2)); target = float(rng.uniform(.15, 2))
    u, s, vt = np.linalg.svd(w)
    weight = torch.tensor(w, requires_grad=True)
    loss = (weight / torch.linalg.svdvals(weight)[0] * torch.tensor(h)).sum()
    gradient = torch.autograd.grad(loss, weight)[0]
    fixtures['matrices'].append(dict(w=w.tolist(), h=h.tolist(), target=target, values=s.tolist(),
        gradient=gradient.tolist(), capped=(u @ np.diag(np.minimum(s, target)) @ vt).tolist(),
        exact=(target*w/s[0]).tolist(), cap=(w/max(1, s[0]/target)).tolist(),
        frobenius=(target*w/np.linalg.norm(w)).tolist()))
for n in range(3, 7):
    for stride in (1, 2):
        for mode in ('valid', 'circular'):
            k = rng.uniform(-4, 4, 2); x = rng.normal(size=n)
            starts = list(range(0, n if mode == 'circular' else n-1, stride))
            # Literal windows applied to every basis input, independent of the JS Gram construction.
            operation = lambda v: np.array([k[0]*v[i] + k[1]*v[(i+1) % n] for i in starts])
            a = np.stack([operation(v) for v in np.eye(n)], axis=1)
            fixtures['convolutions'].append(dict(n=n, stride=stride, mode=mode, kernel=k.tolist(),
                x=x.tolist(), matrix=a.tolist(), output=operation(x).tolist(), sigma=float(np.linalg.svd(a, compute_uv=False)[0])))
for kind in ('target-one', 'one-sided', 'zero'):
    for radius in (.2, .7, 1., 1.7, 4.):
        w = torch.tensor([.6*radius, -.8*radius], requires_grad=True); strength = 1.7; rate = .13
        length = torch.linalg.vector_norm(w)
        loss = strength*(length.square() if kind == 'zero' else torch.relu(length-1).square() if kind == 'one-sided' else (length-1).square())
        derivative = torch.autograd.grad(loss, w)[0]
        fixtures['penalties'].append(dict(kind=kind, w=w.tolist(), strength=strength, rate=rate,
            loss=loss.item(), gradient=derivative.tolist(), updated=(w-rate*derivative).tolist()))
for file in sorted((root/'public/learn-code/spectral-normalization-gradient-penalty').glob('model-*.json')):
    saved = json.loads(file.read_text())
    for kind in ('generator', 'critic'):
        layers = saved[kind+'_layers']
        weights = [(torch.tensor(v['weight']), torch.tensor(v['bias'])) for v in layers]
        def forward(x):
            value = x
            for i, (w, b) in enumerate(weights):
                value = F.linear(value, w, b)
                if i < len(weights)-1:
                    value = F.relu(value) if kind == 'generator' else F.leaky_relu(value, .2)
                elif kind == 'generator':
                    value = torch.sigmoid(value)
            return value
        for point in rng.uniform(-1.4, 1.4, (3, 2)):
            x = torch.tensor(point, requires_grad=True)
            result = forward(x); jac = torch.autograd.functional.jacobian(forward, x)
            fixtures['forwards'].append(dict(file=file.name, kind=kind, point=point.tolist(), output=result.tolist(), jacobian=jac.tolist()))
for _ in range(6):
    layer = torch.nn.utils.parametrizations.spectral_norm(torch.nn.Linear(2, 2, bias=False))
    w = rng.uniform(-2, 2, (2, 2)); u = rng.normal(size=2); u /= np.linalg.norm(u); v = rng.normal(size=2); v /= np.linalg.norm(v)
    with torch.no_grad():
        layer.parametrizations.weight.original.copy_(torch.tensor(w))
        layer.parametrizations.weight[0]._u.copy_(torch.tensor(u)); layer.parametrizations.weight[0]._v.copy_(torch.tensor(v))
    records = []
    for training in (True, True, False, True, False):
        layer.train(training)
        effective = layer.weight.detach().clone()
        p = layer.parametrizations.weight[0]
        records.append(dict(training=training, effective=effective.tolist(), u=p._u.tolist(), v=p._v.tolist()))
    fixtures['library'].append(dict(w=w.tolist(), initial=dict(u=u.tolist(), v=v.tolist()), records=records))
output.write_text(json.dumps(fixtures, indent=2, allow_nan=False)+'\n')
print(json.dumps({k: len(v) for k, v in fixtures.items()}))
