"""Independent reviewer: direct native operators, not the author's model class."""
from pathlib import Path
import hashlib
import json
import math
import subprocess
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[4]
ID = 'grouped-query-attention-gqa-multi-query-attention-mqa'
HERE = Path(__file__).resolve().parent
torch.set_num_threads(2)
torch.set_default_dtype(torch.float64)
torch.manual_seed(291)
checks = []


def check(name, value, **evidence):
    assert value, name
    checks.append({'name': name, 'passed': True, **evidence})


def independent_forward(w, points, groups, shift=0):
    w = {k: torch.tensor(v) for k, v in w.items()}
    def linear(x, name):
        return F.linear(x, w[name + '.weight'], w[name + '.bias'])
    def norm(x, name):
        return F.layer_norm(x, (24,), w[name + '.weight'], w[name + '.bias'], 1e-5)
    def rotary(x):
        theta = torch.arange(shift, shift + len(points))[:, None] * 10000 ** (-torch.arange(0, 6, 2) / 6)
        c, s = theta.cos(), theta.sin()
        return torch.stack((x[..., ::2] * c - x[..., 1::2] * s,
                            x[..., ::2] * s + x[..., 1::2] * c), -1).flatten(-2)
    x = linear(torch.tensor(points, dtype=torch.float64) * 2 - 1, 'stem')
    n = norm(x, 'norm_attention')
    q = rotary(linear(n, 'query').reshape(len(points), 4, 6).transpose(0, 1))
    k = rotary(linear(n, 'key').reshape(len(points), groups, 6).transpose(0, 1))
    v = linear(n, 'value').reshape(len(points), groups, 6).transpose(0, 1)
    # Native SDPA is independent of both lesson's grouped einsum and browser loop.
    h = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    x = x + linear(h.transpose(0, 1).reshape(len(points), 24), 'output')
    x = x + linear(F.gelu(linear(norm(x, 'norm_feedforward'), 'feedforward.0')), 'feedforward.2')
    return (linear(norm(x, 'norm_final'), 'forecast') + 1) / 2


js = r"""
import fs from 'node:fs';
import {groupedForecast,groupedRollout,groupedRead,groupedDefaults} from './src/learn/data/grouped-query-models.js';
const id='grouped-query-attention-gqa-multi-query-attention-mqa';
const cases=[];
for(const h of [1,2,4]) {
 const {weights}=JSON.parse(fs.readFileSync(`public/learn-assets/${id}/runtime-${h}.json`));
 for(const kind of ['constant','alternating','curved']) {
  const points=Array.from({length:kind==='curved'?29:7},(_,i)=>kind==='constant'?[.5,.5]:kind==='alternating'?[i%2,1-i%2]:[.5+.4*Math.sin(i*.71),.5+.4*Math.cos(i*.43)]);
  const out=groupedRollout(weights,points,h,5,37);
  let cache=null,pred=[];
  for(let i=0;i<points.length;i+=3){const s=groupedForecast(weights,points.slice(i,i+3),h,37,cache);cache=s.cache;pred.push(...s.predictions);}
  cases.push({h,kind,points,predictions:out.predictions,rollout:out.rollout,chunked:pred});
 }
}
const future=groupedDefaults();future.positions=[0,1,9];future.queryPosition=1;
const first=groupedRead(future);future.keys.forEach(g=>g[2]=[4000,4000]);future.values.forEach(g=>g[2]=[-999,999]);
console.log(JSON.stringify({cases,futureOriginal:first,futureChanged:groupedRead(future)}));
"""
result = subprocess.run(['node', '--input-type=module', '-e', js], cwd=ROOT, capture_output=True, text=True, check=True)
observed = json.loads(result.stdout)
max_error = 0.
for case in observed['cases']:
    w = json.loads((ROOT / f'public/learn-assets/{ID}/runtime-{case["h"]}.json').read_text())['weights']
    out = independent_forward(w, case['points'], case['h'], 37)
    error = float((out - torch.tensor(case['predictions'])).abs().max())
    max_error = max(max_error, error)
    check(f"Independent SDPA full-model replay: {case['h']} KV, {case['kind']}", error < 2e-5, maxError=error)
    check(f"Three-point chunks match full causal inference: {case['h']} KV, {case['kind']}", torch.allclose(out, torch.tensor(case['chunked']), atol=2e-5, rtol=0))
    prefix = list(case['points'])
    generated = []
    for _ in range(5):
        point = independent_forward(w, prefix, case['h'], 37)[-1].tolist()
        generated.append(point)
        prefix.append(point)
    check(f"Rollout from full recomputation has no future feed: {case['h']} KV, {case['kind']}", torch.allclose(torch.tensor(generated), torch.tensor(case['rollout']), atol=2e-5, rtol=0))
check('Arbitrarily large future keys and values have no effect', observed['futureOriginal'][0]['output'] == observed['futureChanged'][0]['output'] and all(a['weights'] == b['weights'] and a['output'] == b['output'] for a,b in zip(observed['futureOriginal'], observed['futureChanged'])))

# New six-reader, two-memory, unequal key/value width case; compare all input gradients.
q = torch.randn(2, 6, 3, 4, requires_grad=True)
k = torch.randn(2, 2, 5, 4, requires_grad=True)
v = torch.randn(2, 2, 5, 3, requires_grad=True)
mask = torch.arange(5)[None,:] <= torch.tensor([2,3,4])[:,None]
native = F.scaled_dot_product_attention(q,k,v,attn_mask=mask,enable_gqa=True)
expanded = (q @ k.repeat_interleave(3,1).transpose(-1,-2) / 2).masked_fill(~mask,-torch.inf).softmax(-1) @ v.repeat_interleave(3,1)
upstream = torch.randn_like(native)
g1 = torch.autograd.grad((native*upstream).sum(),(q,k,v),retain_graph=True)
g2 = torch.autograd.grad((expanded*upstream).sum(),(q,k,v))
check('Six-reader unequal-width native grouped output and Q/K/V shared gradients match explicit tied-head expansion', torch.allclose(native,expanded,atol=1e-12,rtol=0) and all(torch.allclose(a,b,atol=1e-12,rtol=0) for a,b in zip(g1,g2)))
record = json.loads((ROOT / f'public/learn-assets/{ID}/author-results.json').read_text())
for h, branch in record['branches'].items():
    minimum = min(branch['history'],key=lambda r:r['validation_mse_transformed'])
    check(f'Validation-only earliest minimum selects {h} KV checkpoint', minimum['step']==branch['selected_step'])
draft = ROOT / f'docs/teaching/drafts/{ID}'
for name in ['author-calculations.py','mechanism-calculations.py','author-results.json','forecast-models.json','mechanism-fixtures.json','data-provenance.md','movement_libras.data','movement_libras.names']:
    check(f'Deployed {name} is byte-identical to reviewed source', (draft/name).read_bytes()==(ROOT/f'public/learn-assets/{ID}'/name).read_bytes())
(HERE/'independent-native-results.json').write_text(json.dumps({'passed':True,'checks':checks,'maximumFullModelError':max_error,'environment':{'torch':torch.__version__,'threads':2},'limits':['No new training, no GPU timing, browser acceptance belongs to root.']},indent=2)+'\n')
print(f'{len(checks)} independent native/source checks passed; maximum browser versus independent SDPA model error {max_error:.3g}')
