"""Replay selected native checkpoints and export bounded, source-bound browser fixtures.

No fitting, deployment, network access or foundation-model download occurs here.
"""
from pathlib import Path
import ast
import copy
import hashlib
import importlib.util
import json
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
ID = 'hybrid-ssm-transformer-architectures-jamba'
SOURCE = ROOT / 'docs/teaching/drafts' / ID
OUT = ROOT / 'docs/teaching/deep-learning-completion' / ID
PUBLIC = ROOT / 'public/learn-assets' / ID
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
sys.dont_write_bytecode = True
sys.path.insert(0, str(SOURCE))
from stroke_models import load_fit, data_split, metrics
from hybrid_mechanisms import memory_read, route, cache_bytes

for path in SOURCE.glob('*.py'):
    ast.parse(path.read_text(encoding='utf-8'))
OUT.mkdir(parents=True, exist_ok=True)
PUBLIC.mkdir(parents=True, exist_ok=True)
report = json.loads((SOURCE / 'stroke-results.json').read_text())
author = json.loads((SOURCE / 'author-results.json').read_text())
data, split = data_split()
checks = []


def check(name, passed, **evidence):
    assert passed, (name, evidence)
    checks.append(dict(name=name, passed=True, **evidence))


check('Exact dataset/split identities regenerated', split == report['data'])
for fit in report['fits']:
    model = load_fit(fit['key'])
    check(f"{fit['key']} validation-selected epoch", min(fit['history'], key=lambda row: row['validation'])['epoch'] == fit['selected_epoch'])
    check(f"{fit['key']} parameter count", sum(p.numel() for p in model.parameters()) == fit['parameters'])
    with torch.no_grad():
        for role, (x, y) in data.items():
            current = metrics(model(x), y)
            saved = fit['metrics'][role]
            check(f"{fit['key']} {role} replay", current['errors'] == saved['errors'] and current['confusion'] == saved['confusion'] and abs(current['cross_entropy'] - saved['cross_entropy']) < 1e-7, crossEntropyDifference=abs(current['cross_entropy'] - saved['cross_entropy']))
        validation = model(data['validation'][0]).numpy()
        check(f"{fit['key']} all saved validation logits", np.max(np.abs(validation - np.array(fit['validation_logits']))) < 1e-6, maximumDifference=float(np.max(np.abs(validation - np.array(fit['validation_logits'])))))
    asset = dict(key=fit['key'], pattern=fit['pattern'], seed=fit['seed'], parameters=fit['parameters'], selectedEpoch=fit['selected_epoch'], state={key: value.detach().tolist() for key, value in model.state_dict().items()})
    if fit['pattern'] != 'linear':
        (PUBLIC / f"model-{fit['key']}.json").write_text(json.dumps(asset, separators=(',', ':')) + '\n')

sources = [{"id": value['source_id'], "label": value['label'], "coordinates": value['traces']['carry']['coordinates']} for name, value in author['fixtures'].items() if name in ('worked', 'fresh')]
payload = dict(sources=sources, fits=[{key: fit[key] for key in ('key', 'pattern', 'seed', 'selected_epoch', 'parameters', 'history', 'metrics')} for fit in report['fits']], worked={mode: {key: value[key] for key in ('cache_at_boundary', 'full_probabilities', 'branch_probabilities', 'maximum_logit_difference')} for mode, value in author['fixtures']['worked']['traces'].items()})
(ROOT / 'src/learn/data/hybrid-jamba-study.js').write_text('// Selected displayed measurements only. Selected model weights are fetched on demand.\nexport const hybridStudy = ' + json.dumps(payload, separators=(',', ':')) + ';\n')
downloads = ['hybrid_mechanisms.py', 'stroke_models.py', 'inspect_stroke.py', 'deployment_example.py', 'pendigits.tra', 'pendigits.tes', 'pendigits.names', 'stroke-fits.npz', 'stroke-results.json', 'data-provenance.md']
for name in downloads:
    shutil.copyfile(SOURCE / name, PUBLIC / name)


def native_trace(model, coordinates, boundary, mode, donor):
    points = torch.tensor(np.asarray(coordinates) / 50 - 1, dtype=torch.float64)[None]
    with torch.no_grad():
        full = model(points, True)
        if boundary:
            left, cache = model.stream(points[:, :boundary])
        else:
            left, cache = full[:, :0], [None] * 3
        snapshot = copy.deepcopy(cache)
        if mode == 'cache-swap' and boundary:
            other = torch.tensor(np.asarray(donor) / 50 - 1, dtype=torch.float64)[None]
            _, cache = model.stream(other[:, :boundary])
        for index, kind in enumerate(model.pattern):
            if (mode == 'recurrent-reset' and kind == 'M') or (mode == 'kv-reset' and kind == 'A'):
                cache[index] = None
            if mode == 'convolution-reset' and kind == 'M' and cache[index] is not None:
                cache[index] = (cache[index][0], torch.zeros_like(cache[index][1]))
        if boundary < 8:
            right, after = model.stream(points[:, boundary:], cache, offset=0 if mode == 'position-reset' else boundary)
            branch = torch.cat([left, right], 1)
        else:
            branch, after = left, cache
        converted = []
        for kind, value in zip(model.pattern, snapshot):
            if value is None:
                converted.append(None)
            elif kind == 'M':
                converted.append(dict(state=value[0][0].tolist(), history=value[1][0].tolist()))
            else:
                converted.append(dict(keys=value[0][0].tolist(), values=value[1][0].tolist()))
        return dict(full=full[0].tolist(), branch=branch[0].tolist(), probabilities=branch[0, -1].softmax(-1).tolist(), cache=converted)


fixtures = dict(traces=[], gradients=[], memory=[], routing=[], budgets=[])
worked, fresh = (sources[i]['coordinates'] for i in range(2))
edited = copy.deepcopy(fresh)
edited[3] = [50, 50]
modes = ['carry', 'recurrent-reset', 'kv-reset', 'convolution-reset', 'position-reset', 'cache-swap']
for fit in report['fits']:
    if fit['pattern'] == 'linear':
        continue
    model = load_fit(fit['key']).double()
    for label, coordinates in [('worked', worked), ('fresh', fresh), ('edited', edited), ('constant', [[50, 50]] * 8)]:
        for mode in modes:
            fixtures['traces'].append(dict(key=fit['key'], label=label, coordinates=coordinates, boundary=3, mode=mode, donor=worked, **native_trace(model, coordinates, 3, mode, worked)))
    for boundary in (0, 8):
        for mode in modes:
            fixtures['traces'].append(dict(key=fit['key'], label='boundary', coordinates=fresh, boundary=boundary, mode=mode, donor=worked, **native_trace(model, fresh, boundary, mode, worked)))
    raw = torch.tensor(fresh, dtype=torch.float64, requires_grad=True)
    logits = model((raw / 50 - 1)[None])[0]
    gradient = torch.autograd.grad(logits[2], raw)[0]
    fixtures['gradients'].append(dict(key=fit['key'], coordinates=fresh, classIndex=2, rawGradient=gradient.tolist()))

for values, keys, query, decay, beta in [([3,8,1,5],['A','B','A','C'],'A',.75,np.log(4)),([4,4,4],['A','B','C'],'C',.2,np.log(99)),([2,7,4],['A','B','C'],'Absent',.5,np.log(9)),([3,-5,2],['A','C','A'],'A',0,0),([3,-5,2],['A','C','A'],'A',1,np.log(100))]:
    fixtures['memory'].append(dict(values=values,keys=keys,query=query,decay=decay,beta=beta,result=memory_read(values,keys,query,decay,beta)))
for k in range(1,5):
    for renormalize in (False, True):
        for logits in ([0, np.log(3), np.log(6), np.log(2)], [0,0,0,0], [-8,8,0,2]):
            values=[[1,2],[3,0],[-1,4],[2,-2]]
            fixtures['routing'].append(dict(logits=logits,values=values,k=k,renormalize=renormalize,result=route(logits,values,k,renormalize)))
for length in (0, 1, 2048, 262144):
    for attention in (0, 4, 32):
        for window in (None, 4096):
            fixtures['budgets'].append(dict(length=length,attentionLayers=attention,window=window,result=cache_bytes(length,attention_layers=attention,window=window)))
(OUT/'native-fixtures.json').write_text(json.dumps(fixtures,separators=(',',':'))+'\n')
receipt=dict(passed=True,checks=checks,fixtureTraces=len(fixtures['traces']),nativeGradientCoordinates=sum(np.size(g['rawGradient']) for g in fixtures['gradients']),downloadAllowlist=downloads,limitations=['All six saved checkpoints replayed; the original fitting campaign was retained without retraining.','Large pretrained Jamba deployment is syntax/source checked only; no CUDA kernels or foundation-model weights were run.'],sourceHashes={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SOURCE.iterdir()) if p.is_file()})
(OUT/'native-checks.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(passed=True,checks=len(checks),traces=len(fixtures['traces']),gradientCoordinates=receipt['nativeGradientCoordinates'])))
