"""Replay real saved memories and run the complete published mechanisms on CPU."""
from pathlib import Path
import contextlib
import hashlib
import importlib.util
import io
import json
import shutil
import sys
import tempfile
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
ID = 'modern-hopfield-networks'
PACKET = ROOT / 'docs/teaching/drafts' / ID
OUT = ROOT / 'docs/teaching/deep-learning-completion' / ID
PUBLIC = ROOT / 'public/learn-code' / ID
torch.set_num_threads(2)
checks = []


def check(name, condition):
    assert condition, name
    checks.append({'name': name, 'passed': True})


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def compare_tree(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: compare_tree(a[k], b[k])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b): compare_tree(x, y)
    elif isinstance(a, (float, int)):
        assert abs(a-b) <= 2e-6 + 1e-6*abs(b), (a, b)
    else: assert a == b


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def main():
    PUBLIC.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='hopfield-check-', dir=ROOT/'scratch') as temp:
        temp = Path(temp)
        for name in ['associative_memory.py', 'digit_memory.py', 'author_calculations.py', 'digit-memory-fits.npz', 'optdigits.tra', 'optdigits.tes', 'optdigits.names']:
            shutil.copyfile(PACKET/name, temp/name)
        sys.path.insert(0, str(temp))
        digit = load('digit_memory', temp/'digit_memory.py')
        mechanism = load('associative_memory', temp/'associative_memory.py')
        with contextlib.redirect_stdout(io.StringIO()):
            mechanism.main()
            calculations = load('hopfield_author_calculations', temp/'author_calculations.py')
        # NumPy archives hold a Windows file handle until explicitly closed.
        for value in vars(calculations).values():
            if isinstance(value, np.lib.npyio.NpzFile): value.close()
        for name in ['mechanism-results.json', 'investigation-checks.json']:
            compare_tree(json.loads((temp/name).read_text()), json.loads((PACKET/name).read_text()))
            check('Complete canonical program replay '+name, True)
        roles, metadata = digit.load_roles()
        archive = np.load(PACKET/'digit-memory-fits.npz')
        report = json.loads((PACKET/'digit-results.json').read_text())
        check('Original source bytes and exact declared role identities', metadata == report['data'])
        all_rows = np.vstack([np.loadtxt(PACKET/name, delimiter=',', dtype=int) for name in ['optdigits.tra', 'optdigits.tes']])
        check('All5620 original feature vectors unique', len(np.unique(all_rows[:, :64], axis=0)) == 5620)
        check('Memory/fit/validation disjoint', len(set(sum(metadata['roles_training_source_ids'].values(), []))) == 1300)
        memory, labels = roles['memory']
        for role in ['memory', 'validation']:
            check(role+' archive pixels and labels identical to original source', np.array_equal(archive[role+'_pixels'], roles[role][0].numpy()) and np.array_equal(archive[role+'_labels'], roles[role][1].numpy()))
        port = {'validation': [], 'fresh': [], 'mechanism': json.loads((temp/'mechanism-results.json').read_text()), 'investigations': json.loads((temp/'investigation-checks.json').read_text())}
        with torch.no_grad():
            for candidate in report['fixed_candidates']:
                for role in ['validation', 'test']:
                    for condition in ['clean', 'occluded']:
                        clean, truth = roles[role]
                        inputs = clean if condition == 'clean' else digit.obscure(clean)
                        log_class, weights = digit.read_memory(inputs, memory, labels, candidate['beta'])
                        compare_tree(digit.summarize(log_class, truth), candidate[role+'_'+condition])
                        nearest = (digit.unit_rows(inputs) @ digit.unit_rows(memory).T).argmax(dim=1)
                        check('Fixed '+str(candidate['beta'])+' '+role+' '+condition+' complete metrics and cosine baseline', int((labels[nearest] != truth).sum()) == report['nearest_memory'][role+'_'+condition]['errors'])
            check('Fixed beta selected only by clean validation CE', min(report['fixed_candidates'], key=lambda x: x['validation_clean']['cross_entropy'])['beta'] == 64)
            for model_id in ['fixed', 'seed17', 'seed41']:
                projection = None
                beta = 64. if model_id == 'fixed' else 16.
                if model_id != 'fixed':
                    projection = nn.Linear(64, 16, bias=False)
                    projection.weight.copy_(torch.from_numpy(archive[model_id+'_projection']))
                    run = next(r for r in report['learned'] if r['seed'] == int(model_id[4:]))
                    check(model_id+' validation-only epoch selection', min(run['training_curve'], key=lambda row: row['validation_loss_after_update'])['epoch'] == run['selected_epoch'])
                    for role in ['fit_queries', 'validation', 'test']:
                        for condition in ['clean', 'occluded']:
                            clean, truth = roles[role]
                            inputs = clean if condition == 'clean' else digit.obscure(clean)
                            lc, weights = digit.read_memory(inputs, memory, labels, beta, projection)
                            result = digit.summarize(lc, truth)
                            result['reconstruction_mse_to_clean'] = float(((weights @ memory-clean)**2).mean())
                            result['input_mse_to_clean'] = float(((inputs-clean)**2).mean())
                            compare_tree(result, run[role+'_'+condition])
                            check(model_id+' '+role+' '+condition+' full selected-fit replay', True)
                for condition in ['clean', 'occluded']:
                    clean = roles['validation'][0]
                    inputs = clean if condition == 'clean' else digit.obscure(clean)
                    lc, weights = digit.read_memory(inputs, memory, labels, beta, projection)
                    assert torch.allclose(lc, torch.from_numpy(archive[model_id+'_'+condition+'_log_class']), atol=1e-5, rtol=1e-6)
                    assert torch.allclose(weights, torch.from_numpy(archive[model_id+'_'+condition+'_weights']), atol=1e-5, rtol=1e-6)
                    port['validation'].append({'model': model_id, 'condition': condition, 'logClasses': lc.tolist(), 'weights': weights.tolist(), 'readPixels': (weights @ memory).tolist()})
                save(PUBLIC/('model-'+model_id+'.json'), {'id': model_id, 'beta': beta, 'projection': None if projection is None else archive[model_id+'_projection'].tolist(), 'epoch': None if projection is None else run['selected_epoch']})
        # Independent functional float64 fresh queries and derivatives.
        raw_cases = [[(i*7+3)%17 for i in range(64)], [0]*64, [16 if 2 <= i%8 <= 4 else 0 for i in range(64)]]
        for model_id in ['fixed', 'seed17', 'seed41']:
            w = None if model_id == 'fixed' else torch.tensor(archive[model_id+'_projection'], dtype=torch.float64)
            m = memory.double()
            k = F.normalize(m if w is None else F.linear(m, w), dim=-1, eps=1e-12)
            for raw in raw_cases:
                q = torch.tensor(raw, dtype=torch.float64, requires_grad=True) / 16
                q.retain_grad()
                embedding = q if w is None else F.linear(q, w)
                unit = F.normalize(embedding, dim=-1, eps=1e-12)
                logits = (64 if model_id == 'fixed' else 16)*k@unit
                lw = F.log_softmax(logits, dim=-1)
                lc = torch.stack([torch.logsumexp(lw[labels == i], dim=0) for i in range(10)])
                result = {'model': model_id, 'pixels': raw, 'query': unit.detach().tolist(), 'logClasses': lc.detach().tolist(), 'weights': lw.exp().detach().tolist(), 'readPixels': (lw.exp()@m).detach().tolist()}
                if any(raw):
                    (-lc[3]).backward()
                    result['lossGradientPerIntensity'] = (q.grad/16).tolist()
                port['fresh'].append(result)
        # The framework primitive is checked with distinct values and a nondefault scale.
        q = torch.tensor([[[.4, -.3], [-.2, .7]]], dtype=torch.float64, requires_grad=True)
        k = torch.tensor([[[1., .2], [-.3, .8], [.5, -.7]]], dtype=torch.float64)
        v = torch.tensor([[[2., -1.], [.3, .9], [-.8, 1.1]]], dtype=torch.float64)
        direct = torch.softmax(.7*q@k.transpose(-2, -1), -1) @ v
        native = F.scaled_dot_product_attention(q, k, v, scale=.7, dropout_p=0.)
        dg = torch.autograd.grad(direct.square().sum(), q, retain_graph=True)[0]
        ng = torch.autograd.grad(native.square().sum(), q)[0]
        check('Distinct-value SDPA scale=.7 complete values and query gradients', torch.allclose(direct, native, atol=1e-12, rtol=1e-12) and torch.allclose(dg, ng, atol=1e-12, rtol=1e-12))
        port['sdpa'] = {'query': q.detach().tolist(), 'keys': k.tolist(), 'values': v.tolist(), 'scale': .7, 'output': native.detach().tolist(), 'queryGradient': ng.tolist()}
        # Both shared projection uses receive a loss gradient; one actual Adam update.
        projection = nn.Linear(64, 16, bias=False)
        with torch.no_grad(): projection.weight.copy_(torch.from_numpy(archive['seed17_projection']))
        opt = torch.optim.Adam(projection.parameters(), lr=.005)
        queries, truth = roles['fit_queries']
        before = projection.weight.detach().clone()
        loss = F.nll_loss(digit.read_memory(queries, memory, labels, 16., projection)[0], truth)
        loss.backward()
        check('Actual shared-projection class loss has finite nonzero gradient', bool(torch.isfinite(projection.weight.grad).all() and projection.weight.grad.norm()>0))
        opt.step()
        check('One bounded real Adam update changes projection', not torch.equal(before, projection.weight))
        sys.path.pop(0)
    rows = lambda role: [{'sourceId': sid, 'label': int(label), 'pixels': (pixels.numpy()*16).astype(int).tolist()} for sid, pixels, label in zip(metadata['roles_training_source_ids'][role], *roles[role])]
    save(PUBLIC/'digit-bank.json', {'memory': rows('memory'), 'validation': rows('validation'), 'normalizationDivisor': 16, 'source': 'UCI Optical Recognition of Handwritten Digits; Alpaydin and Kaynak; CC BY4.0'})
    save(OUT/'native-port-fixtures.json', port)
    save(OUT/'native-checks.json', {'topicId': ID, 'passed': True, 'checks': checks, 'environment': {'torch': torch.__version__, 'numpy': np.__version__, 'maximumThreads': 2}, 'limits': ['Original two100-epoch fits reused, not rerun.', 'No optional research package, GPU or browser execution claimed.'], 'sdpaMaximumValueError': float((direct-native).abs().max().detach()), 'sdpaMaximumGradientError': float((dg-ng).abs().max())})
    print(json.dumps({'passed': True, 'checks': len(checks), 'validationArrays': len(port['validation']), 'validationReads': 1800, 'freshReads': len(port['fresh'])}))


if __name__ == '__main__':
    main()
