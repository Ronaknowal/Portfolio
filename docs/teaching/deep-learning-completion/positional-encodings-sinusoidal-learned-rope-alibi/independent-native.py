"""Independent reviewer: new inputs through native SDPA, no author forward import or fit."""
import json
import math
from pathlib import Path
import torch
from torch.nn import functional as F

torch.set_num_threads(2)
torch.set_default_dtype(torch.float64)
ID = 'positional-encodings-sinusoidal-learned-rope-alibi'
ROOT = Path(__file__).resolve().parent
PUBLIC = Path('public/learn-code') / ID


def rotary(x, ids):
    # Complex multiplication independently expresses the adjacent-pair rotation.
    paired = torch.view_as_complex(x.contiguous().reshape(*x.shape[:-1], -1, 2))
    frequency = torch.exp(-math.log(10000) * torch.arange(paired.shape[-1]) / paired.shape[-1])
    phase = torch.polar(torch.ones((len(ids), len(frequency))), ids[:, None] * frequency)
    return torch.view_as_real(paired * phase).flatten(-2)


def forward(model, points, ids, pads, browser_gelu=False):
    w = {k: torch.tensor(v) for k, v in model['state_dict'].items()}
    linear = lambda x, n: F.linear(x, w[n + '.weight'], w[n + '.bias'])
    norm = lambda x, n: F.layer_norm(x, (24,), w[n + '.weight'], w[n + '.bias'], 1e-5)
    z = linear(points * 2 - 1, 'stem')
    if model['mode'] == 'learned':
        z = z + F.embedding(ids.long(), w['position_table'])
    if model['mode'] == 'sinusoidal':
        phase = ids[:, None] * torch.exp(-math.log(10000) * torch.arange(12) / 12)
        z = z + torch.stack((phase.sin(), phase.cos()), -1).flatten(-2)
    packed = linear(norm(z, 'norm_attention'), 'query_key_value')
    q, k, v = (x.reshape(len(ids), 2, 12).transpose(0, 1) for x in packed.chunk(3, dim=-1))
    if model['mode'] == 'rope':
        q, k = rotary(q, ids), rotary(k, ids)
    bias = torch.zeros((2, len(ids), len(ids)))
    if model['mode'] == 'alibi':
        bias = -torch.tensor([1/16, 1/256])[:, None, None] * (ids[:, None] - ids[None, :]).abs()
    bias = bias.masked_fill(pads[None, None, :], -torch.inf)
    mixed = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0)
    attention = (q @ k.transpose(-1, -2) / math.sqrt(12) + bias).softmax(-1)
    torch.testing.assert_close(mixed, attention @ v, atol=2e-14, rtol=2e-14)
    z = z + linear(mixed.transpose(0, 1).reshape(len(ids), 24), 'attention_output')
    hidden = linear(norm(z, 'norm_feedforward'), 'feedforward.0')
    if browser_gelu:
        x = hidden.abs() / math.sqrt(2)
        t = 1 / (1 + .3275911 * x)
        erf = hidden.sign() * (1 - ((((1.061405429*t-1.453152027)*t+1.421413741)*t-.284496736)*t+.254829592)*t*torch.exp(-x*x))
        activation = .5 * hidden * (1 + erf)
    else:
        activation = F.gelu(hidden, approximate='none')
    z = z + linear(activation, 'feedforward.2')
    pooled = norm(z, 'final_norm')[~pads].mean(0)
    logits = linear(pooled, 'classifier')
    return {'logits': logits, 'probabilities': logits.softmax(-1), 'pooled': pooled, 'attention': attention}


def main():
    cases = []
    checks = []
    for mode in ['none', 'sinusoidal', 'learned', 'rope', 'alibi']:
        model = json.loads((PUBLIC / f'movement-{mode}.json').read_text(encoding='utf-8'))
        curved = [[.5 + .3 * math.cos(i * .37), .5 + .25 * math.sin(i * .51)] for i in range(11)]
        edited = [row[:] for row in model['points']]
        edited[3] = [.07, .91]
        edited[17] = [.84, .12]
        edited[39] = [.3, .7]
        interleaved, ids_pad, mask = [], [], []
        for i, p in enumerate(model['points']):
            interleaved.append(p); ids_pad.append(i); mask.append(False)
            if i in [1, 8, 19, 31, 42]:
                interleaved.append([.99, .02]); ids_pad.append(0); mask.append(True)
        samples = [('curved11', curved, [3 + 2*i for i in range(11)], [False]*11),
                   ('edited45', edited, list(range(45)), [False]*45),
                   ('interleavedPads50', interleaved, ids_pad, mask)]
        for name, points, ids, pads in samples:
            x = torch.tensor(points, requires_grad=True)
            result = forward(model, x, torch.tensor(ids), torch.tensor(pads))
            out = {k: v.detach().tolist() for k, v in result.items()}
            item = {'mode': mode, 'name': name, 'points': points, 'ids': ids, 'pads': pads, **out}
            approximate = forward(model, x, torch.tensor(ids), torch.tensor(pads), browser_gelu=True)
            item['browserGeluLogits'] = approximate['logits'].detach().tolist()
            if name == 'curved11':
                coefficients = torch.linspace(-.7, 1.1, 15)
                gradient = torch.autograd.grad(result['logits'] @ coefficients, x)[0]
                item['gradient'] = gradient.tolist()
                item['browserGeluGradient'] = torch.autograd.grad(approximate['logits'] @ coefficients, x)[0].tolist()
                permutation = torch.tensor([7, 2, 10, 0, 5, 1, 9, 3, 8, 6, 4])
                permuted = forward(model, x[permutation], torch.tensor(ids)[permutation], torch.tensor(pads)[permutation])
                torch.testing.assert_close(permuted['logits'], result['logits'], atol=1e-12, rtol=1e-12)
                checks.append(mode + ': arbitrary paired record permutation')
            cases.append(item)
        raw = forward(model, torch.tensor(model['points']), torch.arange(45), torch.zeros(45, dtype=torch.bool))
        torch.testing.assert_close(torch.tensor(cases[-1]['logits']), raw['logits'], atol=1e-12, rtol=1e-12)
        checks.append(mode + ': interleaved masked pads excluded from keys and pooling')
    author = json.loads(Path(f'docs/teaching/drafts/{ID}/author-results.json').read_text(encoding='utf-8'))
    for fit in author['fits']:
        best = max(fit['history'], key=lambda row: (row['validation']['macro_f1'], -row['validation']['loss'], -row['epoch']))
        assert best['epoch'] == fit['selected_epoch']
        checks.append(fit['mode'] + ': selection uses validation F1/loss and earliest tie')
    result = {'passed': True, 'torch': torch.__version__, 'threads': 2, 'cases': cases, 'checks': checks,
              'scope': 'Separate native functional SDPA/complex rotation/exact GELU, fifteen new model cases, five input gradients; no refit.'}
    (ROOT / 'independent-native.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'passed': True, 'cases': len(cases), 'checks': len(checks), 'torch': torch.__version__}))


if __name__ == '__main__':
    main()
