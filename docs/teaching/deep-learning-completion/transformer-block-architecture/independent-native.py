"""Independent review: functional torch forward, without author/model imports."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F

torch.set_num_threads(2)
torch.set_num_interop_threads(2)
root = Path('docs/teaching/deep-learning-completion/transformer-block-architecture')
models = json.loads(Path('public/learn-code/transformer-block-architecture/movement-runtime.json').read_text(encoding='utf-8'))

def forward(model, points, times, mask):
    weights = {name: torch.tensor(value, dtype=torch.float64) for name, value in model['state_dict'].items()}
    def linear(x, name):
        return F.linear(x, weights[name + '.weight'], weights[name + '.bias'])
    def norm(x, name):
        return F.layer_norm(x, (24,), weights[name + '.weight'], weights[name + '.bias'], 1e-5)
    coordinates = torch.tensor(points, dtype=torch.float64)
    tag = torch.tensor(times, dtype=torch.float64)[:, None]
    state = linear(torch.cat([2 * coordinates - 1, tag], dim=1), 'stem')
    allowed = ~torch.tensor(mask, dtype=torch.bool)
    stages = []
    for layer in range(2):
        prefix = f'blocks.{layer}'
        branch = norm(state, prefix + '.norm_attention') if model['preNorm'] else state
        packed = F.linear(branch, weights[prefix + '.attention.in_proj_weight'], weights[prefix + '.attention.in_proj_bias'])
        q, k, v = [chunk.reshape(-1, 2, 12).transpose(0, 1) for chunk in packed.chunk(3, dim=-1)]
        update = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed[None, None, :], dropout_p=0).transpose(0, 1).reshape(-1, 24)
        context = state + linear(update, prefix + '.attention.out_proj')
        if not model['preNorm']:
            context = norm(context, prefix + '.norm_attention')
        branch = norm(context, prefix + '.norm_feedforward') if model['preNorm'] else context
        state = context + linear(F.gelu(linear(branch, prefix + '.feedforward.0')), prefix + '.feedforward.2')
        if not model['preNorm']:
            state = norm(state, prefix + '.norm_feedforward')
        stages.append(state.tolist())
    logits = linear(norm(state, 'final_norm')[allowed].mean(0), 'classifier')
    return {'logits': logits.tolist(), 'stages': stages}

cases = []
for name, model in models.items():
    points = [row[:] for row in model['points']]
    times = model['times'][:]
    points[7] = [.37, .92]
    times[7] = .15
    for case, p, t, mask in [
        ('changed-point-and-tag', points, times, [False] * 45),
        ('constant-records', [[.2, .7]] * 45, [.4] * 45, [False] * 45),
        ('interleaved-padding', sum(([row, [.6, .1]] if i < 5 else [row] for i, row in enumerate(points)), []), sum(([time, -1.5] if i < 5 else [time] for i, time in enumerate(times)), []), sum(([False, True] if i < 5 else [False] for i in range(45)), [])),
    ]:
        cases.append({'placement': name, 'case': case, 'points': p, 'times': t, 'padding': mask, **forward(model, p, t, mask)})

probes = []
for values, probe, epsilon in [([.03, -.02, .01, -.02], [1., -2., .5, .1], .001), ([2., 2., 2., 2.], [1., -1., 0., 0.], .001), ([-3., 1., 5., 0.], [1., 1., 1., 1.], 1e-5)]:
    x = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    scalar = (F.layer_norm(x, (4,), eps=epsilon) * torch.tensor(probe)).sum()
    gradient, = torch.autograd.grad(scalar, x)
    probes.append({'input': values, 'probe': probe, 'epsilon': epsilon, 'gradient': gradient.tolist()})
result = {'environment': {'torch': torch.__version__, 'threads': 2}, 'cases': cases, 'probes': probes}
(root / 'independent-fixtures.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print('Independent native reference: six new full-network cases and three autograd probes executed.')
