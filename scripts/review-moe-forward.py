"""Independent float64 functional oracle; no author model import or fitting."""
from pathlib import Path
import json
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
TOPIC = 'mixture-of-experts-transformers-moe'
torch.set_num_threads(1)
state = {k: torch.tensor(v, dtype=torch.float64) for k, v in json.loads(
    (ROOT / 'public' / 'learn-assets' / TOPIC / 'moe-001-17.json').read_text()).items()}
examples = json.loads((ROOT / 'src/learn/data/moe-examples.json').read_text())['examples']

def forward(pixels, temperature, disabled):
    def linear(x, name):
        return F.linear(x, state[name + '.weight'], state.get(name + '.bias'))
    def norm(x, name):
        return F.layer_norm(x, (16,), state[name + '.weight'], state[name + '.bias'], 1e-5)
    patches = F.unfold(pixels.reshape(1, 1, 8, 8), kernel_size=2, stride=2).transpose(1, 2)
    hidden = linear(patches, 'project') + state['position']
    q, k, v = linear(norm(hidden, 'norm_attention'), 'qkv').chunk(3, -1)
    q, k, v = [x.reshape(1, 16, 2, 8).transpose(1, 2) for x in (q, k, v)]
    context = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(1, 16, 16)
    hidden = hidden + linear(context, 'attention_output')
    expert_input = norm(hidden, 'norm_ffn')
    scores = linear(expert_input, 'router') / temperature
    chosen_scores, chosen = scores.topk(2, -1)
    gate = torch.zeros_like(scores).scatter(-1, chosen, chosen_scores.softmax(-1))
    # Dense branch construction is an independent oracle for the sparse live path.
    branches = torch.stack([linear(F.silu(linear(expert_input, f'experts.{i}.gate')) *
                                  linear(expert_input, f'experts.{i}.value'), f'experts.{i}.down')
                            for i in range(4)], -2)
    if disabled is not None:
        gate[..., disabled] = 0
    combined = (gate.unsqueeze(-1) * branches).sum(-2)
    logits = linear(norm(hidden + combined, 'norm_final').mean(1), 'classifier')[0]
    return {'logits': logits.tolist(), 'combined': combined[0].tolist(),
            'selected': chosen[0].tolist(), 'probabilities': logits.softmax(-1).tolist()}

generator = torch.Generator().manual_seed(92741)
cases = []
for case_id in range(10):
    if case_id < 6:
        pixels = torch.tensor(examples[case_id % 2]['pixels'], dtype=torch.float64)
        positions = torch.randperm(64, generator=generator)[:7]
        pixels[positions] = torch.randint(0, 17, (7,), generator=generator).double() / 16
    else:
        pixels = torch.randint(0, 17, (64,), generator=generator).double() / 16
    temperature = [.63, 1.37, 2.71][case_id % 3]
    disabled = None if case_id % 5 == 0 else (case_id - 1) % 4
    cases.append({'id': case_id, 'pixels': pixels.tolist(), 'temperature': temperature,
                  'disabled': disabled, 'expected': forward(pixels, temperature, disabled)})
assert len(cases) == 10
path = ROOT / 'docs/teaching/deep-learning-completion' / TOPIC / 'independent-forward-fixtures.json'
path.write_text(json.dumps({'torch': torch.__version__, 'dtype': 'float64', 'cases': cases}, indent=2) + '\n')
print(f'Wrote {len(cases)} independent full-model cases without fitting')
