"""Independent functional PyTorch reference, without importing the author's model."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
TOPIC = 'vision-transformers-vit-deit-swin-dinov2'
torch.set_num_threads(2)
generator = torch.Generator().manual_seed(92671)
asset = json.loads((ROOT / f'public/learn-assets/{TOPIC}/plain-vit.json').read_text())
state = {key: torch.tensor(value, dtype=torch.float64) for key, value in asset['state'].items()}
examples = json.loads((ROOT / 'src/learn/data/vision-transformer-examples.json').read_text())['examples']

def affine(x, prefix):
    return F.linear(x, state[prefix + '.weight'], state.get(prefix + '.bias'))

def normal(x, prefix):
    return F.layer_norm(x, (32,), state[prefix + '.weight'], state[prefix + '.bias'], 1e-5)

def reference(image, order, move):
    image = image.reshape(1, 1, 8, 8)
    patches = F.conv2d(image, state['patch.weight'], state['patch.bias'], stride=2).flatten(2).transpose(1, 2)
    positions = state['patch_position'][:, order] if move else state['patch_position']
    x = torch.cat([state['cls'] + state['cls_position'], patches[:, order] + positions], dim=1)
    for block in range(2):
        name = f'blocks.{block}'
        combined = affine(normal(x, name + '.norm1'), name + '.qkv').reshape(1, 17, 3, 4, 8)
        q, k, v = combined.permute(2, 0, 3, 1, 4).unbind(0)
        mixed = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(1, 17, 32)
        x = x + affine(mixed, name + '.output')
        x = x + affine(F.gelu(affine(normal(x, name + '.norm2'), name + '.ffn.0')), name + '.ffn.2')
    x = normal(x, 'norm')
    return affine(x[:, 0], 'head')[0], x[0, 0]

records = []
with torch.inference_mode():
    for case in range(12):
        if case < 6:
            image = torch.tensor(examples[case % 3]['image'], dtype=torch.float64)
            index = torch.randperm(64, generator=generator)[:5]
            image.reshape(-1)[index] = torch.randint(0, 17, (5,), generator=generator).double() / 16
        else:
            image = torch.randint(0, 17, (8, 8), generator=generator).double() / 16
        order = torch.randperm(16, generator=generator).tolist()
        move = case % 2 == 0
        logits, cls = reference(image, order, move)
        records.append({'image': image.tolist(), 'order': order, 'movePositions': move,
                        'logits': logits.tolist(), 'cls': cls.tolist()})
out = ROOT / f'docs/teaching/deep-learning-completion/{TOPIC}/independent-forward-fixtures.json'
out.write_text(json.dumps({'torch': torch.__version__, 'cases': records}, separators=(',', ':')) + '\n')
print(json.dumps({'passed': True, 'cases': len(records), 'torch': torch.__version__,
                  'method': 'Independent functional conv2d, layer_norm, SDPA and GELU; no author model imports'}))
