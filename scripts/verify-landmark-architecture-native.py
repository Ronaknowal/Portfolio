"""Check compositions, library structure, customization boundaries and CAM.

Use --library-path for the isolated Torchvision target. The actual twelve-fit
producer and per-family --compare CLI invocations are separate recorded runs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import runpy
import sys
parser = argparse.ArgumentParser()
parser.add_argument('--library-path')
args = parser.parse_args()
if args.library_path:
    sys.path.insert(0, str(Path(args.library_path).resolve()))
import numpy as np
import sklearn
import torch
from torch import nn
import torchvision
from torchvision.models import get_model
torch.set_num_threads(1)
root = Path(__file__).resolve().parents[1]
assets = root / 'public/learn-assets/landmark-architectures-lenet-alexnet-vgg-resnet-efficientnet'
packet = root / 'docs/teaching/drafts/landmark-architectures-lenet-alexnet-vgg-resnet-efficientnet'
report_path = root / 'docs/teaching/evidence/landmark-architecture-native.json'
report_path.write_text(json.dumps({'passed': False, 'status': 'running'}), encoding='utf-8')
checks, constructions = [], []
hash_file = lambda file: hashlib.sha256(file.read_bytes()).hexdigest()
data = json.loads((assets / 'calculated-inputs.json').read_text(encoding='utf-8'))
assert data == json.loads((packet / 'calculated-inputs.json').read_text(encoding='utf-8'))
assert hash_file(assets / 'architecture-experiments.py') == hash_file(packet / 'architecture-experiments.py')
assert hash_file(assets / 'digits-400.csv') == hash_file(packet / 'digits-400.csv')
checks.append('Complete saved twelve-fit replay equals every conserved packet value; producer/data bytes unchanged')
source = runpy.run_path(str(assets / 'landmark_builders.py'))
expected_counts = {'lenet': 61706, 'alexnet': 61100840, 'vgg16': 138357544, 'resnet18': 11689512, 'efficientnet_b0': 5288548}
for family, builder in source['BUILDERS'].items():
    shape = (1, 1, 32, 32) if family == 'lenet' else (1, 3, 224, 224)
    with torch.device('meta'):
        model = builder().eval()
        output = model(torch.empty(shape))
        count = sum(parameter.numel() for parameter in model.parameters())
        assert count == expected_counts[family]
        assert tuple(output.shape) == (1, 10 if family == 'lenet' else 1000)
        if family != 'lenet':
            reference = get_model(family, weights=None).eval()
            assert sum(parameter.numel() for parameter in reference.parameters()) == count
            kinds = (nn.Conv2d, nn.Linear, nn.BatchNorm2d)
            left = [layer for layer in model.modules() if isinstance(layer, kinds)]
            right = [layer for layer in reference.modules() if isinstance(layer, kinds)]
            assert len(left) == len(right)
            for a, b in zip(left, right):
                assert type(a) is type(b)
                assert {key: tuple(value.shape) for key, value in a.state_dict().items()} == {key: tuple(value.shape) for key, value in b.state_dict().items()}
        constructions.append({'family': family, 'parameters': count, 'output_shape': list(output.shape)})
checks.append('Allfive full constructions run on meta tensors; four library counterparts match every copied component type/state shape and parameter count')
with torch.device('meta'):
    changed = source['vgg16'](7)
    changed.pool = nn.AdaptiveAvgPool2d(1)
    changed.head = nn.Linear(512, 7)
    assert sum(parameter.numel() for parameter in changed.head.parameters()) == 3591
    for side in (64, 96):
        assert tuple(changed(torch.empty(2, 3, side, side)).shape) == (2, 7)
checks.append('Seven-class GAP modification has3591head parameters and batch2x7output at both64and96pixel sizes')
for probability in (-.1, 1.1, float('nan')):
    try:
        source['MobileBlock'](4, 4, 1, 3, 1, probability)
    except ValueError:
        pass
    else:
        raise AssertionError('invalid probability accepted')
block = source['MobileBlock'](4, 4, 1, 3, 1, 1.).train()
inputs = torch.ones(2, 4, 4, 4, requires_grad=True)
output = block(inputs)
torch.testing.assert_close(output, inputs, atol=0, rtol=0)
output.sum().backward()
torch.testing.assert_close(inputs.grad, torch.ones_like(inputs), atol=0, rtol=0)
assert all(parameter.grad is not None and torch.count_nonzero(parameter.grad) == 0 for parameter in block.parameters())
checks.append('Drop probability1 gives exact finite identity/input gradient and zero branch parameter gradients; invalid probabilities reject')
maximum_error = 0.
for fit in data['fits']:
    if fit['seed'] != 1:
        continue
    for sample in fit['observations']:
        for label in range(10):
            explicit_map = [[sum(fit['head_weight'][label][c] * sample['feature_maps'][c][r][s] for c in range(16)) for s in range(2)] for r in range(2)]
            actual = sum(map(sum, explicit_map)) / 4 + fit['head_bias'][label]
            maximum_error = max(maximum_error, abs(actual - sample['logits'][label]))
            assert abs(actual - sample['logits'][label]) < 8e-6
checks.append('Independent scalar-loop CAM reconstruction agrees with all80saved class logits within8e-6')
maps = np.array([[[1., 2.], [0., 3.]], [[0., 1.], [2., 1.]]])
weights, bias = np.array([2., -1.]), .5
class_map = np.einsum('c,chw->hw', weights, maps)
assert np.array_equal(class_map, [[2, 3], [-2, 5]])
assert weights @ maps.mean(axis=(1, 2)) + bias == class_map.mean() + bias == 2.5
checks.append('StandaloneCAMcode reproduces the exact map and both2.5score routes')
record = {'passed': True, 'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'torchvision': torchvision.__version__, 'numpy': np.__version__, 'sklearn': sklearn.__version__}, 'checks': checks, 'constructions': constructions, 'maximumDoubleCAMDifference': maximum_error,
          'sources': {str(file.relative_to(root)).replace('\\', '/'): hash_file(file) for file in [assets/'architecture-experiments.py', assets/'landmark_builders.py', assets/'digits-400.csv', assets/'calculated-inputs.json']},
          'limits': ['This verifier reuses saved fits; full producer was executed separately22September', 'The four actual CPU pairedforward --compare invocations are recorded in the design; this command checks allfive constructions and component shapes on meta tensors', 'No pretrained ImageNet weights or photograph example executed; no hardware benchmark claim']}
text = json.dumps(record, indent=2) + '\n'
report_path.write_text(text, encoding='utf-8')
(assets / 'native-verification.json').write_text(text, encoding='utf-8')
print(json.dumps({'passed': True, 'checks': checks, 'constructions': constructions, 'maximumDoubleCAMDifference': maximum_error}, indent=2))
