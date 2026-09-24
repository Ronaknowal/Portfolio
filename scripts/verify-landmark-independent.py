"""Complementary construction and intervention checks; never repeats the 12 fits."""
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'public/learn-assets/landmark-architectures-lenet-alexnet-vgg-resnet-efficientnet'
REPORT = ROOT / 'docs/teaching/evidence/landmark-architectures-independent-native.json'
spec = importlib.util.spec_from_file_location('landmark_builders', ASSETS/'landmark_builders.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
torch.set_num_threads(1)
checks = []

# Meta execution establishes whole-model contracts without allocating the large
# AlexNet/VGG heads, as taught. Check counts against independently stated family
# specifications, not the lesson's own count helper.
expected = {'lenet': 61706, 'alexnet': 61100840, 'vgg16': 138357544,
            'resnet18': 11689512, 'efficientnet_b0': 5288548}
counts = {}
for name, constructor in builder.BUILDERS.items():
    with torch.device('meta'):
        model = constructor().eval()
        shape = (2, 1, 32, 32) if name == 'lenet' else (2, 3, 224, 224)
        output = model(torch.empty(shape))
    counts[name] = sum(value.numel() for value in model.parameters())
    assert counts[name] == expected[name], (name, counts[name])
    assert output.shape == (2, 10 if name == 'lenet' else 1000)
checks.append('All five full builders: declared meta shapes and independent parameter totals')

# Execute the exact independent modification problem, rather than just checking
# its formula. The unchanged VGG trunk accepts different legal input grids.
with torch.device('meta'):
    model = builder.vgg16().eval()
    model.pool = nn.AdaptiveAvgPool2d(1)
    model.head = nn.Linear(512, 7)
    assert sum(value.numel() for value in model.head.parameters()) == 3591
    for side in (224, 256):
        assert model(torch.empty(3, 3, side, side)).shape == (3, 7)
checks.append('Executed seven-class GAP head exercise at two image sizes; 3,591 head parameters')

# Channel gating is spatially equivariant: a common spatial permutation changes
# positions, while the global mean and every channel multiplier stay the same.
torch.manual_seed(724)
gate = builder.ChannelGate(6, 2).double().eval()
features = torch.randn(2, 6, 3, 4, dtype=torch.float64)
permutation = torch.tensor([7, 2, 10, 1, 11, 4, 6, 8, 0, 9, 5, 3])
permuted = features.flatten(2)[:, :, permutation].reshape_as(features)
expected_output = gate(features).flatten(2)[:, :, permutation].reshape_as(features)
torch.testing.assert_close(gate(permuted), expected_output, rtol=1e-12, atol=1e-12)
checks.append('Actual ChannelGate spatial-permutation equivariance preserves global context')

# The public MobileBlock rate is a learner customization point. The reviewer
# originally found p=1 divided by zero; this check closes that exact failure.
value = torch.ones(16, 4, 3, 3)
for probability in (0., .5, 1.):
    block = builder.MobileBlock(4, 4, 1, 3, 1, probability)
    block.branch = nn.Identity()  # isolate the mask and direct route
    torch.manual_seed(4)
    result = block.train()(value)
    assert torch.isfinite(result).all()
    if probability == 0:
        torch.testing.assert_close(result, 2*value)
    elif probability == 1:
        torch.testing.assert_close(result, value)
    else:
        factors = (result-value).flatten(1)
        assert all(bool((row == row[0]).all()) for row in factors)
        assert set(factors.flatten().tolist()) == {0., 2.}
    torch.testing.assert_close(block.eval()(value), 2*value)
for probability in (-.1, 1.1, float('nan')):
    try:
        builder.MobileBlock(4, 4, 1, 3, 1, probability)
    except ValueError:
        pass
    else:
        raise AssertionError(f'Invalid block probability accepted: {probability}')
checks.append('Fixed p1 division-by-zero; p0/p1/eval identities, real row-sharing and invalid-rate rejection')

# Check the composition-specific B0 details promised in the implementation map.
with torch.device('meta'):
    model = builder.efficientnet_b0()
blocks = [module for module in model.modules() if isinstance(module, builder.MobileBlock)]
assert len(blocks) == 16
assert [block.drop_probability for block in blocks] == [.2*i/16 for i in range(16)]
incoming_widths = [32,16,24,24,40,40,80,80,80,112,112,112,192,192,192,192]
for block, incoming in zip(blocks, incoming_widths):
    channel_gate = next(module for module in block.modules() if isinstance(module, builder.ChannelGate))
    assert channel_gate.reduce.out_channels == max(1, incoming//4)
    assert channel_gate.expand.in_channels == max(1, incoming//4)
checks.append('B0 complete16-block schedule and squeeze width from each block incoming width')

# A spatial permutation is a meaningful CAM null; changing a single channel's
# feature independently still has a signed, exact linear consequence.
saved = json.loads((ASSETS/'calculated-inputs.json').read_text(encoding='utf-8'))
largest_error = 0.
for fit in saved['fits']:
    if fit['seed'] != 1:
        continue
    for observation in fit['observations']:
        features = torch.tensor(observation['feature_maps'], dtype=torch.float64)
        weights = torch.tensor(fit['head_weight'], dtype=torch.float64)
        bias = torch.tensor(fit['head_bias'], dtype=torch.float64)
        scores = weights @ features.mean((-2, -1)) + bias
        # Independent scalar accumulation, rather than calling browser code.
        maps = torch.empty(10, 2, 2, dtype=torch.float64)
        for target in range(10):
            for row in range(2):
                for column in range(2):
                    maps[target,row,column] = sum(weights[target,channel]*features[channel,row,column] for channel in range(16))
        torch.testing.assert_close(maps.mean((-2,-1))+bias, scores, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(weights@features.flip(-1).mean((-2,-1))+bias, scores, atol=1e-12, rtol=1e-12)
        recorded = torch.tensor(observation['logits'], dtype=torch.float64)
        largest_error = max(largest_error, float((scores-recorded).abs().max()))
        channel, target, delta = 3, 8, .7
        changed = features.clone(); changed[channel,0,1] += delta
        actual_change = (weights@changed.mean((-2,-1))+bias-scores)[target]
        torch.testing.assert_close(actual_change, weights[target,channel]*delta/4, atol=1e-12, rtol=1e-12)
assert largest_error < 1e-5
checks.append('Every saved image/class: scalar CAM reconstruction, spatial-permutation null and signed single-cell intervention')

result = {'passed':True,'reviewer':'/root/decision_depth_prose','torch':torch.__version__,
          'sourceHash':hashlib.sha256((ASSETS/'landmark_builders.py').read_bytes()).hexdigest(),
          'completeBuilderParameters':counts,'maxSavedCAMError':largest_error,'checks':checks,
          'limits':['No12-fit rerun','Meta shape/count execution is not numerical full-model fitting',
                    'No pretrained weights downloaded','No browser or screenshot review']}
REPORT.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
