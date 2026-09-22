"""Reviewer-owned changed-state checks; no training or external downloads."""
from pathlib import Path
import hashlib
import json
import runpy
import torch
from torch.nn import functional as F

root = Path(__file__).resolve().parents[1]
assets = root / 'public/learn-assets/convnext-modern-cnn-designs'
definitions = runpy.run_path(str(assets / 'convnext-blocks.py'))
torch.set_num_threads(1)
torch.manual_seed(43)
fixtures = {'normalization': [], 'grn': [], 'folding': [], 'budgets': []}
for values in [[[.123, -7.2, 11.7], [103.7, 96.2, -8.1]], [[-3., -3., -3.], [9., 9., 9.]]]:
    x = torch.tensor(values, dtype=torch.float64)
    fixtures['normalization'].append({'input': values, 'channels': F.layer_norm(x, (3,), eps=1e-6).tolist(), 'whole': F.layer_norm(x, (2, 3), eps=1e-6).tolist()})
for maps, gamma, beta in [([[-2., 4.], [3., -9.]], [.7, -.9], [.123, -.4]), ([[0., 0.], [0., 0.]], [.2, -.5], [-1., 2.])]:
    x = torch.tensor(maps, dtype=torch.float64).T.reshape(1, 1, 2, 2).requires_grad_()
    module = definitions['ResponseNorm'](2).double()
    with torch.no_grad():
        module.scale.copy_(torch.tensor(gamma)); module.shift.copy_(torch.tensor(beta))
    # Set exact double values instead of the default float32 constructor intermediates.
    with torch.no_grad():
        module.scale.copy_(torch.tensor(gamma, dtype=torch.float64)); module.shift.copy_(torch.tensor(beta, dtype=torch.float64))
    output = module(x)
    fixtures['grn'].append({'maps': maps, 'scale': gamma, 'shift': beta, 'output': output.detach().reshape(2,2).T.tolist()})
    if torch.count_nonzero(x):
        assert torch.autograd.gradcheck(module, (x,), atol=1e-5)
for channels in [4, 11, 64]:
    for version in [1, 2]:
        block = definitions['ConvNeXtBlock'](channels, version, 0).double()
        fixtures['budgets'].append({'channels': channels, 'version': version, 'parameters': sum(p.numel() for p in block.parameters())})
        with torch.no_grad():
            block.project.weight.zero_(); block.project.bias.zero_()
        x = torch.randn(2, channels, 3, 5, dtype=torch.float64)
        torch.testing.assert_close(block(x), x, atol=0, rtol=0)
for gamma in [-2.7, 0., 1.4]:
    config = {'kernel': [-.3, 1.7, .2, -2., .1, .9, 1.2, -.5, .7], 'small': -.8, 'bias': -.17, 'smallBias': .9, 'mean': .6, 'variance': .03, 'gamma': gamma, 'beta': .23, 'smallMean': -.45, 'smallVariance': .8, 'smallGamma': -1.3, 'smallBeta': -.17, 'image': torch.randn(25, dtype=torch.float64).tolist()}
    x = torch.tensor(config['image'], dtype=torch.float64).reshape(1,1,5,5)
    conv = F.conv2d(x, torch.tensor(config['kernel'], dtype=torch.float64).reshape(1,1,3,3), torch.tensor([config['bias']], dtype=torch.float64), padding=1)
    first = F.batch_norm(conv, torch.tensor([config['mean']], dtype=torch.float64), torch.tensor([config['variance']], dtype=torch.float64), torch.tensor([gamma], dtype=torch.float64), torch.tensor([config['beta']], dtype=torch.float64), training=False, eps=1e-5)
    second = F.batch_norm(config['small']*x + config['smallBias'], torch.tensor([config['smallMean']], dtype=torch.float64), torch.tensor([config['smallVariance']], dtype=torch.float64), torch.tensor([config['smallGamma']], dtype=torch.float64), torch.tensor([config['smallBeta']], dtype=torch.float64), training=False, eps=1e-5)
    fixtures['folding'].append({'config': config, 'expected': (first+second+x).reshape(-1).tolist()})
files = [assets/'convnext-blocks.py', root/'scripts/verify-convnext-independent.py']
report = {'passed': True, 'environment': {'torch': torch.__version__}, 'checks': ['Changed three-channel and constant-group LayerNorm native references', 'Signed and zero-response GRN native references; changed-state Jacobian', 'Six native parameter counts and exact zero-projection residual identities', 'Three signed/zero-gamma full 25-position eval BatchNorm branch references'], 'fixtures': fixtures, 'sourceHashes': {str(file.relative_to(root)).replace('\\','/'): hashlib.sha256(file.read_bytes()).hexdigest() for file in files}}
(root/'docs/teaching/evidence/convnext-independent-native.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')
print(json.dumps({'passed': True, 'groups': len(report['checks'])}))
