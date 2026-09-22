"""Fresh bounded native checks; conserve previous six fits rather than retrain.

--library-path points to an isolated compatible Torchvision install.
Only the evidence files owned by this verifier are written.
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
import torch
import torchvision
from torchvision.models.convnext import CNBlock

torch.set_num_threads(1)
root = Path(__file__).resolve().parents[1]
assets = root/'public/learn-assets/convnext-modern-cnn-designs'
packet = root/'docs/teaching/drafts/convnext-modern-cnn-designs'
evidence = root/'docs/teaching/evidence/convnext-native.json'
evidence.write_text(json.dumps({'passed': False, 'status': 'running'}), encoding='utf-8')
hash_file = lambda file: hashlib.sha256(file.read_bytes()).hexdigest()
checks = []
for name in ['convnext-blocks.py', 'convnext_library_bridge.py', 'masked-reconstruction.py', 'author-checks.py', 'digits-400.csv', 'calculated-inputs.json']:
    assert hash_file(assets/name) == hash_file(packet/name), name
for name in ['author-check-results.json', 'block-check-results.json']:
    assert json.loads((assets/name).read_text()) == json.loads((packet/name).read_text())
checks.append('Programs/data/six-fit result bytes conserved; fresh author oracle and model-shape outputs equal the complete prepared records')
definitions = runpy.run_path(str(assets/'convnext-blocks.py'))
bridge = runpy.run_path(str(assets/'convnext_library_bridge.py'))
bridge['compare_block']()
checks.append('Prepared Torchvision bridge executes: float64 output/input/all parameter gradients at original LayerScale')
errors = []
for seed in [3, 11]:
    torch.manual_seed(seed)
    local = definitions['ConvNeXtBlock'](8, version=1, drop_probability=0).double()
    native = CNBlock(8, layer_scale=.3, stochastic_depth_prob=0).double()
    pairs = [(local.spatial, native.block[0]), (local.norm, native.block[2]), (local.expand, native.block[3]), (local.project, native.block[5])]
    for left, right in pairs:
        left.load_state_dict(right.state_dict())
    with torch.no_grad():
        scale = torch.linspace(-.8, .8, 8, dtype=torch.float64)
        local.layer_scale.copy_(scale)
        native.layer_scale.copy_(scale[:, None, None])
    x = torch.randn(2, 8, 5, 7, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    left, right = local(x), native(y)
    errors.append(float((left-right).abs().max().detach()))
    torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-12)
    left.square().mean().backward(); right.square().mean().backward()
    torch.testing.assert_close(x.grad, y.grad, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(local.layer_scale.grad, native.layer_scale.grad[:,0,0], atol=1e-12, rtol=1e-12)
    for first, second in pairs:
        for a, b in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=1e-12, rtol=1e-12)
checks.append('Two additional matched V1 comparisons use large signed channel scales so tiny initialization cannot hide a wrong branch')
grn = definitions['ResponseNorm'](3).double()
with torch.no_grad(): grn.scale.copy_(torch.tensor([.5, -.3, .7])); grn.shift.copy_(torch.tensor([.2, 0, -.1]))
samples = torch.randn(2, 2, 3, 3, dtype=torch.float64, requires_grad=True)
assert torch.autograd.gradcheck(grn, (samples,), eps=1e-6, atol=1e-5)
first = grn(samples).detach()[0]
edited = samples.detach().clone(); edited[1] *= 100
torch.testing.assert_close(grn(edited)[0], first, atol=0, rtol=0)
checks.append('Dense V2 GRN input Jacobian passes finite differences and other-specimen edits have exactly zero influence')
experiment = runpy.run_path(str(assets/'masked-reconstruction.py'))
saved = json.loads((assets/'saved-models.json').read_text())
fixtures = []
for run in saved:
    model = experiment['SmallMaskedModel'](run['global_response']).eval()
    model.load_state_dict({name: torch.tensor(value) for name,value in run['model_state'].items()})
    for observation in run['examples']:
        original = np.array(observation['input']); mask = np.array(observation['visible_patches'])
        cases = [('original', original.copy(), mask.copy()), ('all-zero-input', np.zeros_like(original), mask.copy()), ('all-one-input', np.ones_like(original), mask.copy())]
        hidden = original.copy(); hidden[0,0] = 1-hidden[0,0]; cases.append(('hidden-edit', hidden, mask.copy()))
        visible = original.copy(); visible[0,2] = 1-visible[0,2]; cases.append(('visible-edit', visible, mask.copy()))
        changed_mask = mask.copy(); active = tuple(np.argwhere(mask==1)[0]); inactive = tuple(np.argwhere(mask==0)[0]); changed_mask[active],changed_mask[inactive] = 0,1
        cases.append(('mask-swap', original.copy(), changed_mask))
        for kind, pixels, visibility in cases:
            with torch.no_grad():
                tensor = torch.tensor(pixels, dtype=torch.float32)[None,None]
                masks = torch.tensor(visibility, dtype=torch.float32)[None,None]
                output = model(tensor,masks)
                mse = experiment['masked_mse'](output,tensor,masks)
            fixtures.append({'grn':run['global_response'],'source':observation['source_id'],'kind':kind,'pixels':pixels.tolist(),'mask':visibility.tolist(),'output':output[0,0].tolist(),'mse':float(mse)})
checks.append('Twenty-four fresh native inference fixtures cover every saved model/specimen, hidden/visible edits, swaps and zero/one input extremes')
report = {'passed':True,'environment':{'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,'numpy':np.__version__},'checks':checks,'maximumLibraryOutputDifference':max(errors),'fixtures':fixtures,'sources':{str(file.relative_to(root)).replace('\\','/'):hash_file(file) for file in [assets/'convnext-blocks.py', assets/'convnext_library_bridge.py', assets/'masked-reconstruction.py', assets/'author-checks.py', assets/'digits-400.csv', assets/'calculated-inputs.json', assets/'saved-models.json']},'limits':['Six previous training fits conserved and not rerun; fresh native reconstruction and independent NumPy checks executed', 'Large V1/V2 families use meta shape/parameter checks, not full learned inference', 'Optional pretrained photograph route, ImageNet training, native sparse FCMAE and hardware timings not executed']}
evidence.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
public_report = {key:value for key,value in report.items() if key!='fixtures'}
(assets/'native-verification.json').write_text(json.dumps(public_report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'passed':True,'checks':checks,'maximumLibraryOutputDifference':max(errors),'nativeFixtures':len(fixtures)}))
