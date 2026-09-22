"""Focused phase-two checks; reuse the completed six-fit replay unless its sources change."""
from pathlib import Path
import hashlib
import importlib.util
import json
import platform
import re
import sys
import numpy as np
import torch
import sklearn

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'public/learn-assets/capsule-networks'
DRAFT = ROOT / 'docs/teaching/drafts/capsule-networks'
OUT = ROOT / 'docs/teaching/evidence/capsule-native.json'
OUT.write_text(json.dumps({'passed': False, 'status': 'running'}), encoding='utf-8')
sys.path.insert(0, str(ASSETS))
from capsule_learning_import import load_learning
learning = load_learning()
spec = importlib.util.spec_from_file_location('capsule_mechanics', ASSETS / 'capsule-mechanics.py')
mechanics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mechanics)
torch.set_num_threads(1)
groups = []
for filename in ('calculated-inputs.json', 'mechanics-results.json', 'author-check-results.json'):
    assert json.loads((ASSETS / filename).read_text()) == json.loads((DRAFT / filename).read_text())
for filename in ('capsule-learning.py', 'capsule-mechanics.py', 'capsule_learning_import.py', 'author-checks.py', 'digits-400.csv'):
    assert (ASSETS / filename).read_bytes() == (DRAFT / filename).read_bytes()
groups.append('Actual six600-update fits, mechanics and NumPy reconstruction replay match all conserved output and source bytes')
saved = json.loads((ASSETS / 'calculated-inputs.json').read_text())
model = learning.TinyCapsules().double()
model.load_state_dict({name: torch.tensor(value, dtype=torch.float64) for name, value in saved['saved_models']['3'].items()})
model.eval()
run = next(row for row in saved['runs'] if row['seed'] == 1 and row['training_iterations'] == 3)
examples = run['first_two_examples']
fixtures = []
with torch.no_grad():
    for index, image in enumerate(examples['images']):
        original = torch.tensor(image, dtype=torch.float64).reshape(1, 1, 8, 8)
        variants = [('original', original)]
        flipped = original.clone(); flipped[0, 0, 3, 3] = 1 - flipped[0, 0, 3, 3]
        variants.append(('pixel-flip', flipped))
        for dy, dx in ((0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)):
            variants.append((f'shift-{dy},{dx}', learning.shifted(original, dy, dx)))
        for kind, current in variants:
            capsules, _ = model.encode(current)
            lengths = torch.linalg.vector_norm(capsules, dim=-1)
            predicted = lengths.argmax(dim=1)
            reconstruction = model.reconstruct(capsules, predicted)
            fixtures.append({'specimen': index, 'kind': kind, 'image': current.flatten().tolist(),
                             'capsules': capsules[0].tolist(), 'predicted': int(predicted[0]),
                             'reconstruction': reconstruction.flatten().tolist()})
    batched = torch.tensor(examples['images'], dtype=torch.float64)[:, None]
    together = model.encode(batched)[0]
    for index in range(2):
        torch.testing.assert_close(together[index], model.encode(batched[index:index+1])[0][0], atol=1e-14, rtol=1e-13)
    original = together[:1]
    mask = torch.tensor([4])
    before = model.reconstruct(original, mask)
    changed = original.clone(); changed[:, 5, 0] += .137
    assert torch.equal(before, model.reconstruct(changed, mask))
    changed = original.clone(); changed[:, 4, 2] -= .137
    assert not torch.equal(before, model.reconstruct(changed, mask))
groups.append('Fourteen actual float64 frozen encodes and decodes including all directional shifts, zero shift and changed pixels; batch separation and independent decoder masks')
manuscript = (DRAFT / 'lesson.md').read_text(encoding='utf-8')
snippet = re.search(r'```python\n(.*?)\n```', manuscript, re.S).group(1)
environment = {'torch': torch}
exec(compile(snippet, '<displayed-routing>', 'exec'), environment)
torch.manual_seed(37)
votes = torch.randn(2, 4, 3, 5, dtype=torch.float64)
for count in (1, 3, 8):
    torch.testing.assert_close(environment['route'](votes, count), learning.route(votes, count)[0], rtol=0, atol=0)
groups.append('Actual displayed Python route executes and exactly matches the full ordinary Torch route for changed batch, child, parent and coordinate dimensions')
for temperature, expected in ((.5, [.01798620996209156, .9820137900379085]), (1, [.11920292202211755, .8807970779778823]), (2, [.2689414213699951, .7310585786300049])):
    np.testing.assert_allclose(mechanics.softmax(np.array([0., 2.]) / temperature), expected, atol=1e-14)
    np.testing.assert_allclose(torch.softmax(torch.tensor([0., 2.]) / temperature, 0).numpy(), expected, atol=6e-8)
assert np.array_equal(mechanics.softmax(np.array([[8.], [-3.]]), axis=1), np.ones((2, 1)))
groups.append('Positive-temperature modification and one-parent normalization have independently calculated values')
capsules = torch.zeros(1, 10, 8, dtype=torch.float64)
capsules[0, :3, 0] = torch.tensor([.8, .2, .4])
assert abs(float(learning.margin_loss(capsules, torch.tensor([0]))) - .06) < 1e-8
zero = torch.zeros(2, 3, dtype=torch.float64, requires_grad=True)
learning.squash(zero).sum().backward()
assert torch.equal(zero.grad, torch.zeros_like(zero))
for vector in (np.array([.3, .4]), np.array([3., 4.]), np.array([0., 0.])):
    x = torch.tensor(vector, dtype=torch.float64, requires_grad=True)
    jacobian = torch.autograd.functional.jacobian(learning.squash, x).detach().numpy()
    radius = np.linalg.norm(vector)
    tangent = radius / (1 + radius ** 2)
    radial = 2 * radius / (1 + radius ** 2) ** 2
    expected = tangent * np.eye(2)
    if radius:
        expected += (radial - tangent) * np.outer(vector / radius, vector / radius)
    np.testing.assert_allclose(jacobian, expected, atol=1e-14)
groups.append('Margin objective units and exact zero/radial/tangent Torch derivatives are checked independently')
fixture = np.array([[[0., 1.], [2., -1.]], [[4., -2.], [1., 1.]]])
em = mechanics.diagonal_em(fixture, np.array([1., .25]), iterations=1)[0]
assert abs(em['mass'][0] - .625) < 1e-14
np.testing.assert_allclose(em['means'][0], [.8, .4], atol=1e-14)
np.testing.assert_allclose(em['variance'][0], [2.56, 1.44], atol=1e-14)
groups.append('Changed two-coordinate EM weighted mean and variance match direct independent sufficient-statistic arithmetic')
reference = ROOT / 'docs/teaching/evidence/capsule-native-inference.json'
reference.write_text(json.dumps(fixtures, separators=(',', ':')) + '\n', encoding='utf-8')
sources = {str(path.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(path.read_bytes()).hexdigest() for path in [ASSETS / name for name in ('capsule-learning.py', 'capsule-mechanics.py', 'author-checks.py', 'digits-400.csv', 'calculated-inputs.json')]}
result = {'passed': True, 'groups': groups, 'versions': {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__, 'sklearn': sklearn.__version__}, 'sources': sources, 'limits': 'CPU small-model/mechanics execution only; no full MNIST, EM-CapsNet, STAR-Caps, VB, GPU or pretrained reproduction. Browser integration and independent review are separate.'}
OUT.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
(ASSETS / 'native-verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'passed': True, 'groups': len(groups), 'native_inference_fixtures': len(fixtures)}))
