"""Integration owner's complementary checks; author owns the broader parity suite."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


activation_path = Path('public/learn-assets/perceptrons/activation-mechanisms.py')
lora_path = Path('public/learn-assets/transfer-learning/lora-mechanism-bridge.py')
activation = load('depth_activations', activation_path)
lora = load('depth_lora', lora_path)
# Scalar coordinate oracle independent of BLAS/matrix derivative expressions.
x = np.array([[1., 2., -1.], [-2., .5, 3.]])
w = np.array([[.2, .5, -.1], [.7, -.4, .3]])
a = np.array([[.5, -.2, .1], [.3, .4, -.6]])
b = np.array([[.2, -.1], [-.4, .3]])
target = np.array([[.4, -.2], [.8, .1]])
scale = .7
out, loss, da, db, dx = lora.forward_and_gradients(x, target, w, a, b, scale)
expected = np.empty_like(out)
for row in range(len(x)):
    for output in range(len(w)):
        base = sum(x[row, feature] * w[output, feature] for feature in range(x.shape[1]))
        adapter = sum(b[output, rank] * sum(a[rank, feature] * x[row, feature]
                      for feature in range(x.shape[1])) for rank in range(len(a)))
        expected[row, output] = base + scale * adapter
np.testing.assert_allclose(out, expected, atol=1e-14)
delta = 2 * (expected-target) / target.size
expected_da = np.zeros_like(a)
expected_db = np.zeros_like(b)
for rank in range(len(a)):
    for feature in range(x.shape[1]):
        expected_da[rank, feature] = scale * sum(delta[row, output] * b[output, rank] * x[row, feature]
            for row in range(len(x)) for output in range(len(w)))
for output in range(len(w)):
    for rank in range(len(a)):
        expected_db[output, rank] = scale * sum(delta[row, output] * a[rank, feature] * x[row, feature]
            for row in range(len(x)) for feature in range(x.shape[1]))
np.testing.assert_allclose(da, expected_da, atol=1e-14)
np.testing.assert_allclose(db, expected_db, atol=1e-14)
# Equal BA gives equal forward map but factor gradients transform reciprocally.
out2, loss2, da2, db2, dx2 = lora.forward_and_gradients(x, target, w, 2*a, b/2, scale)
np.testing.assert_allclose(out2, out, atol=1e-14)
np.testing.assert_allclose(da2, da/2, atol=1e-14)
np.testing.assert_allclose(db2, db*2, atol=1e-14)
np.testing.assert_allclose(dx2, dx, atol=1e-14)
# Independent finite differences on smooth activations, no framework derivative.
points = np.array([-3.1, -.7, .4, 2.3])
errors = {}
for name in ('sigmoid', 'tanh', 'relu', 'leaky_relu', 'elu', 'gelu', 'gelu_tanh', 'silu', 'mish'):
    _, derivative = activation.activation(name, points)
    plus, _ = activation.activation(name, points + 1e-5)
    minus, _ = activation.activation(name, points - 1e-5)
    numeric = (plus-minus) / 2e-5
    np.testing.assert_allclose(derivative, numeric, atol=1e-8, rtol=1e-8)
    errors[name] = float(np.max(abs(derivative-numeric)))
report = {
    'status': 'passed', 'reviewer': 'integration owner, not program author',
    'checks': ['LoRA scalar coordinate oracle for rectangular forward and both factor gradients',
               'LoRA factor rescaling preserves map/input gradient and changes factor gradients as derived',
               'Nine activation slopes independently checked with central differences away from corners'],
    'activationFiniteDifferenceErrors': errors,
    'sourceHashes': {str(path).replace('\\', '/'): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in (activation_path, lora_path)},
}
Path('docs/teaching/evidence/neuron-implementation-depth-independent.json').write_text(
    json.dumps(report, indent=2)+'\n', encoding='utf8')
print(json.dumps(report))
