"""Scoped execution and complementary checks for the three implementation bridges."""
from pathlib import Path
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import numpy as np
import torch
from torch.nn import functional as F
import sklearn
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
torch.set_num_threads(1)
records = []


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def close(name, actual, expected, tolerance=1e-11):
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)
    records.append(name)


activation_path = "public/learn-assets/perceptrons/activation-mechanisms.py"
backprop_path = "public/learn-assets/backpropagation/engine-library-bridge.py"
lora_path = "public/learn-assets/transfer-learning/lora-mechanism-bridge.py"
activations = module("activation_mechanisms", activation_path)
lora = module("lora_mechanism_bridge", lora_path)
scores = np.linspace(-40, 40, 1601)
functions = {"sigmoid": torch.sigmoid, "tanh": torch.tanh, "relu": F.relu,
             "leaky_relu": lambda z: F.leaky_relu(z, .1), "elu": F.elu,
             "gelu": lambda z: F.gelu(z, approximate="none"),
             "gelu_tanh": lambda z: F.gelu(z, approximate="tanh"),
             "silu": F.silu, "mish": F.mish}
for name, function in functions.items():
    values, slopes = activations.activation(name, scores)
    tensor = torch.tensor(scores, dtype=torch.float64, requires_grad=True)
    result = function(tensor)
    derivative = torch.autograd.grad(result.sum(), tensor)[0]
    close(f"{name}: 1601 forward values", values, result.detach().numpy())
    close(f"{name}: 1601 local slopes including zero convention", slopes, derivative.numpy())
    shaped = np.array([[-2., 0., 2.], [3., -1., .5]])
    shaped_values = activations.activation(name, shaped)[0]
    close(f"{name}: shape-preserving element permutation", shaped_values.T,
          activations.activation(name, shaped.T)[0])
close("sigmoid: large finite inputs avoid exponential overflow",
      activations.sigmoid(np.array([-1000., 0., 1000.])), [0., .5, 1.])
assert activations.activation("gelu", np.array([-10.]))[0][0] < 0
records.append("erfc retains the small negative GELU tail at -10")
assert activations.activation("sigmoid", np.array([40.]))[1][0] > 0
records.append("stable sigmoid slope retains its positive tail at +40")

rng = np.random.default_rng(71)
x, target = rng.normal(size=(4, 3)), rng.normal(size=(4, 5))
weight, a, b = rng.normal(size=(5, 3)), rng.normal(size=(2, 3)), rng.normal(size=(5, 2))
scale = .7
output, loss, ga, gb, gx = lora.forward_and_gradients(x, target, weight, a, b, scale)
tx, ta, tb = [torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in (x, a, b)]
prediction = F.linear(tx, torch.from_numpy(weight)) + scale * F.linear(F.linear(tx, ta), tb)
objective = F.mse_loss(prediction, torch.from_numpy(target)); objective.backward()
close("LoRA changed rectangular batch: forward", output, prediction.detach().numpy())
close("LoRA changed rectangular batch: mean objective", loss, objective.item())
for name, manual, reference in [("A", ga, ta.grad), ("B", gb, tb.grad), ("input", gx, tx.grad)]:
    close(f"LoRA changed rectangular batch: {name} pullback", manual, reference.numpy())
for name, array, analytic in [("A", a, ga), ("B", b, gb), ("input", x, gx)]:
    numerical = np.empty_like(array)
    for index in np.ndindex(array.shape):
        original = array[index]
        array[index] = original + 1e-6
        plus = lora.forward_and_gradients(x, target, weight, a, b, scale)[1]
        array[index] = original - 1e-6
        minus = lora.forward_and_gradients(x, target, weight, a, b, scale)[1]
        array[index] = original
        numerical[index] = (plus - minus) / 2e-6
    close(f"LoRA {name}: every entry central difference on changed fixture", analytic, numerical, 1e-8)
duplicated = lora.forward_and_gradients(np.tile(x, (2, 1)), np.tile(target, (2, 1)), weight, a, b, scale)
close("LoRA mean reduction: duplicated batch preserves parameter gradient A", duplicated[2], ga)
close("LoRA mean reduction: duplicated batch preserves parameter gradient B", duplicated[3], gb)
close("LoRA mean reduction: each duplicated input receives half the derivative", duplicated[4][:len(x)], gx / 2)
rescaled = lora.forward_and_gradients(x, target, weight, 3 * a, b / 3, scale)
close("LoRA reciprocal factor scaling preserves the function", rescaled[0], output)
close("LoRA reciprocal factor scaling changes the A gradient inversely", rescaled[2], ga / 3)
close("LoRA reciprocal factor scaling changes the B gradient proportionally", rescaled[3], 3 * gb)
for invalid_target in (target[:1], target[:, :1]):
    try:
        lora.forward_and_gradients(x, invalid_target, weight, a, b, scale)
    except ValueError:
        pass
    else:
        raise AssertionError("A broadcastable but differently shaped target was accepted")
records.append("LoRA rejects accidental target broadcasting before forming its objective")

examples = {}
for name, path in [("perceptron", activation_path), ("backprop", backprop_path), ("transfer", lora_path)]:
    run = subprocess.run([sys.executable, str(ROOT / path)], check=True, capture_output=True, text=True)
    stdout = run.stdout.strip()
    errors = [float(value) for value in re.findall(r"(?:\w*_)?max_error ([\deE+.-]+)", stdout)]
    assert errors and max(errors) < 1e-11, (name, errors)
    if name == "perceptron":
        assert "AND weights [[3.0, 2.0]] bias [-4.0] predictions [-1, -1, -1, 1]" in stdout
        assert "XOR weights [[0.0, 0.0]] bias [0.0] predictions [-1, -1, -1, -1]" in stdout
    elif name == "backprop":
        assert "after_loss 0.774784850062 0.774784850062" in stdout
        assert "after_loss 0.840015027835 0.840015027835" in stdout
    else:
        assert "zero_B loss 2.5 after 2.025" in stdout
        assert "both_zero loss 2.5 after 2.5" in stdout
        assert "null_measurement loss 1.0 after 1.0" in stdout
        assert "zero_rate loss 2.5 after 2.5" in stdout
        assert "two_rows_nonzero_B loss 2.375 after 1.517758789" in stdout
        assert stdout.count("base_has_gradient False base_max_change 0.0") == 5
    records.append(f"{name}: complete program executed, parity errors and declared controls passed")
    examples[name] = {"source": "/" + path.removeprefix("public/"), "output": stdout}

sources = [activation_path, backprop_path, lora_path, "public/learn-assets/backpropagation/teaching-autodiff.py"]
evidence = {"status": "passed", "checks": records, "checkGroups": len(records),
            "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__, "sklearn": sklearn.__version__, "scipy": scipy.__version__},
            "sourceHashes": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in sources},
            "examples": examples}
(ROOT / "docs/teaching/evidence/neuron-implementation-depth.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
metadata_files = {"perceptron": "perceptron-mechanism-program.js",
                  "backprop": "backprop-mechanism-program.js",
                  "transfer": "transfer-learning-mechanism-program.js"}
for name, filename in metadata_files.items():
    (ROOT / "src/learn/data" / filename).write_text(
        "// Executed by scripts/verify-neuron-implementation-depth.py.\nexport default "
        + json.dumps(examples[name], indent=2) + ";\n", encoding="utf-8")
print(json.dumps({"status": "passed", "groups": len(records), "programs": len(examples), "denseActivationPoints": len(scores)}))
