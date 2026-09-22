"""Run the complete published teaching programs in the existing Python runtime.

No installs, no network, no mutation of the prepared packet. Outputs and hashes
are retained under docs/teaching/evidence/backprop-native.json.
"""
from pathlib import Path
import contextlib
import hashlib
import io
import json
import math
import os
import runpy
import shutil
import sys
import tempfile
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.dont_write_bytecode = True
import numpy as np
import sklearn
import torch

torch.set_num_threads(1)
root = Path(__file__).resolve().parents[1]
assets = root / "public/learn-assets/backpropagation"
record = {"versions": {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__, "sklearn": sklearn.__version__}, "programs": {}, "assertions": 0}


def check(condition, message):
    assert condition, message
    record["assertions"] += 1


def execute(filename):
    output = io.StringIO()
    with contextlib.redirect_stdout(output), warnings.catch_warnings(record=True) as emitted:
        namespace = runpy.run_path(str(filename), run_name="__main__")
    return namespace, output.getvalue(), [str(warning.message) for warning in emitted]


for name in json.loads((assets / "programs.json").read_text()):
    namespace, output, emitted = execute(assets / name)
    record["programs"][name] = {"stdout": output, "warnings": emitted}
    if name == "finite-differences.py":
        rows = [line.split() for line in output.splitlines()]
        check(float(rows[2][2]) == 0, "Offset cancellation is reproduced")
        check(float(rows[2][1]) < float(rows[3][1]), "Sine worsens at smaller perturbation")
    elif name == "pytorch-accumulation.py":
        check([float(line) for line in output.splitlines()] == [6, 12, 6], "Accumulation/reset outputs")
    elif name == "directional-products.py":
        check(torch.allclose(namespace["tangent"], torch.tensor([1.3, math.cos(.3), 2.8], dtype=torch.float64)), "JVP coordinates")
        check(torch.allclose(namespace["sensitivity"], torch.tensor([.7-math.cos(.3), 3.1], dtype=torch.float64)), "VJP coordinates")
    elif name == "hessian-vector.py":
        check(torch.equal(namespace["hessian_vector"], torch.tensor([4., 13.], dtype=torch.float64)), "Nondiagonal HVP")
    elif name == "custom-hard-sigmoid.py":
        check(output.startswith("True\n"), "Hard sigmoid gradcheck")
        check(torch.equal(namespace["x"].grad, torch.tensor([0., .2, .2, .2, 0.], dtype=torch.float64)), "Hard sigmoid gradient")
    elif name == "microbatch-means.py":
        results = {line.split()[0]: float(line.split()[1]) for line in output.splitlines()}
        check(abs(results["full"] + 44) < 1e-12 and abs(results["weighted"] + 44) < 1e-12, "Weighted mean gradients agree")
        check(abs(results["unweighted"] + 230/3) < 1e-12, "Unweighted means fail")
    elif name == "activation-checkpoint.py":
        check(output.strip() == "True", "Checkpoint gradient equivalence")

_, engine_output, engine_warnings = execute(assets / "teaching-autodiff.py")
record["programs"]["teaching-autodiff.py"] = {"stdout": engine_output, "warnings": engine_warnings}
saved = json.loads((assets / "calculated-inputs.json").read_text())
traces = {line.split(" ", 1)[0]: json.loads(line.split(" ", 1)[1]) for line in engine_output.splitlines()}
for actual, expected in zip(traces["XOR"], saved["xorTraining"]):
    check(actual["step"] == expected["step"] and np.isclose(actual["mse"], expected["mse"], rtol=1e-8, atol=1e-28), "XOR recorded trace")
    check(np.allclose(actual["outputs"], expected["outputs"], rtol=1e-8, atol=1e-12), "XOR recorded outputs")
for actual, expected in zip(traces["Digits"], saved["digitTraining"]):
    check(actual["validationCorrect"] == expected["validationCorrect"], "Digit validation count")
    check(np.isclose(actual["trainLoss"], expected["trainLoss"], rtol=1e-8, atol=1e-12), "Digit training loss")

# The authored calculation script writes beside itself: replay an isolated copy.
with tempfile.TemporaryDirectory(prefix="backprop-native-") as working:
    directory = Path(working)
    for name in ["author-calculations.py", "teaching-autodiff.py", "digits-400.csv"]:
        shutil.copyfile(assets / name, directory / name)
    _, author_output, author_warnings = execute(directory / "author-calculations.py")
    actual = json.loads((directory / "calculated-inputs.json").read_text())
    record["programs"]["author-calculations.py"] = {"stdout": author_output, "warnings": author_warnings}
    for case in actual["primitiveChecks"]:
        check(case["maxAbsError"] < 1e-12, case["case"])
    for case in actual["networkGradientChecks"]:
        check(case["maxTorchAbsError"] < 1e-12, "Network versus PyTorch")
        check(case["maxFiniteDifferenceAbsError"] < 1.2e-11, "Network central difference")
    record["networkGradientChecks"] = actual["networkGradientChecks"]
    check(actual["finiteDifferences"][4]["offsetDerivative"] == 0, "Author offset cancellation")

check(hashlib.sha256((assets / "digits-400.csv").read_bytes()).hexdigest() == "a5b50ff0418e2b470140153a399c9200b2bba68468232507fd393ba89c4fc672", "Licensed input byte identity")
record["sourceHashes"] = {str(file.relative_to(root)).replace("\\", "/"): hashlib.sha256(file.read_bytes()).hexdigest() for file in sorted(assets.iterdir()) if file.is_file()}
record["sourceHashes"]["scripts/verify-backprop-native.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
destination = root / "docs/teaching/evidence/backprop-native.json"
destination.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(f"Backprop native: {len(record['programs'])} complete programs, {record['assertions']} assertions passed.")
