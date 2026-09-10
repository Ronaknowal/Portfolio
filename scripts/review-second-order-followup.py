"""Verify the author's scoped changed-data practice addition independently."""
import ast
import contextlib
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import subprocess

import numpy as np

root = Path(__file__).resolve().parents[1]
example = json.loads(subprocess.check_output([
    "node", "--input-type=module", "-e",
    "import {secondOrderMethodsExamples as e} from './src/learn/data/second-order-methods-examples.js'; console.log(JSON.stringify(e.changedFitComparison));"
], cwd=root, text=True, encoding="utf-8"))
tree = ast.parse(example["code"])
method_loop = next(node for node in tree.body if isinstance(node, ast.For))
method_loop.body.extend(ast.parse("captured.append((method, parameters.detach().numpy().copy(), gradient_evaluations, final_loss.item(), final_gradient.detach().numpy().copy()))").body)
namespace = {"captured": []}
output = io.StringIO()
with contextlib.redirect_stdout(output):
    exec(compile(ast.fix_missing_locations(tree), "authored-changed-data-program", "exec"), namespace)
assert output.getvalue().strip() == example["expected"]
times = np.linspace(0, 6, 21)
observations = 1.5 * np.exp(-0.4 * times) + 0.2 + 0.01 * np.sin(3 * times)
results = []
for method, parameters, evaluations, actual_loss, actual_gradient in namespace["captured"]:
    amplitude, rate = np.exp(parameters[:2])
    decay = amplitude * np.exp(-rate * times)
    residual = decay + parameters[2] - observations
    jacobian = np.column_stack((decay, -rate * times * decay, np.ones_like(times)))
    expected_gradient = 2 * jacobian.T @ residual / len(times)
    expected_loss = np.mean(residual ** 2)
    assert np.allclose(actual_gradient, expected_gradient, atol=1e-14, rtol=1e-12)
    assert abs(actual_loss - expected_loss) < 1e-15
    results.append({"method": method, "evaluations_including_final_check": evaluations,
                    "numpy_mse": expected_loss, "numpy_gradient_infinity_norm": float(np.max(np.abs(expected_gradient))),
                    "gradient_error": float(np.max(np.abs(actual_gradient - expected_gradient)))})
record = {"status": "passed", "reviewed_at": datetime.now(timezone.utc).isoformat(),
          "exact_authored_stdout": True, "independent_analytic_numpy_checks": results}
destination = root / "docs/teaching/evidence/second-order-changed-comparison-independent.json"
destination.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(json.dumps(record, indent=2))
