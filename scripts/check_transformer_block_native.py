"""Run current teaching code and retained-state checks without overwriting history."""
import importlib.util
import json
from pathlib import Path
import re
import platform

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "docs/teaching/drafts/transformer-block-architecture"
OUTPUT = ROOT / "docs/teaching/deep-learning-completion/transformer-block-architecture"
torch.set_num_threads(2)
spec = importlib.util.spec_from_file_location("block_reference", PACKET / "author-calculations.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
checks = []
source = (PACKET / "lesson.md").read_text(encoding="utf-8")
programs = re.findall(r"```python\n(.*?)\n```", source, re.S)
assert len(programs) == 3
for index, program in enumerate(programs):
    exec(compile(program, f"lesson-program-{index + 1}", "exec"), {"__name__": "__main__"})
    checks.append({"name": f"Complete displayed Python program {index + 1}", "passed": True})
fixtures = module.exact_fixtures()
assert fixtures["reference_max_abs_errors"]["True"] < 1e-12
assert fixtures["reference_max_abs_errors"]["False"] < 1e-12
checks.append({"name": "Both block placements against installed TransformerEncoderLayer", "passed": True, "maxErrors": fixtures["reference_max_abs_errors"]})
models = json.loads((PACKET / "block-models.json").read_text())
points, labels, splits, contract = module.load_data()
recorded = json.loads((PACKET / "author-results.json").read_text())
assert contract == recorded["data"]
checks.append({"name": "Original data, duplicate groups and split identity", "passed": True})
for actual, saved in zip(fixtures["gradient_depth"], recorded["fixtures"]["gradient_depth"], strict=True):
    for key in ("state_gradient_l2", "state_rms"):
        np.testing.assert_allclose(actual[key], saved[key], atol=1e-10, rtol=1e-10)
checks.append({"name": "All saved initialization gradient-depth and carried-state measurements", "passed": True})
for placement, saved in models.items():
    model = module.MovementClassifier(placement == "pre-norm")
    model.load_state_dict({key: torch.tensor(value) for key, value in saved["state_dict"].items()})
    actual = module.score(model, points[splits["test"]], labels[splits["test"]])
    prior = next(row for row in recorded["fits"] if row["placement"] == placement and row["seed"] == 101)["test"]
    assert actual["correct"] == prior["correct"]
    assert abs(actual["macro_f1"] - prior["macro_f1"]) < 1e-12
    checks.append({"name": f"{placement} retained checkpoint on all 60 test records", "passed": True, "actual": actual})
OUTPUT.mkdir(parents=True, exist_ok=True)
(OUTPUT / "native-checks.json").write_text(json.dumps({"topicId": "transformer-block-architecture", "passed": True, "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__}, "checks": checks, "limits": ["Historical six-fit training histories retained unchanged; no redundant refit.", "Both actual display checkpoints re-evaluated; all three current displayed programs executed including gradients and equal updates."]}, indent=2) + "\n")
