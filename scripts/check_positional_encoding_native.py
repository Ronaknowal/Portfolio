"""Verify current programs and retained models without rewriting saved studies."""
import importlib.util
import json
from pathlib import Path
import platform
import re
import runpy

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
ID = "positional-encodings-sinusoidal-learned-rope-alibi"
PACKET = ROOT / "docs/teaching/drafts" / ID
OUT = ROOT / "docs/teaching/deep-learning-completion" / ID
checks = []
torch.set_num_threads(2)
spec = importlib.util.spec_from_file_location("position_reference", PACKET / "author-calculations.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
programs = re.findall(r"```python\n(.*?)\n```", (PACKET / "lesson.md").read_text(encoding="utf-8"), re.S)
assert len(programs) == 1
namespace = {"__name__": "__main__"}
exec(compile(programs[0], "complete-lesson-program", "exec"), namespace)
checks.append({"name": "Complete displayed NumPy position/cache program", "passed": True})
runpy.run_path(str(PACKET / "position_library_bridge.py"), run_name="__main__")
checks.append({"name": "SDPA direct/library output and input-gradient parity for RoPE/ALiBi; repeated Embedding gradients", "passed": True})
points, labels, splits, contract = module.load_data()
recorded = json.loads((PACKET / "author-results.json").read_text())
assert contract == recorded["data"]
checks.append({"name": "Original data hash, duplicate groups and all split rows", "passed": True})
for mode, saved in json.loads((PACKET / "position-models.json").read_text()).items():
    model = module.PositionClassifier(mode)
    model.load_state_dict({name: torch.tensor(value) for name, value in saved["state_dict"].items()})
    actual = module.score(model, points[splits["test"]], labels[splits["test"]])
    expected = next(row for row in recorded["fits"] if row["mode"] == mode)["test"]
    assert actual["correct"] == expected["correct"]
    assert abs(actual["macro_f1"] - expected["macro_f1"]) < 1e-12
    checks.append({"name": f"{mode} checkpoint on all 60 retained test records", "passed": True, "actual": actual})
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "native-checks.json").write_text(json.dumps({"topicId": ID, "passed": True, "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__}, "checks": checks, "limits": ["All five historical fits retained unchanged; actual display checkpoints reevaluated, no new tuning.", "No large language-model context-extension experiment claimed."]}, indent=2) + "\n")
