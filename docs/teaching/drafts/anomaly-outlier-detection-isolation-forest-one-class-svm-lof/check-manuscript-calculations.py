"""Content-stage arithmetic spot checks, not a phase-two verification suite."""
from contextlib import redirect_stdout
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
import ast
import hashlib
import io
import json
import math
import re

base = Path(__file__).resolve().parent
manuscript = (base / "lesson.md").read_text(encoding="utf-8")
programs = re.findall(r"~~~python\n(.*?)\n~~~", manuscript, re.S)
small_runs = []
for i, code in enumerate(programs):
    ast.parse(code)
    if i == 4:
        continue  # Real-data arithmetic was already probed; do not refit unchanged.
    captured = io.StringIO()
    with redirect_stdout(captured):
        exec(compile(code, f"lesson-example-{i + 1}", "exec"), {})
    small_runs.append({
        "example": i + 1,
        "codeSHA256": hashlib.sha256(code.encode()).hexdigest(),
        "stdout": captured.getvalue(),
    })

from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location("calculations", base / "author-calculations.py")
probe = module_from_spec(spec)
spec.loader.exec_module(probe)
q4 = float(probe.score_new([0, 1, 2, 20, 24, 28], 2, [4])[0])
assert math.isclose(q4, float(Fraction(35, 24)), rel_tol=1e-14)
score = 2 ** (-Fraction(18, 13))
checks = {
    "reachableLeafState": {"reference": [0, 1, 2, 3], "cuts": [0.5, 1.5],
        "query": 2, "leafMembers": [2, 3], "depth": 2, "correctedPath": 3,
        "normalizer": "13/6", "score": score},
    "query4LOF": q4,
    "practiceF": {"trueAlerts": 90, "falseAlertsExpected": 249.5,
                  "totalAlertsExpected": 339.5, "precision": 90 / 339.5},
    "practiceHWorkloadRatio": 7913 / 445,
}
result = {
    "checkedAt": datetime.now(timezone.utc).isoformat(),
    "scope": "Content-stage elementary calculations, four compact teaching programs and syntax of the real-data program. No formal runtime/model/browser verification.",
    "smallProgramRuns": small_runs,
    "arithmetic": checks,
    "realData": "Existing author-calculations.json reused. The presentation program was syntax-checked but not rerun; full displayed-program verification remains phase two.",
}
(base / "manuscript-calculations.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
