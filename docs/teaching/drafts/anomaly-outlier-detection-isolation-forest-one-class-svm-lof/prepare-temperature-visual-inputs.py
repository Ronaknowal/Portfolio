"""Supply real point-level data for the written visual specification.

This content-stage probe executes the manuscript's fixed temperature calculation
once to retain its scores, rather than inventing a curve from aggregate counts.
It is not a runtime implementation or formal verification suite.
"""
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import io
import json
import re

import numpy as np
from threadpoolctl import threadpool_limits

base = Path(__file__).resolve().parent
lesson = (base / "lesson.md").read_text(encoding="utf-8")
program = re.findall(r"~~~python\n(.*?)\n~~~", lesson, re.S)[4]
namespace = {"__file__": str(base / "temperature_monitor.py")}
captured = io.StringIO()
with threadpool_limits(limits=1), redirect_stdout(captured):
    exec(compile(program, "temperature_monitor.py", "exec"), namespace)

features = namespace["features"]
fit, cal, test = (namespace[name] for name in ["fit", "cal", "test"])
table = features.copy()
table["period"] = np.where(fit, "reference", np.where(cal, "calibration", "test"))
for name, (cal_score, test_score) in namespace["scores"].items():
    key = {"Baseline": "baseline", "Isolation Forest": "isolation",
           "One-Class SVM": "one_class_svm", "LOF novelty": "lof"}[name]
    table[key] = np.nan
    table.loc[cal, key] = cal_score
    table.loc[test, key] = test_score
table["window"] = 0
for index, (start, end) in enumerate(namespace["windows"], 1):
    table.loc[(table.index >= start) & (table.index <= end), "window"] = index
csv_path = base / "nab-derived-scores.csv"
table.to_csv(csv_path, index_label="timestamp", float_format="%.17g")

hidden_cases = []
for name, (cal_score, test_score) in namespace["scores"].items():
    threshold = float(np.quantile(cal_score, .975, method="higher"))
    alert = test_score > threshold
    hidden_cases.append({
        "method": name, "q": .975, "threshold": threshold,
        "calibrationAlerts": int((cal_score > threshold).sum()),
        "testAlerts": int(alert.sum()),
        "outsideWindows": int((alert & ~namespace["inside"]).sum()),
        "windowsHit": sum(bool((alert & m).any()) for m in namespace["window_masks"]),
    })
record = {
    "createdAt": datetime.now(timezone.utc).isoformat(),
    "scope": "One fixed author calculation supplying real visual-input scores and an unpublished-to-learner contrast. No runtime/browser or independent verification.",
    "programSHA256": hashlib.sha256(program.encode()).hexdigest(),
    "stdout": captured.getvalue(),
    "sourceCalculation": "The complete temperature program in lesson.md, with unchanged settings.",
    "csvSHA256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
    "rows": len(table),
    "columns": list(table.columns),
    "missingScorePolicy": "Reference rows deliberately have blank scores; calibration/test rows are scored as new observations.",
    "q975AuthorOnlyFixtures": hidden_cases,
}
(base / "visual-input-calculation.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
